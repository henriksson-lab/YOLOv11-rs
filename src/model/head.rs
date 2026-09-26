use burn::module::Module;
use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, softmax};

use crate::model::anchors::make_anchors;
use crate::model::conv::{Activation, Conv, FuseModule};

// ---------------------------------------------------------------------------
// DFL  (Distribution Focal Loss decoder)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct DFL {
    /// Fixed weight tensor [1, ch, 1, 1] containing [0, 1, 2, ..., ch-1].
    weight: Tensor<4>,
}

impl DFL {
    pub fn new(ch: usize, device: &Device) -> Self {
        let w: Vec<f32> = (0..ch).map(|i| i as f32).collect();
        let weight = Tensor::<1>::from_floats(w.as_slice(), device).reshape([1, ch, 1, 1]);
        Self { weight }
    }

    /// Input: [B, 4*ch, A].  Output: [B, 4, A].
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, _c, a] = x.dims();
        let ch = self.weight.dims()[1];
        let x = x.reshape([b, 4, ch, a]);
        let x = x.swap_dims(1, 2);
        let x = softmax(x, 1);
        let weight = self.weight.clone().reshape([1, ch, 1, 1]);
        let x = x * weight;
        x.sum_dim(1).reshape([b, 4, a])
    }
}

// ---------------------------------------------------------------------------
// Head  (decoupled detection head)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct Head {
    nc: usize,
    nl: usize,
    ch: usize,
    stride: Tensor<1>,
    dfl: DFL,
    box_c0: Vec<Conv>,
    box_c1: Vec<Conv>,
    box_c2: Vec<Conv2d>,
    cls_c0: Vec<Conv>,
    cls_c1: Vec<Conv>,
    cls_c2: Vec<Conv>,
    cls_c3: Vec<Conv>,
    cls_c4: Vec<Conv2d>,
}

#[derive(Debug)]
pub enum HeadOutput {
    Train(Vec<Tensor<4>>),
    Infer(Tensor<3>),
}

impl Head {
    pub fn new(nc: usize, filters: &[usize], stride: Tensor<1>, device: &Device) -> Self {
        let ch: usize = 16;
        let nl = filters.len();
        let dfl = DFL::new(ch, device);

        let box_hidden = 64.max(filters[0] / 4);
        let cls_hidden = 80.max(filters[0]).max(nc);

        let mut box_c0 = Vec::with_capacity(nl);
        let mut box_c1 = Vec::with_capacity(nl);
        let mut box_c2 = Vec::with_capacity(nl);
        let mut cls_c0 = Vec::with_capacity(nl);
        let mut cls_c1 = Vec::with_capacity(nl);
        let mut cls_c2 = Vec::with_capacity(nl);
        let mut cls_c3 = Vec::with_capacity(nl);
        let mut cls_c4 = Vec::with_capacity(nl);

        for &f in filters {
            box_c0.push(Conv::new(
                f,
                box_hidden,
                Activation::SiLU,
                3,
                1,
                1,
                1,
                device,
            ));
            box_c1.push(Conv::new(
                box_hidden,
                box_hidden,
                Activation::SiLU,
                3,
                1,
                1,
                1,
                device,
            ));
            box_c2.push(Conv2dConfig::new([box_hidden, 4 * ch], [1, 1]).init(device));

            cls_c0.push(Conv::new(f, f, Activation::SiLU, 3, 1, 1, f, device));
            cls_c1.push(Conv::new(
                f,
                cls_hidden,
                Activation::SiLU,
                1,
                1,
                0,
                1,
                device,
            ));
            cls_c2.push(Conv::new(
                cls_hidden,
                cls_hidden,
                Activation::SiLU,
                3,
                1,
                1,
                cls_hidden,
                device,
            ));
            cls_c3.push(Conv::new(
                cls_hidden,
                cls_hidden,
                Activation::SiLU,
                1,
                1,
                0,
                1,
                device,
            ));
            cls_c4.push(Conv2dConfig::new([cls_hidden, nc], [1, 1]).init(device));
        }

        Self {
            nc,
            nl,
            ch,
            stride,
            dfl,
            box_c0,
            box_c1,
            box_c2,
            cls_c0,
            cls_c1,
            cls_c2,
            cls_c3,
            cls_c4,
        }
    }

    pub fn forward(&self, xs: &[Tensor<4>], training: bool) -> HeadOutput {
        let profile = std::env::var_os("YOLOV11_PROFILE").is_some();
        let device = xs[0].device();
        let branches_started = std::time::Instant::now();
        let mut outputs = Vec::with_capacity(self.nl);
        for (i, x) in xs.iter().enumerate() {
            let box_out = self.box_c0[i].forward(x.clone());
            let box_out = self.box_c1[i].forward(box_out);
            let box_out = self.box_c2[i].forward(box_out);

            let cls_out = self.cls_c0[i].forward(x.clone());
            let cls_out = self.cls_c1[i].forward(cls_out);
            let cls_out = self.cls_c2[i].forward(cls_out);
            let cls_out = self.cls_c3[i].forward(cls_out);
            let cls_out = self.cls_c4[i].forward(cls_out);

            let cat = Tensor::cat(vec![box_out, cls_out], 1);
            outputs.push(cat);
        }

        if profile {
            device.sync().expect("synchronizing YOLO head branches");
        }
        let branches_elapsed = branches_started.elapsed();

        if training {
            return HeadOutput::Train(outputs);
        }

        let nc = self.nc;
        let ch = self.ch;
        let no = nc + ch * 4;

        let anchors_started = std::time::Instant::now();
        let (anchors, strides) = make_anchors(&outputs, &self.stride, 0.5, &device);
        if profile {
            device.sync().expect("synchronizing YOLO anchors");
        }
        let anchors_elapsed = anchors_started.elapsed();
        let anchors = anchors.swap_dims(0, 1); // [2, total]
        let strides = strides.swap_dims(0, 1); // [1, total]
                                               // Add batch dim for broadcasting
        let anchors: Tensor<3> = anchors.unsqueeze_dim(0); // [1, 2, total]
        let strides: Tensor<3> = strides.unsqueeze_dim(0); // [1, 1, total]

        let arrange_started = std::time::Instant::now();
        let b = outputs[0].dims()[0];
        let mut flat: Vec<Tensor<3>> = Vec::new();
        for o in &outputs {
            let [_, _, h, w] = o.dims();
            flat.push(o.clone().reshape([b, no, h * w]));
        }
        let x = Tensor::cat(flat, 2);

        let box_ch = 4 * ch;
        let box_raw = x.clone().narrow(1, 0, box_ch);
        let cls_raw = x.narrow(1, box_ch, nc);
        if profile {
            device.sync().expect("synchronizing YOLO head arrangement");
        }
        let arrange_elapsed = arrange_started.elapsed();

        let decode_started = std::time::Instant::now();
        let dfl_out = self.dfl.forward(box_raw);
        let a_part = dfl_out.clone().narrow(1, 0, 2);
        let b_part = dfl_out.narrow(1, 2, 2);

        let a_decoded = anchors.clone() - a_part;
        let b_decoded = anchors + b_part;
        let center = (a_decoded.clone() + b_decoded.clone()) * 0.5;
        let size = b_decoded - a_decoded;
        let box_decoded = Tensor::cat(vec![center, size], 1);
        let box_scaled = box_decoded * strides;

        let cls_sigmoid = sigmoid(cls_raw);
        let output = Tensor::cat(vec![box_scaled, cls_sigmoid], 1);
        if profile {
            device.sync().expect("synchronizing YOLO head decode");
            eprintln!(
                "YOLOV11_HEAD_PROFILE branches_us={} anchors_us={} arrange_us={} decode_us={}",
                branches_elapsed.as_micros(),
                anchors_elapsed.as_micros(),
                arrange_elapsed.as_micros(),
                decode_started.elapsed().as_micros(),
            );
        }
        HeadOutput::Infer(output)
    }

    pub fn initialize_biases(&mut self) {
        let nc = self.nc;
        let ch = self.ch;
        let stride_values: Vec<f32> = self.stride.clone().to_data().try_to_vec::<f32>().unwrap();

        for i in 0..self.nl {
            let device = self.box_c2[i].weight.val().device();
            self.box_c2[i].bias = Some(Param::from_tensor(Tensor::<1>::full(
                [4 * ch],
                1.0,
                &device,
            )));

            let s = stride_values[i] as f64;
            let cls_bias = (5.0 / nc as f64 / (640.0 / s).powi(2)).ln() as f32;
            self.cls_c4[i].bias = Some(Param::from_tensor(Tensor::<1>::full(
                [nc],
                cls_bias,
                &device,
            )));
        }
    }
}

impl FuseModule for Head {
    fn fuse_module(self) -> Self {
        Self {
            nc: self.nc,
            nl: self.nl,
            ch: self.ch,
            stride: self.stride,
            dfl: self.dfl,
            box_c0: self
                .box_c0
                .into_iter()
                .map(FuseModule::fuse_module)
                .collect(),
            box_c1: self
                .box_c1
                .into_iter()
                .map(FuseModule::fuse_module)
                .collect(),
            box_c2: self.box_c2,
            cls_c0: self
                .cls_c0
                .into_iter()
                .map(FuseModule::fuse_module)
                .collect(),
            cls_c1: self
                .cls_c1
                .into_iter()
                .map(FuseModule::fuse_module)
                .collect(),
            cls_c2: self
                .cls_c2
                .into_iter()
                .map(FuseModule::fuse_module)
                .collect(),
            cls_c3: self
                .cls_c3
                .into_iter()
                .map(FuseModule::fuse_module)
                .collect(),
            cls_c4: self.cls_c4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Head, HeadOutput, DFL};
    use burn::prelude::*;

    #[test]
    fn dfl_forward_matches_python_projection_fixture() {
        let device = burn::tensor::Device::flex();
        let dfl = DFL::new(4, &device);
        let x = Tensor::<1>::from_floats(
            [
                0.2, 0.3, -0.1, 0.1, 0.4, -0.2, 1.0, 0.5, 1.2, 0.7, 0.3, -0.4, -0.7, 0.2, 0.0, 0.0,
                -0.5, -0.1, 0.8, 0.2, 0.1, 0.4, -1.0, -0.8, 0.0, 1.0, 0.0, 0.0, 0.0, -0.5, 0.0,
                0.2,
            ],
            &device,
        )
        .reshape([1, 16, 2]);

        let out = dfl.forward(x).to_data().try_to_vec::<f32>().unwrap();
        let expected = [
            1.9006745, 1.5619756, 0.86645204, 1.2461841, 1.286728, 1.3652573, 1.5, 1.0596901,
        ];

        for (actual, expected) in out.iter().zip(expected) {
            assert!((*actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn initialize_biases_sets_python_head_bias_values() {
        let device = burn::tensor::Device::flex();
        let stride = Tensor::<1>::from_floats([8.0, 16.0, 32.0], &device);
        let mut head = Head::new(3, &[16, 32, 64], stride, &device);

        head.initialize_biases();

        let box_bias = head.box_c2[0]
            .bias
            .as_ref()
            .unwrap()
            .val()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(box_bias.iter().all(|v| (*v - 1.0).abs() < 1e-6));

        let cls_bias = head.cls_c4[0]
            .bias
            .as_ref()
            .unwrap()
            .val()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        let expected = (5.0f32 / 3.0 / (640.0f32 / 8.0).powi(2)).ln();
        assert!(cls_bias.iter().all(|v| (*v - expected).abs() < 1e-6));
    }

    #[test]
    fn head_forward_training_and_inference_shapes_match_python_layout() {
        let device = burn::tensor::Device::flex();
        let stride = Tensor::<1>::from_floats([8.0, 16.0, 32.0], &device);
        let mut head = Head::new(2, &[8, 16, 32], stride, &device);
        head.initialize_biases();
        let xs = vec![
            Tensor::<4>::zeros([1, 8, 4, 4], &device),
            Tensor::<4>::zeros([1, 16, 2, 2], &device),
            Tensor::<4>::zeros([1, 32, 1, 1], &device),
        ];

        match head.forward(&xs, true) {
            HeadOutput::Train(outputs) => {
                assert_eq!(outputs.len(), 3);
                assert_eq!(outputs[0].dims(), [1, 66, 4, 4]);
                assert_eq!(outputs[1].dims(), [1, 66, 2, 2]);
                assert_eq!(outputs[2].dims(), [1, 66, 1, 1]);
            }
            HeadOutput::Infer(_) => panic!("training forward returned inference output"),
        }

        match head.forward(&xs, false) {
            HeadOutput::Infer(output) => assert_eq!(output.dims(), [1, 6, 21]),
            HeadOutput::Train(_) => panic!("inference forward returned training output"),
        }
    }
}
