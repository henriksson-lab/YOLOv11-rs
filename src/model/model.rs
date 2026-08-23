use burn::module::Module;
use burn::prelude::*;

use crate::model::backbone::DarkNet;
use crate::model::conv::FuseModule;
use crate::model::head::{Head, HeadOutput};
use crate::model::neck::DarkFPN;

#[derive(Module, Debug)]
pub struct YOLO {
    net: DarkNet,
    fpn: DarkFPN,
    head: Head,
    pub stride: Tensor<1>,
}

#[derive(Debug)]
pub enum YOLOOutput {
    Train(Vec<Tensor<4>>),
    Infer(Tensor<3>),
}

impl YOLO {
    pub fn new(
        width: &[usize],
        depth: &[usize],
        csp: &[bool],
        num_classes: usize,
        device: &Device,
    ) -> Self {
        let net = DarkNet::new(width, depth, csp, device);
        let fpn = DarkFPN::new(width, depth, csp, device);

        // Compute strides via dummy forward pass
        let dummy = Tensor::<4>::zeros([1, width[0], 256, 256], device);
        let (p3, p4, p5) = net.forward(dummy);
        let (p3, p4, p5) = fpn.forward(p3, p4, p5);

        let s3 = 256.0 / p3.dims()[3] as f64;
        let s4 = 256.0 / p4.dims()[3] as f64;
        let s5 = 256.0 / p5.dims()[3] as f64;
        let stride = Tensor::<1>::from_floats([s3 as f32, s4 as f32, s5 as f32].as_slice(), device);

        let filters = [width[3], width[4], width[5]];
        let mut head = Head::new(num_classes, &filters, stride.clone(), device);
        head.initialize_biases();

        Self {
            net,
            fpn,
            head,
            stride,
        }
    }

    pub fn forward(&self, x: Tensor<4>, training: bool) -> YOLOOutput {
        let (p3, p4, p5) = self.net.forward(x);
        let (p3, p4, p5) = self.fpn.forward(p3, p4, p5);
        match self.head.forward(&[p3, p4, p5], training) {
            HeadOutput::Train(outputs) => YOLOOutput::Train(outputs),
            HeadOutput::Infer(output) => YOLOOutput::Infer(output),
        }
    }

    pub fn fuse(self) -> Self {
        Self {
            net: self.net.fuse_module(),
            fpn: self.fpn.fuse_module(),
            head: self.head.fuse_module(),
            stride: self.stride,
        }
    }
}

// ---------------------------------------------------------------------------
// Model variant constructors
// ---------------------------------------------------------------------------

pub fn yolo_v11_n(num_classes: usize, device: &Device) -> YOLO {
    let csp = [false, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 16, 32, 64, 128, 256];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_t(num_classes: usize, device: &Device) -> YOLO {
    let csp = [false, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 24, 48, 96, 192, 384];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_s(num_classes: usize, device: &Device) -> YOLO {
    let csp = [false, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 32, 64, 128, 256, 512];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_m(num_classes: usize, device: &Device) -> YOLO {
    let csp = [true, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 64, 128, 256, 512, 512];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_l(num_classes: usize, device: &Device) -> YOLO {
    let csp = [true, true];
    let depth = [2, 2, 2, 2, 2, 2];
    let width = [3, 64, 128, 256, 512, 512];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_x(num_classes: usize, device: &Device) -> YOLO {
    let csp = [true, true];
    let depth = [2, 2, 2, 2, 2, 2];
    let width = [3, 96, 192, 384, 768, 768];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

#[cfg(test)]
mod tests {
    use super::{yolo_v11_n, YOLOOutput};
    use burn::prelude::*;

    #[test]
    fn yolo_v11_n_forward_training_smoke_matches_three_scale_head() {
        let device = burn::tensor::Device::flex();
        let model = yolo_v11_n(2, &device);
        let x = Tensor::<4>::zeros([1, 3, 64, 64], &device);

        match model.forward(x, true) {
            YOLOOutput::Train(outputs) => {
                assert_eq!(outputs.len(), 3);
                assert_eq!(outputs[0].dims()[0], 1);
                assert_eq!(outputs[1].dims()[0], 1);
                assert_eq!(outputs[2].dims()[0], 1);
                assert_eq!(outputs[0].dims()[1], 66);
                assert_eq!(outputs[1].dims()[1], 66);
                assert_eq!(outputs[2].dims()[1], 66);
            }
            YOLOOutput::Infer(_) => panic!("training forward returned inference output"),
        }
    }

    #[test]
    fn yolo_fuse_preserves_training_forward_shape() {
        let device = burn::tensor::Device::flex();
        let model = yolo_v11_n(2, &device).fuse();
        let x = Tensor::<4>::zeros([1, 3, 64, 64], &device);

        match model.forward(x, true) {
            YOLOOutput::Train(outputs) => {
                assert_eq!(outputs.len(), 3);
                assert_eq!(outputs[0].dims()[1], 66);
                assert_eq!(outputs[1].dims()[1], 66);
                assert_eq!(outputs[2].dims()[1], 66);
            }
            YOLOOutput::Infer(_) => panic!("training forward returned inference output"),
        }
    }
}
