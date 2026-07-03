use burn::module::Module;
use burn::prelude::*;

use crate::model::backbone::DarkNet;
use crate::model::conv::FuseModule;
use crate::model::head::{Head, HeadOutput};
use crate::model::neck::DarkFPN;

#[derive(Module, Debug)]
pub struct YOLO<B: Backend> {
    net: DarkNet<B>,
    fpn: DarkFPN<B>,
    head: Head<B>,
    pub stride: Tensor<B, 1>,
}

#[derive(Debug)]
pub enum YOLOOutput<B: Backend> {
    Train(Vec<Tensor<B, 4>>),
    Infer(Tensor<B, 3>),
}

impl<B: Backend> YOLO<B> {
    pub fn new(
        width: &[usize],
        depth: &[usize],
        csp: &[bool],
        num_classes: usize,
        device: &B::Device,
    ) -> Self {
        let net = DarkNet::new(width, depth, csp, device);
        let fpn = DarkFPN::new(width, depth, csp, device);

        // Compute strides via dummy forward pass
        let dummy = Tensor::<B, 4>::zeros([1, width[0], 256, 256], device);
        let (p3, p4, p5) = net.forward(dummy);
        let (p3, p4, p5) = fpn.forward(p3, p4, p5);

        let s3 = 256.0 / p3.dims()[3] as f64;
        let s4 = 256.0 / p4.dims()[3] as f64;
        let s5 = 256.0 / p5.dims()[3] as f64;
        let stride =
            Tensor::<B, 1>::from_floats([s3 as f32, s4 as f32, s5 as f32].as_slice(), device);

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

    pub fn forward(&self, x: Tensor<B, 4>, training: bool) -> YOLOOutput<B> {
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

pub fn yolo_v11_n<B: Backend>(num_classes: usize, device: &B::Device) -> YOLO<B> {
    let csp = [false, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 16, 32, 64, 128, 256];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_t<B: Backend>(num_classes: usize, device: &B::Device) -> YOLO<B> {
    let csp = [false, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 24, 48, 96, 192, 384];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_s<B: Backend>(num_classes: usize, device: &B::Device) -> YOLO<B> {
    let csp = [false, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 32, 64, 128, 256, 512];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_m<B: Backend>(num_classes: usize, device: &B::Device) -> YOLO<B> {
    let csp = [true, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 64, 128, 256, 512, 512];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_l<B: Backend>(num_classes: usize, device: &B::Device) -> YOLO<B> {
    let csp = [true, true];
    let depth = [2, 2, 2, 2, 2, 2];
    let width = [3, 64, 128, 256, 512, 512];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

pub fn yolo_v11_x<B: Backend>(num_classes: usize, device: &B::Device) -> YOLO<B> {
    let csp = [true, true];
    let depth = [2, 2, 2, 2, 2, 2];
    let width = [3, 96, 192, 384, 768, 768];
    YOLO::new(&width, &depth, &csp, num_classes, device)
}

#[cfg(test)]
mod tests {
    use super::{yolo_v11_n, YOLOOutput};
    use burn::backend::NdArray;
    use burn::prelude::*;

    type TestBackend = NdArray;

    #[test]
    fn yolo_v11_n_forward_training_smoke_matches_three_scale_head() {
        let device = Default::default();
        let model = yolo_v11_n::<TestBackend>(2, &device);
        let x = Tensor::<TestBackend, 4>::zeros([1, 3, 64, 64], &device);

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
        let device = Default::default();
        let model = yolo_v11_n::<TestBackend>(2, &device).fuse();
        let x = Tensor::<TestBackend, 4>::zeros([1, 3, 64, 64], &device);

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
