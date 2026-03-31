use burn::module::Module;
use burn::prelude::*;

use crate::model::backbone::DarkNet;
use crate::model::head::Head;
use crate::model::neck::DarkFPN;

#[derive(Module, Debug)]
pub struct YOLO<B: Backend> {
    pub net: DarkNet<B>,
    pub fpn: DarkFPN<B>,
    pub head: Head<B>,
    pub stride: Tensor<B, 1>,
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
        let stride = Tensor::<B, 1>::from_floats(
            [s3 as f32, s4 as f32, s5 as f32].as_slice(),
            device,
        );

        let filters = [width[3], width[4], width[5]];
        let head = Head::new(num_classes, &filters, stride.clone(), device);

        Self { net, fpn, head, stride }
    }

    /// Forward for training — returns per-scale feature outputs.
    pub fn forward_train(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let (p3, p4, p5) = self.net.forward(x);
        let (p3, p4, p5) = self.fpn.forward(p3, p4, p5);
        self.head.forward_train(&[p3, p4, p5])
    }

    /// Forward for inference — returns decoded [B, 4+nc, total_anchors].
    pub fn forward_infer(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let (p3, p4, p5) = self.net.forward(x);
        let (p3, p4, p5) = self.fpn.forward(p3, p4, p5);
        self.head.forward_infer(&[p3, p4, p5])
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
