use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::backbone::DarkNet;
use crate::head::Head;
use crate::neck::DarkFPN;

pub struct YOLO {
    pub net: DarkNet,
    pub fpn: DarkFPN,
    pub head: Head,
    pub stride: Tensor,
}

impl YOLO {
    pub fn new(
        width: &[usize],
        depth: &[usize],
        csp: &[bool],
        num_classes: usize,
        device: &Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let net = DarkNet::new(width, depth, csp, vb.pp("net"))?;
        let fpn = DarkFPN::new(width, depth, csp, vb.pp("fpn"))?;

        // Compute strides via dummy forward pass
        let dummy = Tensor::zeros((1, width[0], 256, 256), DType::F32, device)?;
        let (p3, p4, p5) = net.forward(&dummy, false)?;
        let (p3, p4, p5) = fpn.forward(&p3, &p4, &p5, false)?;

        let s3 = 256.0 / p3.dims()[3] as f64;
        let s4 = 256.0 / p4.dims()[3] as f64;
        let s5 = 256.0 / p5.dims()[3] as f64;
        let stride = Tensor::from_vec(vec![s3 as f32, s4 as f32, s5 as f32], 3, device)?;

        let filters = [width[3], width[4], width[5]];
        let head = Head::new(num_classes, &filters, stride.clone(), device, vb.pp("head"))?;

        Ok(Self { net, fpn, head, stride })
    }

    /// Forward for training — returns per-scale feature outputs.
    pub fn forward_train(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let (p3, p4, p5) = self.net.forward(x, true)?;
        let (p3, p4, p5) = self.fpn.forward(&p3, &p4, &p5, true)?;
        self.head.forward_train(&[p3, p4, p5], true)
    }

    /// Forward for inference — returns decoded [B, 4+nc, total_anchors].
    pub fn forward_infer(&self, x: &Tensor) -> Result<Tensor> {
        let (p3, p4, p5) = self.net.forward(x, false)?;
        let (p3, p4, p5) = self.fpn.forward(&p3, &p4, &p5, false)?;
        self.head.forward_infer(&[p3, p4, p5], false)
    }
}

// ---------------------------------------------------------------------------
// Model variant constructors
// ---------------------------------------------------------------------------

pub fn yolo_v11_n(num_classes: usize, device: &Device, vb: VarBuilder) -> Result<YOLO> {
    let csp = [false, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 16, 32, 64, 128, 256];
    YOLO::new(&width, &depth, &csp, num_classes, device, vb)
}

pub fn yolo_v11_t(num_classes: usize, device: &Device, vb: VarBuilder) -> Result<YOLO> {
    let csp = [false, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 24, 48, 96, 192, 384];
    YOLO::new(&width, &depth, &csp, num_classes, device, vb)
}

pub fn yolo_v11_s(num_classes: usize, device: &Device, vb: VarBuilder) -> Result<YOLO> {
    let csp = [false, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 32, 64, 128, 256, 512];
    YOLO::new(&width, &depth, &csp, num_classes, device, vb)
}

pub fn yolo_v11_m(num_classes: usize, device: &Device, vb: VarBuilder) -> Result<YOLO> {
    let csp = [true, true];
    let depth = [1, 1, 1, 1, 1, 1];
    let width = [3, 64, 128, 256, 512, 512];
    YOLO::new(&width, &depth, &csp, num_classes, device, vb)
}

pub fn yolo_v11_l(num_classes: usize, device: &Device, vb: VarBuilder) -> Result<YOLO> {
    let csp = [true, true];
    let depth = [2, 2, 2, 2, 2, 2];
    let width = [3, 64, 128, 256, 512, 512];
    YOLO::new(&width, &depth, &csp, num_classes, device, vb)
}

pub fn yolo_v11_x(num_classes: usize, device: &Device, vb: VarBuilder) -> Result<YOLO> {
    let csp = [true, true];
    let depth = [2, 2, 2, 2, 2, 2];
    let width = [3, 96, 192, 384, 768, 768];
    YOLO::new(&width, &depth, &csp, num_classes, device, vb)
}
