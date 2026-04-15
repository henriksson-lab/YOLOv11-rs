use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, softmax};

use crate::model::anchors::make_anchors;
use crate::model::conv::ConvBn;

// ---------------------------------------------------------------------------
// DFL  (Distribution Focal Loss decoder)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct DFL<B: Backend> {
    /// Fixed weight tensor [1, ch, 1, 1] containing [0, 1, 2, ..., ch-1].
    weight: Tensor<B, 4>,
}

impl<B: Backend> DFL<B> {
    pub fn new(ch: usize, device: &B::Device) -> Self {
        let w: Vec<f32> = (0..ch).map(|i| i as f32).collect();
        let weight = Tensor::<B, 1>::from_floats(w.as_slice(), device).reshape([1, ch, 1, 1]);
        Self { weight }
    }

    pub fn ch(&self) -> usize {
        self.weight.dims()[1]
    }

    /// Input: [B, 4*ch, A].  Output: [B, 4, A].
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, _c, a] = x.dims();
        let ch = self.ch();
        let x = x.reshape([b, 4, ch, a]);
        let x = x.swap_dims(1, 2);
        let x = softmax(x, 1);
        let weight = self.weight.clone().reshape([1, ch, 1, 1]);
        let x = x * weight;
        x.sum_dim(1).squeeze::<3>()
    }
}

// ---------------------------------------------------------------------------
// Box and Cls head branches
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct BoxBranch<B: Backend> {
    c0: ConvBn<B>,
    c1: ConvBn<B>,
    c2: Conv2d<B>,
}

#[derive(Module, Debug)]
pub struct ClsBranch<B: Backend> {
    c0: ConvBn<B>,
    c1: ConvBn<B>,
    c2: ConvBn<B>,
    c3: ConvBn<B>,
    c4: Conv2d<B>,
}

// ---------------------------------------------------------------------------
// Head  (decoupled detection head)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct Head<B: Backend> {
    pub stride: Tensor<B, 1>,
    dfl: DFL<B>,
    box_branches: Vec<BoxBranch<B>>,
    cls_branches: Vec<ClsBranch<B>>,
}

impl<B: Backend> Head<B> {
    pub fn new(
        nc: usize,
        filters: &[usize],
        stride: Tensor<B, 1>,
        device: &B::Device,
    ) -> Self {
        let ch: usize = 16;
        let nl = filters.len();
        let dfl = DFL::new(ch, device);

        let box_hidden = 64.max(filters[0] / 4);
        let cls_hidden = 80.max(filters[0]).max(nc);

        let mut box_branches = Vec::with_capacity(nl);
        let mut cls_branches = Vec::with_capacity(nl);

        for &f in filters {
            let b0 = ConvBn::new(f, box_hidden, 3, 1, 1, 1, device);
            let b1 = ConvBn::new(box_hidden, box_hidden, 3, 1, 1, 1, device);
            let b2 = Conv2dConfig::new([box_hidden, 4 * ch], [1, 1]).init(device);
            box_branches.push(BoxBranch { c0: b0, c1: b1, c2: b2 });

            let c0 = ConvBn::new(f, f, 3, 1, 1, f, device);
            let c1 = ConvBn::new(f, cls_hidden, 1, 1, 0, 1, device);
            let c2 = ConvBn::new(cls_hidden, cls_hidden, 3, 1, 1, cls_hidden, device);
            let c3 = ConvBn::new(cls_hidden, cls_hidden, 1, 1, 0, 1, device);
            let c4 = Conv2dConfig::new([cls_hidden, nc], [1, 1]).init(device);
            cls_branches.push(ClsBranch { c0, c1, c2, c3, c4 });
        }

        Self {
            stride,
            dfl,
            box_branches,
            cls_branches,
        }
    }

    pub fn nc(&self) -> usize {
        if let Some(cb) = self.cls_branches.first() {
            // nc is the output channels of the last conv
            // We can infer from c4's weight shape
            let w = cb.c4.weight.val();
            w.dims()[0]
        } else {
            0
        }
    }

    pub fn nl(&self) -> usize {
        self.box_branches.len()
    }

    pub fn ch(&self) -> usize {
        self.dfl.ch()
    }

    /// Returns list of per-scale outputs (for training).
    pub fn forward_train(&self, xs: &[Tensor<B, 4>]) -> Vec<Tensor<B, 4>> {
        let mut outputs = Vec::with_capacity(self.nl());
        for (i, x) in xs.iter().enumerate() {
            let bb = &self.box_branches[i];
            let box_out = bb.c0.forward_silu(x.clone());
            let box_out = bb.c1.forward_silu(box_out);
            let box_out = bb.c2.forward(box_out);

            let cb = &self.cls_branches[i];
            let cls_out = cb.c0.forward_silu(x.clone());
            let cls_out = cb.c1.forward_silu(cls_out);
            let cls_out = cb.c2.forward_silu(cls_out);
            let cls_out = cb.c3.forward_silu(cls_out);
            let cls_out = cb.c4.forward(cls_out);

            let cat = Tensor::cat(vec![box_out, cls_out], 1);
            outputs.push(cat);
        }
        outputs
    }

    /// Returns decoded [B, 4+nc, total_anchors] tensor (for inference).
    pub fn forward_infer(&self, xs: &[Tensor<B, 4>]) -> Tensor<B, 3> {
        let outputs = self.forward_train(xs);
        let nc = self.nc();
        let ch = self.ch();
        let no = nc + ch * 4;

        let device = outputs[0].device();
        let (anchors, strides) = make_anchors::<B>(&outputs, &self.stride, 0.5, &device);
        let anchors = anchors.swap_dims(0, 1); // [2, total]
        let strides = strides.swap_dims(0, 1); // [1, total]
        // Add batch dim for broadcasting
        let anchors: Tensor<B, 3> = anchors.unsqueeze_dim(0); // [1, 2, total]
        let strides: Tensor<B, 3> = strides.unsqueeze_dim(0); // [1, 1, total]

        let b = outputs[0].dims()[0];
        let mut flat: Vec<Tensor<B, 3>> = Vec::new();
        for o in &outputs {
            let [_, _, h, w] = o.dims();
            flat.push(o.clone().reshape([b, no, h * w]));
        }
        let x = Tensor::cat(flat, 2);

        let box_ch = 4 * ch;
        let box_raw = x.clone().narrow(1, 0, box_ch);
        let cls_raw = x.narrow(1, box_ch, nc);

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
        Tensor::cat(vec![box_scaled, cls_sigmoid], 1)
    }
}
