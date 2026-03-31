use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig};
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    SiLU,
    Identity,
}

pub fn apply_act<B: Backend>(act: Activation, x: Tensor<B, 4>) -> Tensor<B, 4> {
    match act {
        Activation::SiLU => burn::tensor::activation::silu(x),
        Activation::Identity => x,
    }
}

// ---------------------------------------------------------------------------
// ConvBn  (Conv2d + BatchNorm, no activation — activation applied at call site)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ConvBn<B: Backend> {
    pub conv: Conv2d<B>,
    pub norm: Option<BatchNorm<B, 2>>,
}

impl<B: Backend> ConvBn<B> {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        k: usize,
        s: usize,
        p: usize,
        g: usize,
        device: &B::Device,
    ) -> Self {
        let conv = Conv2dConfig::new([in_ch, out_ch], [k, k])
            .with_stride([s, s])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(p, p))
            .with_groups(g)
            .with_bias(false)
            .init(device);
        let norm = Some(
            BatchNormConfig::new(out_ch)
                .with_epsilon(0.001)
                .with_momentum(0.03)
                .init(device),
        );
        Self { conv, norm }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        if let Some(ref norm) = self.norm {
            norm.forward(x)
        } else {
            x
        }
    }

    /// Forward with SiLU activation.
    pub fn forward_silu(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        burn::tensor::activation::silu(self.forward(x))
    }
}

/// Create a ConvBn with default k=1, s=1, p=0, g=1.
pub fn conv_bn<B: Backend>(in_ch: usize, out_ch: usize, device: &B::Device) -> ConvBn<B> {
    ConvBn::new(in_ch, out_ch, 1, 1, 0, 1, device)
}
