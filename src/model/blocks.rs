use burn::module::Module;
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::prelude::*;

use crate::model::conv::{Activation, Conv, FuseModule};

// ---------------------------------------------------------------------------
// Residual
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct Residual<B: Backend> {
    conv1: Conv<B>,
    conv2: Conv<B>,
}

impl<B: Backend> Residual<B> {
    pub fn new(ch: usize, e: f64, device: &B::Device) -> Self {
        let hid = (ch as f64 * e) as usize;
        let conv1 = Conv::new(ch, hid, Activation::SiLU, 3, 1, 1, 1, device);
        let conv2 = Conv::new(hid, ch, Activation::SiLU, 3, 1, 1, 1, device);
        Self { conv1, conv2 }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.conv1.forward(x.clone());
        let y = self.conv2.forward(y);
        x + y
    }
}

impl<B: Backend> FuseModule for Residual<B> {
    fn fuse_module(self) -> Self {
        Self {
            conv1: self.conv1.fuse_module(),
            conv2: self.conv2.fuse_module(),
        }
    }
}

// ---------------------------------------------------------------------------
// CSPModule
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct CSPModule<B: Backend> {
    conv1: Conv<B>,
    conv2: Conv<B>,
    conv3: Conv<B>,
    res_m: Vec<Residual<B>>,
}

impl<B: Backend> CSPModule<B> {
    pub fn new(in_ch: usize, out_ch: usize, device: &B::Device) -> Self {
        let half = out_ch / 2;
        let conv1 = Conv::new(in_ch, half, Activation::SiLU, 1, 1, 0, 1, device);
        let conv2 = Conv::new(in_ch, half, Activation::SiLU, 1, 1, 0, 1, device);
        let conv3 = Conv::new(2 * half, out_ch, Activation::SiLU, 1, 1, 0, 1, device);
        let r0 = Residual::new(half, 1.0, device);
        let r1 = Residual::new(half, 1.0, device);
        Self {
            conv1,
            conv2,
            conv3,
            res_m: vec![r0, r1],
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut y = self.conv1.forward(x.clone());
        for r in &self.res_m {
            y = r.forward(y);
        }
        let z = self.conv2.forward(x);
        let cat = Tensor::cat(vec![y, z], 1);
        self.conv3.forward(cat)
    }
}

impl<B: Backend> FuseModule for CSPModule<B> {
    fn fuse_module(self) -> Self {
        Self {
            conv1: self.conv1.fuse_module(),
            conv2: self.conv2.fuse_module(),
            conv3: self.conv3.fuse_module(),
            res_m: self
                .res_m
                .into_iter()
                .map(FuseModule::fuse_module)
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// CSP  (uses either Residual blocks or CSPModule blocks, never both)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct CSP<B: Backend> {
    conv1: Conv<B>,
    conv2: Conv<B>,
    res_blocks: Vec<Residual<B>>,
    csp_blocks: Vec<CSPModule<B>>,
}

impl<B: Backend> CSP<B> {
    /// `n` = number of blocks, `csp` = true → CSPModule blocks, false → Residual blocks.
    /// `r` = channel reduction ratio (2 or 4).
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        n: usize,
        csp: bool,
        r: usize,
        device: &B::Device,
    ) -> Self {
        let mid = out_ch / r;
        let conv1 = Conv::new(in_ch, 2 * mid, Activation::SiLU, 1, 1, 0, 1, device);
        let cat_ch = (2 + n) * mid;
        let conv2 = Conv::new(cat_ch, out_ch, Activation::SiLU, 1, 1, 0, 1, device);

        let mut res_blocks = Vec::new();
        let mut csp_blocks = Vec::new();
        for _ in 0..n {
            if csp {
                csp_blocks.push(CSPModule::new(mid, mid, device));
            } else {
                res_blocks.push(Residual::new(mid, 0.5, device));
            }
        }

        Self {
            conv1,
            conv2,
            res_blocks,
            csp_blocks,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let c1 = self.conv1.forward(x);
        let [_b, c, _h, _w] = c1.dims();
        let half = c / 2;
        let first = c1.clone().narrow(1, 0, half);
        let second = c1.narrow(1, half, c - half);
        let mut parts: Vec<Tensor<B, 4>> = vec![first, second];

        if !self.csp_blocks.is_empty() {
            for block in &self.csp_blocks {
                let last = parts.last().unwrap().clone();
                parts.push(block.forward(last));
            }
        } else {
            for block in &self.res_blocks {
                let last = parts.last().unwrap().clone();
                parts.push(block.forward(last));
            }
        }

        let cat = Tensor::cat(parts, 1);
        self.conv2.forward(cat)
    }
}

impl<B: Backend> FuseModule for CSP<B> {
    fn fuse_module(self) -> Self {
        Self {
            conv1: self.conv1.fuse_module(),
            conv2: self.conv2.fuse_module(),
            res_blocks: self
                .res_blocks
                .into_iter()
                .map(FuseModule::fuse_module)
                .collect(),
            csp_blocks: self
                .csp_blocks
                .into_iter()
                .map(FuseModule::fuse_module)
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// SPP  (Spatial Pyramid Pooling)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct SPP<B: Backend> {
    conv1: Conv<B>,
    conv2: Conv<B>,
    pool: MaxPool2d,
}

impl<B: Backend> SPP<B> {
    pub fn new(in_ch: usize, out_ch: usize, k: usize, device: &B::Device) -> Self {
        let conv1 = Conv::new(in_ch, in_ch / 2, Activation::SiLU, 1, 1, 0, 1, device);
        let conv2 = Conv::new(in_ch * 2, out_ch, Activation::SiLU, 1, 1, 0, 1, device);
        let p = k / 2;
        let pool = MaxPool2dConfig::new([k, k])
            .with_strides([1, 1])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(p, p, p, p))
            .init();
        Self { conv1, conv2, pool }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(x);
        let y1 = self.pool.forward(x.clone());
        let y2 = self.pool.forward(y1.clone());
        let y3 = self.pool.forward(y2.clone());
        let cat = Tensor::cat(vec![x, y1, y2, y3], 1);
        self.conv2.forward(cat)
    }
}

impl<B: Backend> FuseModule for SPP<B> {
    fn fuse_module(self) -> Self {
        Self {
            conv1: self.conv1.fuse_module(),
            conv2: self.conv2.fuse_module(),
            pool: self.pool,
        }
    }
}
