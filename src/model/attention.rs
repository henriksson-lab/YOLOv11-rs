use burn::module::Module;
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::model::conv::{Activation, Conv, FuseModule};

// ---------------------------------------------------------------------------
// Attention (multi-head with QKV projection)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct Attention {
    qkv: Conv,
    conv1: Conv,
    conv2: Conv,
    num_head: usize,
    dim_head: usize,
    dim_key: usize,
    scale: f64,
}

impl Attention {
    pub fn new(ch: usize, num_head: usize, device: &Device) -> Self {
        let dim_head = ch / num_head;
        let dim_key = dim_head / 2;
        let scale = (dim_key as f64).powf(-0.5);
        let qkv_ch = ch + dim_key * num_head * 2;
        let qkv = Conv::new(ch, qkv_ch, Activation::Identity, 1, 1, 0, 1, device);
        let conv1 = Conv::new(ch, ch, Activation::Identity, 3, 1, 1, ch, device);
        let conv2 = Conv::new(ch, ch, Activation::Identity, 1, 1, 0, 1, device);
        Self {
            qkv,
            conv1,
            conv2,
            num_head,
            dim_head,
            dim_key,
            scale,
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [b, c, h, w] = x.dims();

        let qkv = self.qkv.forward(x.clone()); // no activation (identity)
        let qkv = qkv.reshape([b, self.num_head, self.dim_key * 2 + self.dim_head, h * w]);

        let q = qkv.clone().narrow(2, 0, self.dim_key);
        let k = qkv.clone().narrow(2, self.dim_key, self.dim_key);
        let v = qkv.narrow(2, self.dim_key * 2, self.dim_head);

        let attn = q.swap_dims(2, 3).matmul(k);
        let attn = attn * self.scale;
        let attn = softmax(attn, 3);

        let out = v.clone().matmul(attn.swap_dims(2, 3));
        let out = out.reshape([b, c, h, w]);

        let v_spatial = v.reshape([b, c, h, w]);
        let conv_path = self.conv1.forward(v_spatial); // identity
        let out = out + conv_path;

        self.conv2.forward(out) // identity
    }
}

impl FuseModule for Attention {
    fn fuse_module(self) -> Self {
        Self {
            qkv: self.qkv.fuse_module(),
            conv1: self.conv1.fuse_module(),
            conv2: self.conv2.fuse_module(),
            num_head: self.num_head,
            dim_head: self.dim_head,
            dim_key: self.dim_key,
            scale: self.scale,
        }
    }
}

// ---------------------------------------------------------------------------
// PSABlock  (Attention + FFN with residuals)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct PSABlock {
    attn: Attention,
    ffn1: Conv,
    ffn2: Conv,
}

impl PSABlock {
    pub fn new(ch: usize, num_head: usize, device: &Device) -> Self {
        let attn = Attention::new(ch, num_head, device);
        let ffn1 = Conv::new(ch, ch * 2, Activation::SiLU, 1, 1, 0, 1, device);
        let ffn2 = Conv::new(ch * 2, ch, Activation::Identity, 1, 1, 0, 1, device);
        Self { attn, ffn1, ffn2 }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let attn_out = self.attn.forward(x.clone());
        let x = x + attn_out;
        let ffn = self.ffn1.forward(x.clone());
        let ffn = self.ffn2.forward(ffn); // identity
        x + ffn
    }
}

impl FuseModule for PSABlock {
    fn fuse_module(self) -> Self {
        Self {
            attn: self.attn.fuse_module(),
            ffn1: self.ffn1.fuse_module(),
            ffn2: self.ffn2.fuse_module(),
        }
    }
}

// ---------------------------------------------------------------------------
// PSA  (Partial Self-Attention)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct PSA {
    conv1: Conv,
    conv2: Conv,
    res_m: Vec<PSABlock>,
}

impl PSA {
    pub fn new(ch: usize, n: usize, device: &Device) -> Self {
        let half = ch / 2;
        let conv1 = Conv::new(ch, 2 * half, Activation::SiLU, 1, 1, 0, 1, device);
        let conv2 = Conv::new(2 * half, ch, Activation::SiLU, 1, 1, 0, 1, device);
        let num_head = ch / 128;
        let mut res_m = Vec::with_capacity(n);
        for _ in 0..n {
            res_m.push(PSABlock::new(half, num_head, device));
        }
        Self {
            conv1,
            conv2,
            res_m,
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let c1 = self.conv1.forward(x);
        let [_b, c, _h, _w] = c1.dims();
        let half = c / 2;
        let x_pass = c1.clone().narrow(1, 0, half);
        let mut y = c1.narrow(1, half, c - half);
        for block in &self.res_m {
            y = block.forward(y);
        }
        let cat = Tensor::cat(vec![x_pass, y], 1);
        self.conv2.forward(cat)
    }
}

impl FuseModule for PSA {
    fn fuse_module(self) -> Self {
        Self {
            conv1: self.conv1.fuse_module(),
            conv2: self.conv2.fuse_module(),
            res_m: self
                .res_m
                .into_iter()
                .map(FuseModule::fuse_module)
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PSA;

    #[test]
    fn psa_new_uses_python_integer_head_count() {
        let device = burn::tensor::Device::flex();
        let psa = PSA::new(256, 1, &device);

        assert_eq!(psa.res_m[0].attn.num_head, 2);
        assert_eq!(psa.res_m[0].attn.dim_head, 64);
        assert_eq!(psa.res_m[0].attn.dim_key, 32);
    }
}
