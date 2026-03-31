use burn::module::Module;
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::model::conv::ConvBn;

// ---------------------------------------------------------------------------
// Attention (multi-head with QKV projection)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    qkv: ConvBn<B>,
    conv1: ConvBn<B>,
    conv2: ConvBn<B>,
}

impl<B: Backend> Attention<B> {
    pub fn new(ch: usize, num_head: usize, device: &B::Device) -> Self {
        let dim_head = ch / num_head;
        let dim_key = dim_head / 2;
        let qkv_ch = ch + dim_key * num_head * 2;
        let qkv = ConvBn::new(ch, qkv_ch, 1, 1, 0, 1, device);
        let conv1 = ConvBn::new(ch, ch, 3, 1, 1, ch, device);
        let conv2 = ConvBn::new(ch, ch, 1, 1, 0, 1, device);
        Self { qkv, conv1, conv2 }
    }

    pub fn forward(&self, x: Tensor<B, 4>, num_head: usize) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let dim_head = c / num_head;
        let dim_key = dim_head / 2;
        let scale = (dim_key as f64).powf(-0.5);

        let qkv = self.qkv.forward(x.clone()); // no activation (identity)
        let qkv = qkv.reshape([b, num_head, dim_key * 2 + dim_head, h * w]);

        let q = qkv.clone().narrow(2, 0, dim_key);
        let k = qkv.clone().narrow(2, dim_key, dim_key);
        let v = qkv.narrow(2, dim_key * 2, dim_head);

        let attn = q.swap_dims(2, 3).matmul(k);
        let attn = attn * scale;
        let attn = softmax(attn, 3);

        let out = v.clone().matmul(attn.swap_dims(2, 3));
        let out = out.reshape([b, c, h, w]);

        let v_spatial = v.reshape([b, c, h, w]);
        let conv_path = self.conv1.forward(v_spatial); // identity
        let out = out + conv_path;

        self.conv2.forward(out) // identity
    }
}

// ---------------------------------------------------------------------------
// PSABlock  (Attention + FFN with residuals)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct PSABlock<B: Backend> {
    attn: Attention<B>,
    ffn1: ConvBn<B>,
    ffn2: ConvBn<B>,
}

impl<B: Backend> PSABlock<B> {
    pub fn new(ch: usize, num_head: usize, device: &B::Device) -> Self {
        let attn = Attention::new(ch, num_head, device);
        let ffn1 = ConvBn::new(ch, ch * 2, 1, 1, 0, 1, device);
        let ffn2 = ConvBn::new(ch * 2, ch, 1, 1, 0, 1, device);
        Self { attn, ffn1, ffn2 }
    }

    pub fn forward(&self, x: Tensor<B, 4>, num_head: usize) -> Tensor<B, 4> {
        let attn_out = self.attn.forward(x.clone(), num_head);
        let x = x + attn_out;
        let ffn = self.ffn1.forward_silu(x.clone());
        let ffn = self.ffn2.forward(ffn); // identity
        x + ffn
    }
}

// ---------------------------------------------------------------------------
// PSA  (Partial Self-Attention)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct PSA<B: Backend> {
    conv1: ConvBn<B>,
    conv2: ConvBn<B>,
    res_m: Vec<PSABlock<B>>,
}

impl<B: Backend> PSA<B> {
    pub fn new(ch: usize, n: usize, device: &B::Device) -> Self {
        let half = ch / 2;
        let conv1 = ConvBn::new(ch, 2 * half, 1, 1, 0, 1, device);
        let conv2 = ConvBn::new(2 * half, ch, 1, 1, 0, 1, device);
        let num_head = (ch / 128).max(1);
        let mut res_m = Vec::with_capacity(n);
        for _ in 0..n {
            res_m.push(PSABlock::new(half, num_head, device));
        }
        Self { conv1, conv2, res_m }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let c1 = self.conv1.forward_silu(x);
        let [_b, c, _h, _w] = c1.dims();
        let half = c / 2;
        let num_head = (c / 128).max(1);
        let x_pass = c1.clone().narrow(1, 0, half);
        let mut y = c1.narrow(1, half, c - half);
        for block in &self.res_m {
            y = block.forward(y, num_head);
        }
        let cat = Tensor::cat(vec![x_pass, y], 1);
        self.conv2.forward_silu(cat)
    }
}
