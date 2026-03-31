use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::conv::{Activation, Conv};

// ---------------------------------------------------------------------------
// Attention (multi-head with QKV projection)
// ---------------------------------------------------------------------------

pub struct Attention {
    num_head: usize,
    dim_head: usize,
    dim_key: usize,
    scale: f64,
    qkv: Conv,
    conv1: Conv, // depthwise conv path
    conv2: Conv, // output projection
}

impl Attention {
    pub fn new(ch: usize, num_head: usize, vb: VarBuilder) -> Result<Self> {
        let dim_head = ch / num_head;
        let dim_key = dim_head / 2;
        let scale = (dim_key as f64).powf(-0.5);

        let qkv_ch = ch + dim_key * num_head * 2;
        let qkv = Conv::new(ch, qkv_ch, Activation::Identity, 1, 1, 0, 1, vb.pp("qkv"))?;
        let conv1 = Conv::new(ch, ch, Activation::Identity, 3, 1, 1, ch, vb.pp("conv1"))?;
        let conv2 = Conv::new(ch, ch, Activation::Identity, 1, 1, 0, 1, vb.pp("conv2"))?;

        Ok(Self {
            num_head,
            dim_head,
            dim_key,
            scale,
            qkv,
            conv1,
            conv2,
        })
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;

        let qkv = self.qkv.forward(x, training)?;
        // Reshape to [B, num_head, dim_key*2 + dim_head, H*W]
        let qkv = qkv.reshape((b, self.num_head, self.dim_key * 2 + self.dim_head, h * w))?;

        // Split into Q [B,nh,dk,HW], K [B,nh,dk,HW], V [B,nh,dh,HW]
        let q = qkv.narrow(2, 0, self.dim_key)?;
        let k = qkv.narrow(2, self.dim_key, self.dim_key)?;
        let v = qkv.narrow(2, self.dim_key * 2, self.dim_head)?;

        // Attention: (Q^T @ K) * scale -> softmax
        let attn = q.transpose(2, 3)?.matmul(&k)?; // [B,nh,HW,HW]
        let attn = (attn * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        // V @ attn^T  -> [B,nh,dh,HW]
        let out = v.matmul(&attn.transpose(2, 3)?)?;
        let out = out.reshape((b, c, h, w))?;

        // Add depthwise conv path on V
        let v_spatial = v.reshape((b, c, h, w))?;
        let conv_path = self.conv1.forward(&v_spatial, training)?;
        let out = out.add(&conv_path)?;

        self.conv2.forward(&out, training)
    }
}

// ---------------------------------------------------------------------------
// PSABlock  (Attention + FFN with residuals)
// ---------------------------------------------------------------------------

pub struct PSABlock {
    attn: Attention,
    ffn1: Conv,
    ffn2: Conv,
}

impl PSABlock {
    pub fn new(ch: usize, num_head: usize, vb: VarBuilder) -> Result<Self> {
        let attn = Attention::new(ch, num_head, vb.pp("conv1"))?;
        let ffn_vb = vb.pp("conv2");
        let ffn1 = Conv::new(ch, ch * 2, Activation::SiLU, 1, 1, 0, 1, ffn_vb.pp("0"))?;
        let ffn2 = Conv::new(ch * 2, ch, Activation::Identity, 1, 1, 0, 1, ffn_vb.pp("1"))?;
        Ok(Self { attn, ffn1, ffn2 })
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let x = x.add(&self.attn.forward(x, training)?)?;
        let ffn = self.ffn1.forward(&x, training)?;
        let ffn = self.ffn2.forward(&ffn, training)?;
        x.add(&ffn)
    }
}

// ---------------------------------------------------------------------------
// PSA  (Partial Self-Attention)
// ---------------------------------------------------------------------------

pub struct PSA {
    conv1: Conv,
    conv2: Conv,
    res_m: Vec<PSABlock>,
}

impl PSA {
    pub fn new(ch: usize, n: usize, vb: VarBuilder) -> Result<Self> {
        let half = ch / 2;
        let conv1 = Conv::new(ch, 2 * half, Activation::SiLU, 1, 1, 0, 1, vb.pp("conv1"))?;
        let conv2 = Conv::new(2 * half, ch, Activation::SiLU, 1, 1, 0, 1, vb.pp("conv2"))?;
        let num_head = half / 128;
        let num_head = if num_head == 0 { 1 } else { num_head };
        let vb_res = vb.pp("res_m");
        let mut res_m = Vec::with_capacity(n);
        for i in 0..n {
            res_m.push(PSABlock::new(half, num_head, vb_res.pp(i))?);
        }
        Ok(Self { conv1, conv2, res_m })
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let c1 = self.conv1.forward(x, training)?;
        let chunks = c1.chunk(2, 1)?;
        let x_pass = &chunks[0];
        let mut y = chunks[1].clone();
        for block in &self.res_m {
            y = block.forward(&y, training)?;
        }
        let cat = Tensor::cat(&[x_pass, &y], 1)?;
        self.conv2.forward(&cat, training)
    }
}
