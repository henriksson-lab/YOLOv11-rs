use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::model::conv::{Activation, Conv};

// ---------------------------------------------------------------------------
// Residual
// ---------------------------------------------------------------------------

pub struct Residual {
    conv1: Conv,
    conv2: Conv,
}

impl Residual {
    /// `e` controls the hidden-channel expansion ratio (0.5 or 1.0).
    pub fn new(ch: usize, e: f64, vb: VarBuilder) -> Result<Self> {
        let hid = (ch as f64 * e) as usize;
        let conv1 = Conv::new(ch, hid, Activation::SiLU, 3, 1, 1, 1, vb.pp("conv1"))?;
        let conv2 = Conv::new(hid, ch, Activation::SiLU, 3, 1, 1, 1, vb.pp("conv2"))?;
        Ok(Self { conv1, conv2 })
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let y = self.conv1.forward(x, training)?;
        let y = self.conv2.forward(&y, training)?;
        x.add(&y)
    }
}

// ---------------------------------------------------------------------------
// CSPModule
// ---------------------------------------------------------------------------

pub struct CSPModule {
    conv1: Conv,
    conv2: Conv,
    conv3: Conv,
    res_m: Vec<Residual>,
}

impl CSPModule {
    pub fn new(in_ch: usize, out_ch: usize, vb: VarBuilder) -> Result<Self> {
        let half = out_ch / 2;
        let conv1 = Conv::new(in_ch, half, Activation::SiLU, 1, 1, 0, 1, vb.pp("conv1"))?;
        let conv2 = Conv::new(in_ch, half, Activation::SiLU, 1, 1, 0, 1, vb.pp("conv2"))?;
        let conv3 = Conv::new(2 * half, out_ch, Activation::SiLU, 1, 1, 0, 1, vb.pp("conv3"))?;
        let vb_res = vb.pp("res_m");
        let r0 = Residual::new(half, 1.0, vb_res.pp("0"))?;
        let r1 = Residual::new(half, 1.0, vb_res.pp("1"))?;
        Ok(Self {
            conv1,
            conv2,
            conv3,
            res_m: vec![r0, r1],
        })
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let mut y = self.conv1.forward(x, training)?;
        for r in &self.res_m {
            y = r.forward(&y, training)?;
        }
        let z = self.conv2.forward(x, training)?;
        let cat = Tensor::cat(&[&y, &z], 1)?;
        self.conv3.forward(&cat, training)
    }
}

// ---------------------------------------------------------------------------
// CSP
// ---------------------------------------------------------------------------

pub enum CSPBlock {
    Residual(Residual),
    Module(CSPModule),
}

impl CSPBlock {
    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        match self {
            CSPBlock::Residual(r) => r.forward(x, training),
            CSPBlock::Module(m) => m.forward(x, training),
        }
    }
}

pub struct CSP {
    conv1: Conv,
    conv2: Conv,
    res_m: Vec<CSPBlock>,
}

impl CSP {
    /// `n` = number of blocks, `csp` = true → CSPModule blocks, false → Residual blocks.
    /// `r` = channel reduction ratio (2 or 4).
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        n: usize,
        csp: bool,
        r: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mid = out_ch / r;
        let conv1 = Conv::new(in_ch, 2 * mid, Activation::SiLU, 1, 1, 0, 1, vb.pp("conv1"))?;
        let cat_ch = (2 + n) * mid;
        let conv2 = Conv::new(cat_ch, out_ch, Activation::SiLU, 1, 1, 0, 1, vb.pp("conv2"))?;

        let vb_res = vb.pp("res_m");
        let mut res_m = Vec::with_capacity(n);
        for i in 0..n {
            let block = if csp {
                CSPBlock::Module(CSPModule::new(mid, mid, vb_res.pp(i))?)
            } else {
                CSPBlock::Residual(Residual::new(mid, 0.5, vb_res.pp(i))?)
            };
            res_m.push(block);
        }

        Ok(Self { conv1, conv2, res_m })
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let c1 = self.conv1.forward(x, training)?;
        let chunks = c1.chunk(2, 1)?;
        let mut parts: Vec<Tensor> = chunks.into_iter().collect();
        for block in &self.res_m {
            let last = parts.last().unwrap();
            let next = block.forward(last, training)?;
            parts.push(next);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let cat = Tensor::cat(&refs, 1)?;
        self.conv2.forward(&cat, training)
    }
}

// ---------------------------------------------------------------------------
// SPP  (Spatial Pyramid Pooling)
// ---------------------------------------------------------------------------

pub struct SPP {
    conv1: Conv,
    conv2: Conv,
    k: usize,
}

impl SPP {
    pub fn new(in_ch: usize, out_ch: usize, k: usize, vb: VarBuilder) -> Result<Self> {
        let conv1 = Conv::new(in_ch, in_ch / 2, Activation::SiLU, 1, 1, 0, 1, vb.pp("conv1"))?;
        let conv2 = Conv::new(in_ch * 2, out_ch, Activation::SiLU, 1, 1, 0, 1, vb.pp("conv2"))?;
        Ok(Self { conv1, conv2, k })
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let x = self.conv1.forward(x, training)?;
        let p = self.k / 2;
        // Pad and max-pool with stride=1 to keep spatial dims
        let y1 = x.pad_with_zeros(2, p, p)?.pad_with_zeros(3, p, p)?;
        let y1 = y1.max_pool2d_with_stride(self.k, 1)?;
        let y2 = y1.pad_with_zeros(2, p, p)?.pad_with_zeros(3, p, p)?;
        let y2 = y2.max_pool2d_with_stride(self.k, 1)?;
        let y3 = y2.pad_with_zeros(2, p, p)?.pad_with_zeros(3, p, p)?;
        let y3 = y3.max_pool2d_with_stride(self.k, 1)?;
        let cat = Tensor::cat(&[&x, &y1, &y2, &y3], 1)?;
        self.conv2.forward(&cat, training)
    }
}
