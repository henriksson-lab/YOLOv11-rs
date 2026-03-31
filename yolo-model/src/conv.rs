use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// BatchNorm2d  (not provided by candle-nn)
// ---------------------------------------------------------------------------

pub struct BatchNorm2d {
    pub weight: Tensor,   // gamma  [C]
    pub bias: Tensor,     // beta   [C]
    running_mean: RefCell<Tensor>, // [C]
    running_var: RefCell<Tensor>,  // [C]
    eps: f64,
    momentum: f64,
}

impl BatchNorm2d {
    pub fn new(num_features: usize, eps: f64, momentum: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(num_features, "weight", candle_nn::Init::Const(1.0))?;
        let bias = vb.get_with_hints(num_features, "bias", candle_nn::Init::Const(0.0))?;
        let running_mean = Tensor::zeros(num_features, DType::F32, vb.device())?;
        let running_var = Tensor::ones(num_features, DType::F32, vb.device())?;
        Ok(Self {
            weight,
            bias,
            running_mean: RefCell::new(running_mean),
            running_var: RefCell::new(running_var),
            eps,
            momentum,
        })
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        if training {
            // Compute batch statistics over (N, H, W)
            let x_flat = x.reshape((b, c, h * w))?;        // [B, C, H*W]
            let x_flat = x_flat.transpose(0, 1)?;           // [C, B, H*W]
            let x_flat = x_flat.reshape((c, b * h * w))?;   // [C, N]
            let mean = x_flat.mean_keepdim(1)?;              // [C, 1]
            let var = x_flat.var_keepdim(1)?;                // [C, 1]
            let mean_1d = mean.squeeze(1)?;                  // [C]
            let var_1d = var.squeeze(1)?;                    // [C]

            // Update running stats (detached, no gradient)
            {
                let rm = self.running_mean.borrow();
                let new_rm = ((1.0 - self.momentum) * &*rm)?.add(
                    &(self.momentum * mean_1d.detach())?,
                )?;
                drop(rm);
                *self.running_mean.borrow_mut() = new_rm;
            }
            {
                let rv = self.running_var.borrow();
                let new_rv = ((1.0 - self.momentum) * &*rv)?.add(
                    &(self.momentum * var_1d.detach())?,
                )?;
                drop(rv);
                *self.running_var.borrow_mut() = new_rv;
            }

            // Normalize: (x - mean) / sqrt(var + eps)
            let mean_4d = mean_1d.reshape((1, c, 1, 1))?;
            let var_4d = var_1d.reshape((1, c, 1, 1))?;
            let x_norm = x.broadcast_sub(&mean_4d)?
                .broadcast_div(&(var_4d + self.eps)?.sqrt()?)?;

            // Scale and shift
            let w = self.weight.reshape((1, c, 1, 1))?;
            let b = self.bias.reshape((1, c, 1, 1))?;
            x_norm.broadcast_mul(&w)?.broadcast_add(&b)
        } else {
            let rm = self.running_mean.borrow();
            let rv = self.running_var.borrow();
            let mean_4d = rm.reshape((1, c, 1, 1))?;
            let var_4d = rv.reshape((1, c, 1, 1))?;
            let x_norm = x.broadcast_sub(&mean_4d)?
                .broadcast_div(&(var_4d + self.eps)?.sqrt()?)?;
            let w = self.weight.reshape((1, c, 1, 1))?;
            let b = self.bias.reshape((1, c, 1, 1))?;
            x_norm.broadcast_mul(&w)?.broadcast_add(&b)
        }
    }
}

// ---------------------------------------------------------------------------
// Activation enum
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    SiLU,
    Identity,
}

impl Activation {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Activation::SiLU => candle_nn::ops::silu(x),
            Activation::Identity => Ok(x.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// Conv  (Conv2d + BatchNorm2d + Activation)
// ---------------------------------------------------------------------------

pub struct Conv {
    pub conv: Conv2d,
    pub norm: Option<BatchNorm2d>,
    pub act: Activation,
}

impl Conv {
    /// Create a Conv block matching PyTorch `Conv(in_ch, out_ch, activation, k, s, p, g)`.
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        act: Activation,
        k: usize,
        s: usize,
        p: usize,
        g: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            stride: s,
            padding: p,
            groups: g,
            ..Default::default()
        };
        let conv = candle_nn::conv2d_no_bias(in_ch, out_ch, k, cfg, vb.pp("conv"))?;
        let norm = Some(BatchNorm2d::new(out_ch, 0.001, 0.03, vb.pp("norm"))?);
        Ok(Self { conv, norm, act })
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = if let Some(ref norm) = self.norm {
            norm.forward(&x, training)?
        } else {
            x
        };
        self.act.forward(&x)
    }

    /// Fuse BatchNorm into the Conv2d weights for inference.
    pub fn fuse(&mut self) -> Result<()> {
        let fused = if let Some(ref norm) = self.norm {
            let rv = norm.running_var.borrow();
            let rm = norm.running_mean.borrow();
            let std_inv = (&*rv + norm.eps)?.sqrt()?.recip()?;
            let w_scale = norm.weight.mul(&std_inv)?;

            let conv_w = self.conv.weight();
            let (out_ch, _, _kh, _kw) = conv_w.dims4()?;
            let w_scale_4d = w_scale.reshape((out_ch, 1, 1, 1))?;
            let fused_w = conv_w.broadcast_mul(&w_scale_4d)?;

            let conv_b = Tensor::zeros(out_ch, DType::F32, conv_w.device())?;
            let fused_b = (&w_scale * &conv_b)?
                .add(&norm.bias)?
                .sub(&(&norm.weight * &*rm)?.mul(&std_inv)?)?;

            Some((fused_w, fused_b))
        } else {
            None
        };
        if let Some((fused_w, fused_b)) = fused {
            self.conv = Conv2d::new(fused_w, Some(fused_b), *self.conv.config());
            self.norm = None;
        }
        Ok(())
    }
}

/// Convenience constructor with default k=1, s=1, p=0, g=1.
pub fn conv(
    in_ch: usize,
    out_ch: usize,
    act: Activation,
    vb: VarBuilder,
) -> Result<Conv> {
    Conv::new(in_ch, out_ch, act, 1, 1, 0, 1, vb)
}
