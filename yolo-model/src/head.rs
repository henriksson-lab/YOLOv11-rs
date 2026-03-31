use candle_core::{Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::anchors::make_anchors;
use crate::conv::{Activation, Conv};

// ---------------------------------------------------------------------------
// DFL  (Distribution Focal Loss decoder)
// ---------------------------------------------------------------------------

pub struct DFL {
    ch: usize,
    /// Fixed weight tensor [1, ch, 1, 1] containing [0, 1, 2, ..., ch-1].
    weight: Tensor,
}

impl DFL {
    pub fn new(ch: usize, device: &Device) -> Result<Self> {
        let w: Vec<f32> = (0..ch).map(|i| i as f32).collect();
        let weight = Tensor::from_vec(w, (1, ch, 1, 1), device)?;
        Ok(Self { ch, weight })
    }

    /// Input: [B, 4*ch, A].  Output: [B, 4, A].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _c, a) = x.dims3()?;
        let x = x.reshape((b, 4, self.ch, a))?; // [B, 4, ch, A]
        let x = x.transpose(1, 2)?;              // [B, ch, 4, A]
        let x = candle_nn::ops::softmax(&x, 1)?; // softmax over ch dim
        // Weighted sum: sum(x * [0..ch]) over dim 1
        let x = x.broadcast_mul(&self.weight)?;   // [B, ch, 4, A]
        x.sum(1)                                   // [B, 4, A]
    }
}

// ---------------------------------------------------------------------------
// Head  (decoupled detection head)
// ---------------------------------------------------------------------------

pub struct Head {
    pub nc: usize,
    pub nl: usize,
    ch: usize,
    pub stride: Tensor,
    dfl: DFL,
    box_convs: Vec<(Conv, Conv, candle_nn::Conv2d)>,
    cls_convs: Vec<(Conv, Conv, Conv, Conv, candle_nn::Conv2d)>,
}

impl Head {
    pub fn new(
        nc: usize,
        filters: &[usize],
        stride: Tensor,
        device: &Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ch: usize = 16; // DFL channels
        let nl = filters.len();
        let dfl = DFL::new(ch, device)?;

        let box_hidden = 64.max(filters[0] / 4);
        let cls_hidden = 80.max(filters[0]).max(nc);

        let mut box_convs = Vec::with_capacity(nl);
        let mut cls_convs = Vec::with_capacity(nl);

        for (i, &f) in filters.iter().enumerate() {
            let vb_box = vb.pp("box").pp(i);
            let b0 = Conv::new(f, box_hidden, Activation::SiLU, 3, 1, 1, 1, vb_box.pp("0"))?;
            let b1 = Conv::new(box_hidden, box_hidden, Activation::SiLU, 3, 1, 1, 1, vb_box.pp("1"))?;
            let b2 = candle_nn::conv2d(
                box_hidden,
                4 * ch,
                1,
                Default::default(),
                vb_box.pp("2"),
            )?;
            box_convs.push((b0, b1, b2));

            let vb_cls = vb.pp("cls").pp(i);
            let c0 = Conv::new(f, f, Activation::SiLU, 3, 1, 1, f, vb_cls.pp("0"))?;
            let c1 = Conv::new(f, cls_hidden, Activation::SiLU, 1, 1, 0, 1, vb_cls.pp("1"))?;
            let c2 = Conv::new(cls_hidden, cls_hidden, Activation::SiLU, 3, 1, 1, cls_hidden, vb_cls.pp("2"))?;
            let c3 = Conv::new(cls_hidden, cls_hidden, Activation::SiLU, 1, 1, 0, 1, vb_cls.pp("3"))?;
            let c4 = candle_nn::conv2d(
                cls_hidden,
                nc,
                1,
                Default::default(),
                vb_cls.pp("4"),
            )?;
            cls_convs.push((c0, c1, c2, c3, c4));
        }

        Ok(Self {
            nc,
            nl,
            ch,
            stride,
            dfl,
            box_convs,
            cls_convs,
        })
    }

    /// Returns list of per-scale outputs (for training).
    pub fn forward_train(
        &self,
        xs: &[Tensor],
        training: bool,
    ) -> Result<Vec<Tensor>> {
        let mut outputs = Vec::with_capacity(self.nl);
        for (i, x) in xs.iter().enumerate() {
            let (b0, b1, b2) = &self.box_convs[i];
            let box_out = b0.forward(x, training)?;
            let box_out = b1.forward(&box_out, training)?;
            let box_out = b2.forward(&box_out)?;

            let (c0, c1, c2, c3, c4) = &self.cls_convs[i];
            let cls_out = c0.forward(x, training)?;
            let cls_out = c1.forward(&cls_out, training)?;
            let cls_out = c2.forward(&cls_out, training)?;
            let cls_out = c3.forward(&cls_out, training)?;
            let cls_out = c4.forward(&cls_out)?;

            let cat = Tensor::cat(&[&box_out, &cls_out], 1)?;
            outputs.push(cat);
        }
        Ok(outputs)
    }

    /// Returns decoded [B, 4+nc, total_anchors] tensor (for inference).
    pub fn forward_infer(
        &self,
        xs: &[Tensor],
        training: bool,
    ) -> Result<Tensor> {
        let outputs = self.forward_train(xs, training)?;
        let no = self.nc + self.ch * 4;

        // Generate anchors from feature map sizes
        let (anchors, strides) = make_anchors(&outputs, &self.stride, 0.5)?;
        let anchors = anchors.transpose(0, 1)?; // [2, total_anchors]
        let strides = strides.transpose(0, 1)?; // [1, total_anchors]

        // Flatten and concat all scale outputs: [B, no, total_anchors]
        let b = outputs[0].dims()[0];
        let mut flat: Vec<Tensor> = Vec::new();
        for o in &outputs {
            let (_, _, h, w) = o.dims4()?;
            flat.push(o.reshape((b, no, h * w))?);
        }
        let refs: Vec<&Tensor> = flat.iter().collect();
        let x = Tensor::cat(&refs, 2)?; // [B, no, total]

        // Split box and cls
        let box_ch = 4 * self.ch;
        let box_raw = x.narrow(1, 0, box_ch)?;
        let cls_raw = x.narrow(1, box_ch, self.nc)?;

        // DFL decode
        let dfl_out = self.dfl.forward(&box_raw)?; // [B, 4, total]
        let ab = dfl_out.chunk(2, 1)?;
        let a = &ab[0]; // first 2 channels
        let b_half = &ab[1]; // last 2 channels

        let anchors_unsq = anchors.unsqueeze(0)?; // [1, 2, total]
        let a_decoded = anchors_unsq.broadcast_sub(a)?;
        let b_decoded = anchors_unsq.broadcast_add(b_half)?;
        let center = ((&a_decoded + &b_decoded)? * 0.5)?;
        let size = b_decoded.sub(&a_decoded)?;
        let box_decoded = Tensor::cat(&[&center, &size], 1)?; // [B, 4, total]
        let box_scaled = box_decoded.broadcast_mul(&strides)?;

        let cls_sigmoid = candle_nn::ops::sigmoid(&cls_raw)?;
        Tensor::cat(&[&box_scaled, &cls_sigmoid], 1) // [B, 4+nc, total]
    }
}
