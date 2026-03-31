use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::model::blocks::CSP;
use crate::model::conv::{Activation, Conv};

/// DarkFPN neck – fuses multi-scale features from backbone.
pub struct DarkFPN {
    h1: CSP,
    h2: CSP,
    h3: Conv,
    h4: CSP,
    h5: Conv,
    h6: CSP,
}

impl DarkFPN {
    pub fn new(width: &[usize], depth: &[usize], csp: &[bool], vb: VarBuilder) -> Result<Self> {
        let h1 = CSP::new(width[4] + width[5], width[4], depth[5], csp[0], 2, vb.pp("h1"))?;
        let h2 = CSP::new(width[4] + width[4], width[3], depth[5], csp[0], 2, vb.pp("h2"))?;
        let h3 = Conv::new(width[3], width[3], Activation::SiLU, 3, 2, 1, 1, vb.pp("h3"))?;
        let h4 = CSP::new(width[3] + width[4], width[4], depth[5], csp[0], 2, vb.pp("h4"))?;
        let h5 = Conv::new(width[4], width[4], Activation::SiLU, 3, 2, 1, 1, vb.pp("h5"))?;
        let h6 = CSP::new(width[4] + width[5], width[5], depth[5], csp[1], 2, vb.pp("h6"))?;
        Ok(Self { h1, h2, h3, h4, h5, h6 })
    }

    /// Takes (p3, p4, p5) and returns refined (p3, p4, p5).
    pub fn forward(
        &self,
        p3: &Tensor,
        p4: &Tensor,
        p5: &Tensor,
        training: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (_, _, p4h, p4w) = p4.dims4()?;
        let (_, _, p3h, p3w) = p3.dims4()?;

        // Upsample p5 -> p4 size, concat, CSP
        let up5 = p5.upsample_nearest2d(p4h, p4w)?;
        let p4 = Tensor::cat(&[&up5, p4], 1)?;
        let p4 = self.h1.forward(&p4, training)?;

        // Upsample p4 -> p3 size, concat, CSP
        let up4 = p4.upsample_nearest2d(p3h, p3w)?;
        let p3 = Tensor::cat(&[&up4, p3], 1)?;
        let p3 = self.h2.forward(&p3, training)?;

        // Downsample p3 -> concat with p4, CSP
        let down3 = self.h3.forward(&p3, training)?;
        let p4 = Tensor::cat(&[&down3, &p4], 1)?;
        let p4 = self.h4.forward(&p4, training)?;

        // Downsample p4 -> concat with p5, CSP
        let down4 = self.h5.forward(&p4, training)?;
        let p5 = Tensor::cat(&[&down4, p5], 1)?;
        let p5 = self.h6.forward(&p5, training)?;

        Ok((p3, p4, p5))
    }
}
