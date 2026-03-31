use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::attention::PSA;
use crate::blocks::{CSP, SPP};
use crate::conv::{Activation, Conv};

/// DarkNet backbone. Outputs feature maps at 3 scales: p3 (8x), p4 (16x), p5 (32x).
pub struct DarkNet {
    p1: Conv,
    p2_conv: Conv,
    p2_csp: CSP,
    p3_conv: Conv,
    p3_csp: CSP,
    p4_conv: Conv,
    p4_csp: CSP,
    p5_conv: Conv,
    p5_csp: CSP,
    p5_spp: SPP,
    p5_psa: PSA,
}

impl DarkNet {
    pub fn new(
        width: &[usize],
        depth: &[usize],
        csp: &[bool],
        vb: VarBuilder,
    ) -> Result<Self> {
        // p1/2
        let p1 = Conv::new(width[0], width[1], Activation::SiLU, 3, 2, 1, 1, vb.pp("p1").pp("0"))?;
        // p2/4
        let p2_conv = Conv::new(width[1], width[2], Activation::SiLU, 3, 2, 1, 1, vb.pp("p2").pp("0"))?;
        let p2_csp = CSP::new(width[2], width[3], depth[0], csp[0], 4, vb.pp("p2").pp("1"))?;
        // p3/8
        let p3_conv = Conv::new(width[3], width[3], Activation::SiLU, 3, 2, 1, 1, vb.pp("p3").pp("0"))?;
        let p3_csp = CSP::new(width[3], width[4], depth[1], csp[0], 4, vb.pp("p3").pp("1"))?;
        // p4/16
        let p4_conv = Conv::new(width[4], width[4], Activation::SiLU, 3, 2, 1, 1, vb.pp("p4").pp("0"))?;
        let p4_csp = CSP::new(width[4], width[4], depth[2], csp[1], 2, vb.pp("p4").pp("1"))?;
        // p5/32
        let p5_conv = Conv::new(width[4], width[5], Activation::SiLU, 3, 2, 1, 1, vb.pp("p5").pp("0"))?;
        let p5_csp = CSP::new(width[5], width[5], depth[3], csp[1], 2, vb.pp("p5").pp("1"))?;
        let p5_spp = SPP::new(width[5], width[5], 5, vb.pp("p5").pp("2"))?;
        let p5_psa = PSA::new(width[5], depth[4], vb.pp("p5").pp("3"))?;

        Ok(Self {
            p1,
            p2_conv,
            p2_csp,
            p3_conv,
            p3_csp,
            p4_conv,
            p4_csp,
            p5_conv,
            p5_csp,
            p5_spp,
            p5_psa,
        })
    }

    /// Returns (p3, p4, p5) feature maps.
    pub fn forward(&self, x: &Tensor, training: bool) -> Result<(Tensor, Tensor, Tensor)> {
        let p1 = self.p1.forward(x, training)?;
        let p2 = self.p2_conv.forward(&p1, training)?;
        let p2 = self.p2_csp.forward(&p2, training)?;
        let p3 = self.p3_conv.forward(&p2, training)?;
        let p3 = self.p3_csp.forward(&p3, training)?;
        let p4 = self.p4_conv.forward(&p3, training)?;
        let p4 = self.p4_csp.forward(&p4, training)?;
        let p5 = self.p5_conv.forward(&p4, training)?;
        let p5 = self.p5_csp.forward(&p5, training)?;
        let p5 = self.p5_spp.forward(&p5, training)?;
        let p5 = self.p5_psa.forward(&p5, training)?;
        Ok((p3, p4, p5))
    }
}
