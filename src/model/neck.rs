use burn::module::Module;
use burn::prelude::*;
use burn::tensor::module::interpolate;
use burn::tensor::ops::InterpolateOptions;

use crate::model::blocks::CSP;
use crate::model::conv::ConvBn;

/// DarkFPN neck – fuses multi-scale features from backbone.
#[derive(Module, Debug)]
pub struct DarkFPN<B: Backend> {
    h1: CSP<B>,
    h2: CSP<B>,
    h3: ConvBn<B>,
    h4: CSP<B>,
    h5: ConvBn<B>,
    h6: CSP<B>,
}

impl<B: Backend> DarkFPN<B> {
    pub fn new(width: &[usize], depth: &[usize], csp: &[bool], device: &B::Device) -> Self {
        let h1 = CSP::new(width[4] + width[5], width[4], depth[5], csp[0], 2, device);
        let h2 = CSP::new(width[4] + width[4], width[3], depth[5], csp[0], 2, device);
        let h3 = ConvBn::new(width[3], width[3], 3, 2, 1, 1, device);
        let h4 = CSP::new(width[3] + width[4], width[4], depth[5], csp[0], 2, device);
        let h5 = ConvBn::new(width[4], width[4], 3, 2, 1, 1, device);
        let h6 = CSP::new(width[4] + width[5], width[5], depth[5], csp[1], 2, device);
        Self { h1, h2, h3, h4, h5, h6 }
    }

    pub fn forward(
        &self,
        p3: Tensor<B, 4>,
        p4: Tensor<B, 4>,
        p5: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [_, _, p4h, p4w] = p4.dims();
        let [_, _, p3h, p3w] = p3.dims();

        let up5 = interpolate(
            p5.clone(),
            [p4h, p4w],
            InterpolateOptions::new(burn::tensor::ops::InterpolateMode::Nearest),
        );
        let p4_out = Tensor::cat(vec![up5, p4], 1);
        let p4_out = self.h1.forward(p4_out);

        let up4 = interpolate(
            p4_out.clone(),
            [p3h, p3w],
            InterpolateOptions::new(burn::tensor::ops::InterpolateMode::Nearest),
        );
        let p3_out = Tensor::cat(vec![up4, p3], 1);
        let p3_out = self.h2.forward(p3_out);

        let down3 = self.h3.forward_silu(p3_out.clone());
        let p4_out = Tensor::cat(vec![down3, p4_out], 1);
        let p4_out = self.h4.forward(p4_out);

        let down4 = self.h5.forward_silu(p4_out.clone());
        let p5_out = Tensor::cat(vec![down4, p5], 1);
        let p5_out = self.h6.forward(p5_out);

        (p3_out, p4_out, p5_out)
    }
}
