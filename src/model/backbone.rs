use burn::module::Module;
use burn::prelude::*;

use crate::model::attention::PSA;
use crate::model::blocks::{CSP, SPP};
use crate::model::conv::{Activation, Conv, FuseModule};

/// DarkNet backbone. Outputs feature maps at 3 scales: p3 (8x), p4 (16x), p5 (32x).
#[derive(Module, Debug)]
pub struct DarkNet<B: Backend> {
    p1: Conv<B>,
    p2_conv: Conv<B>,
    p2_csp: CSP<B>,
    p3_conv: Conv<B>,
    p3_csp: CSP<B>,
    p4_conv: Conv<B>,
    p4_csp: CSP<B>,
    p5_conv: Conv<B>,
    p5_csp: CSP<B>,
    p5_spp: SPP<B>,
    p5_psa: PSA<B>,
}

impl<B: Backend> DarkNet<B> {
    pub fn new(width: &[usize], depth: &[usize], csp: &[bool], device: &B::Device) -> Self {
        let p1 = Conv::new(width[0], width[1], Activation::SiLU, 3, 2, 1, 1, device);
        let p2_conv = Conv::new(width[1], width[2], Activation::SiLU, 3, 2, 1, 1, device);
        let p2_csp = CSP::new(width[2], width[3], depth[0], csp[0], 4, device);
        let p3_conv = Conv::new(width[3], width[3], Activation::SiLU, 3, 2, 1, 1, device);
        let p3_csp = CSP::new(width[3], width[4], depth[1], csp[0], 4, device);
        let p4_conv = Conv::new(width[4], width[4], Activation::SiLU, 3, 2, 1, 1, device);
        let p4_csp = CSP::new(width[4], width[4], depth[2], csp[1], 2, device);
        let p5_conv = Conv::new(width[4], width[5], Activation::SiLU, 3, 2, 1, 1, device);
        let p5_csp = CSP::new(width[5], width[5], depth[3], csp[1], 2, device);
        let p5_spp = SPP::new(width[5], width[5], 5, device);
        let p5_psa = PSA::new(width[5], depth[4], device);

        Self {
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
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let p1 = self.p1.forward(x);
        let p2 = self.p2_conv.forward(p1);
        let p2 = self.p2_csp.forward(p2);
        let p3 = self.p3_conv.forward(p2);
        let p3 = self.p3_csp.forward(p3);
        let p4 = self.p4_conv.forward(p3.clone());
        let p4 = self.p4_csp.forward(p4);
        let p5 = self.p5_conv.forward(p4.clone());
        let p5 = self.p5_csp.forward(p5);
        let p5 = self.p5_spp.forward(p5);
        let p5 = self.p5_psa.forward(p5);
        (p3, p4, p5)
    }
}

impl<B: Backend> FuseModule for DarkNet<B> {
    fn fuse_module(self) -> Self {
        Self {
            p1: self.p1.fuse_module(),
            p2_conv: self.p2_conv.fuse_module(),
            p2_csp: self.p2_csp.fuse_module(),
            p3_conv: self.p3_conv.fuse_module(),
            p3_csp: self.p3_csp.fuse_module(),
            p4_conv: self.p4_conv.fuse_module(),
            p4_csp: self.p4_csp.fuse_module(),
            p5_conv: self.p5_conv.fuse_module(),
            p5_csp: self.p5_csp.fuse_module(),
            p5_spp: self.p5_spp.fuse_module(),
            p5_psa: self.p5_psa.fuse_module(),
        }
    }
}
