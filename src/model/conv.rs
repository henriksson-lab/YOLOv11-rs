use burn::module::Module;
use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig};
use burn::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    SiLU,
    Identity,
}

pub fn fuse_conv<B: Backend>(conv: Conv2d<B>, norm: BatchNorm<B>) -> Conv2d<B> {
    let device = conv.weight.val().device();
    let [out_ch, _in_ch, _k1, _k2] = conv.weight.dims();

    let gamma = norm.gamma.val();
    let beta = norm.beta.val();
    let running_mean = norm.running_mean.value().to_device(&device);
    let running_var = norm.running_var.value().to_device(&device);
    let scale = gamma / (running_var + norm.epsilon).sqrt();

    let conv_bias = match conv.bias.as_ref() {
        Some(bias) => bias.val(),
        None => Tensor::<B, 1>::zeros([out_ch], &device),
    };

    let fused_weight = conv.weight.val() * scale.clone().reshape([out_ch, 1, 1, 1]);
    let fused_bias = beta + (conv_bias - running_mean) * scale;

    let mut fused = Conv2dConfig::new(
        [conv.weight.dims()[1] * conv.groups, out_ch],
        conv.kernel_size,
    )
    .with_stride(conv.stride)
    .with_padding(conv.padding.clone())
    .with_groups(conv.groups)
    .with_bias(true)
    .init(&device);
    fused.weight = Param::from_tensor(fused_weight);
    fused.bias = Some(Param::from_tensor(fused_bias));
    fused
}

#[derive(Module, Debug)]
pub struct Conv<B: Backend> {
    conv: Conv2d<B>,
    norm: Option<BatchNorm<B>>,
    relu: Activation,
}

pub(crate) trait FuseModule {
    fn fuse_module(self) -> Self;
}

impl<B: Backend> Conv<B> {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        activation: Activation,
        k: usize,
        s: usize,
        p: usize,
        g: usize,
        device: &B::Device,
    ) -> Self {
        let conv = Conv2dConfig::new([in_ch, out_ch], [k, k])
            .with_stride([s, s])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(p, p, p, p))
            .with_groups(g)
            .with_bias(false)
            .init(device);
        let norm = Some(
            BatchNormConfig::new(out_ch)
                .with_epsilon(0.001)
                .with_momentum(0.03)
                .init(device),
        );
        Self {
            conv,
            norm,
            relu: activation,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = if let Some(ref norm) = self.norm {
            norm.forward::<4>(x)
        } else {
            x
        };
        match self.relu {
            Activation::SiLU => burn::tensor::activation::silu(x),
            Activation::Identity => x,
        }
    }

    pub fn fuse_forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        match self.relu {
            Activation::SiLU => burn::tensor::activation::silu(x),
            Activation::Identity => x,
        }
    }
}

impl<B: Backend> FuseModule for Conv<B> {
    fn fuse_module(self) -> Self {
        let Self { conv, norm, relu } = self;
        match norm {
            Some(norm) => Self {
                conv: fuse_conv(conv, norm),
                norm: None,
                relu,
            },
            None => Self { conv, norm, relu },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fuse_conv;
    use burn::backend::NdArray;
    use burn::module::{Param, RunningState};
    use burn::nn::conv::Conv2dConfig;
    use burn::nn::BatchNormConfig;
    use burn::nn::PaddingConfig2d;
    use burn::prelude::*;

    type TestBackend = NdArray;

    #[test]
    fn fuse_conv_matches_python_weight_and_bias_formula() {
        let device = Default::default();
        let mut conv = Conv2dConfig::new([1, 2], [1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
            .with_bias(false)
            .init::<TestBackend>(&device);
        conv.weight = Param::from_tensor(
            Tensor::<TestBackend, 1>::from_floats([2.0, -3.0], &device).reshape([2, 1, 1, 1]),
        );

        let mut norm = BatchNormConfig::new(2)
            .with_epsilon(0.001)
            .init::<TestBackend>(&device);
        norm.gamma =
            Param::from_tensor(Tensor::<TestBackend, 1>::from_floats([1.5, -2.0], &device));
        norm.beta =
            Param::from_tensor(Tensor::<TestBackend, 1>::from_floats([0.25, 0.75], &device));
        norm.running_mean =
            RunningState::new(Tensor::<TestBackend, 1>::from_floats([0.5, -1.0], &device));
        norm.running_var =
            RunningState::new(Tensor::<TestBackend, 1>::from_floats([4.0, 0.25], &device));

        let fused = fuse_conv(conv, norm);
        let weight = fused.weight.val().to_data().to_vec::<f32>().unwrap();
        let bias = fused
            .bias
            .as_ref()
            .unwrap()
            .val()
            .to_data()
            .to_vec::<f32>()
            .unwrap();

        let scale0 = 1.5_f32 / (4.0_f32 + 0.001).sqrt();
        let scale1 = -2.0_f32 / (0.25_f32 + 0.001).sqrt();
        let expected_weight = [2.0 * scale0, -3.0 * scale1];
        let expected_bias = [0.25 + (0.0 - 0.5) * scale0, 0.75 + (0.0 - -1.0) * scale1];

        for (actual, expected) in weight.iter().zip(expected_weight) {
            assert!(
                (*actual - expected).abs() < 1e-6,
                "weight actual={actual} expected={expected}"
            );
        }
        for (actual, expected) in bias.iter().zip(expected_bias) {
            assert!(
                (*actual - expected).abs() < 1e-6,
                "bias actual={actual} expected={expected}"
            );
        }
    }
}
