use burn::module::Module;
use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig};
use burn::prelude::*;
use std::cell::Cell;

thread_local! {
    static FROZEN_BATCH_NORM: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn with_frozen_batch_norm<T>(frozen: bool, operation: impl FnOnce() -> T) -> T {
    FROZEN_BATCH_NORM.with(|state| {
        let previous = state.replace(frozen);
        let result = operation();
        state.set(previous);
        result
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    SiLU,
    Identity,
}

pub fn fuse_conv(conv: Conv2d, norm: BatchNorm) -> Conv2d {
    let device = conv.weight.val().device();
    let [out_ch, _in_ch, _k1, _k2] = conv.weight.dims();

    let gamma = norm.gamma.val();
    let beta = norm.beta.val();
    let running_mean = norm.running_mean.value().to_device(&device);
    let running_var = norm.running_var.value().to_device(&device);
    let scale = gamma / (running_var + norm.epsilon).sqrt();

    let conv_bias = match conv.bias.as_ref() {
        Some(bias) => bias.val(),
        None => Tensor::<1>::zeros([out_ch], &device),
    };

    // Folding is a model transformation. The resulting values must become new
    // trainable parameter leaves rather than retaining the temporary folding
    // graph.
    let fused_weight = (conv.weight.val() * scale.clone().reshape([out_ch, 1, 1, 1])).detach();
    let fused_bias = (beta + (conv_bias - running_mean) * scale).detach();

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
pub struct Conv {
    conv: Conv2d,
    norm: Option<BatchNorm>,
    #[module(skip)]
    relu: Activation,
}

pub(crate) trait FuseModule {
    fn fuse_module(self) -> Self;
}

impl Conv {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        activation: Activation,
        k: usize,
        s: usize,
        p: usize,
        g: usize,
        device: &Device,
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

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.conv.forward(x);
        let x = if let Some(ref norm) = self.norm {
            if FROZEN_BATCH_NORM.with(Cell::get) {
                norm.forward_inference::<4>(x)
            } else {
                norm.forward::<4>(x)
            }
        } else {
            x
        };
        match self.relu {
            Activation::SiLU => burn::tensor::activation::silu(x),
            Activation::Identity => x,
        }
    }

    pub fn fuse_forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.conv.forward(x);
        match self.relu {
            Activation::SiLU => burn::tensor::activation::silu(x),
            Activation::Identity => x,
        }
    }
}

impl FuseModule for Conv {
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
    use super::{fuse_conv, with_frozen_batch_norm, Activation, Conv};
    use burn::module::{Param, RunningState};
    use burn::nn::conv::Conv2dConfig;
    use burn::nn::BatchNormConfig;
    use burn::nn::PaddingConfig2d;
    use burn::prelude::*;

    #[test]
    fn fuse_conv_matches_python_weight_and_bias_formula() {
        let device = burn::tensor::Device::flex();
        let mut conv = Conv2dConfig::new([1, 2], [1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
            .with_bias(false)
            .init(&device);
        conv.weight = Param::from_tensor(
            Tensor::<1>::from_floats([2.0, -3.0], &device).reshape([2, 1, 1, 1]),
        );

        let mut norm = BatchNormConfig::new(2).with_epsilon(0.001).init(&device);
        norm.gamma = Param::from_tensor(Tensor::<1>::from_floats([1.5, -2.0], &device));
        norm.beta = Param::from_tensor(Tensor::<1>::from_floats([0.25, 0.75], &device));
        norm.running_mean = RunningState::new(Tensor::<1>::from_floats([0.5, -1.0], &device));
        norm.running_var = RunningState::new(Tensor::<1>::from_floats([4.0, 0.25], &device));

        let fused = fuse_conv(conv, norm);
        let weight = fused.weight.val().to_data().try_to_vec::<f32>().unwrap();
        let bias = fused
            .bias
            .as_ref()
            .unwrap()
            .val()
            .to_data()
            .try_to_vec::<f32>()
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

    #[test]
    fn frozen_batch_norm_uses_running_statistics_without_updating_them() {
        let device = Device::flex().autodiff();
        let conv = Conv::new(1, 1, Activation::Identity, 1, 1, 0, 1, &device);
        let input = Tensor::<4>::ones([2, 1, 4, 4], &device);
        let before = conv
            .norm
            .as_ref()
            .unwrap()
            .running_mean
            .value_sync()
            .to_data();

        let output = with_frozen_batch_norm(true, || conv.forward(input));
        let _gradients = output.sum().backward();
        let after = conv
            .norm
            .as_ref()
            .unwrap()
            .running_mean
            .value_sync()
            .to_data();

        assert_eq!(before, after);
    }
}
