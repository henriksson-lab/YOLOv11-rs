use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::prelude::*;
use burn::tensor::TensorData;
use std::collections::HashMap;

/// Collects all float parameter data from a module.
struct ParamCollector {
    params: HashMap<u64, TensorData>,
}

impl ParamCollector {
    fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }
}

impl ModuleVisitor for ParamCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.params.insert(param.id.val(), param.val().to_data());
    }
}

/// Maps each float parameter by blending: decay * current + (1 - decay) * source.
struct EmaMapper {
    source_params: HashMap<u64, TensorData>,
    decay: f64,
    device: Device,
}

impl ModuleMapper for EmaMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        if let Some(source_data) = self.source_params.get(&param.id.val()) {
            let source = Tensor::<D>::from_data(source_data.clone(), &self.device);
            // ema = decay * ema + (1 - decay) * model
            let blended = param.val() * self.decay + source * (1.0 - self.decay);
            param.map(|_| blended)
        } else {
            param
        }
    }
}

/// Exponential Moving Average of model parameters.
pub struct EMA<M: Module> {
    shadow: M,
    decay: f64,
    tau: f64,
    updates: usize,
}

impl<M: Module> EMA<M> {
    pub fn new(model: &M, decay: f64, tau: f64) -> Self {
        let shadow = model.clone();
        Self {
            shadow,
            decay,
            tau,
            updates: 0,
        }
    }

    /// Update EMA parameters from current model.
    /// ema = d * ema + (1-d) * model, where d ramps up over time.
    pub fn update(&mut self, model: &M, device: &Device) {
        self.updates += 1;
        let d = self.decay * (1.0 - (-((self.updates as f64) / self.tau)).exp());

        // Collect current model parameters
        let mut collector = ParamCollector::new();
        model.visit(&mut collector);

        // Blend shadow parameters with model parameters
        let mut mapper = EmaMapper {
            source_params: collector.params,
            decay: d,
            device: device.clone(),
        };
        self.shadow = self.shadow.clone().map(&mut mapper);
    }

    /// Get a reference to the EMA model for evaluation.
    pub(crate) fn model(&self) -> &M {
        &self.shadow
    }
}

#[cfg(test)]
mod tests {
    use super::EMA;
    use burn::module::{Module, Param};
    use burn::prelude::*;

    #[derive(Module, Debug)]
    struct TinyModule {
        weight: Param<Tensor<1>>,
    }

    impl TinyModule {
        fn new(value: f32, device: &Device) -> Self {
            Self {
                weight: Param::from_tensor(Tensor::from_floats([value], device)),
            }
        }

        fn with_weight(mut self, value: f32, device: &Device) -> Self {
            self.weight = self.weight.map(|_| Tensor::from_floats([value], device));
            self
        }

        fn weight_value(&self) -> f32 {
            self.weight.val().to_data().try_to_vec::<f32>().unwrap()[0]
        }
    }

    #[test]
    fn ema_update_matches_python_decay_ramp_fixture() {
        let device = burn::tensor::Device::flex();
        let model = TinyModule::new(2.0, &device);
        let mut ema = EMA::new(&model, 0.9, 2.0);

        let model_after_first_update = model.clone().with_weight(10.0, &device);
        ema.update(&model_after_first_update, &device);
        let d1 = 0.9 * (1.0 - (-1.0f64 / 2.0).exp());
        let expected_first = 2.0 * d1 as f32 + 10.0 * (1.0 - d1 as f32);
        assert!((ema.model().weight_value() - expected_first).abs() < 1e-6);

        let model_after_second_update = model_after_first_update.with_weight(-4.0, &device);
        ema.update(&model_after_second_update, &device);
        let d2 = 0.9 * (1.0 - (-2.0f64 / 2.0).exp());
        let expected_second = expected_first * d2 as f32 + -4.0 * (1.0 - d2 as f32);
        assert!((ema.model().weight_value() - expected_second).abs() < 1e-6);
    }
}
