use burn::module::{Module, ModuleMapper, ModuleVisitor, ParamId};
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

impl<B: Backend> ModuleVisitor<B> for ParamCollector {
    fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D>) {
        self.params.insert(id.val(), tensor.to_data());
    }
}

/// Maps each float parameter by blending: decay * current + (1 - decay) * source.
struct EmaMapper<B: Backend> {
    source_params: HashMap<u64, TensorData>,
    decay: f64,
    device: B::Device,
}

impl<B: Backend> ModuleMapper<B> for EmaMapper<B> {
    fn map_float<const D: usize>(&mut self, id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        if let Some(source_data) = self.source_params.get(&id.val()) {
            let source = Tensor::<B, D>::from_data(source_data.clone(), &self.device);
            // ema = decay * ema + (1 - decay) * model
            tensor * self.decay + source * (1.0 - self.decay)
        } else {
            tensor
        }
    }
}

/// Exponential Moving Average of model parameters.
pub struct EMA<B: Backend, M: Module<B>> {
    shadow: M,
    decay: f64,
    tau: f64,
    updates: usize,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend, M: Module<B>> EMA<B, M> {
    pub fn new(model: &M, decay: f64, tau: f64) -> Self {
        let shadow = model.clone();
        Self {
            shadow,
            decay,
            tau,
            updates: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Update EMA parameters from current model.
    /// ema = d * ema + (1-d) * model, where d ramps up over time.
    pub fn update(&mut self, model: &M, device: &B::Device) {
        self.updates += 1;
        let d = self.decay * (1.0 - (-((self.updates as f64) / self.tau)).exp());

        // Collect current model parameters
        let mut collector = ParamCollector::new();
        model.visit(&mut collector);

        // Blend shadow parameters with model parameters
        let mut mapper = EmaMapper::<B> {
            source_params: collector.params,
            decay: d,
            device: device.clone(),
        };
        self.shadow = self.shadow.clone().map(&mut mapper);
    }

    /// Get a reference to the EMA model for evaluation.
    pub fn model(&self) -> &M {
        &self.shadow
    }
}
