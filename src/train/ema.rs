use candle_core::{Result, Tensor};
use candle_nn::VarMap;

/// Exponential Moving Average of model parameters.
pub struct EMA {
    /// Stored EMA parameter values, keyed by variable name.
    params: Vec<(String, Tensor)>,
    decay: f64,
    tau: f64,
    updates: usize,
}

impl EMA {
    pub fn new(varmap: &VarMap, decay: f64, tau: f64) -> Result<Self> {
        let data = varmap.data().lock().unwrap();
        let params: Vec<(String, Tensor)> = data
            .iter()
            .map(|(name, var)| {
                let t = var.as_tensor().detach().clone();
                (name.clone(), t)
            })
            .collect();
        Ok(Self {
            params,
            decay,
            tau,
            updates: 0,
        })
    }

    /// Update EMA parameters from current model.
    pub fn update(&mut self, varmap: &VarMap) -> Result<()> {
        self.updates += 1;
        let d = self.decay * (1.0 - (-((self.updates as f64) / self.tau)).exp());

        let data = varmap.data().lock().unwrap();
        for (name, ema_val) in &mut self.params {
            if let Some(var) = data.get(name) {
                let model_val = var.as_tensor();
                // ema = d * ema + (1-d) * model
                let new_val = ((d * &*ema_val)? + ((1.0 - d) * model_val)?)?;
                *ema_val = new_val.detach();
            }
        }
        Ok(())
    }

    /// Apply EMA weights to a VarMap (for evaluation).
    pub fn apply_to(&self, varmap: &VarMap) -> Result<()> {
        let mut data = varmap.data().lock().unwrap();
        for (name, ema_val) in &self.params {
            if let Some(var) = data.get_mut(name) {
                var.set(ema_val)?;
            }
        }
        Ok(())
    }

    /// Save current model weights, apply EMA, return guard to restore.
    pub fn apply_and_save(&self, varmap: &VarMap) -> Result<Vec<(String, Tensor)>> {
        let data = varmap.data().lock().unwrap();
        let saved: Vec<(String, Tensor)> = data
            .iter()
            .map(|(name, var)| (name.clone(), var.as_tensor().detach().clone()))
            .collect();
        drop(data);
        self.apply_to(varmap)?;
        Ok(saved)
    }

    /// Restore previously saved weights.
    pub fn restore(saved: Vec<(String, Tensor)>, varmap: &VarMap) -> Result<()> {
        let mut data = varmap.data().lock().unwrap();
        for (name, val) in saved {
            if let Some(var) = data.get_mut(&name) {
                var.set(&val)?;
            }
        }
        Ok(())
    }
}
