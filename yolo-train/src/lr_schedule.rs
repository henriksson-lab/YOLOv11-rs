/// Linear warmup then linear decay learning rate scheduler.
pub struct LinearLR {
    min_lr: f64,
    max_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    warmup_momentum_start: f64,
    target_momentum: f64,
}

impl LinearLR {
    pub fn new(
        min_lr: f64,
        max_lr: f64,
        momentum: f64,
        warmup_epochs: f64,
        num_steps_per_epoch: usize,
        total_epochs: usize,
    ) -> Self {
        let warmup_steps = (warmup_epochs * num_steps_per_epoch as f64) as usize;
        let total_steps = num_steps_per_epoch * total_epochs;
        Self {
            min_lr,
            max_lr,
            warmup_steps,
            total_steps,
            warmup_momentum_start: 0.8,
            target_momentum: momentum,
        }
    }

    /// Get (learning_rate, momentum) for a given global step.
    pub fn get(&self, step: usize) -> (f64, f64) {
        if step < self.warmup_steps {
            // Linear warmup
            let t = step as f64 / self.warmup_steps as f64;
            let lr = self.min_lr + (self.max_lr - self.min_lr) * t;
            let mom = self.warmup_momentum_start
                + (self.target_momentum - self.warmup_momentum_start) * t;
            (lr, mom)
        } else {
            // Linear decay from max_lr to min_lr
            let remaining = self.total_steps - self.warmup_steps;
            if remaining == 0 {
                return (self.min_lr, self.target_momentum);
            }
            let t = (step - self.warmup_steps) as f64 / remaining as f64;
            let lr = self.max_lr - (self.max_lr - self.min_lr) * t;
            (lr, self.target_momentum)
        }
    }
}
