pub struct CosineLR {
    total_lr: Vec<f64>,
}

impl CosineLR {
    pub fn new(
        min_lr: f64,
        max_lr: f64,
        warmup_epochs: f64,
        num_steps_per_epoch: usize,
        total_epochs: usize,
    ) -> Self {
        let warmup_steps = ((warmup_epochs * num_steps_per_epoch as f64) as usize).max(100);
        let total_steps = num_steps_per_epoch * total_epochs;
        let decay_steps = total_steps.saturating_sub(warmup_steps);

        let mut total_lr = Vec::with_capacity(total_steps);
        for step in 0..warmup_steps {
            let t = if warmup_steps > 1 {
                step as f64 / (warmup_steps - 1) as f64
            } else {
                0.0
            };
            total_lr.push(min_lr + (max_lr - min_lr) * t);
        }
        if decay_steps > 0 {
            for step in 1..=decay_steps {
                let alpha = (std::f64::consts::PI * step as f64 / decay_steps as f64).cos();
                total_lr.push(min_lr + 0.5 * (max_lr - min_lr) * (1.0 + alpha));
            }
        }

        Self { total_lr }
    }

    pub fn step(&self, step: usize) -> f64 {
        self.total_lr[step]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn cosine_lr_matches_python_warmup_and_decay_values() {
        let lr = CosineLR::new(0.0, 1.0, 0.0, 10, 12);

        assert_close(lr.step(0), 0.0);
        assert_close(lr.step(99), 1.0);
        assert_close(
            lr.step(100),
            0.5 * (1.0 + (std::f64::consts::PI / 20.0).cos()),
        );
        assert_close(lr.step(119), 0.0);
    }

    #[test]
    fn cosine_lr_handles_python_empty_decay_range() {
        let lr = CosineLR::new(0.0, 1.0, 0.0, 10, 5);

        assert_close(lr.step(0), 0.0);
        assert_close(lr.step(99), 1.0);
    }

    #[test]
    fn linear_lr_matches_python_endpoint_choices() {
        let lr = LinearLR::new(0.0, 1.0, 0.0, 10, 12);

        assert_close(lr.step(0), 0.0);
        assert_close(lr.step(99), 0.99);
        assert_close(lr.step(100), 1.0);
        assert_close(lr.step(119), 0.0);
    }

    #[test]
    #[should_panic(expected = "Number of samples")]
    fn linear_lr_rejects_python_negative_decay_sample_count() {
        let _ = LinearLR::new(0.0, 1.0, 0.0, 10, 5);
    }
}

pub struct LinearLR {
    total_lr: Vec<f64>,
}

impl LinearLR {
    pub fn new(
        min_lr: f64,
        max_lr: f64,
        warmup_epochs: f64,
        num_steps_per_epoch: usize,
        total_epochs: usize,
    ) -> Self {
        let warmup_steps = ((warmup_epochs * num_steps_per_epoch as f64) as usize).max(100);
        let total_steps = num_steps_per_epoch * total_epochs;
        let decay_steps = total_steps as isize - warmup_steps as isize;
        assert!(
            decay_steps >= 0,
            "Number of samples, {decay_steps}, must be non-negative"
        );
        let decay_steps = decay_steps as usize;

        let mut total_lr = Vec::with_capacity(total_steps);
        for step in 0..warmup_steps {
            let t = step as f64 / warmup_steps as f64;
            total_lr.push(min_lr + (max_lr - min_lr) * t);
        }
        for step in 0..decay_steps {
            let t = if decay_steps > 1 {
                step as f64 / (decay_steps - 1) as f64
            } else {
                0.0
            };
            total_lr.push(max_lr + (min_lr - max_lr) * t);
        }

        Self { total_lr }
    }

    pub fn step(&self, step: usize) -> f64 {
        self.total_lr[step]
    }
}
