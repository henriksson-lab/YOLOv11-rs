use anyhow::Result;
use burn::grad_clipping::GradientClippingConfig;

pub fn setup_seed() {
    crate::rng::set_seed(0);
}

pub fn setup_multi_processes() {
    std::env::set_var(
        "OMP_NUM_THREADS",
        std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "1".to_string()),
    );
    std::env::set_var(
        "MKL_NUM_THREADS",
        std::env::var("MKL_NUM_THREADS").unwrap_or_else(|_| "1".to_string()),
    );
}

pub fn export_onnx<A>(_args: &A) -> Result<()> {
    anyhow::bail!("ONNX export is not supported by the Burn translation")
}

pub fn smooth(y: &[f32], f: f32) -> Vec<f32> {
    let nf = ((y.len() as f32 * f * 2.0).round() as usize) / 2 + 1;
    let pad = nf / 2;
    let mut padded = Vec::with_capacity(y.len() + pad * 2);
    padded.extend(std::iter::repeat(y[0]).take(pad));
    padded.extend_from_slice(y);
    padded.extend(std::iter::repeat(*y.last().unwrap()).take(pad));

    (0..=padded.len() - nf)
        .map(|i| padded[i..i + nf].iter().sum::<f32>() / nf as f32)
        .collect()
}

pub fn plot_pr_curve(
    _px: &[f32],
    _py: &[Vec<f32>],
    _ap: &[Vec<f32>],
    _names: &[String],
    _save_dir: &str,
) -> Result<()> {
    anyhow::bail!("plot_pr_curve requires matplotlib-style plotting support not present in this Burn translation")
}

pub fn plot_curve(
    _px: &[f32],
    _py: &[Vec<f32>],
    _names: &[String],
    _save_dir: &str,
    _x_label: &str,
    _y_label: &str,
) -> Result<()> {
    anyhow::bail!("plot_curve requires matplotlib-style plotting support not present in this Burn translation")
}

pub fn strip_optimizer(_filename: &str) -> Result<()> {
    anyhow::bail!("strip_optimizer is not supported for Burn checkpoints")
}

pub fn clip_gradients(max_norm: f64) -> GradientClippingConfig {
    GradientClippingConfig::Norm(max_norm as f32)
}

pub fn load_weight<M>(_model: M, _ckpt: &str) -> Result<M> {
    anyhow::bail!("load_weight is handled by Burn record/safetensors loading in the CLI")
}

pub fn set_params<M>(_model: &M, _decay: f64) -> Result<()> {
    anyhow::bail!("set_params parameter grouping is not directly supported by Burn's SGD optimizer")
}

pub fn plot_lr<A, O, S>(
    _args: &A,
    _optimizer: &O,
    _scheduler: &S,
    _num_steps: usize,
) -> Result<()> {
    anyhow::bail!(
        "plot_lr requires matplotlib-style plotting support not present in this Burn translation"
    )
}

pub struct AverageMeter {
    pub num: usize,
    pub sum: f64,
    pub avg: f64,
}

impl AverageMeter {
    pub fn new() -> Self {
        Self {
            num: 0,
            sum: 0.0,
            avg: 0.0,
        }
    }

    pub fn update(&mut self, v: f64, n: usize) {
        if !v.is_nan() {
            self.num += n;
            self.sum += v * n as f64;
            assert_ne!(self.num, 0, "division by zero");
            self.avg = self.sum / self.num as f64;
        }
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
    fn smooth_matches_python_box_filter_with_edge_padding() {
        let y = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let smoothed = smooth(&y, 0.2);

        let expected = [1.0_f32, 1.5, 2.5, 3.5, 4.5, 5.0];
        assert_eq!(smoothed.len(), expected.len());
        for (actual, expected) in smoothed.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic]
    fn smooth_empty_input_matches_python_index_error() {
        let _ = smooth(&[], 0.1);
    }

    #[test]
    fn average_meter_matches_python_update_and_nan_skip() {
        let mut meter = AverageMeter::new();

        meter.update(2.0, 3);
        assert_eq!(meter.num, 3);
        assert_close(meter.sum, 6.0);
        assert_close(meter.avg, 2.0);

        meter.update(f64::NAN, 10);
        assert_eq!(meter.num, 3);
        assert_close(meter.sum, 6.0);
        assert_close(meter.avg, 2.0);

        meter.update(5.0, 1);
        assert_eq!(meter.num, 4);
        assert_close(meter.sum, 11.0);
        assert_close(meter.avg, 2.75);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn average_meter_update_zero_count_matches_python_zero_division() {
        let mut meter = AverageMeter::new();
        meter.update(1.0, 0);
    }

    #[test]
    fn clip_gradients_returns_burn_norm_clipping_config() {
        match clip_gradients(10.0) {
            GradientClippingConfig::Norm(value) => assert!((value - 10.0).abs() < f32::EPSILON),
            GradientClippingConfig::Value(_) => panic!("clip_gradients should use norm clipping"),
        }
    }

    #[test]
    fn unsupported_utility_stubs_preserve_python_call_shape() {
        let px = [0.0_f32, 1.0];
        let py = vec![vec![0.0_f32, 1.0]];
        let ap = vec![vec![0.5_f32]];
        let names = vec!["class".to_string()];
        let args = ();
        let optimizer = ();
        let scheduler = ();

        assert!(export_onnx(&args).is_err());
        assert!(plot_pr_curve(&px, &py, &ap, &names, "./weights/PR_curve.png").is_err());
        assert!(plot_curve(
            &px,
            &py,
            &names,
            "./weights/F1_curve.png",
            "Confidence",
            "Metric"
        )
        .is_err());
        assert!(strip_optimizer("./weights/best.pt").is_err());
        assert!(load_weight((), "./weights/best.pt").is_err());
        assert!(set_params(&(), 0.0005).is_err());
        assert!(plot_lr(&args, &optimizer, &scheduler, 10).is_err());
    }

    #[test]
    fn setup_seed_resets_shared_augmentation_rng() {
        let _lock = crate::rng::test_lock();
        setup_seed();
        let first = crate::data::resize::resample();
        let second = crate::data::resize::resample();

        setup_seed();
        assert_eq!(crate::data::resize::resample(), first);
        assert_eq!(crate::data::resize::resample(), second);
    }
}
