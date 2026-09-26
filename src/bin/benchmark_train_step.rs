use std::time::Instant;

use anyhow::{bail, Result};
use burn::optim::{GradientsParams, SgdConfig};
use burn::prelude::*;
use clap::Parser;
use serde::Serialize;
use yolov11::model::loss::ComputeLoss;
use yolov11::model::model::{yolo_v11_n, YOLOOutput, YOLO};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 4)]
    batch_size: usize,
    #[arg(long, default_value_t = 640)]
    input_size: usize,
    #[arg(long, default_value_t = 32)]
    targets_per_image: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 5)]
    iterations: usize,
    /// Optimizer learning rate. Defaults to the reference trainer's initial LR.
    #[arg(long, default_value_t = 0.0001)]
    learning_rate: f64,
    /// Replace the YOLO loss with the sum of every raw model output.
    #[arg(long, default_value_t = false)]
    simple_loss: bool,
    /// Fold BatchNorm into convolution weights before training.
    #[arg(long, default_value_t = false)]
    fused_batch_norm: bool,
    /// Diagnostic: all, box, cls, or dfl.
    #[arg(long, default_value = "all")]
    loss_kind: String,
}

#[derive(Serialize)]
struct Report {
    backend: &'static str,
    release_build: bool,
    batch_size: usize,
    input_size: usize,
    targets_per_image: usize,
    iterations: usize,
    learning_rate: f64,
    simple_loss: bool,
    fused_batch_norm: bool,
    forward_ms: Vec<f64>,
    loss_ms: Vec<f64>,
    backward_ms: Vec<f64>,
    optimizer_ms: Vec<f64>,
    step_ms: Vec<f64>,
    loss_values: Vec<f32>,
}

fn sync(device: &Device) -> Result<()> {
    device
        .sync()
        .map_err(|error| anyhow::anyhow!("device sync: {error}"))
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        bail!("training benchmark requires a release build");
    }
    let args = Args::parse();
    if args.batch_size == 0 || args.iterations == 0 {
        bail!("batch size and iterations must be positive");
    }

    #[cfg(feature = "libtorch")]
    let device = Device::libtorch_cuda(burn::tensor::DeviceIndex::Default).autodiff();
    #[cfg(all(feature = "cuda", not(feature = "libtorch")))]
    let device = Device::cuda(burn::tensor::DeviceIndex::Default).autodiff();
    #[cfg(not(any(feature = "cuda", feature = "libtorch")))]
    let device = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice).autodiff();

    #[cfg(feature = "libtorch")]
    let backend = "libtorch-cuda";
    #[cfg(all(feature = "cuda", not(feature = "libtorch")))]
    let backend = "cubecl-cuda";
    #[cfg(not(any(feature = "cuda", feature = "libtorch")))]
    let backend = "wgpu";

    device.seed(0);
    let mut model = yolo_v11_n(1, &device);
    if args.fused_batch_norm {
        model = model.fuse();
    }
    let criterion = ComputeLoss::new(1, 3, &model.stride, 7.5, 0.5, 1.5);
    let mut optimizer = SgdConfig::new().init();

    let images = Tensor::<4>::random(
        [args.batch_size, 3, args.input_size, args.input_size],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let target_count = args.batch_size * args.targets_per_image;
    let cls = Tensor::<2>::zeros([target_count.max(1), 1], &device);
    let mut boxes = Vec::with_capacity(target_count.max(1) * 4);
    let mut indices = Vec::with_capacity(target_count.max(1));
    for batch in 0..args.batch_size {
        for target in 0..args.targets_per_image {
            let column = target % 8;
            let row = (target / 8) % 8;
            boxes.extend_from_slice(&[
                (column as f32 + 0.5) / 8.0,
                (row as f32 + 0.5) / 8.0,
                0.04,
                0.04,
            ]);
            indices.push(batch as f32);
        }
    }
    if target_count == 0 {
        boxes.extend_from_slice(&[0.5, 0.5, 0.1, 0.1]);
        indices.push(-1.0);
    }
    let bbox =
        Tensor::<1>::from_floats(boxes.as_slice(), &device).reshape([target_count.max(1), 4]);
    let idx = Tensor::<1>::from_floats(indices.as_slice(), &device);

    let run_step =
        |model: YOLO, mut optimizer: burn::optim::ModuleOptimizer, record: bool| -> Result<_> {
            sync(&device)?;
            let step_started = Instant::now();
            let started = Instant::now();
            let outputs = match model.forward(images.clone(), true) {
                YOLOOutput::Train(outputs) => outputs,
                YOLOOutput::Infer(_) => unreachable!(),
            };
            sync(&device)?;
            let forward = started.elapsed();

            let started = Instant::now();
            let loss = if args.simple_loss {
                let mut loss = Tensor::<1>::zeros([1], &device);
                for output in &outputs {
                    loss = loss + output.clone().sum().reshape([1]);
                }
                loss
            } else {
                let (box_loss, cls_loss, dfl_loss) = criterion.call(
                    &outputs,
                    &cls,
                    &bbox,
                    &idx,
                    args.batch_size,
                    args.input_size,
                    &device,
                );
                match args.loss_kind.as_str() {
                    "all" => box_loss + cls_loss + dfl_loss,
                    "box" => box_loss,
                    "cls" => cls_loss,
                    "dfl" => dfl_loss,
                    other => bail!("unknown loss kind {other:?}; use all, box, cls, or dfl"),
                }
            };
            sync(&device)?;
            let loss_value = loss.clone().into_data().try_to_vec::<f32>().unwrap()[0];
            if !loss_value.is_finite() {
                bail!("training loss became non-finite: {loss_value}");
            }
            let loss_time = started.elapsed();

            let started = Instant::now();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            sync(&device)?;
            let backward = started.elapsed();

            let started = Instant::now();
            let model = optimizer.step(args.learning_rate, model, grads);
            sync(&device)?;
            let optimizer_time = started.elapsed();
            let step = step_started.elapsed();
            Ok((
                model,
                optimizer,
                record.then_some((
                    forward,
                    loss_time,
                    backward,
                    optimizer_time,
                    step,
                    loss_value,
                )),
            ))
        };

    for _ in 0..args.warmup {
        (model, optimizer, _) = run_step(model, optimizer, false)?;
    }

    let mut report = Report {
        backend,
        release_build: true,
        batch_size: args.batch_size,
        input_size: args.input_size,
        targets_per_image: args.targets_per_image,
        iterations: args.iterations,
        learning_rate: args.learning_rate,
        simple_loss: args.simple_loss,
        fused_batch_norm: args.fused_batch_norm,
        forward_ms: Vec::new(),
        loss_ms: Vec::new(),
        backward_ms: Vec::new(),
        optimizer_ms: Vec::new(),
        step_ms: Vec::new(),
        loss_values: Vec::new(),
    };
    for _ in 0..args.iterations {
        let timing;
        (model, optimizer, timing) = run_step(model, optimizer, true)?;
        let (forward, loss, backward, optimizer_time, step, loss_value) = timing.unwrap();
        report.forward_ms.push(forward.as_secs_f64() * 1000.0);
        report.loss_ms.push(loss.as_secs_f64() * 1000.0);
        report.backward_ms.push(backward.as_secs_f64() * 1000.0);
        report
            .optimizer_ms
            .push(optimizer_time.as_secs_f64() * 1000.0);
        report.step_ms.push(step.as_secs_f64() * 1000.0);
        report.loss_values.push(loss_value);
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
