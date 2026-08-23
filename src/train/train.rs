use anyhow::Result;
use burn::optim::{decay::WeightDecayConfig, GradientsParams, SgdConfig};
use burn::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::Write;
use std::path::Path;

use crate::data::dataset::{Dataset, Sample};
use crate::model::loss::ComputeLoss;
use crate::model::model::{yolo_v11_n, YOLOOutput, YOLO};

use crate::train::config::Config;
use crate::train::ema::EMA;
use crate::train::eval;
use crate::train::lr_schedule::LinearLR;
use crate::train::util::AverageMeter;

pub fn train(
    config: &Config,
    data_dir: &str,
    input_size: usize,
    batch_size: usize,
    epochs: usize,
    // Evaluate every this many epochs (1 = every epoch, as before). Validation
    // is the expensive phase on a small dataset, both in time and — because the
    // inference shape set is not the training one — in device memory.
    eval_interval: usize,
    device: &Device,
) -> Result<()> {
    let num_classes = config.num_classes();

    // Create model
    let model: YOLO = yolo_v11_n(num_classes, device);

    // Optimizer
    let world_size = 1usize;
    // Dataset
    let data_dir_path = Path::new(data_dir);
    let train_filenames = std::fs::read_to_string(data_dir_path.join("train2017.txt"))?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let basename = Path::new(line.trim()).file_name().unwrap().to_owned();
            data_dir_path
                .join("images")
                .join("train2017")
                .join(basename)
        })
        .collect();
    let mut train_dataset = Dataset::new(
        train_filenames,
        input_size as u32,
        true,
        &config.to_augment_params(),
    )?;

    let val_filenames = std::fs::read_to_string(data_dir_path.join("val2017.txt"))?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let basename = Path::new(line.trim()).file_name().unwrap().to_owned();
            data_dir_path.join("images").join("val2017").join(basename)
        })
        .collect();
    let val_dataset = Dataset::new(
        val_filenames,
        input_size as u32,
        false,
        &config.to_augment_params(),
    )?;

    // Optimizer
    //
    // `accumulate` is capped at the number of steps in an epoch. The nominal
    // `64 / batch_size` assumes an epoch long enough to contain it; on a small
    // dataset it is not, and the optimizer would then step roughly once per
    // epoch while `pending_loss` carried an autodiff graph over every step of
    // it. That is a fourfold difference in device memory here, and it is what
    // used to exhaust a 16 GB card by epoch 5.
    let num_steps = (train_dataset.len() + batch_size - 1) / batch_size;
    let accumulate = ((64.0 / (batch_size * world_size) as f64).round().max(1.0) as usize)
        .min(num_steps.max(1));
    let weight_decay = config.weight_decay * (batch_size * world_size * accumulate) as f64 / 64.0;

    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig {
            penalty: weight_decay as f32,
        }))
        .with_momentum(Some(burn::optim::momentum::MomentumConfig {
            momentum: config.momentum,
            dampening: 0.0,
            nesterov: true,
        }))
        .init();

    // Scheduler
    let scheduler = LinearLR::new(
        config.min_lr,
        config.max_lr,
        config.warmup_epochs,
        num_steps,
        epochs,
    );

    // EMA
    let mut ema = EMA::new(&model, 0.9999, 2000.0);

    // Loss
    let criterion = ComputeLoss::new(
        num_classes,
        3,
        &model.stride,
        config.box_gain(),
        config.cls_gain(),
        config.dfl_gain(),
    );

    // CSV logger
    std::fs::create_dir_all("weights")?;
    let mut csv_file = std::fs::File::create("weights/step.csv")?;
    writeln!(csv_file, "epoch,box,cls,dfl,Recall,Precision,mAP@50,mAP")?;

    let mut best_map = 0.0f32;
    let mut model = model;

    for epoch in 0..epochs {
        if epochs - epoch == 10 {
            train_dataset.mosaic = false;
        }

        println!(
            "\n{:>10}{:>10}{:>10}{:>10}{:>10}",
            "epoch", "memory", "box", "cls", "dfl"
        );

        let pb = ProgressBar::new(num_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40}] {pos}/{len}")
                .unwrap(),
        );

        let mut avg_box_loss = AverageMeter::new();
        let mut avg_cls_loss = AverageMeter::new();
        let mut avg_dfl_loss = AverageMeter::new();
        let mut pending_loss: Option<Tensor<1>> = None;

        let mut indices: Vec<usize> = (0..train_dataset.len()).collect();
        crate::rng::shuffle(&mut indices);

        for step_in_epoch in 0..num_steps {
            let global_step = step_in_epoch + num_steps * epoch;
            let lr = scheduler.step(global_step);

            let start = step_in_epoch * batch_size;
            let end = (start + batch_size).min(train_dataset.len());
            if start >= train_dataset.len() {
                break;
            }

            let mut samples: Vec<Sample> = Vec::new();
            for &idx in &indices[start..end] {
                samples.push(train_dataset.get_item(idx, device)?);
            }

            let batch = Dataset::collate_fn(&samples, device);
            let images = batch.images / 255.0;

            // Forward
            let outputs = match model.forward(images, true) {
                YOLOOutput::Train(outputs) => outputs,
                YOLOOutput::Infer(_) => unreachable!("training forward returned inference output"),
            };

            // Loss
            let (loss_box, loss_cls, loss_dfl) = criterion.call(
                &outputs,
                &batch.cls,
                &batch.bbox,
                &batch.idx,
                samples.len(),
                input_size,
                device,
            );

            let total_loss = loss_box.clone() + loss_cls.clone() + loss_dfl.clone();
            let scaled_loss = total_loss * (batch_size as f64 * world_size as f64);
            pending_loss = Some(match pending_loss.take() {
                Some(loss) => loss + scaled_loss,
                None => scaled_loss,
            });

            // Optimizer step with gradient accumulation.
            if global_step % accumulate == 0 {
                let grads = pending_loss
                    .take()
                    .expect("pending loss is set before optimizer step")
                    .backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(lr, model, grads);
                ema.update(&model, device);
            }

            let lb: f32 = loss_box.into_data().try_to_vec::<f32>().unwrap()[0];
            let lc: f32 = loss_cls.into_data().try_to_vec::<f32>().unwrap()[0];
            let ld: f32 = loss_dfl.into_data().try_to_vec::<f32>().unwrap()[0];
            avg_box_loss.update(lb as f64, samples.len());
            avg_cls_loss.update(lc as f64, samples.len());
            avg_dfl_loss.update(ld as f64, samples.len());

            pb.set_message(format!(
                "{:>5}/{:<5} {:>8} {:>8.3} {:>8.3} {:>8.3}",
                epoch + 1,
                epochs,
                "",
                avg_box_loss.avg,
                avg_cls_loss.avg,
                avg_dfl_loss.avg
            ));
            pb.inc(1);
        }
        pb.finish();

        // Evaluate with EMA model. The last epoch always evaluates, so a run
        // ends with a measured model however the interval divides.
        let evaluated = eval_interval <= 1
            || epoch + 1 == epochs
            || (epoch + 1) % eval_interval == 0;

        let save_model = ema.model().clone();
        save_model
            .clone()
            .save_file("weights/last")
            .map_err(|e| anyhow::anyhow!("Failed to save model: {}", e))?;

        if evaluated {
            let ema_model = ema.model();
            let (mean_ap, map50, recall, precision) =
                eval::test(ema_model, &val_dataset, device, 4)?;

            // Log
            writeln!(
                csv_file,
                "{:03},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
                epoch + 1,
                avg_box_loss.avg,
                avg_cls_loss.avg,
                avg_dfl_loss.avg,
                recall,
                precision,
                map50,
                mean_ap
            )?;
            csv_file.flush()?;

            // Save EMA checkpoint like Python's copied `ema.ema` model.
            if mean_ap > best_map {
                best_map = mean_ap;
                save_model
                    .save_file("weights/best")
                    .map_err(|e| anyhow::anyhow!("Failed to save model: {}", e))?;
            }
        }
    }

    println!("\nTraining complete. Best mAP: {:.3}", best_map);
    Ok(())
}
