use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{Optimizer, VarMap};
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use std::io::Write;

use crate::data::dataset::{self, Dataset, Sample};
use crate::model::loss::ComputeLoss;

use crate::train::config::Config;
use crate::train::ema::EMA;
use crate::train::eval;
use crate::train::lr_schedule::LinearLR;

pub fn train(
    config: &Config,
    data_dir: &str,
    input_size: usize,
    batch_size: usize,
    epochs: usize,
    device: &Device,
) -> Result<()> {
    let num_classes = config.num_classes();

    // Create model
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = crate::model::model::yolo_v11_n(num_classes, device, vb)?;

    // Optimizer
    let world_size = 1usize;
    let accumulate = (64.0 / (batch_size * world_size) as f64).round().max(1.0) as usize;

    let mut optimizer = candle_nn::SGD::new(varmap.all_vars(), config.min_lr)?;

    // Dataset
    let train_filenames = dataset::Dataset::load_filenames(
        &std::path::PathBuf::from(format!("{}/train2017.txt", data_dir)),
        &std::path::PathBuf::from(data_dir),
        "train2017",
    )?;
    let mut train_dataset = Dataset::new(
        train_filenames,
        input_size as u32,
        true,
        &config.to_augment_params(),
    )?;

    let val_filenames = dataset::Dataset::load_filenames(
        &std::path::PathBuf::from(format!("{}/val2017.txt", data_dir)),
        &std::path::PathBuf::from(data_dir),
        "val2017",
    )?;
    let val_dataset = Dataset::new(
        val_filenames,
        input_size as u32,
        false,
        &config.to_augment_params(),
    )?;

    // Scheduler
    let num_steps = (train_dataset.len() + batch_size - 1) / batch_size;
    let scheduler = LinearLR::new(
        config.min_lr,
        config.max_lr,
        config.momentum,
        config.warmup_epochs,
        num_steps,
        epochs,
    );

    // EMA
    let mut ema = EMA::new(&varmap, 0.9999, 2000.0)?;

    // Loss
    let criterion = ComputeLoss::new(
        num_classes,
        3, // nl = 3 detection layers
        model.stride.clone(),
        config.box_gain(),
        config.cls_gain(),
        config.dfl_gain(),
    );

    // CSV logger
    std::fs::create_dir_all("weights")?;
    let mut csv_file = std::fs::File::create("weights/step.csv")?;
    writeln!(csv_file, "epoch,box,cls,dfl,Recall,Precision,mAP@50,mAP")?;

    let mut best_map = 0.0f32;

    for epoch in 0..epochs {
        // Disable mosaic for last 10 epochs
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

        let mut avg_box = 0.0f64;
        let mut avg_cls = 0.0f64;
        let mut avg_dfl = 0.0f64;
        let mut count = 0usize;

        // Shuffle indices
        let mut indices: Vec<usize> = (0..train_dataset.len()).collect();
        indices.shuffle(&mut rand::thread_rng());

        for step_in_epoch in 0..num_steps {
            let global_step = step_in_epoch + num_steps * epoch;
            let (lr, _mom) = scheduler.get(global_step);
            optimizer.set_learning_rate(lr);

            // Build batch
            let start = step_in_epoch * batch_size;
            let end = (start + batch_size).min(train_dataset.len());
            if start >= train_dataset.len() {
                break;
            }

            let mut samples: Vec<Sample> = Vec::new();
            for &idx in &indices[start..end] {
                match train_dataset.get(idx, device) {
                    Ok(s) => samples.push(s),
                    Err(_) => continue,
                }
            }
            if samples.is_empty() {
                continue;
            }

            let batch = dataset::collate(&samples, device)?;
            let images = batch.images;

            // Forward
            let outputs = model.forward_train(&images)?;

            // Loss
            let (loss_box, loss_cls, loss_dfl) = criterion.compute(
                &outputs,
                &batch.cls,
                &batch.bbox,
                &batch.idx,
                samples.len(),
                input_size,
            )?;

            let total_loss = (&loss_box + &loss_cls)?.add(&loss_dfl)?;
            let scaled_loss =
                (total_loss * (batch_size as f64 * world_size as f64))?;

            // Optimizer step with gradient accumulation
            if global_step % accumulate == 0 {
                optimizer.backward_step(&scaled_loss)?;
                ema.update(&varmap)?;
            } else {
                // Accumulate: just compute backward without stepping
                let _grads = scaled_loss.backward()?;
            }

            // Track averages
            let lb: f32 = loss_box.to_scalar()?;
            let lc: f32 = loss_cls.to_scalar()?;
            let ld: f32 = loss_dfl.to_scalar()?;
            count += 1;
            avg_box += (lb as f64 - avg_box) / count as f64;
            avg_cls += (lc as f64 - avg_cls) / count as f64;
            avg_dfl += (ld as f64 - avg_dfl) / count as f64;

            pb.set_message(format!(
                "{:>5}/{:<5} {:>8} {:>8.3} {:>8.3} {:>8.3}",
                epoch + 1,
                epochs,
                "",
                avg_box,
                avg_cls,
                avg_dfl
            ));
            pb.inc(1);
        }
        pb.finish();

        // Evaluate with EMA weights
        let saved = ema.apply_and_save(&varmap)?;
        let (mean_ap, map50, recall, precision) =
            eval::test(&model, &val_dataset, num_classes, device, 4)?;
        EMA::restore(saved, &varmap)?;

        // Log
        writeln!(
            csv_file,
            "{:03},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            epoch + 1,
            avg_box,
            avg_cls,
            avg_dfl,
            recall,
            precision,
            map50,
            mean_ap
        )?;
        csv_file.flush()?;

        // Save checkpoint
        if mean_ap > best_map {
            best_map = mean_ap;
            varmap.save("weights/best.safetensors")?;
        }
        varmap.save("weights/last.safetensors")?;
    }

    println!("\nTraining complete. Best mAP: {:.3}", best_map);
    Ok(())
}
