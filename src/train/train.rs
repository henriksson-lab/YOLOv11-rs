use anyhow::{Context, Result};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, ParamGroup};
use burn::optim::{decay::WeightDecayConfig, GradientsAccumulator, GradientsParams, SgdConfig};
use burn::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::data::loader::{orientation_for, ParallelBatchLoader, PlannedSample};
use crate::data::source::{FileSampleStore, GeometryPolicy, SampleStore, SharedSampleStore};
use crate::model::loss::ComputeLoss;
use crate::model::metrics;
use crate::model::model::{yolo_v11_n, YOLOOutput, YOLO};
use crate::model::nms;
use crate::train::config::Config;
use crate::train::ema::EMA;
use crate::train::lr_schedule::LinearLR;
use crate::train::progress::write_training_progress_svg;
use crate::train::util::AverageMeter;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrainableLayers {
    #[default]
    All,
    /// Freeze the DarkNet feature extractor while training the neck and head.
    HeadAndNeck,
    /// Freeze the DarkNet feature extractor and feature pyramid, training only
    /// the decoupled detection head.
    HeadOnly,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BatchNormPolicy {
    #[default]
    Update,
    /// Normalize with stored population statistics and do not update them.
    Frozen,
}

#[derive(Clone, Debug)]
pub struct TrainingOptions {
    pub input_size: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub eval_interval: usize,
    pub loader_workers: usize,
    pub queue_batches: usize,
    pub seed: u64,
    pub geometry: GeometryPolicy,
    pub weights: Option<PathBuf>,
    pub reset_class_head: bool,
    pub gradient_clip: Option<f32>,
    pub trainable_layers: TrainableLayers,
    pub batch_norm: BatchNormPolicy,
    pub evaluation_confidence_threshold: f32,
    pub evaluation_iou_threshold: f32,
    pub evaluation_max_detections: usize,
    /// Total epoch horizon used by the learning-rate schedule.
    pub schedule_epochs: Option<usize>,
    /// Resume an exact staged run from this run directory.
    pub resume: Option<PathBuf>,
    /// Evaluate the held-out test store once after this stage.
    pub finalize: bool,
    pub output_dir: PathBuf,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub mean_ap: f32,
    pub map50: f32,
    pub recall: f32,
    pub precision: f32,
    #[serde(default)]
    pub confidence_threshold: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TrainingSummary {
    pub stage: usize,
    pub completed_epochs: usize,
    pub schedule_epochs: usize,
    pub best_epoch: usize,
    pub best_validation: EvaluationMetrics,
    pub test: Option<EvaluationMetrics>,
}

impl Default for TrainingOptions {
    fn default() -> Self {
        Self {
            input_size: 640,
            batch_size: 32,
            epochs: 600,
            eval_interval: 1,
            loader_workers: 4,
            queue_batches: 2,
            seed: 0,
            geometry: GeometryPolicy::None,
            weights: None,
            reset_class_head: false,
            gradient_clip: None,
            trainable_layers: TrainableLayers::All,
            batch_norm: BatchNormPolicy::Update,
            evaluation_confidence_threshold: 0.001,
            evaluation_iou_threshold: 0.65,
            evaluation_max_detections: 300,
            schedule_epochs: None,
            resume: None,
            finalize: true,
            output_dir: PathBuf::from("weights"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ResumeState {
    version: u32,
    stage: usize,
    completed_epochs: usize,
    optimizer_steps: usize,
    ema_updates: usize,
    schedule_epochs: usize,
    nominal_steps: usize,
    train_samples: usize,
    batch_size: usize,
    input_size: usize,
    seed: u64,
    geometry: String,
    eval_interval: usize,
    training_config: String,
    best_epoch: usize,
    best_validation: EvaluationMetrics,
    checkpoint_dir: String,
}

fn load_weights(mut model: YOLO, path: &Path, reset_class_head: bool) -> Result<YOLO> {
    let path_string = path.to_string_lossy();
    if matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("pt" | "pth")
    ) {
        anyhow::bail!("direct .pt loading is unsupported; convert it to .safetensors first");
    }
    if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
        use burn_store::ModuleSnapshot;
        let mut store = burn_store::SafetensorsStore::from_file(path_string.as_ref());
        if reset_class_head {
            store = store
                .with_predicate(|path, _| !path.starts_with("head.cls_c4."))
                .allow_partial(true);
        }
        let result = model
            .load_from(&mut store)
            .map_err(|error| anyhow::anyhow!("failed to load {}: {error}", path.display()))?;
        println!(
            "Loaded {} compatible tensors{}",
            result.applied.len(),
            if reset_class_head {
                "; class output head remains newly initialized"
            } else {
                ""
            }
        );
        Ok(model)
    } else {
        model
            .try_load_file(path_string.as_ref())
            .map_err(|error| anyhow::anyhow!("failed to load {}: {error}", path.display()))
    }
}

fn save_weights(model: &YOLO, path: &Path) -> Result<()> {
    model
        .clone()
        .save_file(path)
        .map_err(|error| anyhow::anyhow!("failed to save {}: {error}", path.display()))
}

fn geometry_name(geometry: GeometryPolicy) -> &'static str {
    match geometry {
        GeometryPolicy::None => "none",
        GeometryPolicy::D4 => "d4",
    }
}

fn training_config_signature(config: &Config, options: &TrainingOptions) -> String {
    let mut names = config.names.iter().collect::<Vec<_>>();
    names.sort_by_key(|(index, _)| **index);
    let mut signature = format!(
        "min_lr={:.17};max_lr={:.17};momentum={:.17};weight_decay={:.17};warmup_epochs={:.17};box={:.17};cls={:.17};dfl={:.17};gradient_clip={gradient_clip:?};names={names:?}",
        config.min_lr,
        config.max_lr,
        config.momentum,
        config.weight_decay,
        config.warmup_epochs,
        config.box_gain(),
        config.cls_gain(),
        config.dfl_gain(),
        gradient_clip = options.gradient_clip,
    )
    ;
    // Preserve compatibility with version-2 checkpoints created before these
    // policies existed. Non-default policies are always recorded.
    if options.trainable_layers != TrainableLayers::All {
        signature.push_str(&format!(";trainable_layers={:?}", options.trainable_layers));
    }
    if options.batch_norm != BatchNormPolicy::Update {
        signature.push_str(&format!(";batch_norm={:?}", options.batch_norm));
    }
    signature
}

fn read_resume_state(run: &Path) -> Result<ResumeState> {
    let path = run.join("resume.json");
    let bytes =
        std::fs::read(&path).with_context(|| format!("reading resume state {}", path.display()))?;
    let state: ResumeState = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing resume state {}", path.display()))?;
    anyhow::ensure!(
        state.version == 2,
        "unsupported resume format version {}",
        state.version
    );
    Ok(state)
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}

fn open_history(
    output: &Path,
    resume: Option<&Path>,
    completed_epochs: usize,
) -> Result<std::fs::File> {
    let output_history = output.join("step.csv");
    if let Some(run) = resume {
        let source = run.join("step.csv");
        anyhow::ensure!(
            source.is_file(),
            "resume history {} is missing",
            source.display()
        );
        let contents = std::fs::read_to_string(&source)?;
        let mut lines = contents.lines();
        let header = lines.next().context("resume history is empty")?;
        anyhow::ensure!(
            header == "stage,epoch,learning_rate,evaluated,box,cls,dfl,Recall,Precision,mAP@50,mAP",
            "resume history has an unsupported header"
        );
        let mut committed = String::from(header);
        committed.push('\n');
        for line in lines.filter(|line| !line.trim().is_empty()) {
            let epoch = line
                .split(',')
                .nth(1)
                .context("resume history row has no epoch")?
                .parse::<usize>()
                .context("resume history epoch is invalid")?;
            if epoch <= completed_epochs {
                committed.push_str(line);
                committed.push('\n');
            }
        }
        std::fs::write(&output_history, committed)
            .with_context(|| format!("preparing resume history from {}", source.display()))?;
        return Ok(OpenOptions::new().append(true).open(&output_history)?);
    }
    let mut file = std::fs::File::create(output_history)?;
    writeln!(
        file,
        "stage,epoch,learning_rate,evaluated,box,cls,dfl,Recall,Precision,mAP@50,mAP"
    )?;
    Ok(file)
}

fn epoch_plan(
    store: &dyn SampleStore,
    epoch: usize,
    seed: u64,
    geometry: GeometryPolicy,
) -> Vec<PlannedSample> {
    store
        .epoch_keys(epoch, seed)
        .into_iter()
        .map(|key| PlannedSample {
            key,
            orientation: match geometry {
                GeometryPolicy::None => 0,
                GeometryPolicy::D4 => orientation_for(seed, epoch, key),
            },
        })
        .collect()
}

/// Train from source-neutral stores. Storage reads and CPU transforms run in a
/// bounded worker pool; model updates remain ordered on the caller thread.
pub fn train_with_stores(
    config: &Config,
    train_store: SharedSampleStore,
    validation_store: Option<SharedSampleStore>,
    options: &TrainingOptions,
    device: &Device,
) -> Result<()> {
    train_with_stores_and_test(config, train_store, validation_store, None, options, device)?;
    Ok(())
}

/// Evaluate a saved model on a source-neutral store without modifying it.
pub fn evaluate_checkpoint_with_store(
    config: &Config,
    checkpoint: &Path,
    store: SharedSampleStore,
    options: &TrainingOptions,
    device: &Device,
) -> Result<EvaluationMetrics> {
    let model = load_weights(yolo_v11_n(config.num_classes(), device), checkpoint, false)?;
    evaluate_store(&model, store, options, device)
}

/// Train from source-neutral stores and evaluate the validation-selected model
/// once on an optional held-out test store.
pub fn train_with_stores_and_test(
    config: &Config,
    train_store: SharedSampleStore,
    validation_store: Option<SharedSampleStore>,
    test_store: Option<SharedSampleStore>,
    options: &TrainingOptions,
    device: &Device,
) -> Result<TrainingSummary> {
    anyhow::ensure!(!train_store.is_empty(), "training store is empty");
    anyhow::ensure!(options.batch_size > 0, "batch size must be positive");
    anyhow::ensure!(options.epochs > 0, "epoch count must be positive");
    anyhow::ensure!(
        options.weights.is_none() || options.resume.is_none(),
        "--weights is a warm start and cannot be combined with --resume"
    );

    let num_classes = config.num_classes();
    let nominal_steps = train_store.len().div_ceil(options.batch_size);
    let training_config = training_config_signature(config, options);
    let resume_state = options
        .resume
        .as_deref()
        .map(read_resume_state)
        .transpose()?;
    let (stage, start_epoch, schedule_epochs, mut optimizer_steps) =
        if let Some(state) = &resume_state {
            anyhow::ensure!(
                state.train_samples == train_store.len(),
                "training sample count changed"
            );
            anyhow::ensure!(
                state.nominal_steps == nominal_steps,
                "steps per epoch changed"
            );
            anyhow::ensure!(state.batch_size == options.batch_size, "batch size changed");
            anyhow::ensure!(state.input_size == options.input_size, "input size changed");
            anyhow::ensure!(state.seed == options.seed, "training seed changed");
            anyhow::ensure!(
                state.eval_interval == options.eval_interval,
                "validation interval changed"
            );
            anyhow::ensure!(
                state.training_config == training_config,
                "optimizer, loss, learning-rate, clipping, or class configuration changed"
            );
            anyhow::ensure!(
                state.geometry == geometry_name(options.geometry),
                "geometry policy changed"
            );
            if let Some(requested) = options.schedule_epochs {
                anyhow::ensure!(
                    requested == state.schedule_epochs,
                    "schedule horizon changed"
                );
            }
            (
                state.stage + 1,
                state.completed_epochs,
                state.schedule_epochs,
                state.optimizer_steps,
            )
        } else {
            (1, 0, options.schedule_epochs.unwrap_or(options.epochs), 0)
        };
    anyhow::ensure!(
        start_epoch + options.epochs <= schedule_epochs,
        "this stage would end at epoch {}, past the schedule horizon of {}",
        start_epoch + options.epochs,
        schedule_epochs
    );

    let checkpoint = resume_state.as_ref().map(|state| {
        options
            .resume
            .as_ref()
            .expect("resume state requires a run directory")
            .join(&state.checkpoint_dir)
    });
    let mut model: YOLO = yolo_v11_n(num_classes, device);
    if let Some(checkpoint) = &checkpoint {
        model = load_weights(model, &checkpoint.join("train.bpk"), false)?;
    } else if let Some(path) = &options.weights {
        println!("Loading warm-start weights from {}", path.display());
        model = load_weights(model, path, options.reset_class_head)?;
    }
    model = match options.trainable_layers {
        TrainableLayers::All => model,
        TrainableLayers::HeadAndNeck => model.freeze_backbone(),
        TrainableLayers::HeadOnly => model.freeze_backbone_and_neck(),
    };
    let accumulate =
        ((64.0 / options.batch_size as f64).round().max(1.0) as usize).min(nominal_steps.max(1));
    let weight_decay = config.weight_decay * (options.batch_size * accumulate) as f64 / 64.0;
    let momentum = burn::optim::momentum::MomentumConfig {
        momentum: config.momentum,
        dampening: 0.0,
        nesterov: true,
    };
    // Match Python set_params(): biases and normalization parameters use no
    // decay. Burn names convolution/linear kernels `weight`, while BatchNorm
    // trainable parameters are `gamma` and `beta`.
    let gradient_clipping = options
        .gradient_clip
        .map(|limit| GradientClippingConfig::Norm(limit).init());
    let mut optimizer_config = SgdConfig::new().with_momentum(Some(momentum.clone()));
    if let Some(limit) = options.gradient_clip {
        anyhow::ensure!(
            limit.is_finite() && limit > 0.0,
            "gradient clip must be positive"
        );
        optimizer_config =
            optimizer_config.with_gradient_clipping(Some(GradientClippingConfig::Norm(limit)));
    }
    let mut optimizer = optimizer_config.init().with_group(
        ParamGroup::from_predicate("weight"),
        SgdConfig::new()
            .with_weight_decay(Some(WeightDecayConfig {
                penalty: weight_decay as f32,
            }))
            .with_momentum(Some(momentum))
            .build(),
        gradient_clipping,
    );
    if let Some(checkpoint) = &checkpoint {
        optimizer = optimizer
            .load(checkpoint.join("optimizer.bpk"))
            .map_err(|error| anyhow::anyhow!("failed to load optimizer state: {error}"))?;
    }
    let scheduler = LinearLR::new(
        config.min_lr,
        config.max_lr,
        config.warmup_epochs,
        nominal_steps,
        schedule_epochs,
    );
    let criterion = ComputeLoss::new(
        num_classes,
        3,
        &model.stride,
        config.box_gain(),
        config.cls_gain(),
        config.dfl_gain(),
    );
    let mut ema = if let (Some(state), Some(checkpoint)) = (&resume_state, &checkpoint) {
        let shadow = load_weights(
            yolo_v11_n(num_classes, device),
            &checkpoint.join("ema.bpk"),
            false,
        )?
        .valid();
        EMA::from_shadow(shadow, 0.9999, 2000.0, state.ema_updates)
    } else {
        EMA::new(&model, 0.9999, 2000.0)
    };

    std::fs::create_dir_all(&options.output_dir)?;
    if let Some(run) = &options.resume {
        let source = run.join("best.bpk");
        let destination = options.output_dir.join("best.bpk");
        if source != destination {
            std::fs::copy(&source, &destination).with_context(|| {
                format!("copying previous best checkpoint from {}", source.display())
            })?;
        }
    }
    let mut csv_file = open_history(&options.output_dir, options.resume.as_deref(), start_epoch)?;
    let mut best_epoch = resume_state.as_ref().map_or(0, |state| state.best_epoch);
    let mut best_validation = resume_state
        .as_ref()
        .map_or_else(EvaluationMetrics::default, |state| state.best_validation);
    let mut best_map = resume_state
        .as_ref()
        .map_or(-1.0, |state| state.best_validation.mean_ap);
    let mut best_model = if let Some(run) = &options.resume {
        Some(load_weights(
            yolo_v11_n(num_classes, device),
            &run.join("best.bpk"),
            false,
        )?)
    } else {
        None
    };
    for stage_epoch in 0..options.epochs {
        let epoch = start_epoch + stage_epoch;
        let plan = epoch_plan(train_store.as_ref(), epoch, options.seed, options.geometry);
        let mut loader = ParallelBatchLoader::new(
            Arc::clone(&train_store),
            plan,
            options.input_size,
            options.batch_size,
            options.loader_workers,
            options.queue_batches,
        )?;
        let num_steps = loader.num_batches();
        let pb = ProgressBar::new(num_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40}] {pos}/{len}")
                .unwrap(),
        );
        let mut avg_box_loss = AverageMeter::new();
        let mut avg_cls_loss = AverageMeter::new();
        let mut avg_dfl_loss = AverageMeter::new();
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulated = 0usize;
        let mut step_in_epoch = 0usize;

        while let Some(host_batch) = loader.next_batch()? {
            let samples = host_batch.len();
            let batch = host_batch.upload(device);
            let outputs = match model.forward_with_batch_norm(
                batch.images,
                true,
                options.batch_norm == BatchNormPolicy::Frozen,
            ) {
                YOLOOutput::Train(outputs) => outputs,
                YOLOOutput::Infer(_) => unreachable!("training forward returned inference output"),
            };
            let (loss_box, loss_cls, loss_dfl) = criterion.call(
                &outputs,
                &batch.cls,
                &batch.bbox,
                &batch.idx,
                samples,
                options.input_size,
                device,
            );
            let lb = loss_box.clone().into_data().try_to_vec::<f32>().unwrap()[0];
            let lc = loss_cls.clone().into_data().try_to_vec::<f32>().unwrap()[0];
            let ld = loss_dfl.clone().into_data().try_to_vec::<f32>().unwrap()[0];
            anyhow::ensure!(
                lb.is_finite() && lc.is_finite() && ld.is_finite(),
                "non-finite training loss at epoch {} step {}: box={lb} cls={lc} dfl={ld}",
                epoch + 1,
                step_in_epoch + 1,
            );
            let scaled_loss = (loss_box.clone() + loss_cls.clone() + loss_dfl.clone())
                * options.batch_size as f64;
            let grads = GradientsParams::from_grads(scaled_loss.backward(), &model);
            accumulator.accumulate(&model, grads);
            accumulated += 1;

            let last_batch = step_in_epoch + 1 == num_steps;
            if accumulated == accumulate || last_batch {
                // Use the current global microbatch index, matching the
                // reference scheduler at each optimizer boundary.
                let global_step = epoch * nominal_steps + step_in_epoch;
                let lr = scheduler.step(global_step);
                model = optimizer.step(lr, model, accumulator.grads());
                ema.update(&model);
                optimizer_steps += 1;
                accumulated = 0;
            }

            avg_box_loss.update(lb as f64, samples);
            avg_cls_loss.update(lc as f64, samples);
            avg_dfl_loss.update(ld as f64, samples);
            pb.set_message(format!(
                "{:>5}/{:<5} box {:>7.3} cls {:>7.3} dfl {:>7.3}",
                epoch + 1,
                schedule_epochs,
                avg_box_loss.avg,
                avg_cls_loss.avg,
                avg_dfl_loss.avg
            ));
            pb.inc(1);
            step_in_epoch += 1;
        }
        pb.finish();

        save_weights(&ema.model().valid(), &options.output_dir.join("last"))?;

        let evaluated = options.eval_interval <= 1
            || stage_epoch + 1 == options.epochs
            || (epoch + 1) % options.eval_interval == 0;
        let metrics = if evaluated {
            if let Some(store) = &validation_store {
                evaluate_store(ema.model(), Arc::clone(store), options, device)?
            } else {
                EvaluationMetrics::default()
            }
        } else {
            EvaluationMetrics::default()
        };
        let last_lr = scheduler.step(epoch * nominal_steps + nominal_steps.saturating_sub(1));
        writeln!(
            csv_file,
            "{},{},{:.8},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            stage,
            epoch + 1,
            last_lr,
            evaluated,
            avg_box_loss.avg,
            avg_cls_loss.avg,
            avg_dfl_loss.avg,
            metrics.recall,
            metrics.precision,
            metrics.map50,
            metrics.mean_ap
        )?;
        csv_file.flush()?;
        if evaluated && metrics.mean_ap > best_map {
            best_map = metrics.mean_ap;
            best_epoch = epoch + 1;
            best_validation = metrics;
            // `Module::clone` shares BatchNorm RunningState storage. Take a
            // synchronized, independent snapshot so later EMA updates cannot
            // mutate the normalization state belonging to the best weights.
            let snapshot = ema.model().valid();
            save_weights(&snapshot, &options.output_dir.join("best"))?;
            best_model = Some(snapshot);
        }
    }
    let completed_epochs = start_epoch + options.epochs;
    let best_model = best_model.context("training produced no best model")?;

    let checkpoint_relative = format!("checkpoints/stage-{stage:04}");
    let checkpoint_dir = options.output_dir.join(&checkpoint_relative);
    std::fs::create_dir_all(&checkpoint_dir)?;
    save_weights(&model.valid(), &checkpoint_dir.join("train"))?;
    save_weights(&ema.model().valid(), &checkpoint_dir.join("ema"))?;
    optimizer
        .save(checkpoint_dir.join("optimizer.bpk"))
        .map_err(|error| anyhow::anyhow!("failed to save optimizer state: {error}"))?;
    let state = ResumeState {
        version: 2,
        stage,
        completed_epochs,
        optimizer_steps,
        ema_updates: ema.updates(),
        schedule_epochs,
        nominal_steps,
        train_samples: train_store.len(),
        batch_size: options.batch_size,
        input_size: options.input_size,
        seed: options.seed,
        geometry: geometry_name(options.geometry).to_owned(),
        eval_interval: options.eval_interval,
        training_config,
        best_epoch,
        best_validation,
        checkpoint_dir: checkpoint_relative,
    };
    if let Some(store) = &validation_store {
        let reloaded_best = load_weights(
            yolo_v11_n(num_classes, device),
            &options.output_dir.join("best.bpk"),
            false,
        )?;
        let reloaded_metrics = evaluate_store(&reloaded_best, Arc::clone(store), options, device)?;
        let close = |left: f32, right: f32| (left - right).abs() <= 1e-5;
        anyhow::ensure!(
            close(reloaded_metrics.mean_ap, best_validation.mean_ap)
                && close(reloaded_metrics.map50, best_validation.map50)
                && close(reloaded_metrics.recall, best_validation.recall)
                && close(reloaded_metrics.precision, best_validation.precision),
            "saved best checkpoint failed validation fidelity: before save {:?}, after reload {:?}",
            best_validation,
            reloaded_metrics
        );
    }
    write_json_atomic(&checkpoint_dir.join("state.json"), &state)?;
    drop(csv_file);
    write_training_progress_svg(
        &options.output_dir.join("step.csv"),
        &options.output_dir.join("training-progress.svg"),
    )?;
    write_json_atomic(&options.output_dir.join("resume.json"), &state)?;

    println!("Stage {stage} complete. Best mAP: {:.3}", best_map.max(0.0));
    let test = match (options.finalize, test_store) {
        (true, Some(store)) => Some(evaluate_store(&best_model, store, options, device)?),
        _ => None,
    };
    Ok(TrainingSummary {
        stage,
        completed_epochs,
        schedule_epochs,
        best_epoch,
        best_validation,
        test,
    })
}

fn evaluate_store(
    model: &YOLO,
    store: SharedSampleStore,
    options: &TrainingOptions,
    device: &Device,
) -> Result<EvaluationMetrics> {
    // BatchNorm selects training or inference behavior from the tensor's
    // autodiff state in Burn. Strip autodiff for validation, as model.eval()
    // does in the PyTorch trainer.
    let model = model.clone().valid();
    let inference_device = device.clone().inner();
    let plan: Vec<_> = store
        .epoch_keys(0, options.seed)
        .into_iter()
        .map(|key| PlannedSample {
            key,
            orientation: 0,
        })
        .collect();
    let mut loader = ParallelBatchLoader::new(
        store,
        plan,
        options.input_size,
        options.batch_size,
        options.loader_workers,
        options.queue_batches,
    )?;
    let iou_thresholds: Vec<f32> = (0..10).map(|index| 0.5 + index as f32 * 0.05).collect();
    let mut tp = Vec::new();
    let mut conf = Vec::new();
    let mut pred_cls = Vec::new();
    let mut gt_cls_all = Vec::new();
    while let Some(host_batch) = loader.next_batch()? {
        let batch_size = host_batch.len();
        let batch = host_batch.upload(&inference_device);
        let output = match model.forward(batch.images, false) {
            YOLOOutput::Infer(output) => output,
            YOLOOutput::Train(_) => unreachable!("evaluation forward returned training output"),
        };
        let detections = nms::non_max_suppression_with_options(
            &output,
            nms::NmsOptions {
                confidence_threshold: options.evaluation_confidence_threshold,
                iou_threshold: options.evaluation_iou_threshold,
                max_detections: options.evaluation_max_detections,
                ..nms::NmsOptions::default()
            },
        );
        let gt_cls = batch
            .cls
            .squeeze_dim::<1>(1)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        let gt_bbox = batch.bbox.to_data().try_to_vec::<f32>().unwrap();
        let gt_idx = batch.idx.to_data().try_to_vec::<f32>().unwrap();
        for (batch_index, predicted) in detections.iter().enumerate().take(batch_size) {
            let mut gt_boxes = Vec::new();
            for target in 0..gt_idx.len() {
                if gt_idx[target] >= 0.0 && gt_idx[target] as usize == batch_index {
                    let cx = gt_bbox[target * 4] * options.input_size as f32;
                    let cy = gt_bbox[target * 4 + 1] * options.input_size as f32;
                    let width = gt_bbox[target * 4 + 2] * options.input_size as f32;
                    let height = gt_bbox[target * 4 + 3] * options.input_size as f32;
                    gt_boxes.push([
                        gt_cls[target],
                        cx - width * 0.5,
                        cy - height * 0.5,
                        cx + width * 0.5,
                        cy + height * 0.5,
                    ]);
                }
            }
            let predicted: Vec<[f32; 6]> = predicted.clone();
            tp.extend(metrics::compute_metric(
                &predicted,
                &gt_boxes,
                &iou_thresholds,
            ));
            conf.extend(predicted.iter().map(|row| row[4]));
            pred_cls.extend(predicted.iter().map(|row| row[5]));
            gt_cls_all.extend(gt_boxes.iter().map(|row| row[0]));
        }
    }
    if !tp.is_empty() && tp.iter().any(|row| row.iter().any(|&matched| matched)) {
        let (_, _, precision, recall, map50, mean_ap, confidence_threshold) =
            metrics::compute_ap_with_confidence(&tp, &conf, &pred_cls, &gt_cls_all);
        Ok(EvaluationMetrics {
            mean_ap,
            map50,
            recall,
            precision,
            confidence_threshold,
        })
    } else {
        Ok(EvaluationMetrics::default())
    }
}

/// File-backed command-line entry point.
pub fn train(
    config: &Config,
    data_dir: &str,
    options: &TrainingOptions,
    device: &Device,
) -> Result<()> {
    let data_dir = Path::new(data_dir);
    let paths = |split: &str| -> Result<Vec<PathBuf>> {
        std::fs::read_to_string(data_dir.join(format!("{split}.txt")))?
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let basename = Path::new(line.trim())
                    .file_name()
                    .context("dataset list contains a path without a filename")?;
                Ok(data_dir.join("images").join(split).join(basename))
            })
            .collect()
    };
    let train_store: SharedSampleStore = Arc::new(FileSampleStore::new(paths("train2017")?)?);
    let validation_store: SharedSampleStore = Arc::new(FileSampleStore::new(paths("val2017")?)?);
    train_with_stores(config, train_store, Some(validation_store), options, device)
}
