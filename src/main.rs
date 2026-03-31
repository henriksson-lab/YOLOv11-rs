use anyhow::Result;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::prelude::*;
use burn::record::Recorder;
use clap::{Parser, Subcommand};

use yolov11::data;
use yolov11::model;
use yolov11::train;

type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;

#[derive(Parser, Debug)]
#[command(name = "yolov11", about = "YOLOv11 training and evaluation in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Train the model
    Train {
        /// Path to config YAML (see default_args.yaml)
        #[arg(long)]
        config: String,

        /// Path to dataset root (COCO format)
        #[arg(long)]
        data_dir: String,

        /// Input image size
        #[arg(long, default_value_t = 640)]
        input_size: usize,

        /// Batch size
        #[arg(long, default_value_t = 32)]
        batch_size: usize,

        /// Number of training epochs
        #[arg(long, default_value_t = 600)]
        epochs: usize,

        /// Path to weights file to resume from
        #[arg(long)]
        weights: Option<String>,
    },
    /// Evaluate the model
    Test {
        /// Path to config YAML (see default_args.yaml)
        #[arg(long)]
        config: String,

        /// Path to dataset root (COCO format)
        #[arg(long)]
        data_dir: String,

        /// Input image size
        #[arg(long, default_value_t = 640)]
        input_size: usize,

        /// Path to weights file
        #[arg(long)]
        weights: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = WgpuDevice::default();
    println!("Using device: Wgpu");

    match cli.command {
        Command::Train {
            config,
            data_dir,
            input_size,
            batch_size,
            epochs,
            weights: _,
        } => {
            let config = train::config::Config::load(&std::path::PathBuf::from(&config))?;
            println!("Loaded config with {} classes", config.num_classes());

            // Profile model
            {
                let model: model::model::YOLO<MyBackend> =
                    model::model::yolo_v11_n(config.num_classes(), &device);
                let total_params = model.num_params();
                println!("Number of parameters: {:.3}M", total_params as f64 / 1e6);
            }

            train::train::train::<MyAutodiffBackend>(
                &config, &data_dir, input_size, batch_size, epochs, &device,
            )?;
        }
        Command::Test {
            config,
            data_dir,
            input_size,
            weights,
        } => {
            let config = train::config::Config::load(&std::path::PathBuf::from(&config))?;
            let num_classes = config.num_classes();
            println!("Loaded config with {} classes", num_classes);

            let model: model::model::YOLO<MyBackend> =
                model::model::yolo_v11_n(num_classes, &device);

            // Load weights if provided
            let model = if let Some(weights_path) = weights {
                if weights_path.ends_with(".pt") || weights_path.ends_with(".pth") {
                    println!("Loading PyTorch weights from {}", weights_path);
                    println!("  (use convert_weights.py first if keys are not remapped)");
                    let recorder = burn_import::pytorch::PyTorchFileRecorder::<burn::record::FullPrecisionSettings>::default();
                    let args = burn_import::pytorch::LoadArgs::new(weights_path.into());
                    let record = recorder.load(args, &device)
                        .map_err(|e| anyhow::anyhow!("Failed to load PyTorch weights: {}", e))?;
                    model.load_record(record)
                } else {
                    println!("Loading weights from {}", weights_path);
                    model
                        .load_file(&weights_path, &burn::record::DefaultFileRecorder::<burn::record::FullPrecisionSettings>::new(), &device)
                        .map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?
                }
            } else {
                model
            };

            let val_filenames = data::dataset::Dataset::load_filenames(
                &std::path::PathBuf::from(format!("{}/val2017.txt", data_dir)),
                &std::path::PathBuf::from(&data_dir),
                "val2017",
            )?;
            let val_dataset = data::dataset::Dataset::new(
                val_filenames,
                input_size as u32,
                false,
                &config.to_augment_params(),
            )?;
            train::eval::test(&model, &val_dataset, num_classes, &device, 4)?;
        }
    }

    Ok(())
}
