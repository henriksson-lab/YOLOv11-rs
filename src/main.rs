use anyhow::Result;
use burn::backend::Autodiff;
use burn::prelude::*;
use clap::{Parser, Subcommand};

#[cfg(feature = "cuda")]
use burn::backend::{Cuda, cuda::CudaDevice};
#[cfg(not(feature = "cuda"))]
use burn::backend::wgpu::{Wgpu, WgpuDevice};

use yolov11::data;
use yolov11::model;
use yolov11::model::model::ModelVariant;
use yolov11::train;

#[cfg(feature = "cuda")]
type MyBackend = Cuda;
#[cfg(not(feature = "cuda"))]
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

        /// Model variant
        #[arg(long, value_enum, default_value_t = ModelVariant::N)]
        model: ModelVariant,

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

        /// Model variant
        #[arg(long, value_enum, default_value_t = ModelVariant::N)]
        model: ModelVariant,

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
    #[cfg(feature = "cuda")]
    let device = CudaDevice::default();
    #[cfg(not(feature = "cuda"))]
    let device = WgpuDevice::default();

    #[cfg(feature = "cuda")]
    println!("Using device: CUDA");
    #[cfg(not(feature = "cuda"))]
    println!("Using device: Wgpu");

    match cli.command {
        Command::Train {
            config,
            data_dir,
            model: variant,
            input_size,
            batch_size,
            epochs,
            weights: _,
        } => {
            let config = train::config::Config::load(&std::path::PathBuf::from(&config))?;
            println!("Loaded config with {} classes", config.num_classes());
            println!("Model variant: {:?}", variant);

            // Profile model
            {
                let model: model::model::YOLO<MyBackend> =
                    model::model::build_yolo(variant, config.num_classes(), &device);
                let total_params = model.num_params();
                println!("Number of parameters: {:.3}M", total_params as f64 / 1e6);
            }

            train::train::train::<MyAutodiffBackend>(
                &config, &data_dir, variant, input_size, batch_size, epochs, &device,
            )?;
        }
        Command::Test {
            config,
            data_dir,
            model: variant,
            input_size,
            weights,
        } => {
            let config = train::config::Config::load(&std::path::PathBuf::from(&config))?;
            let num_classes = config.num_classes();
            println!("Loaded config with {} classes", num_classes);
            println!("Model variant: {:?}", variant);

            let model: model::model::YOLO<MyBackend> =
                model::model::build_yolo(variant, num_classes, &device);

            // Load weights if provided
            let model = if let Some(weights_path) = weights {
                if weights_path.ends_with(".pt") || weights_path.ends_with(".pth") {
                    anyhow::bail!(
                        "Direct .pt loading is not supported in burn 0.21 pre-release \
                         (broken transitive dep). Convert to .safetensors first."
                    );
                } else if weights_path.ends_with(".safetensors") {
                    println!("Loading safetensors weights from {}", weights_path);
                    use burn_store::ModuleSnapshot;
                    let mut model = model;
                    let mut store = burn_store::SafetensorsStore::from_file(&weights_path);
                    model.load_from(&mut store)
                        .map_err(|e| anyhow::anyhow!("Failed to load safetensors weights: {}", e))?;
                    model
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
