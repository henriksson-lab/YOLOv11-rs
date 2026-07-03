use anyhow::Result;
use burn::backend::Autodiff;
use burn::prelude::*;
use clap::{Parser, Subcommand};

#[cfg(not(feature = "cuda"))]
use burn::backend::wgpu::{Wgpu, WgpuDevice};
#[cfg(feature = "cuda")]
use burn::backend::{cuda::CudaDevice, Cuda};

use yolov11::data;
use yolov11::model;
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
    /// Profile the model
    Profile {
        /// Path to config YAML (see default_args.yaml)
        #[arg(long)]
        config: String,

        /// Input image size
        #[arg(long, default_value_t = 640)]
        input_size: usize,
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

    train::util::setup_seed();
    train::util::setup_multi_processes();

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

            train::profile::profile(&config, input_size)?;

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
            train::profile::profile(&config, input_size)?;

            let model: model::model::YOLO<MyBackend> =
                model::model::yolo_v11_n(num_classes, &device);

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
                    model.load_from(&mut store).map_err(|e| {
                        anyhow::anyhow!("Failed to load safetensors weights: {}", e)
                    })?;
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

            let data_dir_path = std::path::Path::new(&data_dir);
            let val_filenames = std::fs::read_to_string(data_dir_path.join("val2017.txt"))?
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| {
                    let basename = std::path::Path::new(line.trim())
                        .file_name()
                        .unwrap()
                        .to_owned();
                    data_dir_path.join("images").join("val2017").join(basename)
                })
                .collect();
            let val_dataset = data::dataset::Dataset::new(
                val_filenames,
                input_size as u32,
                false,
                &config.to_augment_params(),
            )?;
            train::eval::test(&model, &val_dataset, &device, 4)?;
        }
        Command::Profile { config, input_size } => {
            let config = train::config::Config::load(&std::path::PathBuf::from(&config))?;
            println!("Loaded config with {} classes", config.num_classes());
            train::profile::profile(&config, input_size)?;
        }
    }

    Ok(())
}
