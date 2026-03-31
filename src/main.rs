use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarMap;
use clap::{Parser, Subcommand};

use yolov11::data;
use yolov11::model;
use yolov11::train;

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

        /// Path to weights file to resume from (.safetensors or .pt)
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

        /// Path to weights file (.safetensors or .pt)
        #[arg(long)]
        weights: Option<String>,
    },
}

fn get_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = get_device();
    println!("Using device: {:?}", device);

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
                let varmap = VarMap::new();
                let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
                let _model =
                    model::model::yolo_v11_n(config.num_classes(), &device, vb)?;
                let total_params: usize = varmap
                    .all_vars()
                    .iter()
                    .map(|v| v.as_tensor().elem_count())
                    .sum();
                println!("Number of parameters: {:.3}M", total_params as f64 / 1e6);
            }

            train::train::train(&config, &data_dir, input_size, batch_size, epochs, &device)?;
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

            let mut varmap = VarMap::new();

            let weights_path = weights.unwrap_or_else(|| "weights/best.safetensors".to_string());

            if weights_path.ends_with(".pt") || weights_path.ends_with(".pth") {
                println!("Loading PyTorch weights from {}", weights_path);
                let vb_pt =
                    candle_nn::VarBuilder::from_pth(&weights_path, DType::F32, &device)?;
                let model =
                    model::model::yolo_v11_n(num_classes, &device, vb_pt)?;

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
            } else {
                let vb =
                    candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
                let model =
                    model::model::yolo_v11_n(num_classes, &device, vb)?;
                varmap.load(&weights_path)?;
                println!("Loaded weights from {}", weights_path);

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
    }

    Ok(())
}
