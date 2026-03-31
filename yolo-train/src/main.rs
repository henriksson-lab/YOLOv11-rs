mod config;
mod ema;
mod eval;
mod lr_schedule;
mod train;

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarMap;
use clap::Parser;

use crate::config::Config;

#[derive(Parser, Debug)]
#[command(name = "yolo-train", about = "YOLOv11 training and evaluation in Rust")]
struct Args {
    /// Input image size
    #[arg(long, default_value_t = 640)]
    input_size: usize,

    /// Batch size
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// Number of training epochs
    #[arg(long, default_value_t = 600)]
    epochs: usize,

    /// Path to dataset root (COCO format)
    #[arg(long, default_value = "../Dataset/COCO")]
    data_dir: String,

    /// Path to config yaml
    #[arg(long, default_value = "config/args.yaml")]
    config: String,

    /// Run training
    #[arg(long)]
    train: bool,

    /// Run testing/evaluation
    #[arg(long)]
    test: bool,

    /// Path to weights file (.safetensors or .pt) to load
    #[arg(long)]
    weights: Option<String>,
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
    let args = Args::parse();
    let device = get_device();
    println!("Using device: {:?}", device);

    let config = Config::load(&std::path::PathBuf::from(&args.config))?;
    println!("Loaded config with {} classes", config.num_classes());

    // Profile model
    {
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _model = yolo_model::model::yolo_v11_n(config.num_classes(), &device, vb)?;

        let total_params: usize = varmap
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum();
        println!("Number of parameters: {:.3}M", total_params as f64 / 1e6);
    }

    if args.train {
        train::train(
            &config,
            &args.data_dir,
            args.input_size,
            args.batch_size,
            args.epochs,
            &device,
        )?;
    }

    if args.test {
        let num_classes = config.num_classes();
        let mut varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = yolo_model::model::yolo_v11_n(num_classes, &device, vb)?;

        // Load weights
        if let Some(ref weights_path) = args.weights {
            if weights_path.ends_with(".safetensors") {
                varmap.load(weights_path)?;
                println!("Loaded weights from {}", weights_path);
            } else if weights_path.ends_with(".pt") || weights_path.ends_with(".pth") {
                // Load PyTorch weights via VarBuilder::from_pth
                println!("Loading PyTorch weights from {}", weights_path);
                let vb_pt = candle_nn::VarBuilder::from_pth(weights_path, DType::F32, &device)?;
                // Re-create model with pretrained VarBuilder
                let _model = yolo_model::model::yolo_v11_n(num_classes, &device, vb_pt)?;
                println!("Note: model re-created with .pt weights");
                // For .pt loading, we rebuild the model with the loaded weights
                let val_filenames = yolo_data::dataset::Dataset::load_filenames(
                    &std::path::PathBuf::from(format!("{}/val2017.txt", args.data_dir)),
                    &std::path::PathBuf::from(&args.data_dir),
                    "val2017",
                )?;
                let val_dataset = yolo_data::dataset::Dataset::new(
                    val_filenames,
                    args.input_size as u32,
                    false,
                    &config.to_augment_params(),
                )?;
                eval::test(&_model, &val_dataset, num_classes, &device, 4)?;
                return Ok(());
            } else {
                anyhow::bail!("Unsupported weights format: {}", weights_path);
            }
        } else {
            varmap.load("weights/best.safetensors")?;
            println!("Loaded weights from weights/best.safetensors");
        }

        let val_filenames = yolo_data::dataset::Dataset::load_filenames(
            &std::path::PathBuf::from(format!("{}/val2017.txt", args.data_dir)),
            &std::path::PathBuf::from(&args.data_dir),
            "val2017",
        )?;
        let val_dataset = yolo_data::dataset::Dataset::new(
            val_filenames,
            args.input_size as u32,
            false,
            &config.to_augment_params(),
        )?;

        eval::test(&model, &val_dataset, num_classes, &device, 4)?;
    }

    Ok(())
}
