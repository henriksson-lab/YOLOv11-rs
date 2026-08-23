use anyhow::Result;
use burn_store::ModuleSnapshot;
use std::path::Path;
use yolov11::data::dataset::Dataset;
use yolov11::model::model;
use yolov11::train;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let limit = args
        .get(1)
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let config = train::config::Config::load(Path::new("default_args.yaml"))?;
    let data_dir = Path::new("Dataset/COCO");
    let data_dir_str = data_dir.to_string_lossy();
    let data_dir_path = Path::new(data_dir_str.as_ref());

    let mut val_filenames: Vec<_> = std::fs::read_to_string(data_dir_path.join("val2017.txt"))?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let basename = Path::new(line.trim()).file_name().unwrap().to_owned();
            data_dir_path.join("images").join("val2017").join(basename)
        })
        .collect();
    if let Some(limit) = limit {
        val_filenames.truncate(limit);
    }

    let device = burn::tensor::Device::flex();
    let model = model::yolo_v11_n(config.num_classes(), &device);
    let mut model = model;
    let mut store = burn_store::SafetensorsStore::from_file("weights/model.safetensors");
    model
        .load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("Failed to load safetensors weights: {}", e))?;

    let dataset = Dataset::new(val_filenames, 640, false, &config.to_augment_params())?;
    let (mean_ap, map50, recall, precision) = train::eval::test(&model, &dataset, &device, 4)?;
    println!(
        "RESULT precision={precision:.6} recall={recall:.6} map50={map50:.6} map={mean_ap:.6}"
    );
    Ok(())
}
