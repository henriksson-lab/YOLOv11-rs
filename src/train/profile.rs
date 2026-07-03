use anyhow::Result;
use burn::backend::NdArray;
use burn::prelude::*;

use crate::model::model::yolo_v11_n;
use crate::train::config::Config;

pub fn profile(config: &Config, input_size: usize) -> Result<()> {
    anyhow::ensure!(
        input_size >= 64,
        "profile input_size must be at least 64 for the YOLO downsampling path"
    );
    let device = Default::default();
    let model = yolo_v11_n::<NdArray>(config.num_classes(), &device).fuse();
    let num_params = model.num_params();

    let shape = [1, 3, input_size, input_size];
    let x = Tensor::<NdArray, 4>::zeros(shape, &device);
    let _ = model.forward(x, false);

    println!("Profile backend: NdArray CPU");
    println!("Number of parameters: {:.3}M", num_params as f64 / 1e6);
    println!("Number of FLOPs: unavailable in Burn without a THOP-equivalent profiler");

    Ok(())
}
