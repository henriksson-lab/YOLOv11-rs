use anyhow::{bail, Result};
use burn::prelude::*;
use burn_store::ModuleSnapshot;
use clap::Parser;
use serde::Serialize;
use yolov11::model::loss::ComputeLoss;
use yolov11::model::model::{yolo_v11_n, YOLOOutput};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    fused_batch_norm: bool,
}

#[derive(Serialize)]
struct TensorReport {
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Serialize)]
struct Report {
    backend: &'static str,
    release_build: bool,
    loaded_tensors: usize,
    fused_batch_norm: bool,
    loss: f32,
    outputs: Vec<TensorReport>,
    input_gradient: TensorReport,
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        bail!("model gradient fixture requires a release build");
    }

    let args = Args::parse();

    #[cfg(feature = "libtorch")]
    let device = Device::libtorch_cuda(burn::tensor::DeviceIndex::Default).autodiff();
    #[cfg(all(feature = "cuda", not(feature = "libtorch")))]
    let device = Device::cuda(burn::tensor::DeviceIndex::Default).autodiff();
    #[cfg(not(any(feature = "cuda", feature = "libtorch")))]
    let device = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice).autodiff();

    #[cfg(feature = "libtorch")]
    let backend = "libtorch-cuda";
    #[cfg(all(feature = "cuda", not(feature = "libtorch")))]
    let backend = "cubecl-cuda";
    #[cfg(not(any(feature = "cuda", feature = "libtorch")))]
    let backend = "wgpu";

    let mut model = yolo_v11_n(80, &device);
    let mut store = burn_store::SafetensorsStore::from_file("weights/model.safetensors");
    let loaded = model
        .load_from(&mut store)
        .map_err(|error| anyhow::anyhow!("loading matched weights: {error}"))?;
    if args.fused_batch_norm {
        model = model.fuse();
    }

    let count = 3 * 64 * 64;
    let values = (0..count)
        .map(|index| ((index % 251) as f32 - 125.0) / 125.0)
        .collect::<Vec<_>>();
    let input = Tensor::<1>::from_floats(values.as_slice(), &device)
        .reshape([1, 3, 64, 64])
        .require_grad();
    let outputs = match model.forward(input.clone(), true) {
        YOLOOutput::Train(outputs) => outputs,
        YOLOOutput::Infer(_) => unreachable!("training forward returned inference output"),
    };
    let output_report = outputs
        .iter()
        .map(|output| TensorReport {
            shape: output.dims().to_vec(),
            values: output
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .expect("output must be f32"),
        })
        .collect();

    let criterion = ComputeLoss::new(80, 3, &model.stride, 7.5, 0.5, 1.5);
    let cls = Tensor::<1>::from_floats([0.0, 1.0, 2.0, 3.0].as_slice(), &device).reshape([4, 1]);
    let bbox = Tensor::<1>::from_floats(
        [
            0.25, 0.25, 0.16, 0.16, 0.70, 0.25, 0.12, 0.18, 0.30, 0.72, 0.20, 0.14, 0.72, 0.70,
            0.18, 0.18,
        ]
        .as_slice(),
        &device,
    )
    .reshape([4, 4]);
    let idx = Tensor::<1>::zeros([4], &device);
    let (box_loss, cls_loss, dfl_loss) =
        criterion.call(&outputs, &cls, &bbox, &idx, 1, 64, &device);
    let loss = box_loss + cls_loss + dfl_loss;
    let loss_value = loss.clone().into_data().try_to_vec::<f32>()?[0];
    let grads = loss.backward();
    let input_gradient = input
        .grad(&grads)
        .expect("input must have a gradient")
        .into_data()
        .try_to_vec::<f32>()?;
    println!(
        "{}",
        serde_json::to_string(&Report {
            backend,
            release_build: true,
            loaded_tensors: loaded.applied.len(),
            fused_batch_norm: args.fused_batch_norm,
            loss: loss_value,
            outputs: output_report,
            input_gradient: TensorReport {
                shape: vec![1, 3, 64, 64],
                values: input_gradient,
            },
        })?
    );
    Ok(())
}
