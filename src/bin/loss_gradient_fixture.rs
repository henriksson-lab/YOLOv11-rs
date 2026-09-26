use anyhow::{bail, Result};
use burn::prelude::*;
use clap::Parser;
use serde::Serialize;
use yolov11::model::loss::ComputeLoss;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = "all")]
    loss_kind: String,
}

#[derive(Serialize)]
struct ScaleGradient {
    shape: [usize; 4],
    values: Vec<f32>,
}

#[derive(Serialize)]
struct Report {
    backend: &'static str,
    release_build: bool,
    loss_kind: String,
    loss: f32,
    gradients: Vec<ScaleGradient>,
}

fn fixture_output(shape: [usize; 4], offset: usize, device: &Device) -> Tensor<4> {
    let count = shape.iter().product();
    let values = (0..count)
        .map(|index| (((index + offset) % 37) as f32 - 18.0) / 20.0)
        .collect::<Vec<_>>();
    Tensor::<1>::from_floats(values.as_slice(), device)
        .reshape(shape)
        .require_grad()
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        bail!("loss gradient fixture requires a release build");
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

    let shapes = [[1, 65, 8, 8], [1, 65, 4, 4], [1, 65, 2, 2]];
    let outputs = vec![
        fixture_output(shapes[0], 0, &device),
        fixture_output(shapes[1], 11, &device),
        fixture_output(shapes[2], 23, &device),
    ];
    let stride = Tensor::<1>::from_floats([8.0, 16.0, 32.0].as_slice(), &device);
    let criterion = ComputeLoss::new(1, 3, &stride, 7.5, 0.5, 1.5);
    let cls = Tensor::<2>::zeros([4, 1], &device);
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
    let loss = match args.loss_kind.as_str() {
        "all" => box_loss + cls_loss + dfl_loss,
        "box" => box_loss,
        "cls" => cls_loss,
        "dfl" => dfl_loss,
        other => bail!("unknown loss kind {other:?}; use all, box, cls, or dfl"),
    };
    let loss_value = loss.clone().into_data().try_to_vec::<f32>()?[0];
    let grads = loss.backward();
    let gradients = outputs
        .iter()
        .zip(shapes)
        .map(|(output, shape)| {
            let values = output
                .grad(&grads)
                .expect("fixture output must have a gradient")
                .into_data()
                .try_to_vec::<f32>()
                .expect("gradient must be f32");
            ScaleGradient { shape, values }
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string(&Report {
            backend,
            release_build: true,
            loss_kind: args.loss_kind,
            loss: loss_value,
            gradients,
        })?
    );
    Ok(())
}
