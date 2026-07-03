# YOLOv11-rs

YOLOv11 object detection implemented in Rust using the [Candle](https://github.com/huggingface/candle) ML framework. Supports both training and inference. Can be used as a library or as a CLI.

Based on the PyTorch re-implementation in [YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt).


* 2026-07-03: File-by-file audit done. Seems promising but need to update my CUDA and do deeper testing


## This is an LLM-mediated faithful (hopefully) translation, not the original code!

Most users should probably first see if the existing original code works for them, unless they have reason otherwise. The original source
may have newer features and it has had more love in terms of fixing bugs. In fact, we aim to replicate bugs if they are present, for the
sake of reproducibility! (but then we might have added a few more in the process)

There are however cases when you might prefer this Rust version. We generally agree with [this page](https://rewrites.bio/)
but more specifically:
* We have had many issues with ensuring that our software works using existing containers (Docker, PodMan, Singularity). One size does not fit all and it eats our resources trying to keep up with every way of delivering software
* Common package managers do not work well. It was great when we had a few Linux distributions with stable procedures, but now there are just too many ecosystems (Homebrew, Conda). Conda has an NP-complete resolver which does not scale. Homebrew is only so-stable. And our dependencies in Python still break. These can no longer be considered professional serious options. Meanwhile, Cargo enables multiple versions of packages to be available, even within the same program(!)
* The future is the web. We deploy software in the web browser, and until now that has meant Javascript. This is a language where even the == operator is broken. Typescript is one step up, but a game changer is the ability to compile Rust code into webassembly, enabling performance and sharing of code with the backend. Translating code to Rust enables new ways of deployment and running code in the browser has especial benefits for science - researchers do not have deep pockets to run servers, so pushing compute to the user enables deployment that otherwise would be impossible
* Old CLI-based utilities are bad for the environment(!). A large amount of compute resources are spent creating and communicating via small files, which we can bypass by using code as libraries. Even better, we can avoid frequent reloading of databases by hoisting this stage, with up to 100x speedups in some cases. Less compute means faster compute and less electricity wasted
* LLM-mediated translations may actually be safer to use than the original code. This article shows that [running the same code on different operating systems can give somewhat different answers](https://doi.org/10.1038/nbt.3820). This is a gap that Rust+Cargo can reduce. Typesafe interfaces also reduce coding mistakes and error handling, as opposed to typical command-line scripting

But:

* **This approach should still be considered experimental**. The LLM technology is immature and has sharp corners. But there are opportunities to reap, and the genie is not going back to the bottle. This translation is as much aimed to learn how to improve the technology and get feedback on the results.
* Translations are not endorsed by the original authors unless otherwise noted. **Do not send bug reports to the original developers**. Use our Github issues page instead.
* **Do not trust the benchmarks on this page**. They are used to help evaluate the translation. If you want improved performance, you generally have to use this code as a library, and use the additional tricks it offers. We generally accept performance losses in order to reduce our dependency issues
* **Check the original Github pages for information about the package**. This README is kept sparse on purpose. It is not meant to be the primary source of information


## License

License derives from PyTorch reimplementation. It is likely AGPL 3.0

## Requirements

- Rust 1.70+
- (Optional) CUDA toolkit for GPU acceleration

## Building

```bash
# CPU only
cargo build --release

# With CUDA support
cargo build --release --features cuda
```

## Library Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
yolov11 = { path = "../YOLOv11-rs" }
candle-core = "0.8"
candle-nn = "0.8"
```

### Inference with pretrained weights

```rust
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use yolov11::model::model::yolo_v11_n;
use yolov11::model::nms;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let num_classes = 80;

    // Load model with safetensors weights
    let mut varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = yolo_v11_n(num_classes, &device, vb)?;
    varmap.load("weights/best.safetensors")?;

    // Prepare input: [1, 3, 640, 640] normalized to 0..1
    let image = Tensor::zeros((1, 3, 640, 640), DType::F32, &device)?;

    // Run inference
    let output = model.forward_infer(&image)?; // [1, 84, 8400]

    // Apply NMS
    let detections = nms::batch_nms(&output, 0.25, 0.45, 300)?;
    for det in &detections[0] {
        println!(
            "class={} conf={:.2} box=({:.0},{:.0},{:.0},{:.0})",
            det.class, det.confidence, det.x1, det.y1, det.x2, det.y2
        );
    }

    Ok(())
}
```

### Loading PyTorch .pt weights

```rust
use candle_core::{DType, Device};
use yolov11::model::model::yolo_v11_n;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // VarBuilder::from_pth loads the .pt state dict directly
    let vb = candle_nn::VarBuilder::from_pth("weights/best.pt", DType::F32, &device)?;
    let model = yolo_v11_n(80, &device, vb)?;

    // model is ready for inference
    Ok(())
}
```

### Loading and preprocessing an image

```rust
use candle_core::Device;
use yolov11::data::resize::{letterbox, image_to_tensor};

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let img = image::open("photo.jpg")?;

    // Resize to 640x640 with letterboxing
    let (resized, _ratio, _pad) = letterbox(&img, 640, false)?;

    // Convert to tensor [3, 640, 640] in 0..1 range
    let tensor = image_to_tensor(&resized, &device)?;

    // Add batch dimension: [1, 3, 640, 640]
    let input = tensor.unsqueeze(0)?;

    Ok(())
}
```

### Using a different model variant

```rust
use candle_core::{DType, Device};
use candle_nn::VarMap;
use yolov11::model::model::{yolo_v11_s, yolo_v11_m, yolo_v11_x};

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Available: yolo_v11_n, yolo_v11_t, yolo_v11_s, yolo_v11_m, yolo_v11_l, yolo_v11_x
    let model = yolo_v11_s(80, &device, vb)?;

    Ok(())
}
```

### End-to-end: load image, run inference, get detections

```rust
use candle_core::{DType, Device};
use candle_nn::VarMap;
use yolov11::data::resize::{letterbox, image_to_tensor};
use yolov11::model::model::yolo_v11_n;
use yolov11::model::nms;

fn detect(image_path: &str, weights_path: &str) -> anyhow::Result<Vec<nms::Detection>> {
    let device = Device::Cpu;

    // Load model
    let mut varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = yolo_v11_n(80, &device, vb)?;
    varmap.load(weights_path)?;

    // Preprocess
    let img = image::open(image_path)?;
    let (resized, _ratio, _pad) = letterbox(&img, 640, false)?;
    let input = image_to_tensor(&resized, &device)?.unsqueeze(0)?;

    // Inference + NMS
    let output = model.forward_infer(&input)?;
    let mut detections = nms::batch_nms(&output, 0.25, 0.45, 300)?;

    Ok(detections.remove(0))
}
```

## CLI Usage

### Training

```bash
cargo run --release -- train \
    --config default_args.yaml \
    --data-dir /path/to/COCO
```

### Evaluation

```bash
cargo run --release -- test \
    --config default_args.yaml \
    --data-dir /path/to/COCO \
    --weights weights/best.safetensors
```

### Loading PyTorch weights

```bash
cargo run --release -- test \
    --config default_args.yaml \
    --data-dir /path/to/COCO \
    --weights path/to/best.pt
```

Or convert to safetensors first (requires Python with `torch` and `safetensors`):

```bash
python convert_weights.py path/to/best.pt -o weights/best.safetensors
```

### CLI Reference

```
yolov11 <COMMAND>

Commands:
  train  Train the model
  test   Evaluate the model
```

**train options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Path to config YAML (see `default_args.yaml`) |
| `--data-dir` | (required) | Path to dataset root (COCO format) |
| `--input-size` | 640 | Input image size |
| `--batch-size` | 32 | Training batch size |
| `--epochs` | 600 | Number of training epochs |
| `--weights` | | Resume from weights file (`.safetensors` or `.pt`) |

**test options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Path to config YAML (see `default_args.yaml`) |
| `--data-dir` | (required) | Path to dataset root (COCO format) |
| `--input-size` | 640 | Input image size |
| `--weights` | | Path to weights file (`.safetensors` or `.pt`) |

## Dataset Structure

Expects COCO-format layout with YOLO-format label files:

```
COCO/
    train2017.txt          # list of image filenames
    val2017.txt
    images/
        train2017/
            000001.jpg
            ...
        val2017/
            000001.jpg
            ...
    labels/
        train2017/
            000001.txt     # class cx cy w h (normalized)
            ...
        val2017/
            000001.txt
            ...
```

## Model Variants

| Variant | Constructor |
|---------|-------------|
| Nano | `yolo_v11_n` |
| Tiny | `yolo_v11_t` |
| Small | `yolo_v11_s` |
| Medium | `yolo_v11_m` |
| Large | `yolo_v11_l` |
| XLarge | `yolo_v11_x` |

## Project Structure

```
src/
    model/     Model architecture, loss, NMS, metrics
    data/      Dataset loading, augmentations, image preprocessing
    train/     Training loop, evaluation, EMA, LR scheduling
    main.rs    CLI with train/test subcommands
    lib.rs     Library entry point
```

## Training Details

- SGD optimizer with linear warmup and linear decay
- Gradient accumulation (effective batch size 64)
- Exponential moving average (EMA) of model weights
- Mosaic augmentation (disabled for last 10 epochs)
- HSV jitter, random perspective, flips
- Checkpoints saved in safetensors format

## Reference

- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- [jahongir7174/YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)
