# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLOv11 object detection with two implementations:
- **`YOLOv11-pt/`** — Original Python/PyTorch reference implementation
- **`src/`** — Rust port using the Candle ML framework (single crate, binary with `train`/`test` subcommands)

## Building (Rust)

```bash
cargo build --release              # Wgpu backend (default)
cargo build --release --features cuda  # With candle-cuda via Burn
```

## Commands (Rust)

```bash
# Train
cargo run --release -- train --config default_args.yaml --data-dir /path/to/COCO

# Evaluate (with Burn-native weights)
cargo run --release -- test --config default_args.yaml --data-dir /path/to/COCO --weights weights/best

# Load converted .pt weights (see conversion step below)
cargo run --release -- test --config default_args.yaml --data-dir /path/to/COCO --weights weights/model.pt

# Convert PyTorch .pt weights to Burn-compatible key names
python convert_weights.py YOLOv11-pt/weights/best.pt -o weights/model.pt
```

Key flags: `--input-size` (default 640), `--batch-size` (default 32), `--epochs` (default 600).

## Commands (Python)

All commands run from within `YOLOv11-pt/`.

```bash
bash main.sh <NUM_GPUS> --train    # Multi-GPU training
python main.py --train             # Single GPU training
python main.py --test              # Evaluate
```

## Architecture

### Model (`src/model/`)

The YOLO model has three stages:

1. **Backbone (`DarkNet`)** — 5 downsampling stages (p1-p5). Uses `Conv` (Conv2d+BN+SiLU), `CSP` blocks with optional `CSPModule` sub-blocks, `SPP` (spatial pyramid pooling), and `PSA` (partial self-attention). Outputs feature maps at 3 scales: p3 (8x), p4 (16x), p5 (32x).

2. **Neck (`DarkFPN`)** — FPN that fuses multi-scale features via upsample + concat + CSP, then downsample + concat + CSP.

3. **Head** — Decoupled box/class branches per scale. Uses `DFL` for box regression. Training returns raw features; inference decodes anchors.

Model variants (n/t/s/m/l/x) differ only in `width`, `depth`, and `csp` arrays — defined by `yolo_v11_n()` through `yolo_v11_x()` in `src/model/model.rs`.

### Key Rust implementation details

- **BatchNorm2d** is implemented from scratch in `src/model/conv.rs` (not provided by candle-nn). Uses `RefCell` for running stats with `forward(x, training)` to switch behavior.
- **Task-aligned assigner** in `src/model/loss.rs` runs on CPU using pure Rust `Vec` operations (no gradient needed).
- **NMS** in `src/model/nms.rs` is greedy CPU-side implementation.
- Training vs inference mode is explicit via `training: bool` parameter on forward methods.
- Checkpoints saved as `.safetensors`; Python `.pt` weights loaded via `VarBuilder::from_pth`.

### Training (`src/train/`)

- SGD with linear warmup and linear decay LR schedule
- Gradient accumulation (effective batch size 64)
- EMA of model weights
- Mosaic augmentation disabled for last 10 epochs
- Logs to `weights/step.csv`; saves `weights/best.safetensors` and `weights/last.safetensors`
- Single GPU only (no DDP equivalent in Candle)

### Dataset (`src/data/`)

- COCO-format: `images/{split}/` and `labels/{split}/` with YOLO `.txt` label files (class cx cy w h, normalized)
- Augmentations: mosaic, random perspective, HSV jitter, flips
- Image loading via `image` crate

## Configuration

- `default_args.yaml` — example config with all training hyperparameters and COCO class names (same format as `YOLOv11-pt/utils/args.yaml`)
- `--config` is required, no default path
- Model variant is selected by changing the `yolo_v11_n` call in `src/main.rs` and `src/train/train.rs`
