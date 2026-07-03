# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLOv11 object detection with two implementations:
- **`YOLOv11-pt/`** — Original Python/PyTorch reference implementation
- **`src/`** — Rust port using the Burn ML framework (single crate, binary with `train`/`test`/`profile` subcommands)

## Translation Rule

The Rust code is intended to be a faithful translation of `YOLOv11-pt/`. Original Python top-level functions, classes, and class methods should map 1:1 to Rust functions, structs, and methods with systematic snake_case names. Do not add public translated-layer helpers or renamed convenience APIs unless they correspond to an original Python item. Rust-only framework/CLI glue belongs outside the translated model/data/util surface or must be explicitly documented as glue.

Use `python3 tools/translation_conformance.py` after translation-surface changes. The check must report no missing mapped Rust items and no extra public translated-layer Rust items.

## Building (Rust)

```bash
cargo build --release                  # WGPU backend (default)
cargo build --release --features cuda  # CUDA backend via Burn
```

## Commands (Rust)

```bash
# Train
cargo run --release -- train --config default_args.yaml --data-dir /path/to/COCO

# Evaluate with Burn-native weights
cargo run --release -- test --config default_args.yaml --data-dir /path/to/COCO --weights weights/best

# Profile
cargo run --release -- profile --config default_args.yaml --input-size 640
```

Key flags: `--input-size` (default 640), `--batch-size` (default 32), `--epochs` (default 600).
Direct `.pt`/`.pth` loading is not supported by the current Burn 0.21 pre-release path; use Burn record or `.safetensors` weights.

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

- **Conv** wraps Burn `Conv2d` + `BatchNorm` + activation and supports private recursive fuse glue.
- **Task-aligned assigner** in `src/model/loss.rs` runs on CPU using pure Rust `Vec` operations (no gradient needed).
- **NMS** in `src/model/nms.rs` is greedy CPU-side implementation that returns Python-shaped `Nx6` rows.
- Training vs inference mode is explicit via `training: bool` parameters and Rust enum return glue.
- Checkpoints are Burn records or `.safetensors`; Python `.pt`/`.pth` loading is explicitly unsupported in the current CLI.

### Training (`src/train/`)

- SGD with linear warmup and linear decay LR schedule
- Gradient accumulation (effective batch size 64)
- EMA of model weights
- Mosaic augmentation disabled for last 10 epochs
- Logs to `weights/step.csv`; saves Burn records under `weights/best` and `weights/last`
- Single-device training only (no DDP equivalent in this Burn translation)

### Dataset (`src/data/`)

- COCO-format: `images/{split}/` and `labels/{split}/` with YOLO `.txt` label files (class cx cy w h, normalized)
- Augmentations: mosaic, random perspective, HSV jitter, flips
- Image loading via `image` crate

## Configuration

- `default_args.yaml` — example config with all training hyperparameters and COCO class names (same format as `YOLOv11-pt/utils/args.yaml`)
- `--config` is required, no default path
- Training, evaluation, and profiling instantiate `yolo_v11_n`, matching the original Python entrypoints.
