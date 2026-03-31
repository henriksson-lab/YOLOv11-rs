#!/bin/bash
# Benchmark: Rust (Candle) vs Python (PyTorch) YOLOv11 inference
#
# Usage: ./benchmark.sh [DATA_DIR] [WEIGHTS_PT] [WEIGHTS_RS]
#
# Defaults:
#   DATA_DIR     = ../Dataset/COCO    (same as Python default)
#   WEIGHTS_PT   = YOLOv11-pt/weights/best.pt
#   WEIGHTS_RS   = weights/best.safetensors  (or .pt if safetensors not found)

set -euo pipefail

DATA_DIR="${1:-Dataset/COCO}"
WEIGHTS_PT="${2:-YOLOv11-pt/weights/best.pt}"
WEIGHTS_RS="${3:-weights/best.safetensors}"

# Fall back to .pt weights for Rust if safetensors not found
if [ ! -f "$WEIGHTS_RS" ] && [ -f "$WEIGHTS_PT" ]; then
    WEIGHTS_RS="$WEIGHTS_PT"
fi

CONFIG="default_args.yaml"

echo "============================================"
echo " YOLOv11 Benchmark: Rust vs Python"
echo "============================================"
echo "  Data dir:       $DATA_DIR"
echo "  Python weights: $WEIGHTS_PT"
echo "  Rust weights:   $WEIGHTS_RS"
echo "  Config:         $CONFIG"
echo ""

# --- Python (PyTorch) ---
echo "============================================"
echo " Python (PyTorch, FP16, CUDA)"
echo "============================================"
if [ -f "$WEIGHTS_PT" ]; then
    cd YOLOv11-pt
    python main.py --input-size 640 --test
    cd ..
else
    echo "  SKIP: weights not found at $WEIGHTS_PT"
fi

echo ""

# --- Rust (Candle) ---
echo "============================================"
echo " Rust (Candle, FP32)"
echo "============================================"
if [ -f "$WEIGHTS_RS" ]; then
    cargo run --release --features cuda -- test \
        --config "$CONFIG" \
        --data-dir "$DATA_DIR" \
        --input-size 640 \
        --weights "$WEIGHTS_RS"
else
    echo "  SKIP: weights not found at $WEIGHTS_RS"
fi

echo ""
echo "============================================"
echo " Benchmark complete"
echo "============================================"
