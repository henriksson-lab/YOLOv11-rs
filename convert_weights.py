#!/usr/bin/env python3
"""Convert YOLOv11 .pt checkpoint to .safetensors for loading in Rust/Candle."""

import argparse
import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Convert .pt weights to .safetensors")
    parser.add_argument("input", help="Input .pt file path")
    parser.add_argument("--output", "-o", help="Output .safetensors file path")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.rsplit(".", 1)[0] + ".safetensors"

    print(f"Loading {args.input}...")
    ckpt = torch.load(args.input, map_location="cpu")

    if "model" in ckpt:
        state_dict = ckpt["model"].state_dict()
    elif isinstance(ckpt, dict) and any(k.endswith(".weight") for k in ckpt):
        state_dict = ckpt
    else:
        state_dict = ckpt

    # Convert all tensors to float32
    tensors = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            tensors[key] = value.float().contiguous()
            print(f"  {key}: {list(value.shape)}")

    print(f"\nSaving {len(tensors)} tensors to {args.output}...")
    save_file(tensors, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
