#!/usr/bin/env python3
"""Convert YOLOv11 PyTorch weights to Burn-compatible key names.

Usage:
    python convert_weights.py YOLOv11-pt/weights/best.pt -o weights/model.pt

The output .pt file has a flat state_dict with keys matching Burn's
#[derive(Module)] field names.
"""

import argparse
import re
import sys
import os
import torch


def remap_key(key: str) -> str:
    """Remap a single PyTorch state dict key to Burn field names."""

    # --- Backbone: net.pN.idx.rest → net.pN_{name}.rest ---
    # net.p1.0.X → net.p1.X  (single conv, no index split)
    key = re.sub(r'^net\.p1\.0\.', 'net.p1.', key)

    # net.pN.0.X → net.pN_conv.X  (first element = downsample conv)
    # net.pN.1.X → net.pN_csp.X   (second element = CSP block)
    for p in ['p2', 'p3', 'p4']:
        key = re.sub(rf'^net\.{p}\.0\.', f'net.{p}_conv.', key)
        key = re.sub(rf'^net\.{p}\.1\.', f'net.{p}_csp.', key)

    # net.p5.0.X → net.p5_conv.X
    # net.p5.1.X → net.p5_csp.X
    # net.p5.2.X → net.p5_spp.X
    # net.p5.3.X → net.p5_psa.X
    key = re.sub(r'^net\.p5\.0\.', 'net.p5_conv.', key)
    key = re.sub(r'^net\.p5\.1\.', 'net.p5_csp.', key)
    key = re.sub(r'^net\.p5\.2\.', 'net.p5_spp.', key)
    key = re.sub(r'^net\.p5\.3\.', 'net.p5_psa.', key)

    # --- PSA: res_m.N.conv1.X → res_m.N.attn.X (attention block) ---
    #          res_m.N.conv2.0.X → res_m.N.ffn1.X
    #          res_m.N.conv2.1.X → res_m.N.ffn2.X
    key = re.sub(r'\.res_m\.(\d+)\.conv2\.0\.', r'.res_m.\1.ffn1.', key)
    key = re.sub(r'\.res_m\.(\d+)\.conv2\.1\.', r'.res_m.\1.ffn2.', key)
    # Attention is under conv1 in PyTorch, under attn in Burn
    key = re.sub(r'\.res_m\.(\d+)\.conv1\.(qkv|conv1|conv2)\.', r'.res_m.\1.attn.\2.', key)

    # --- Head: head.box.N.M.X → head.box_branches.N.cM.X ---
    key = re.sub(r'^head\.box\.(\d+)\.0\.', r'head.box_branches.\1.c0.', key)
    key = re.sub(r'^head\.box\.(\d+)\.1\.', r'head.box_branches.\1.c1.', key)
    key = re.sub(r'^head\.box\.(\d+)\.2\.', r'head.box_branches.\1.c2.', key)

    key = re.sub(r'^head\.cls\.(\d+)\.0\.', r'head.cls_branches.\1.c0.', key)
    key = re.sub(r'^head\.cls\.(\d+)\.1\.', r'head.cls_branches.\1.c1.', key)
    key = re.sub(r'^head\.cls\.(\d+)\.2\.', r'head.cls_branches.\1.c2.', key)
    key = re.sub(r'^head\.cls\.(\d+)\.3\.', r'head.cls_branches.\1.c3.', key)
    key = re.sub(r'^head\.cls\.(\d+)\.4\.', r'head.cls_branches.\1.c4.', key)

    # --- CSP block type: res_m → res_blocks or csp_blocks ---
    # CSPModule variant (csp=true): p4_csp, p5_csp, fpn.h6
    # Residual variant (csp=false): p2_csp, p3_csp, fpn.h1, fpn.h2, fpn.h4
    # PSA.res_m stays as res_m (not a CSP block)
    #
    # We only rename the CSP-level res_m, identified by being right after the CSP parent.
    csp_module_parents = ['net.p4_csp', 'net.p5_csp', 'fpn.h6']
    csp_residual_parents = ['net.p2_csp', 'net.p3_csp', 'fpn.h1', 'fpn.h2', 'fpn.h4']

    for parent in csp_module_parents:
        prefix = parent + '.res_m.'
        if prefix in key:
            key = key.replace(prefix, parent + '.csp_blocks.', 1)

    for parent in csp_residual_parents:
        prefix = parent + '.res_m.'
        if prefix in key:
            key = key.replace(prefix, parent + '.res_blocks.', 1)

    # NOTE: BatchNorm weight→gamma, bias→beta renaming is handled automatically
    # by Burn's PyTorchFileRecorder adapter. Do NOT rename them here.

    return key


def convert(input_path: str, output_path: str, debug: bool = False):
    sys.path.insert(0, os.path.join(os.path.dirname(input_path), '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'YOLOv11-pt'))

    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

    if 'model' in checkpoint:
        sd = checkpoint['model'].state_dict()
    elif isinstance(checkpoint, dict) and any(k.endswith('.weight') for k in checkpoint):
        sd = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

    new_sd = {}
    for old_key, value in sd.items():
        # Skip num_batches_tracked (not used by Burn)
        if 'num_batches_tracked' in old_key:
            continue

        new_key = remap_key(old_key)
        if debug:
            if old_key != new_key:
                print(f"  {old_key} → {new_key}")
            else:
                print(f"  {old_key} (unchanged)")
        new_sd[new_key] = value

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(new_sd, output_path)
    print(f"Converted {len(new_sd)} tensors: {input_path} → {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert YOLOv11 PyTorch weights to Burn key names')
    parser.add_argument('input', help='Input .pt file')
    parser.add_argument('-o', '--output', required=True, help='Output .pt file')
    parser.add_argument('--debug', action='store_true', help='Print key remapping')
    args = parser.parse_args()
    convert(args.input, args.output, args.debug)
