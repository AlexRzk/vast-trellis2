#!/usr/bin/env python3
"""Convert Meta's official DINOv3 ViT-L/16 LVD-1689M checkpoint to Transformers format.

This is intentionally limited to the exact encoder TRELLIS.2 expects:
  dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

The conversion mirrors Hugging Face's official DINOv3 conversion mapping but accepts
an already-downloaded local checkpoint, avoiding a second gated Hub download.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from transformers import DINOv3ViTConfig, DINOv3ViTImageProcessorFast, DINOv3ViTModel


MAPPING = {
    r"cls_token": r"embeddings.cls_token",
    r"mask_token": r"embeddings.mask_token",
    r"storage_tokens": r"embeddings.register_tokens",
    r"patch_embed.proj": r"embeddings.patch_embeddings",
    r"periods": r"inv_freq",
    r"rope_embed": r"rope_embeddings",
    r"blocks.(\d+).attn.proj": r"layer.\1.attention.o_proj",
    r"blocks.(\d+).attn.": r"layer.\1.attention.",
    r"blocks.(\d+).ls(\d+).gamma": r"layer.\1.layer_scale\2.lambda1",
    r"blocks.(\d+).mlp.fc1": r"layer.\1.mlp.up_proj",
    r"blocks.(\d+).mlp.fc2": r"layer.\1.mlp.down_proj",
    r"blocks.(\d+).mlp": r"layer.\1.mlp",
    r"blocks.(\d+).norm": r"layer.\1.norm",
    r"w1": r"gate_proj",
    r"w2": r"up_proj",
    r"w3": r"down_proj",
}


def rename_key(key: str) -> str:
    new = key
    for pattern, replacement in MAPPING.items():
        new = re.sub(pattern, replacement, new)
    return new


def split_qkv(state: dict[str, torch.Tensor]) -> None:
    for key in [k for k in list(state) if "qkv" in k]:
        qkv = state.pop(key)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        state[key.replace("qkv", "q_proj")] = q
        state[key.replace("qkv", "k_proj")] = k
        state[key.replace("qkv", "v_proj")] = v


def unwrap_checkpoint(obj):
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint is not a state-dict-like mapping")
    for candidate in ("state_dict", "model", "teacher"):
        nested = obj.get(candidate)
        if isinstance(nested, dict) and nested and all(torch.is_tensor(v) for v in nested.values()):
            obj = nested
            break
    if obj and all(k.startswith("module.") for k in obj):
        obj = {k.removeprefix("module."): v for k, v in obj.items()}
    return obj


def build_config() -> DINOv3ViTConfig:
    return DINOv3ViTConfig(
        patch_size=16,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_register_tokens=4,
        use_gated_mlp=False,
        hidden_act="gelu",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    print(f"Loading {checkpoint} ...")
    try:
        raw = torch.load(checkpoint, mmap=True, map_location="cpu")
    except TypeError:
        raw = torch.load(checkpoint, map_location="cpu")
    state = unwrap_checkpoint(raw)
    if not state or not all(torch.is_tensor(v) for v in state.values()):
        raise SystemExit("Checkpoint does not look like the official DINOv3 ViT-L/16 state dict.")

    split_qkv(state)
    converted: dict[str, torch.Tensor] = {}
    for key, weight in state.items():
        new_key = rename_key(key)
        if "bias_mask" in key or "attn.k_proj.bias" in key or "local_cls_norm" in key:
            continue
        if key.startswith("projectors."):
            continue
        if "embeddings.mask_token" in new_key:
            weight = weight.unsqueeze(1)
        if "inv_freq" in new_key:
            continue
        if new_key.startswith("layer."):
            new_key = f"model.{new_key}"
        converted[new_key] = weight

    print("Constructing Transformers DINOv3ViTModel ...")
    model = DINOv3ViTModel(build_config()).eval()
    try:
        result = model.load_state_dict(converted, strict=True)
    except RuntimeError as exc:
        print("Strict load failed. This usually means the checkpoint is not the expected")
        print("dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth or Transformers changed its layout.")
        raise SystemExit(str(exc)) from exc

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    processor = DINOv3ViTImageProcessorFast(
        do_resize=True,
        size={"height": 224, "width": 224},
        resample=2,
    )
    processor.save_pretrained(output)

    print(result)
    print(f"Saved converted DINOv3 to: {output}")
    print("Expected files include config.json, model.safetensors and preprocessor_config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
