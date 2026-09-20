#!/usr/bin/env python3
"""Idempotent compatibility/runtime patches for Microsoft TRELLIS.2.

The patch keeps upstream behavior by default, but adds environment-variable overrides
used by the Vast.ai launcher:
  DINOV3_MODEL_PATH
  TRELLIS_LOW_VRAM
  TRELLIS_DEFAULT_RESOLUTION
  TRELLIS_SERVER_NAME
  TRELLIS_PORT

It also makes the DINOv3 layer traversal compatible with both older and newer
Transformers layouts (model.layer vs model.model.layer).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def patch_feature_extractor(path: Path) -> list[str]:
    changes: list[str] = []
    text = path.read_text()

    if "DINOV3_MODEL_PATH" not in text:
        old = (
            "        self.model_name = model_name\n"
            "        self.model = DINOv3ViTModel.from_pretrained(model_name)"
        )
        new = (
            "        import os\n"
            "        local_model = os.environ.get(\"DINOV3_MODEL_PATH\")\n"
            "        if local_model:\n"
            "            model_name = local_model\n"
            "        self.model_name = model_name\n"
            "        self.model = DINOv3ViTModel.from_pretrained(model_name)"
        )
        if old not in text:
            raise RuntimeError(
                f"Could not find the DINOv3 loader pattern in {path}. "
                "Upstream TRELLIS.2 may have changed."
            )
        text = text.replace(old, new, 1)
        changes.append("DINOv3 local-path override")

    if "self.model.model.layer" not in text:
        old = "        for i, layer_module in enumerate(self.model.layer):"
        new = (
            "        layers = self.model.layer if hasattr(self.model, \"layer\") else self.model.model.layer\n"
            "        for i, layer_module in enumerate(layers):"
        )
        if old not in text:
            raise RuntimeError(
                f"Could not find the DINOv3 layer traversal in {path}. "
                "Upstream TRELLIS.2 may have changed."
            )
        text = text.replace(old, new, 1)
        changes.append("Transformers DINOv3 layer compatibility")

    path.write_text(text)
    return changes


def patch_app(path: Path) -> list[str]:
    changes: list[str] = []
    text = path.read_text()

    marker = "# VAST_TRELLIS_RUNTIME_PATCH"
    if marker not in text:
        old = (
            "    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')\n"
            "    pipeline.cuda()"
        )
        new = (
            "    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')\n"
            "    # VAST_TRELLIS_RUNTIME_PATCH\n"
            "    _low_vram = os.environ.get('TRELLIS_LOW_VRAM', '1').lower() not in ('0', 'false', 'no', 'off')\n"
            "    pipeline.low_vram = _low_vram\n"
            "    pipeline.cuda()"
        )
        if old not in text:
            raise RuntimeError(
                f"Could not find pipeline construction in {path}. Upstream app.py may have changed."
            )
        text = text.replace(old, new, 1)
        changes.append("runtime low-VRAM override")

    old_resolution = (
        '            resolution = gr.Radio(["512", "1024", "1536"], '
        'label="Resolution", value="1024")'
    )
    if "TRELLIS_DEFAULT_RESOLUTION" not in text:
        new_resolution = (
            '            resolution = gr.Radio(["512", "1024", "1536"], '
            'label="Resolution", value=os.environ.get("TRELLIS_DEFAULT_RESOLUTION", "1024"))'
        )
        if old_resolution in text:
            text = text.replace(old_resolution, new_resolution, 1)
            changes.append("default resolution override")
        else:
            raise RuntimeError(
                f"Could not find the resolution selector in {path}. Upstream app.py may have changed."
            )

    old_launch = "    demo.launch(css=css, head=head)"
    if "TRELLIS_SERVER_NAME" not in text:
        new_launch = (
            "    demo.launch(\n"
            "        css=css,\n"
            "        head=head,\n"
            "        server_name=os.environ.get('TRELLIS_SERVER_NAME', '127.0.0.1'),\n"
            "        server_port=int(os.environ.get('TRELLIS_PORT', '7860')),\n"
            "    )"
        )
        if old_launch in text:
            text = text.replace(old_launch, new_launch, 1)
            changes.append("Gradio bind/port override")
        else:
            raise RuntimeError(
                f"Could not find demo.launch in {path}. Upstream app.py may have changed."
            )

    path.write_text(text)
    return changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path, help="Path to the TRELLIS.2 checkout")
    args = parser.parse_args()

    repo = args.repo.resolve()
    feature = repo / "trellis2/modules/image_feature_extractor.py"
    app = repo / "app.py"
    if not feature.is_file() or not app.is_file():
        raise SystemExit(f"Not a TRELLIS.2 checkout: {repo}")

    changes = []
    changes.extend(patch_feature_extractor(feature))
    changes.extend(patch_app(app))

    if changes:
        print("Applied:")
        for change in changes:
            print(f"  - {change}")
    else:
        print("TRELLIS.2 compatibility patches already applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
