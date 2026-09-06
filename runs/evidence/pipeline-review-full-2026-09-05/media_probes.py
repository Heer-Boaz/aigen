"""CPU checks for image orientation/alpha contracts and Hunyuan frame arithmetic."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from aigen import sam_commands
from aigen.generation.image_upscale import _image_to_tensor, _tensor_to_image
from aigen.generation.vosr_backend import _prepare_vosr_file
from aigen.progress import SILENT_STATUS
from aigen.sam_prompt_canvas import SAMPromptCanvas

EVIDENCE = Path(__file__).resolve().parent


def main():
    workspace = EVIDENCE / f"media-data-{uuid4().hex[:8]}"
    workspace.mkdir()
    jpeg = workspace / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (64, 32), "red").save(jpeg, exif=exif)
    displayed = SAMPromptCanvas._load_image(jpeg)
    observed = {}

    def segment(segmenter, image, input_path, **kwargs):
        observed["model_input_hw"] = list(image.shape[:2])
        observed["points"] = kwargs["positive_points"]
        return np.ones(image.shape[:2], dtype=bool)

    with patch.object(sam_commands, "_create_segmenter", return_value=SimpleNamespace(close=lambda: None)), patch.object(sam_commands, "_segment", segment):
        image, mask = sam_commands._build_mask(
            jpeg, engine="sam2", device="cpu", prompt_mode="points", mask_candidate=None,
            box=None, positive_points=((16, 40),), negative_points=(), threshold=28,
            grow=0, feather=0, fill_holes=False, largest_component=False, invert=False,
            progress=SILENT_STATUS,
        )
    observed["canvas_hw"] = list(displayed.shape[:2])
    assert observed["canvas_hw"] == [64, 32]
    assert observed["model_input_hw"] == [32, 64]
    image.close()

    rgba = Image.new("RGBA", (16, 16), (100, 50, 20, 0))
    alpha_path = workspace / "rgba.png"
    rgba.save(alpha_path)
    tensor = _image_to_tensor(rgba, torch=torch, device="cpu")
    restored = _tensor_to_image(tensor, torch=torch)
    assert tensor.shape[1] == 3 and restored.mode == "RGB"
    prepared = _prepare_vosr_file(alpha_path, workspace / "unused.png", 2, None)
    assert prepared.alpha is not None and prepared.alpha.getextrema() == (0, 0)
    prepared.image.close()
    prepared.alpha.close()

    source = EVIDENCE / "upstream/hunyuan_pipeline.py"
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HunyuanVideo_1_5_Pipeline")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "get_latent_size")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    ratios = SimpleNamespace(vae_spatial_compression_ratio=16, vae_temporal_compression_ratio=4)
    frames = {str(n): namespace["get_latent_size"](ratios, n, 768, 512)[0] for n in (49, 50, 51, 52, 53)}
    assert frames == {"49": 13, "50": 13, "51": 13, "52": 13, "53": 14}
    results = {
        "sam_orientation": observed,
        "illustration_upscale_alpha": {"input_mode": "RGBA", "tensor_channels": 3, "output_mode": restored.mode},
        "vosr_rgba_alpha": "preserved by preprocessing; palette tRNS differs, see backend-results.json",
        "hunyuan_requested_to_latent_frames": frames,
        "scope": "Real image preprocessing and pinned Hunyuan shape calculation. SAM neural segmentation doubled; no model execution or actual Hunyuan video-length reproduction.",
    }
    (EVIDENCE / "media-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"status": "passed", "sam_coordinate_mismatch": True, "illustration_alpha_lost": True}))


if __name__ == "__main__":
    main()
