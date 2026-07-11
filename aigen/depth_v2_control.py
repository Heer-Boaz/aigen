from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from aigen.runtime_profiles import MODELS_ROOT

DEFAULT_DEPTH_V2_MODEL = MODELS_ROOT / "depth/depth-anything/Depth-Anything-V2-Large-hf"


class DepthV2ControlError(RuntimeError):
    pass


@dataclass(frozen=True)
class DepthV2Control:
    image: Image.Image
    metadata: dict[str, float | int | str]
    device: str
    model: Path


def render_depth_v2_control(
    image: Image.Image,
    *,
    source_label: str,
    device: str = "cuda",
    model_path: Path = DEFAULT_DEPTH_V2_MODEL,
) -> DepthV2Control:
    if not model_path.is_dir():
        raise DepthV2ControlError(f"Missing Depth Anything V2 model directory: {model_path.as_posix()}")
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as error:
        raise DepthV2ControlError("Depth control rendering requires torch and transformers") from error

    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(model_path, local_files_only=True).to(device).eval()
    try:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        predicted_depth = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )[0]["predicted_depth"]
        depth_min = predicted_depth.min()
        depth_max = predicted_depth.max()
        depth_range = depth_max - depth_min
        if not bool(depth_range > 0):
            raise DepthV2ControlError(f"Depth Anything V2 produced a constant depth map for {source_label}")
        depth_array = (
            ((predicted_depth - depth_min) / depth_range * 255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        control_image = Image.fromarray(depth_array, mode="L").convert("RGB")
        metadata = {
            "depth_min": float(depth_min.item()),
            "depth_max": float(depth_max.item()),
            "width": image.width,
            "height": image.height,
        }
        del inputs, outputs, predicted_depth, depth_min, depth_max, depth_range
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return DepthV2Control(
        image=control_image,
        metadata=metadata,
        device=device,
        model=model_path.resolve(),
    )
