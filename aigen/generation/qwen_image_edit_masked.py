"""Source-canvas preparation and exact-pixel publication for native Qwen inpaint."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from aigen.generation.image_edit_batch import (
    ImageEditBatchCase, ImageEditBatchError, ImageEditBatchOutput, ImageEditBatchRequest,
)
from aigen.generation.image_generation_requests import (
    ImageGenerationCaseRequest, ImageGenerationMaskRequest, ImageGenerationOutputRequest,
)
from aigen.generation.qwen_image_edit_lightx2v import (
    LIGHTX2V_QWEN_EDIT_2511_PROFILE, QWEN_IMAGE_EDIT_LIGHTX2V_PROFILES,
    QwenImageEditLightX2VError, run_lightx2v_qwen_image_edit,
)
from aigen.image_io import image_alpha, open_image
from aigen.keyframe_image_ops import exact_outside_mask_diff
from aigen.lora_weights import LoraLoadSpec
from aigen.manifest_io import atomic_write_json, file_manifest
from aigen.progress import StatusReporter


QWEN_MASKED_EDIT_REVISION = "1"


@dataclass(frozen=True)
class QwenMaskedCanvas:
    source_image: Path
    mask_image: Path
    canvas_size: tuple[int, int]
    content_box: tuple[int, int, int, int]


def qwen_masked_canvas_size(size: tuple[int, int], max_side: int | None) -> tuple[int, int]:
    if max_side is not None:
        if max_side < 16:
            raise ImageEditBatchError("Qwen masked canvas cap must be at least 16 pixels")
        cap = max_side // 16 * 16
        if max(size) > cap:
            size = tuple(max(1, round(value * cap / max(size))) for value in size)
    return tuple((value + 15) // 16 * 16 for value in size)


def prepare_qwen_masked_canvas(source: Path, mask: Path, size: tuple[int, int], directory: Path) -> QwenMaskedCanvas:
    if any(value < 16 or value % 16 for value in size):
        raise ImageEditBatchError("Qwen masked canvas dimensions must be positive multiples of 16")
    with open_image(source) as original, open_image(mask) as original_mask:
        if original.size != original_mask.size:
            raise ImageEditBatchError("the repaint mask must match the source image's displayed dimensions")
        pixels = original.convert("RGB")
        repaint = original_mask.convert("L")
        if repaint.getbbox() is None:
            raise ImageEditBatchError("the repaint mask has no selected pixels")
        if pixels.width > size[0] or pixels.height > size[1]:
            pixels = ImageOps.contain(pixels, size, method=Image.Resampling.LANCZOS)
            repaint = repaint.resize(pixels.size, Image.Resampling.NEAREST)
        left, top = (size[0] - pixels.width) // 2, (size[1] - pixels.height) // 2
        box = (left, top, left + pixels.width, top + pixels.height)
        canvas = Image.new("RGB", size, "white")
        canvas.paste(pixels, (left, top))
        mask_canvas = Image.new("L", size, 0)
        mask_canvas.paste(repaint, (left, top))
        directory.mkdir(parents=True)
        source_path, mask_path = directory / "source.png", directory / "mask.png"
        canvas.save(source_path)
        mask_canvas.save(mask_path)
        canvas.close()
        mask_canvas.close()
        pixels.close()
        repaint.close()
    prepared = QwenMaskedCanvas(source_path, mask_path, size, box)
    atomic_write_json(directory / "canvas.json", {
        "source": file_manifest(source), "mask": file_manifest(mask),
        "canvas_size": size, "content_box": box,
        "prepared_source": file_manifest(source_path), "prepared_mask": file_manifest(mask_path),
    })
    return prepared


def composite_qwen_masked_output(source: Path, mask: Path, decoded: Path,
                                canvas: QwenMaskedCanvas, output: Path) -> dict[str, Any]:
    if output.suffix.lower() != ".png":
        raise ImageEditBatchError("masked edits require lossless PNG output for exact pixel preservation")
    with open_image(source) as original, open_image(mask) as original_mask, open_image(decoded) as generated:
        if generated.size != canvas.canvas_size:
            raise ImageEditBatchError("Qwen masked decode does not match its recorded canvas")
        with generated.crop(canvas.content_box) as crop:
            edited = crop.resize(original.size, Image.Resampling.LANCZOS) if crop.size != original.size else crop.copy()
        alpha = image_alpha(original)
        base = original.convert("RGBA" if alpha is not None else "RGB")
        if alpha is not None:
            edited = edited.convert("RGBA")
            edited.putalpha(alpha)
            alpha.close()
        with original_mask.convert("L") as repaint:
            result = Image.composite(edited, base, repaint)
            preservation = exact_outside_mask_diff(base, result, repaint)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output)
        edited.close()
        base.close()
        result.close()
    if not preservation["passed"]:
        raise ImageEditBatchError("Qwen regional edit changed pixels outside the original repaint mask")
    return preservation


def run_qwen_masked_batch(
    request: ImageEditBatchRequest, *, cases: tuple[ImageEditBatchCase, ...],
    loras: tuple[LoraLoadSpec, ...], progress: StatusReporter, record_dir: Path,
    on_output: Callable[[ImageEditBatchOutput], None] | None,
) -> tuple[ImageEditBatchOutput, ...]:
    profile = QWEN_IMAGE_EDIT_LIGHTX2V_PROFILES[LIGHTX2V_QWEN_EDIT_2511_PROFILE]
    steps = profile.default_steps if request.steps is None else request.steps
    if request.strength is not None:
        raise ImageEditBatchError("regional strength belongs to the explicit mask request")
    prepared = {}
    by_id = {case.id: case for case in cases}
    canvases = {}
    groups = {}
    for case in cases:
        assert case.mask is not None
        if round(steps * case.mask.strength) < 1:
            raise ImageEditBatchError("masked strength selects zero denoising steps; increase strength")
        key = (case.mask.source_image, case.mask.mask_image, case.width, case.height)
        if key not in prepared:
            prepared[key] = prepare_qwen_masked_canvas(*key[:2], key[2:], record_dir / "inputs" / str(len(prepared)))
        canvas = canvases[case.id] = prepared[key]
        group_key = (case.prompt, case.image_paths, *key, case.mask.strength)
        groups.setdefault(group_key, []).append(case)
    native_cases = []
    for group in groups.values():
        case = group[0]
        canvas = canvases[case.id]
        native_cases.append(ImageGenerationCaseRequest(
            name=case.id, prompt=case.prompt, image_paths=case.image_paths, width=case.width, height=case.height,
            outputs=tuple(ImageGenerationOutputRequest(item.id, item.seed, record_dir / "decoded" / f"{item.id}.png") for item in group),
            mask=ImageGenerationMaskRequest(canvas.source_image, canvas.mask_image, case.mask.strength),
        ))
    outputs = []

    def completed(raw: dict[str, Any]) -> None:
        case = by_id[raw["name"]]
        assert case.mask is not None
        preservation = composite_qwen_masked_output(
            case.mask.source_image, case.mask.mask_image, Path(raw["path"]), canvases[case.id], case.output_path,
        )
        with open_image(case.output_path) as image:
            output = ImageEditBatchOutput(case_id=case.id, path=case.output_path,
                                         width=image.width, height=image.height, seed=case.seed)
        atomic_write_json(record_dir / "cases" / f"{case.id}.json", {
            **raw, "case": case.id, "decoded": file_manifest(Path(raw["path"])), "preservation": preservation,
            "composite": output.model_dump(mode="json"), "masked_edit_revision": QWEN_MASKED_EDIT_REVISION,
        })
        outputs.append(output)
        if on_output is not None:
            on_output(output)

    try:
        run_lightx2v_qwen_image_edit(
            profile=profile, cases=tuple(native_cases), steps=steps,
            true_cfg_scale=profile.default_true_cfg_scale if request.guidance is None else request.guidance,
            guidance_scale=1.0 if request.guidance_scale is None else request.guidance_scale,
            max_sequence_length=512 if request.max_sequence_length is None else request.max_sequence_length,
            loras=loras, sampler=request.sampler, scheduler=request.scheduler, progress=progress,
            record_dir=record_dir / "lightx2v", on_output=completed,
        )
    except QwenImageEditLightX2VError as error:
        raise ImageEditBatchError(str(error)) from error
    return tuple(outputs)
