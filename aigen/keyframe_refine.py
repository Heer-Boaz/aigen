from __future__ import annotations

import subprocess
import sys
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from aigen.diffusers_kontext_adapter import kontext_inpaint_text_kwargs
from aigen.generation.runtime_diagnostics import cuda_memory_stats, elapsed_ms, module_device_report, synchronized_time
from aigen.generation.runtime_types import resolve_torch_dtype
from aigen.image_assets import image_asset_json
from aigen.keyframe_image_ops import (
    expanded_aligned_box,
    mask_overlay,
    outside_mask_change,
    paste_refined_crop,
    save_contact_sheet,
)
from aigen.keyframe_memory import NvidiaSmiMemorySampler, nvidia_smi_preflight
from aigen.keyframe_pose import PoseScoreConfig, extract_target_pose_map_keypoints
from aigen.keyframe_profiles import KeyframeRefineProfile
from aigen.keyframe_refine_models import (
    KeyframeRefineError,
    KeyframeRefineJobSpec,
    load_keyframe_refine_job,
)
from aigen.keyframe_segmentation import SamForegroundSegmenter, SamSegmentationConfig
from aigen.manifest_io import (
    read_json,
    resolve_existing_path,
    resolve_output_path,
    sha256_bytes,
    write_json,
)
from aigen.progress import StatusReporter
from aigen.prompt_tokens import count_kontext_prompt_tokens


@dataclass(frozen=True)
class RefineMaskPlan:
    hard_mask: Image.Image
    feather_mask: Image.Image
    crop_box: tuple[int, int, int, int]
    front_arm_indices: tuple[int, int, int]
    arm_line_width_px: int
    fist_radius_px: int


class KontextInpaintRefiner:
    def __init__(self, profile: KeyframeRefineProfile, *, device: str = "cuda") -> None:
        self.profile = profile
        self.device = device
        self.torch, pipeline_class = _load_kontext_inpaint()
        model_load_start = synchronized_time(self.torch)
        pipeline_kwargs: dict[str, Any] = {
            "torch_dtype": _torch_dtype(self.torch, profile.dtype),
            "local_files_only": True,
        }
        from nunchaku import NunchakuFluxTransformer2dModel

        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            profile.nunchaku_transformer_model.resolve().as_posix(),
            torch_dtype=_torch_dtype(self.torch, profile.dtype),
        )
        transformer.set_attention_impl(profile.attention_impl)
        pipeline_kwargs["transformer"] = transformer

        self.pipeline = pipeline_class.from_pretrained(profile.model, **pipeline_kwargs)
        self.pipeline.set_progress_bar_config(disable=True)
        if profile.vae_tiling:
            self.pipeline.vae.enable_tiling()
        else:
            self.pipeline.vae.disable_tiling()
        self.pipeline.vae.disable_slicing()
        if profile.pipeline_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(device)
        self.device_report = _pipeline_device_report(self.pipeline)
        self.model_load_ms = elapsed_ms(model_load_start, synchronized_time(self.torch))

    def refine(
        self,
        *,
        base_crop: Image.Image,
        mask_crop: Image.Image,
        reference_image: Image.Image,
        clip_prompt: str,
        t5_prompt: str,
        negative_prompt: str | None,
        true_cfg_scale: float,
        steps: int,
        guidance_scale: float,
        strength: float,
        max_sequence_length: int,
        seed: int,
    ) -> Image.Image:
        args: dict[str, Any] = {
            "image": base_crop,
            "image_reference": reference_image,
            "mask_image": mask_crop,
            "true_cfg_scale": true_cfg_scale,
            "height": base_crop.height,
            "width": base_crop.width,
            "max_area": base_crop.width * base_crop.height,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
            "_auto_resize": False,
            "generator": self.torch.Generator(device=self.device).manual_seed(seed),
        }
        args.update(
            kontext_inpaint_text_kwargs(
                clip_prompt=clip_prompt,
                t5_prompt=t5_prompt,
                negative_prompt=negative_prompt,
            )
        )
        return self.pipeline(**args).images[0]

    def close(self) -> None:
        del self.pipeline
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

def resolve_keyframe_refine_job(
    job_path: Path,
    profile: KeyframeRefineProfile,
    *,
    project_root: Path,
    check_outputs: bool,
) -> dict[str, Any]:
    spec = load_keyframe_refine_job(job_path)
    if spec.pipeline.profile != profile.name:
        raise KeyframeRefineError(f"Job uses profile {spec.pipeline.profile}, but CLI resolved {profile.name}")

    base_dir = resolve_output_path(spec.base.run_dir, job_path.parent)
    result = read_json(base_dir / "result.json", label="keyframe refine base result")
    base_output = _base_output(result, spec.base.candidate)
    base_image = image_asset_json(Path(base_output["path"]).resolve())
    identity_primer = image_asset_json(resolve_existing_path(spec.character.identity_primer.path, job_path.parent))
    pose = image_asset_json(resolve_existing_path(spec.region.mask_source.pose, job_path.parent))
    contour = image_asset_json(resolve_existing_path(spec.region.mask_source.contour, job_path.parent))
    if pose["width"] != base_image["width"] or pose["height"] != base_image["height"]:
        raise KeyframeRefineError("Refine pose asset must match the base candidate dimensions")
    if contour["width"] != base_image["width"] or contour["height"] != base_image["height"]:
        raise KeyframeRefineError("Refine contour asset must match the base candidate dimensions")

    output_dir = resolve_output_path(spec.output.directory, job_path.parent)
    outputs = _planned_outputs(spec, output_dir)
    if check_outputs and not spec.output.overwrite:
        existing = [output for output in outputs if Path(output["path"]).exists()]
        if existing:
            raise KeyframeRefineError(f"Output exists and overwrite=false: {existing[0]['path']}")

    tokens = count_kontext_prompt_tokens(profile.model, spec.prompt.clip, spec.prompt.t5)
    if tokens.clip > tokens.clip_limit:
        raise KeyframeRefineError(f"CLIP prompt has {tokens.clip} tokens, limit is {tokens.clip_limit}")
    if tokens.t5 > spec.sampling.max_sequence_length:
        raise KeyframeRefineError(
            f"T5 prompt has {tokens.t5} tokens, max_sequence_length is {spec.sampling.max_sequence_length}"
        )
    if spec.prompt.negative is not None and spec.prompt.true_cfg_scale <= 1.0:
        raise KeyframeRefineError("negative prompt is configured but true_cfg_scale <= 1.0")
    if spec.prompt.true_cfg_scale > 1.0 and spec.prompt.negative is None:
        raise KeyframeRefineError("true_cfg_scale > 1.0 requires prompt.negative")

    return {
        "kind": "resolved-keyframe-refine",
        "job_path": job_path.resolve().as_posix(),
        "job_id": spec.id,
        "profile": _profile_json(profile),
        "base": {
            "run_dir": base_dir.as_posix(),
            "candidate": spec.base.candidate,
            "image": base_image,
            "base_job_id": result["job_id"],
        },
        "character": {
            "id": spec.character.id,
            "identity_primer": {
                "view": spec.character.identity_primer.view,
                **identity_primer,
            },
        },
        "region": spec.region.model_dump(mode="json"),
        "assets": {
            "pose": pose,
            "contour": contour,
        },
        "prompt": spec.prompt.model_dump(mode="json", exclude_none=True),
        "sampling": spec.sampling.model_dump(mode="json"),
        "variants": [variant.model_dump(mode="json") for variant in spec.variants],
        "output": {
            **spec.output.model_dump(mode="json"),
            "directory": output_dir.as_posix(),
            "files": outputs,
        },
        "acceptance": spec.acceptance.model_dump(mode="json"),
        "tokens": {
            "clip": tokens.clip,
            "clip_limit": tokens.clip_limit,
            "t5": tokens.t5,
            "t5_limit": spec.sampling.max_sequence_length,
        },
        "git_commit": _git_commit(project_root),
        "spec_sha256": sha256_bytes(job_path.read_bytes()),
    }


def validate_keyframe_refine_job(
    job_path: Path,
    profile: KeyframeRefineProfile,
    *,
    project_root: Path,
) -> dict[str, Any]:
    resolved = resolve_keyframe_refine_job(job_path, profile, project_root=project_root, check_outputs=False)
    return {
        "status": "valid",
        "job_id": resolved["job_id"],
        "profile": resolved["profile"]["name"],
        "tokens": resolved["tokens"],
        "outputs": resolved["output"]["files"],
    }


def plan_keyframe_refine_job(
    job_path: Path,
    profile: KeyframeRefineProfile,
    *,
    project_root: Path,
) -> dict[str, Any]:
    resolved = resolve_keyframe_refine_job(job_path, profile, project_root=project_root, check_outputs=True)
    mask_plan = build_refine_mask_plan(resolved)
    return {
        **resolved,
        "mask_plan": _mask_plan_json(mask_plan),
    }


def run_keyframe_refine_job(
    job_path: Path,
    profile: KeyframeRefineProfile,
    *,
    project_root: Path,
    progress: StatusReporter,
) -> dict[str, Any]:
    progress.phase("resolve refine job")
    spec = load_keyframe_refine_job(job_path)
    resolved = resolve_keyframe_refine_job(job_path, profile, project_root=project_root, check_outputs=True)
    output_dir = Path(resolved["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    progress.step("build refine mask")
    mask_plan = build_refine_mask_plan(resolved)
    _save_mask_artifacts(mask_plan, output_dir)
    resolved = {
        **resolved,
        "mask_plan": _mask_plan_json(mask_plan, output_dir),
    }
    write_json(output_dir / "resolved.json", resolved)
    if spec.output.save_debug_images:
        progress.phase("save refine debug images")
        _save_debug_images(resolved, mask_plan, output_dir)

    memory_sampler = NvidiaSmiMemorySampler(nvidia_smi_preflight())
    memory_sampler.start()
    try:
        total_start = perf_counter()
        variant_results = []
        progress.begin(len(spec.variants), "refine variants")
        if len(spec.variants) == 1:
            progress.phase(f"refine {spec.variants[0].name} (1/1)")
            variant_results.append(
                run_keyframe_refine_variant(
                    job_path,
                    profile,
                    variant_name=spec.variants[0].name,
                    project_root=project_root,
                    resolved=resolved,
                    progress=progress,
                )
            )
            progress.step(f"refined {spec.variants[0].name} (1/1)")
        else:
            variant_results = _run_refine_variants_in_subprocesses(
                job_path,
                spec,
                project_root,
                output_dir / "resolved.json",
                progress=progress,
            )

        outputs = [result["output"] for result in variant_results]
        progress.step("write refine contact sheet")
        if spec.output.save_contact_sheet:
            save_contact_sheet(outputs, output_dir / "contact_sheet.png", thumb_width=256, label_x=8)
        progress.step("write refine result")
        memory = memory_sampler.stop()
        for variant_result in variant_results:
            memory["nvidia_smi_peak_used_mb"] = max(
                memory["nvidia_smi_peak_used_mb"],
                variant_result["memory"]["nvidia_smi_peak_used_mb"],
            )
        result = {
            "status": "completed",
            "job_id": spec.id,
            "spec_sha256": resolved["spec_sha256"],
            "git_commit": resolved["git_commit"],
            "models": resolved["profile"]["models"],
            "assets": {
                "identity_primer": resolved["character"]["identity_primer"],
                "base_image": resolved["base"]["image"],
                "pose": resolved["assets"]["pose"],
                "contour": resolved["assets"]["contour"],
            },
            "outputs": outputs,
            "effective_config": resolved,
            "timings_ms": {
                "model_load_ms": sum(result["timings_ms"]["model_load_ms"] for result in variant_results),
                "model_load_per_variant_ms": [result["timings_ms"]["model_load_ms"] for result in variant_results],
                "total_ms": elapsed_ms(total_start, perf_counter()),
            },
            "memory": memory,
            "environment": variant_results[0]["environment"] if variant_results else {},
        }
        write_json(output_dir / "result.json", result)
        return result
    finally:
        memory_sampler.stop()


def run_keyframe_refine_variant(
    job_path: Path,
    profile: KeyframeRefineProfile,
    *,
    variant_name: str,
    project_root: Path,
    resolved: dict[str, Any],
    progress: StatusReporter,
) -> dict[str, Any]:
    spec = load_keyframe_refine_job(job_path)
    output_dir = Path(resolved["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_plan = _load_refine_mask_plan(resolved["mask_plan"])
    variant_lookup = {variant.name: variant for variant in spec.variants}
    planned_lookup = {planned["name"]: planned for planned in resolved["output"]["files"]}
    if variant_name not in variant_lookup:
        raise KeyframeRefineError(f"Refine job has no variant named {variant_name}")
    variant = variant_lookup[variant_name]
    planned = planned_lookup[variant_name]
    memory_sampler = NvidiaSmiMemorySampler(nvidia_smi_preflight())
    memory_sampler.start()
    try:
        progress.phase(f"load refiner {variant_name}")
        with closing(KontextInpaintRefiner(profile)) as refiner:
            torch_module = refiner.torch
            variant_start = synchronized_time(torch_module)
            with Image.open(resolved["base"]["image"]["path"]) as image:
                base = image.convert("RGB")
            with Image.open(resolved["character"]["identity_primer"]["path"]) as image:
                identity_primer = image.convert("RGB")
            base_crop = base.crop(mask_plan.crop_box)
            mask_crop = mask_plan.feather_mask.crop(mask_plan.crop_box)
            progress.phase(f"inpaint {variant_name}")
            refined_crop = refiner.refine(
                base_crop=base_crop,
                mask_crop=mask_crop,
                reference_image=identity_primer,
                clip_prompt=spec.prompt.clip,
                t5_prompt=spec.prompt.t5,
                negative_prompt=spec.prompt.negative,
                true_cfg_scale=spec.prompt.true_cfg_scale,
                steps=spec.sampling.steps,
                guidance_scale=spec.sampling.guidance_scale,
                strength=spec.sampling.strength,
                max_sequence_length=spec.sampling.max_sequence_length,
                seed=variant.seed,
            )
            composed = paste_refined_crop(base, refined_crop.convert("RGB"), mask_plan.feather_mask, mask_plan.crop_box)
            output_path = Path(planned["path"])
            composed.save(output_path)
            payload = {
                "status": "completed",
                "variant": variant.name,
                "output": {
                    **planned,
                    "seed": variant.seed,
                    "timings_ms": {
                        "refine_ms": elapsed_ms(variant_start, synchronized_time(torch_module)),
                    },
                    "mask_change": outside_mask_change(base, composed, mask_plan.feather_mask),
                },
                "timings_ms": {
                    "model_load_ms": refiner.model_load_ms,
                },
                "device_report": refiner.device_report,
                "memory": cuda_memory_stats(torch_module, "cuda") | memory_sampler.stop(),
                "environment": _generation_environment(torch_module),
            }
            sidecar = _variant_result_path(output_dir, variant.name)
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            write_json(sidecar, payload)
            return payload
    finally:
        memory_sampler.stop()


def _run_refine_variants_in_subprocesses(
    job_path: Path,
    spec: KeyframeRefineJobSpec,
    project_root: Path,
    resolved_path: Path,
    progress: StatusReporter,
) -> list[dict[str, Any]]:
    results = []
    output_dir = resolve_output_path(spec.output.directory, job_path.parent)
    for index, variant in enumerate(spec.variants, start=1):
        progress.phase(f"refine {variant.name} ({index}/{len(spec.variants)})")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "aigen.keyframe_refine_worker",
                job_path.resolve().as_posix(),
                "--variant",
                variant.name,
                "--resolved",
                resolved_path.resolve().as_posix(),
                "--project-root",
                project_root.resolve().as_posix(),
            ],
            cwd=project_root,
        )
        if completed.returncode != 0:
            raise KeyframeRefineError(f"Refine variant failed: {variant.name}")
        results.append(read_json(_variant_result_path(output_dir, variant.name), label="keyframe refine variant result"))
        progress.step(f"refined {variant.name} ({index}/{len(spec.variants)})")
    return results


def _variant_result_path(output_dir: Path, variant_name: str) -> Path:
    return output_dir / "variant_results" / f"{variant_name}.json"


def build_refine_mask_plan(resolved: dict[str, Any]) -> RefineMaskPlan:
    pose = extract_target_pose_map_keypoints(Path(resolved["assets"]["pose"]["path"]), PoseScoreConfig(min_common_keypoints=5))
    width, height = pose.image_size
    direction = _direction_from_resolved(resolved)
    arm_indices = _front_arm_indices(pose.points, direction)
    hard_mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(hard_mask)
    points = [_point_px(pose.points[index], width, height) for index in arm_indices]
    arm_line_width = max(24, round(width * 0.085))
    fist_radius = max(26, round(width * 0.075))
    draw.line(points, fill=255, width=arm_line_width, joint="curve")
    wrist_x, wrist_y = points[-1]
    draw.ellipse(
        (wrist_x - fist_radius, wrist_y - fist_radius, wrist_x + fist_radius, wrist_y + fist_radius),
        fill=255,
    )

    contour = _load_luma(Path(resolved["assets"]["contour"]["path"])) > 32
    arm_band = np.asarray(hard_mask, dtype=np.uint8) > 0
    contour_near_arm = contour & ndimage.binary_dilation(arm_band, iterations=max(8, round(width * 0.04)))
    hard = arm_band | contour_near_arm
    if resolved["region"]["mask_source"]["candidate_foreground"]:
        with ExitStack() as resources:
            segmenter = resources.enter_context(closing(SamForegroundSegmenter(SamSegmentationConfig())))
            foreground = segmenter.segment(Path(resolved["base"]["image"]["path"]))
        if foreground.shape != hard.shape:
            raise KeyframeRefineError(
                f"Candidate foreground mask has shape {foreground.shape}, expected {hard.shape}"
            )
        hard &= ndimage.binary_dilation(foreground, iterations=8)
    dilated = ndimage.binary_dilation(hard, iterations=resolved["region"]["dilate_px"])
    hard_image = Image.fromarray((dilated.astype(np.uint8) * 255), mode="L")
    feather_image = hard_image.filter(ImageFilter.GaussianBlur(radius=resolved["region"]["feather_px"]))
    crop_box = expanded_aligned_box(dilated, resolved["region"]["crop_padding_px"], width, height)
    return RefineMaskPlan(
        hard_mask=hard_image,
        feather_mask=feather_image,
        crop_box=crop_box,
        front_arm_indices=arm_indices,
        arm_line_width_px=arm_line_width,
        fist_radius_px=fist_radius,
    )


def _profile_json(profile: KeyframeRefineProfile) -> dict[str, Any]:
    models = {
        "kontext": {
            **profile.model_revisions["kontext"],
            "path": profile.model,
        },
        "nunchaku_transformer": {
            **profile.model_revisions["nunchaku_transformer"],
            "path": profile.nunchaku_transformer_model.resolve().as_posix(),
        },
    }
    return {
        "name": profile.name,
        "dtype": profile.dtype,
        "attention_impl": profile.attention_impl,
        "pipeline_cpu_offload": profile.pipeline_cpu_offload,
        "vae_tiling": profile.vae_tiling,
        "models": models,
    }


def _base_output(result: dict[str, Any], candidate: str) -> dict[str, Any]:
    for output in result["outputs"]:
        if output["name"] == candidate:
            return output
    raise KeyframeRefineError(f"Base run has no candidate named {candidate}")


def _direction_from_resolved(resolved: dict[str, Any]) -> Literal["left", "right"]:
    source = read_json(Path(resolved["base"]["run_dir"]) / "resolved.json", label="source keyframe resolved manifest")
    return source["keyframe"]["direction"]


def _front_arm_indices(points: np.ndarray, direction: Literal["left", "right"]) -> tuple[int, int, int]:
    arms = ((2, 3, 4), (5, 6, 7))
    candidates = [arm for arm in arms if np.isfinite(points[list(arm), 0]).all()]
    if not candidates:
        raise KeyframeRefineError("Target pose has no complete arm chain for refine masking")
    upper_body_candidates = [arm for arm in candidates if float(points[list(arm), 1].mean()) < 0.65]
    if upper_body_candidates:
        candidates = upper_body_candidates
    edge = min if direction == "left" else max
    return edge(candidates, key=lambda arm: edge(float(points[index, 0]) for index in arm))


def _point_px(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    return int(round(float(point[0]) * width)), int(round(float(point[1]) * height))


def _mask_plan_json(mask_plan: RefineMaskPlan, output_dir: Path | None = None) -> dict[str, Any]:
    payload = {
        "crop_box": list(mask_plan.crop_box),
        "front_arm_indices": list(mask_plan.front_arm_indices),
        "arm_line_width_px": mask_plan.arm_line_width_px,
        "fist_radius_px": mask_plan.fist_radius_px,
    }
    if output_dir is not None:
        payload["hard_mask"] = image_asset_json(_mask_artifact_dir(output_dir) / "hard.png")
        payload["feather_mask"] = image_asset_json(_mask_artifact_dir(output_dir) / "feather.png")
    return payload


def _save_mask_artifacts(mask_plan: RefineMaskPlan, output_dir: Path) -> None:
    mask_dir = _mask_artifact_dir(output_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_plan.hard_mask.save(mask_dir / "hard.png")
    mask_plan.feather_mask.save(mask_dir / "feather.png")


def _load_refine_mask_plan(mask_plan: dict[str, Any]) -> RefineMaskPlan:
    with Image.open(mask_plan["hard_mask"]["path"]) as image:
        hard_mask = image.convert("L")
    with Image.open(mask_plan["feather_mask"]["path"]) as image:
        feather_mask = image.convert("L")
    return RefineMaskPlan(
        hard_mask=hard_mask,
        feather_mask=feather_mask,
        crop_box=tuple(mask_plan["crop_box"]),
        front_arm_indices=tuple(mask_plan["front_arm_indices"]),
        arm_line_width_px=mask_plan["arm_line_width_px"],
        fist_radius_px=mask_plan["fist_radius_px"],
    )


def _mask_artifact_dir(output_dir: Path) -> Path:
    return output_dir / "masks"


def _save_debug_images(resolved: dict[str, Any], mask_plan: RefineMaskPlan, output_dir: Path) -> None:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(resolved["base"]["image"]["path"]) as image:
        base = image.convert("RGB")
    mask_plan.hard_mask.save(debug_dir / "mask_hard.png")
    mask_plan.feather_mask.save(debug_dir / "mask_feather.png")
    base.crop(mask_plan.crop_box).save(debug_dir / "crop.png")
    mask_plan.feather_mask.crop(mask_plan.crop_box).save(debug_dir / "crop_mask.png")
    mask_overlay(base, mask_plan.feather_mask).save(debug_dir / "mask_overlay.png")


def _planned_outputs(spec: KeyframeRefineJobSpec, output_dir: Path) -> list[dict[str, str | int]]:
    return [
        {
            "name": variant.name,
            "seed": variant.seed,
            "path": (output_dir / spec.output.filename.format(id=spec.id, variant=variant.name)).as_posix(),
        }
        for variant in spec.variants
    ]


def _load_luma(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _git_commit(project_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()


def _torch_dtype(torch_module: Any, dtype: str) -> Any:
    return resolve_torch_dtype(torch_module, dtype, auto_value=None)


def _load_kontext_inpaint() -> tuple[Any, Any]:
    try:
        import torch
        from diffusers import FluxKontextInpaintPipeline
        from diffusers.utils import logging as diffusers_logging
    except ImportError as exc:
        raise KeyframeRefineError("keyframe refine requires `pip install -e .[generation]`") from exc
    diffusers_logging.disable_progress_bar()
    return torch, FluxKontextInpaintPipeline


def _generation_environment(torch_module: Any) -> dict[str, Any]:
    import diffusers

    environment = {
        "torch_version": torch_module.__version__,
        "torch_cuda_version": torch_module.version.cuda,
        "diffusers_version": diffusers.__version__,
    }
    if torch_module.cuda.is_available():
        environment["gpu_name"] = torch_module.cuda.get_device_name(0)
        environment["compute_capability"] = list(torch_module.cuda.get_device_capability(0))
    return environment


def _pipeline_device_report(pipeline: Any) -> dict[str, Any]:
    return {
        "pipeline_class": type(pipeline).__qualname__,
        "model_cpu_offload_seq": pipeline.model_cpu_offload_seq,
        "components": {
            "transformer": module_device_report(pipeline.transformer),
            "vae": module_device_report(pipeline.vae),
            "text_encoder": module_device_report(pipeline.text_encoder),
            "text_encoder_2": module_device_report(pipeline.text_encoder_2),
        },
    }
