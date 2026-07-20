from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from PIL import Image

from aigen.character_reference_models import CharacterReferenceError
from aigen.character_reference_pack import load_character_reference_pack
from aigen.command_io import command_error_payload, dump_json
from aigen.image_edit_defaults import (
    BOOGU_DEFAULT_GUIDANCE,
    BOOGU_DEFAULT_STEPS,
    BOOGU_SAMPLER,
    BOOGU_SCHEDULER,
    FLUX2_KLEIN_DEFAULT_SAMPLER,
    FLUX2_KLEIN_SAMPLERS,
    FLUX2_KLEIN_SCHEDULER,
    FLUX2_KLEIN_STEPS,
    HIDREAM_DEFAULT_GUIDANCE,
    HIDREAM_DEFAULT_SAMPLER,
    HIDREAM_DEFAULT_SCHEDULER,
    HIDREAM_DEFAULT_STEPS,
    HIDREAM_SAMPLERS,
    HIDREAM_SCHEDULERS,
    QWEN_2511_BASE_DEFAULT_GUIDANCE,
    QWEN_2511_BASE_DEFAULT_STEPS,
    QWEN_2511_LIGHTNING_DEFAULT_GUIDANCE,
    QWEN_2511_LIGHTNING_DEFAULT_STEPS,
    QWEN_2511_DEFAULT_SCHEDULER,
    QWEN_2511_SAMPLER,
    QWEN_2511_SAMPLERS,
    QWEN_2511_SCHEDULERS,
)
from aigen.image_dimensions import normalized_aspect_ratio, parse_aspect_ratio
from aigen.lora_weights import (
    FLUX2_KLEIN_ARCHITECTURE,
    QWEN_IMAGE_ARCHITECTURE,
    LoraLoadSpec,
    inspect_lora_weights,
)
from aigen.progress import StatusReporter


FLUX2_KLEIN_BACKEND = "flux2-klein"
QWEN_2511_LIGHTNING_BACKEND = "qwen-image-edit-2511-lightning"
QWEN_2511_BASE_BACKEND = "qwen-image-edit-2511-base"
HIDREAM_O1_BACKEND = "hidream-o1-full-fp8"
BOOGU_IMAGE_EDIT_BACKEND = "boogu-image-edit-turbo-fp8"
IMAGE_EDIT_BACKENDS = (
    FLUX2_KLEIN_BACKEND,
    QWEN_2511_LIGHTNING_BACKEND,
    QWEN_2511_BASE_BACKEND,
    HIDREAM_O1_BACKEND,
    BOOGU_IMAGE_EDIT_BACKEND,
)
IMAGE_EDIT_ASPECT_RATIOS = (
    (1, 1),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (9, 16),
    (16, 9),
)
IMAGE_EDIT_BACKEND_LORA_ARCHITECTURES = {
    FLUX2_KLEIN_BACKEND: FLUX2_KLEIN_ARCHITECTURE,
    QWEN_2511_LIGHTNING_BACKEND: QWEN_IMAGE_ARCHITECTURE,
    QWEN_2511_BASE_BACKEND: QWEN_IMAGE_ARCHITECTURE,
}


@dataclass(frozen=True)
class ImageEditBackendSettings:
    steps: int
    guidance: float | None
    sampler: str
    samplers: tuple[str, ...]
    scheduler: str
    schedulers: tuple[str, ...]


IMAGE_EDIT_BACKEND_SETTINGS = {
    FLUX2_KLEIN_BACKEND: ImageEditBackendSettings(
        FLUX2_KLEIN_STEPS,
        None,
        FLUX2_KLEIN_DEFAULT_SAMPLER,
        FLUX2_KLEIN_SAMPLERS,
        FLUX2_KLEIN_SCHEDULER,
        (FLUX2_KLEIN_SCHEDULER,),
    ),
    QWEN_2511_LIGHTNING_BACKEND: ImageEditBackendSettings(
        QWEN_2511_LIGHTNING_DEFAULT_STEPS,
        QWEN_2511_LIGHTNING_DEFAULT_GUIDANCE,
        QWEN_2511_SAMPLER,
        QWEN_2511_SAMPLERS,
        QWEN_2511_DEFAULT_SCHEDULER,
        QWEN_2511_SCHEDULERS,
    ),
    QWEN_2511_BASE_BACKEND: ImageEditBackendSettings(
        QWEN_2511_BASE_DEFAULT_STEPS,
        QWEN_2511_BASE_DEFAULT_GUIDANCE,
        QWEN_2511_SAMPLER,
        QWEN_2511_SAMPLERS,
        QWEN_2511_DEFAULT_SCHEDULER,
        QWEN_2511_SCHEDULERS,
    ),
    HIDREAM_O1_BACKEND: ImageEditBackendSettings(
        HIDREAM_DEFAULT_STEPS,
        HIDREAM_DEFAULT_GUIDANCE,
        HIDREAM_DEFAULT_SAMPLER,
        HIDREAM_SAMPLERS,
        HIDREAM_DEFAULT_SCHEDULER,
        HIDREAM_SCHEDULERS,
    ),
    BOOGU_IMAGE_EDIT_BACKEND: ImageEditBackendSettings(
        BOOGU_DEFAULT_STEPS,
        BOOGU_DEFAULT_GUIDANCE,
        BOOGU_SAMPLER,
        (BOOGU_SAMPLER,),
        BOOGU_SCHEDULER,
        (BOOGU_SCHEDULER,),
    ),
}


class ImageEditCommandError(RuntimeError):
    pass


def add_image_edit_command(subparsers: Any) -> None:
    command = subparsers.add_parser(
        "image-edit",
        help="Edit images with a selected image-generation backend",
    )
    command.add_argument("--backend", choices=IMAGE_EDIT_BACKENDS, required=True)
    command.add_argument("--image", type=Path, action="append")
    command.add_argument(
        "--reference-pack",
        type=Path,
        action="append",
        help="Named visual reference pack; repeat to combine packs",
    )
    command.add_argument("--prompt", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Seed; repeat for a seed sweep (default: 0)",
    )
    command.add_argument("--width", type=int, help="Output width; requires --height")
    command.add_argument("--height", type=int, help="Output height; requires --width")
    command.add_argument(
        "--aspect-ratio",
        type=parse_aspect_ratio,
        help="Recommended backend canvas for W:H; cannot be combined with --width/--height",
    )
    command.add_argument(
        "--steps",
        type=int,
        help="Inference steps; default is backend-native",
    )
    command.add_argument(
        "--guidance",
        type=float,
        help="CFG scale; default is backend-native",
    )
    command.add_argument(
        "--sampler",
        help="Backend sampler; default is backend-native",
    )
    command.add_argument(
        "--scheduler",
        help="Backend sigma scheduler; default is backend-native",
    )
    command.add_argument(
        "--lora",
        type=Path,
        action="append",
        help="Backend-compatible LoRA SafeTensors; repeat to combine LoRAs",
    )
    command.add_argument(
        "--lora-weight",
        type=float,
        action="append",
        help="LoRA strengths in --lora order (default: 1.0 for all)",
    )
    command.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory",
    )


def run_image_edit_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    try:
        prompt = args.prompt.strip()
        if not prompt:
            raise ImageEditCommandError("--prompt must not be empty")
        images = _resolve_images(args.image, args.reference_pack)
        seeds = tuple(args.seed or (0,))
        sampler = _resolve_sampler(args.backend, args.sampler)
        scheduler = _resolve_scheduler(args.backend, args.scheduler)
        width, height = _resolve_dimensions(args.width, args.height, args.aspect_ratio)
        loras = _resolve_loras(args.lora, args.lora_weight, args.backend)
        if width is None:
            inferred_aspect = args.aspect_ratio is None
            aspect_ratio = args.aspect_ratio or _image_aspect_ratio(images[0])
            width, height = _recommended_canvas_size(
                args.backend,
                aspect_ratio,
                closest=inferred_aspect,
            )
        output_dir = _prepare_output_directory(args.output_dir, overwrite=args.overwrite)

        if args.backend == FLUX2_KLEIN_BACKEND:
            payload = _run_flux2_klein(
                prompt=prompt,
                images=images,
                output_dir=output_dir,
                seeds=seeds,
                width=width,
                height=height,
                steps=args.steps,
                guidance=args.guidance,
                sampler=sampler,
                loras=loras,
                progress=progress,
            )
        elif args.backend in (QWEN_2511_LIGHTNING_BACKEND, QWEN_2511_BASE_BACKEND):
            payload = _run_qwen_2511(
                backend=args.backend,
                prompt=prompt,
                images=images,
                output_dir=output_dir,
                seeds=seeds,
                width=width,
                height=height,
                steps=args.steps,
                guidance=args.guidance,
                sampler=sampler,
                scheduler=scheduler,
                loras=loras,
                progress=progress,
            )
        elif args.backend == HIDREAM_O1_BACKEND:
            payload = _run_hidream_o1(
                prompt=prompt,
                images=images,
                output_dir=output_dir,
                seeds=seeds,
                width=width,
                height=height,
                steps=args.steps,
                guidance=args.guidance,
                sampler=sampler,
                scheduler=scheduler,
                progress=progress,
            )
        else:
            payload = _run_boogu_image_edit(
                prompt=prompt,
                images=images,
                output_dir=output_dir,
                seeds=seeds,
                width=width,
                height=height,
                steps=args.steps,
                guidance=args.guidance,
                progress=progress,
            )
        payload["sampler"] = sampler
        payload["scheduler"] = scheduler
    except (CharacterReferenceError, ImageEditCommandError, OSError, ValueError) as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1

    dump_json(stdout, payload, pretty=True)
    return 0


def _run_flux2_klein(
    *,
    prompt: str,
    images: tuple[Path, ...],
    output_dir: Path,
    seeds: tuple[int, ...],
    width: int,
    height: int,
    steps: int | None,
    guidance: float | None,
    sampler: str,
    loras: tuple[LoraLoadSpec, ...],
    progress: StatusReporter,
) -> dict[str, Any]:
    from aigen.generation.flux2_klein import (
        Flux2KleinError,
        generate_flux2_klein_seed_sweep,
    )

    if steps is not None and steps != FLUX2_KLEIN_STEPS:
        raise ImageEditCommandError(
            f"{FLUX2_KLEIN_BACKEND} uses its official {FLUX2_KLEIN_STEPS}-step schedule"
        )
    if guidance is not None:
        raise ImageEditCommandError(f"{FLUX2_KLEIN_BACKEND} does not expose CFG guidance")
    try:
        result = generate_flux2_klein_seed_sweep(
            prompt=prompt,
            output=output_dir / "image.png",
            references=images,
            width=width,
            height=height,
            seeds=seeds,
            sampler=sampler,
            loras=loras,
            progress=progress,
        )
    except Flux2KleinError as error:
        raise ImageEditCommandError(str(error)) from error
    return {
        "status": "completed",
        "kind": "image-edit-result",
        "backend": FLUX2_KLEIN_BACKEND,
        "output_dir": output_dir.as_posix(),
        "seeds": list(seeds),
        **result.to_json(),
    }


def _resolve_sampler(backend: str, sampler: str | None) -> str:
    settings = IMAGE_EDIT_BACKEND_SETTINGS[backend]
    resolved = settings.sampler if sampler is None else sampler
    if resolved not in settings.samplers:
        raise ImageEditCommandError(
            f"{backend} does not support sampler {resolved!r}; choose from: "
            f"{', '.join(settings.samplers)}"
        )
    return resolved


def _resolve_scheduler(backend: str, scheduler: str | None) -> str:
    settings = IMAGE_EDIT_BACKEND_SETTINGS[backend]
    resolved = settings.scheduler if scheduler is None else scheduler
    if resolved not in settings.schedulers:
        raise ImageEditCommandError(
            f"{backend} does not support scheduler {resolved!r}; choose from: "
            f"{', '.join(settings.schedulers)}"
        )
    return resolved


def _run_qwen_2511(
    *,
    backend: str,
    prompt: str,
    images: tuple[Path, ...],
    output_dir: Path,
    seeds: tuple[int, ...],
    width: int,
    height: int,
    steps: int | None,
    guidance: float | None,
    sampler: str,
    scheduler: str,
    loras: tuple[LoraLoadSpec, ...],
    progress: StatusReporter,
) -> dict[str, Any]:
    from aigen.generation.qwen_image_edit_identity import (
        DEFAULT_QWEN_IDENTITY_MAX_SEQUENCE_LENGTH,
        DEFAULT_QWEN_IDENTITY_MAX_SIDE,
        DEFAULT_QWEN_UPSCALE_LONG_SIDE,
        QwenIdentityCase,
        QwenImageEditIdentityError,
        qwen_image_edit_identity_profile_for_name,
        run_qwen_image_edit_cases,
    )
    from aigen.generation.qwen_image_edit_lightx2v import (
        LIGHTX2V_QWEN_EDIT_2511_BASE_PROFILE,
        LIGHTX2V_QWEN_EDIT_2511_PROFILE,
    )

    profile_name = (
        LIGHTX2V_QWEN_EDIT_2511_PROFILE
        if backend == QWEN_2511_LIGHTNING_BACKEND
        else LIGHTX2V_QWEN_EDIT_2511_BASE_PROFILE
    )
    source_names = tuple(f"image_{index}" for index in range(1, len(images) + 1))
    try:
        result = run_qwen_image_edit_cases(
            source_images=dict(zip(source_names, images, strict=True)),
            references={},
            guides={},
            controls={},
            output_dir=output_dir,
            profile=qwen_image_edit_identity_profile_for_name(profile_name),
            edit_cases=(
                QwenIdentityCase(
                    name="image",
                    source_images=source_names,
                    references=(),
                    prompt=prompt,
                    seeds=seeds,
                ),
            ),
            max_side=DEFAULT_QWEN_IDENTITY_MAX_SIDE,
            steps=steps,
            true_cfg_scale=guidance,
            guidance_scale=None,
            seed=seeds[0],
            max_sequence_length=DEFAULT_QWEN_IDENTITY_MAX_SEQUENCE_LENGTH,
            candidates_per_case=len(seeds),
            overwrite=False,
            nunchaku_blocks_on_gpu=None,
            aspect_ratio=None,
            canvas_size=(width, height),
            upscale_long_side=DEFAULT_QWEN_UPSCALE_LONG_SIDE,
            postprocess="none",
            result_kind="image-edit-result",
            manifest_context=None,
            loras=loras,
            sampler=sampler,
            scheduler=scheduler,
            progress=progress,
        )
    except QwenImageEditIdentityError as error:
        raise ImageEditCommandError(str(error)) from error
    return {
        "backend": backend,
        "output_dir": output_dir.as_posix(),
        **result,
    }


def _run_hidream_o1(
    *,
    prompt: str,
    images: tuple[Path, ...],
    output_dir: Path,
    seeds: tuple[int, ...],
    width: int,
    height: int,
    steps: int | None,
    guidance: float | None,
    sampler: str,
    scheduler: str,
    progress: StatusReporter,
) -> dict[str, Any]:
    from aigen.generation.hidream_o1_comfy import (
        HiDreamO1Error,
        generate_hidream_o1_seed_sweep,
    )

    try:
        results = generate_hidream_o1_seed_sweep(
            prompt=prompt,
            references=images,
            output=output_dir / "image.png",
            width=width,
            height=height,
            seeds=seeds,
            steps=HIDREAM_DEFAULT_STEPS if steps is None else steps,
            guidance=HIDREAM_DEFAULT_GUIDANCE if guidance is None else guidance,
            sampler=sampler,
            scheduler=scheduler,
            progress=progress,
        )
    except HiDreamO1Error as error:
        raise ImageEditCommandError(str(error)) from error
    return _image_edit_payload(
        backend=HIDREAM_O1_BACKEND,
        output_dir=output_dir,
        seeds=seeds,
        outputs=[result.to_json() for result in results],
    )


def _run_boogu_image_edit(
    *,
    prompt: str,
    images: tuple[Path, ...],
    output_dir: Path,
    seeds: tuple[int, ...],
    width: int,
    height: int,
    steps: int | None,
    guidance: float | None,
    progress: StatusReporter,
) -> dict[str, Any]:
    from aigen.generation.boogu_image_edit import (
        BooguImageEditError,
        generate_boogu_image_edit_seed_sweep,
    )

    try:
        results = generate_boogu_image_edit_seed_sweep(
            prompt=prompt,
            references=images,
            output=output_dir / "image.png",
            width=width,
            height=height,
            seeds=seeds,
            steps=BOOGU_DEFAULT_STEPS if steps is None else steps,
            guidance=BOOGU_DEFAULT_GUIDANCE if guidance is None else guidance,
            progress=progress,
        )
    except BooguImageEditError as error:
        raise ImageEditCommandError(str(error)) from error
    return _image_edit_payload(
        backend=BOOGU_IMAGE_EDIT_BACKEND,
        output_dir=output_dir,
        seeds=seeds,
        outputs=[result.to_json() for result in results],
    )


def _image_edit_payload(
    *,
    backend: str,
    output_dir: Path,
    seeds: tuple[int, ...],
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "completed",
        "kind": "image-edit-result",
        "backend": backend,
        "output_dir": output_dir.as_posix(),
        "seeds": list(seeds),
        "outputs": outputs,
    }


def _resolve_images(
    paths: list[Path] | None,
    reference_packs: list[Path] | None,
) -> tuple[Path, ...]:
    image_paths = [path.expanduser().resolve() for path in paths or ()]
    for pack_path in reference_packs or ():
        image_paths.extend(load_character_reference_pack(pack_path).references.values())
    if not image_paths:
        raise ImageEditCommandError("at least one --image or --reference-pack is required")
    images = tuple(image_paths)
    missing = next((path for path in images if not path.is_file()), None)
    if missing is not None:
        raise ImageEditCommandError(f"input image does not exist: {missing}")
    return images


def _resolve_dimensions(
    width: int | None,
    height: int | None,
    aspect_ratio: tuple[int, int] | None,
) -> tuple[int | None, int | None]:
    if (width is None) != (height is None):
        raise ImageEditCommandError("--width and --height must be provided together")
    if width is not None and aspect_ratio is not None:
        raise ImageEditCommandError("--aspect-ratio cannot be combined with --width/--height")
    if width is not None and (width < 1 or height is None or height < 1):
        raise ImageEditCommandError("--width and --height must be positive")
    return width, height


def _resolve_loras(
    paths: list[Path] | None,
    weights: list[float] | None,
    backend: str,
) -> tuple[LoraLoadSpec, ...]:
    if not paths:
        if weights:
            raise ImageEditCommandError("--lora-weight requires --lora")
        return ()
    if weights and len(weights) != len(paths):
        raise ImageEditCommandError(
            "Repeat --lora-weight once per --lora, or omit all weights to use 1.0"
        )
    resolved_weights = tuple(weights or (1.0,) * len(paths))
    if any(not math.isfinite(weight) for weight in resolved_weights):
        raise ImageEditCommandError("--lora-weight must be finite")

    expected_architecture = IMAGE_EDIT_BACKEND_LORA_ARCHITECTURES.get(backend)
    if expected_architecture is None:
        raise ImageEditCommandError(f"{backend} does not support --lora")
    resolved = []
    for path, weight in zip(paths, resolved_weights, strict=True):
        info = inspect_lora_weights(path)
        if info.architecture != expected_architecture:
            raise ImageEditCommandError(
                f"{backend} requires a {expected_architecture} LoRA, but {info.path} "
                f"targets {info.architecture}"
            )
        resolved.append(LoraLoadSpec(path=info.path, weight=weight))
    return tuple(resolved)


def _image_aspect_ratio(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return normalized_aspect_ratio(*image.size)


def _recommended_canvas_size(
    backend: str,
    aspect_ratio: tuple[int, int],
    *,
    closest: bool,
) -> tuple[int, int]:
    if backend == FLUX2_KLEIN_BACKEND:
        from aigen.generation.flux2_klein import flux2_klein_recommended_canvas_size

        return flux2_klein_recommended_canvas_size(aspect_ratio)
    if backend in (QWEN_2511_LIGHTNING_BACKEND, QWEN_2511_BASE_BACKEND):
        from aigen.generation.qwen_image_edit_identity import (
            QwenImageEditIdentityError,
            qwen_image_edit_2511_native_canvas_size,
        )

        try:
            return qwen_image_edit_2511_native_canvas_size(aspect_ratio, closest=closest)
        except QwenImageEditIdentityError as error:
            raise ImageEditCommandError(str(error)) from error
    if backend == HIDREAM_O1_BACKEND:
        from aigen.generation.hidream_o1_comfy import hidream_o1_native_canvas_size

        return hidream_o1_native_canvas_size(aspect_ratio)

    from aigen.generation.boogu_image_edit import (
        BooguImageEditError,
        boogu_recommended_1k_canvas_size,
    )

    try:
        return boogu_recommended_1k_canvas_size(aspect_ratio, closest=closest)
    except BooguImageEditError as error:
        raise ImageEditCommandError(str(error)) from error


def _prepare_output_directory(path: Path, *, overwrite: bool) -> Path:
    output_dir = path.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ImageEditCommandError(f"output directory is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise ImageEditCommandError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    return output_dir
