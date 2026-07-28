from __future__ import annotations

import math
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from PIL import Image

from aigen.character_reference_pack import load_character_reference_pack
from aigen.image_edit_defaults import (
    BOOGU_DEFAULT_GUIDANCE,
    BOOGU_DEFAULT_STEPS,
    BOOGU_SAMPLER,
    BOOGU_SCHEDULER,
    FLUX2_KLEIN_DEFAULT_SAMPLER,
    FLUX2_KLEIN_SAMPLERS,
    FLUX2_KLEIN_SCHEDULER,
    FLUX2_KLEIN_STEPS,
    FLUX2_DEV_GUIDANCE,
    FLUX2_DEV_SAMPLER,
    FLUX2_DEV_SCHEDULER,
    FLUX2_DEV_STEPS,
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
    USO_FLUX1_DEFAULT_GUIDANCE,
    USO_FLUX1_DEFAULT_STEPS,
    USO_FLUX1_SAMPLER,
    USO_FLUX1_SCHEDULER,
)
from aigen.image_dimensions import normalized_aspect_ratio
from aigen.lora_weights import (
    FLUX2_DEV_ARCHITECTURE,
    FLUX2_KLEIN_ARCHITECTURE,
    QWEN_IMAGE_ARCHITECTURE,
    LoraLoadSpec,
    inspect_lora_weights,
)
from aigen.progress import StatusReporter


FLUX2_KLEIN_BACKEND = "flux2-klein"
FLUX2_DEV_BACKEND = "flux2-dev-nvfp4"
QWEN_2511_LIGHTNING_BACKEND = "qwen-image-edit-2511-lightning"
QWEN_2511_BASE_BACKEND = "qwen-image-edit-2511-base"
HIDREAM_O1_BACKEND = "hidream-o1-full-fp8"
BOOGU_IMAGE_EDIT_BACKEND = "boogu-image-edit-turbo-fp8"
USO_FLUX1_BACKEND = "uso-flux1-dev-fp8"
IMAGE_EDIT_BACKENDS = (
    FLUX2_KLEIN_BACKEND,
    FLUX2_DEV_BACKEND,
    QWEN_2511_LIGHTNING_BACKEND,
    QWEN_2511_BASE_BACKEND,
    HIDREAM_O1_BACKEND,
    BOOGU_IMAGE_EDIT_BACKEND,
    USO_FLUX1_BACKEND,
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
    FLUX2_DEV_BACKEND: FLUX2_DEV_ARCHITECTURE,
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
    strength: float | None = None
    supports_strength: bool = False
    supports_empty_prompt: bool = False
    image_slot_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedImageEditSettings:
    width: int | None
    height: int | None
    aspect_ratio: tuple[int, int] | None
    steps: int
    guidance: float | None
    strength: float | None
    sampler: str
    scheduler: str


IMAGE_EDIT_BACKEND_SETTINGS = {
    FLUX2_KLEIN_BACKEND: ImageEditBackendSettings(
        FLUX2_KLEIN_STEPS,
        None,
        FLUX2_KLEIN_DEFAULT_SAMPLER,
        FLUX2_KLEIN_SAMPLERS,
        FLUX2_KLEIN_SCHEDULER,
        (FLUX2_KLEIN_SCHEDULER,),
        supports_strength=True,
    ),
    FLUX2_DEV_BACKEND: ImageEditBackendSettings(
        FLUX2_DEV_STEPS,
        FLUX2_DEV_GUIDANCE,
        FLUX2_DEV_SAMPLER,
        (FLUX2_DEV_SAMPLER,),
        FLUX2_DEV_SCHEDULER,
        (FLUX2_DEV_SCHEDULER,),
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
    USO_FLUX1_BACKEND: ImageEditBackendSettings(
        USO_FLUX1_DEFAULT_STEPS,
        USO_FLUX1_DEFAULT_GUIDANCE,
        USO_FLUX1_SAMPLER,
        (USO_FLUX1_SAMPLER,),
        USO_FLUX1_SCHEDULER,
        (USO_FLUX1_SCHEDULER,),
        supports_empty_prompt=True,
        image_slot_labels=("Content image", "Style image 1", "Style image 2"),
    ),
}


class ImageEditError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageEditRequest:
    backend: str
    prompt: str
    output_dir: Path
    images: tuple[Path, ...] = ()
    reference_packs: tuple[Path, ...] = ()
    seeds: tuple[int, ...] = (0,)
    width: int | None = None
    height: int | None = None
    aspect_ratio: tuple[int, int] | None = None
    steps: int | None = None
    guidance: float | None = None
    strength: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    loras: tuple[Path, ...] = ()
    lora_weights: tuple[float, ...] = ()
    overwrite: bool = False


@dataclass(frozen=True)
class ResolvedImageEditRequest:
    backend: str
    prompt: str
    output_dir: Path
    images: tuple[Path, ...]
    seeds: tuple[int, ...]
    width: int
    height: int
    steps: int
    guidance: float | None
    strength: float | None
    sampler: str
    scheduler: str
    loras: tuple[LoraLoadSpec, ...]
    replace_output: bool


@dataclass(frozen=True)
class ImageEditOutput:
    path: Path
    width: int
    height: int
    seed: int


@dataclass(frozen=True)
class ImageEditResult:
    backend: str
    output_dir: Path
    outputs: tuple[ImageEditOutput, ...]
    json_payload: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return dict(self.json_payload)


@dataclass(frozen=True)
class _BackendImageEditResult:
    outputs: tuple[ImageEditOutput, ...]
    payload: dict[str, Any]


class _SeedSweepResult(Protocol):
    output: Path
    width: int
    height: int
    seed: int

    def to_json(self) -> dict[str, Any]: ...


def image_edit_backend_settings(backend: str) -> ImageEditBackendSettings:
    try:
        return IMAGE_EDIT_BACKEND_SETTINGS[backend]
    except KeyError as error:
        raise ImageEditError(
            f"unsupported image-edit backend: {backend}"
        ) from error


def resolve_image_edit_settings(
    *,
    backend: str,
    width: int | None,
    height: int | None,
    aspect_ratio: tuple[int, int] | None,
    steps: int | None,
    guidance: float | None,
    strength: float | None,
    sampler: str | None,
    scheduler: str | None,
) -> ResolvedImageEditSettings:
    settings = image_edit_backend_settings(backend)
    if (width is None) != (height is None):
        raise ImageEditError("--width and --height must be provided together")
    if width is not None and aspect_ratio is not None:
        raise ImageEditError(
            "--aspect-ratio cannot be combined with --width/--height"
        )
    if width is not None and (width < 1 or height is None or height < 1):
        raise ImageEditError("--width and --height must be positive")

    resolved_steps = settings.steps if steps is None else steps
    if resolved_steps < 1:
        raise ImageEditError("--steps must be positive")
    if backend == FLUX2_KLEIN_BACKEND:
        if resolved_steps != FLUX2_KLEIN_STEPS:
            raise ImageEditError(
                f"{FLUX2_KLEIN_BACKEND} uses its official "
                f"{FLUX2_KLEIN_STEPS}-step schedule"
            )
        if guidance is not None:
            raise ImageEditError(
                f"{FLUX2_KLEIN_BACKEND} does not expose CFG guidance"
            )

    resolved_strength = settings.strength if strength is None else strength
    if resolved_strength is not None:
        if not settings.supports_strength:
            raise ImageEditError(
                f"{backend} does not expose image-to-image strength"
            )
        if not 0.0 < resolved_strength <= 1.0:
            raise ImageEditError("--strength must be in (0, 1]")

    resolved_sampler = settings.sampler if sampler is None else sampler
    if resolved_sampler not in settings.samplers:
        raise ImageEditError(
            f"{backend} does not support sampler {resolved_sampler!r}; choose from: "
            f"{', '.join(settings.samplers)}"
        )
    resolved_scheduler = settings.scheduler if scheduler is None else scheduler
    if resolved_scheduler not in settings.schedulers:
        raise ImageEditError(
            f"{backend} does not support scheduler {resolved_scheduler!r}; choose from: "
            f"{', '.join(settings.schedulers)}"
        )
    return ResolvedImageEditSettings(
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        steps=resolved_steps,
        guidance=settings.guidance if guidance is None else guidance,
        strength=resolved_strength,
        sampler=resolved_sampler,
        scheduler=resolved_scheduler,
    )


def resolve_image_edit_canvas_size(
    *,
    backend: str,
    first_reference: Path,
    settings: ResolvedImageEditSettings,
) -> tuple[int, int]:
    if settings.width is not None:
        assert settings.height is not None
        return settings.width, settings.height
    inferred_aspect = settings.aspect_ratio is None
    target_aspect = settings.aspect_ratio or _image_aspect_ratio(first_reference)
    return _recommended_canvas_size(
        backend,
        target_aspect,
        closest=inferred_aspect,
    )


def resolve_image_edit_request(
    request: ImageEditRequest,
) -> ResolvedImageEditRequest:
    backend_settings = image_edit_backend_settings(request.backend)
    prompt = request.prompt.strip()
    if not prompt and not backend_settings.supports_empty_prompt:
        raise ImageEditError("--prompt must not be empty")
    images = _resolve_images(request.images, request.reference_packs)
    if not request.seeds:
        raise ImageEditError("at least one seed is required")
    settings = resolve_image_edit_settings(
        backend=request.backend,
        width=request.width,
        height=request.height,
        aspect_ratio=request.aspect_ratio,
        steps=request.steps,
        guidance=request.guidance,
        strength=request.strength,
        sampler=request.sampler,
        scheduler=request.scheduler,
    )
    loras = _resolve_loras(
        request.loras,
        request.lora_weights,
        request.backend,
    )
    width, height = resolve_image_edit_canvas_size(
        backend=request.backend,
        first_reference=images[0],
        settings=settings,
    )
    output_dir, replace_output = _resolve_output_directory(
        request.output_dir,
        overwrite=request.overwrite,
    )
    return ResolvedImageEditRequest(
        backend=request.backend,
        prompt=prompt,
        output_dir=output_dir,
        images=images,
        seeds=request.seeds,
        width=width,
        height=height,
        steps=settings.steps,
        guidance=settings.guidance,
        strength=settings.strength,
        sampler=settings.sampler,
        scheduler=settings.scheduler,
        loras=loras,
        replace_output=replace_output,
    )


def run_image_edit(
    request: ImageEditRequest,
    *,
    progress: StatusReporter,
) -> ImageEditResult:
    resolved = resolve_image_edit_request(request)
    if resolved.replace_output:
        shutil.rmtree(resolved.output_dir)

    if resolved.backend == FLUX2_KLEIN_BACKEND:
        backend_result = _run_flux2_klein(
            prompt=resolved.prompt,
            images=resolved.images,
            output_dir=resolved.output_dir,
            seeds=resolved.seeds,
            width=resolved.width,
            height=resolved.height,
            sampler=resolved.sampler,
            loras=resolved.loras,
            strength=resolved.strength,
            progress=progress,
        )
    elif resolved.backend == FLUX2_DEV_BACKEND:
        backend_result = _run_flux2_dev(
            prompt=resolved.prompt,
            images=resolved.images,
            output_dir=resolved.output_dir,
            seeds=resolved.seeds,
            width=resolved.width,
            height=resolved.height,
            steps=resolved.steps,
            guidance=cast(float, resolved.guidance),
            loras=resolved.loras,
            progress=progress,
        )
    elif resolved.backend in (
        QWEN_2511_LIGHTNING_BACKEND,
        QWEN_2511_BASE_BACKEND,
    ):
        backend_result = _run_qwen_2511(
            backend=resolved.backend,
            prompt=resolved.prompt,
            images=resolved.images,
            output_dir=resolved.output_dir,
            seeds=resolved.seeds,
            width=resolved.width,
            height=resolved.height,
            steps=resolved.steps,
            guidance=cast(float, resolved.guidance),
            sampler=resolved.sampler,
            scheduler=resolved.scheduler,
            loras=resolved.loras,
            progress=progress,
        )
    elif resolved.backend == HIDREAM_O1_BACKEND:
        backend_result = _run_hidream_o1(
            prompt=resolved.prompt,
            images=resolved.images,
            output_dir=resolved.output_dir,
            seeds=resolved.seeds,
            width=resolved.width,
            height=resolved.height,
            steps=resolved.steps,
            guidance=cast(float, resolved.guidance),
            sampler=resolved.sampler,
            scheduler=resolved.scheduler,
            progress=progress,
        )
    elif resolved.backend == BOOGU_IMAGE_EDIT_BACKEND:
        backend_result = _run_boogu_image_edit(
            prompt=resolved.prompt,
            images=resolved.images,
            output_dir=resolved.output_dir,
            seeds=resolved.seeds,
            width=resolved.width,
            height=resolved.height,
            steps=resolved.steps,
            guidance=cast(float, resolved.guidance),
            progress=progress,
        )
    elif resolved.backend == USO_FLUX1_BACKEND:
        backend_result = _run_uso_flux1(
            prompt=resolved.prompt,
            images=resolved.images,
            output_dir=resolved.output_dir,
            seeds=resolved.seeds,
            width=resolved.width,
            height=resolved.height,
            steps=resolved.steps,
            guidance=cast(float, resolved.guidance),
            progress=progress,
        )
    else:
        raise ImageEditError(
            f"unsupported image-edit backend: {resolved.backend}"
        )

    payload = backend_result.payload
    payload["sampler"] = resolved.sampler
    payload["scheduler"] = resolved.scheduler
    if resolved.strength is not None:
        payload["strength"] = resolved.strength
    return ImageEditResult(
        backend=resolved.backend,
        output_dir=resolved.output_dir,
        outputs=backend_result.outputs,
        json_payload=payload,
    )


def _run_flux2_klein(
    *,
    prompt: str,
    images: tuple[Path, ...],
    output_dir: Path,
    seeds: tuple[int, ...],
    width: int,
    height: int,
    sampler: str,
    loras: tuple[LoraLoadSpec, ...],
    strength: float | None,
    progress: StatusReporter,
) -> _BackendImageEditResult:
    from aigen.generation.flux2_klein import (
        Flux2KleinError,
        generate_flux2_klein_seed_sweep,
    )

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
            strength=strength,
            progress=progress,
        )
    except Flux2KleinError as error:
        raise ImageEditError(str(error)) from error
    return _BackendImageEditResult(
        outputs=tuple(
            ImageEditOutput(
                path=Path(output.output),
                width=output.width,
                height=output.height,
                seed=output.seed,
            )
            for output in result.outputs
        ),
        payload={
            "status": "completed",
            "kind": "image-edit-result",
            "backend": FLUX2_KLEIN_BACKEND,
            "output_dir": output_dir.as_posix(),
            "seeds": list(seeds),
            **result.to_json(),
        },
    )


def _run_flux2_dev(
    *,
    prompt: str,
    images: tuple[Path, ...],
    output_dir: Path,
    seeds: tuple[int, ...],
    width: int,
    height: int,
    steps: int,
    guidance: float,
    loras: tuple[LoraLoadSpec, ...],
    progress: StatusReporter,
) -> _BackendImageEditResult:
    from aigen.generation.flux2_dev_wangp import (
        Flux2DevError,
        generate_flux2_dev_seed_sweep,
    )

    try:
        results = generate_flux2_dev_seed_sweep(
            prompt=prompt,
            references=images,
            output=output_dir / "image.png",
            width=width,
            height=height,
            seeds=seeds,
            steps=steps,
            guidance=guidance,
            loras=loras,
            progress=progress,
        )
    except Flux2DevError as error:
        raise ImageEditError(str(error)) from error
    return _backend_image_edit_result(
        backend=FLUX2_DEV_BACKEND,
        output_dir=output_dir,
        seeds=seeds,
        results=results,
    )


def _run_qwen_2511(
    *,
    backend: str,
    prompt: str,
    images: tuple[Path, ...],
    output_dir: Path,
    seeds: tuple[int, ...],
    width: int,
    height: int,
    steps: int,
    guidance: float,
    sampler: str,
    scheduler: str,
    loras: tuple[LoraLoadSpec, ...],
    progress: StatusReporter,
) -> _BackendImageEditResult:
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
        raise ImageEditError(str(error)) from error
    return _BackendImageEditResult(
        outputs=tuple(
            ImageEditOutput(
                path=Path(output["image"]["path"]),
                width=int(output["width"]),
                height=int(output["height"]),
                seed=int(output["seed"]),
            )
            for output in result["outputs"]
        ),
        payload={
            "backend": backend,
            "output_dir": output_dir.as_posix(),
            **result,
        },
    )


def _run_hidream_o1(
    *,
    prompt: str,
    images: tuple[Path, ...],
    output_dir: Path,
    seeds: tuple[int, ...],
    width: int,
    height: int,
    steps: int,
    guidance: float,
    sampler: str,
    scheduler: str,
    progress: StatusReporter,
) -> _BackendImageEditResult:
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
            steps=steps,
            guidance=guidance,
            sampler=sampler,
            scheduler=scheduler,
            progress=progress,
        )
    except HiDreamO1Error as error:
        raise ImageEditError(str(error)) from error
    return _backend_image_edit_result(
        backend=HIDREAM_O1_BACKEND,
        output_dir=output_dir,
        seeds=seeds,
        results=results,
    )


def _run_boogu_image_edit(
    *,
    prompt: str,
    images: tuple[Path, ...],
    output_dir: Path,
    seeds: tuple[int, ...],
    width: int,
    height: int,
    steps: int,
    guidance: float,
    progress: StatusReporter,
) -> _BackendImageEditResult:
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
            steps=steps,
            guidance=guidance,
            progress=progress,
        )
    except BooguImageEditError as error:
        raise ImageEditError(str(error)) from error
    return _backend_image_edit_result(
        backend=BOOGU_IMAGE_EDIT_BACKEND,
        output_dir=output_dir,
        seeds=seeds,
        results=results,
    )


def _run_uso_flux1(
    *,
    prompt: str,
    images: tuple[Path, ...],
    output_dir: Path,
    seeds: tuple[int, ...],
    width: int,
    height: int,
    steps: int,
    guidance: float,
    progress: StatusReporter,
) -> _BackendImageEditResult:
    from aigen.generation.uso_flux1 import (
        UsoFlux1Error,
        generate_uso_flux1_seed_sweep,
    )

    try:
        results = generate_uso_flux1_seed_sweep(
            prompt=prompt,
            references=images,
            output=output_dir / "image.png",
            width=width,
            height=height,
            seeds=seeds,
            steps=steps,
            guidance=guidance,
            progress=progress,
        )
    except UsoFlux1Error as error:
        raise ImageEditError(str(error)) from error
    return _backend_image_edit_result(
        backend=USO_FLUX1_BACKEND,
        output_dir=output_dir,
        seeds=seeds,
        results=results,
    )


def _backend_image_edit_result(
    *,
    backend: str,
    output_dir: Path,
    seeds: tuple[int, ...],
    results: Sequence[_SeedSweepResult],
) -> _BackendImageEditResult:
    return _BackendImageEditResult(
        outputs=tuple(
            ImageEditOutput(
                path=result.output,
                width=result.width,
                height=result.height,
                seed=result.seed,
            )
            for result in results
        ),
        payload={
            "status": "completed",
            "kind": "image-edit-result",
            "backend": backend,
            "output_dir": output_dir.as_posix(),
            "seeds": list(seeds),
            "outputs": [result.to_json() for result in results],
        },
    )


def _resolve_images(
    paths: Sequence[Path],
    reference_packs: Sequence[Path],
) -> tuple[Path, ...]:
    image_paths = [path.expanduser().resolve() for path in paths]
    for pack_path in reference_packs:
        image_paths.extend(load_character_reference_pack(pack_path).references.values())
    if not image_paths:
        raise ImageEditError("at least one --image or --reference-pack is required")
    images = tuple(image_paths)
    missing = next((path for path in images if not path.is_file()), None)
    if missing is not None:
        raise ImageEditError(f"input image does not exist: {missing}")
    return images


def _resolve_loras(
    paths: Sequence[Path],
    weights: Sequence[float],
    backend: str,
) -> tuple[LoraLoadSpec, ...]:
    if not paths:
        if weights:
            raise ImageEditError("--lora-weight requires --lora")
        return ()
    if weights and len(weights) != len(paths):
        raise ImageEditError(
            "Repeat --lora-weight once per --lora, or omit all weights to use 1.0"
        )
    resolved_weights = tuple(weights or (1.0,) * len(paths))
    if any(not math.isfinite(weight) for weight in resolved_weights):
        raise ImageEditError("--lora-weight must be finite")

    expected_architecture = IMAGE_EDIT_BACKEND_LORA_ARCHITECTURES.get(backend)
    if expected_architecture is None:
        raise ImageEditError(f"{backend} does not support --lora")
    resolved = []
    for path, weight in zip(paths, resolved_weights, strict=True):
        info = inspect_lora_weights(path)
        if info.architecture != expected_architecture:
            raise ImageEditError(
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
    if backend == FLUX2_DEV_BACKEND:
        from aigen.generation.flux2_dimensions import (
            flux2_dev_recommended_canvas_size,
        )

        return flux2_dev_recommended_canvas_size(aspect_ratio)
    if backend in (QWEN_2511_LIGHTNING_BACKEND, QWEN_2511_BASE_BACKEND):
        from aigen.generation.qwen_image_edit_identity import (
            QwenImageEditIdentityError,
            qwen_image_edit_2511_native_canvas_size,
        )

        try:
            return qwen_image_edit_2511_native_canvas_size(aspect_ratio, closest=closest)
        except QwenImageEditIdentityError as error:
            raise ImageEditError(str(error)) from error
    if backend == HIDREAM_O1_BACKEND:
        from aigen.generation.hidream_o1_comfy import hidream_o1_native_canvas_size

        return hidream_o1_native_canvas_size(aspect_ratio)
    if backend == USO_FLUX1_BACKEND:
        from aigen.generation.uso_flux1 import uso_flux1_recommended_canvas_size

        return uso_flux1_recommended_canvas_size(aspect_ratio)
    from aigen.generation.boogu_image_edit import (
        BooguImageEditError,
        boogu_recommended_1k_canvas_size,
    )

    try:
        return boogu_recommended_1k_canvas_size(aspect_ratio, closest=closest)
    except BooguImageEditError as error:
        raise ImageEditError(str(error)) from error


def _resolve_output_directory(
    path: Path,
    *,
    overwrite: bool,
) -> tuple[Path, bool]:
    output_dir = path.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ImageEditError(f"output directory is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise ImageEditError(f"output directory already exists: {output_dir}")
        return output_dir, True
    return output_dir, False
