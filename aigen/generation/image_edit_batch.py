from __future__ import annotations

import shutil
from pathlib import Path
from collections.abc import Callable
from uuid import uuid4
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aigen.generation.image_generation_requests import (
    ImageGenerationCaseRequest,
    ImageGenerationOutputRequest,
)
from aigen.generation.image_edit import (
    FLUX2_KLEIN_BACKEND,
    QWEN_2511_BASE_BACKEND,
    QWEN_2511_LIGHTNING_BACKEND,
    validate_image_edit_inputs,
)
from aigen.image_edit_defaults import (
    FLUX2_KLEIN_SCHEDULER,
    FLUX2_KLEIN_STEPS,
)
from aigen.lora_weights import LoraLoadSpec
from aigen.progress import StatusReporter
from aigen.manifest_io import atomic_write_json


IMAGE_EDIT_BATCH_JOB_KIND = "aigen-image-edit-batch-job"
IMAGE_EDIT_BATCH_RESULT_KIND = "aigen-image-edit-batch-result"
IMAGE_EDIT_BATCH_VERSION = 1
_CASE_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]*$"
ImageEditBatchBackend = Literal[
    "flux2-klein",
    "qwen-image-edit-2511-lightning",
    "qwen-image-edit-2511-base",
]


class ImageEditBatchError(RuntimeError):
    pass


class BatchModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )


class ImageEditBatchLora(BatchModel):
    path: Path
    weight: float = 1.0


class ImageEditMask(BatchModel):
    source_image: Path
    mask_image: Path
    strength: float = Field(default=1.0, gt=0, le=1)


class ImageEditBatchCase(BatchModel):
    id: str = Field(
        min_length=1,
        max_length=128,
        pattern=_CASE_ID_PATTERN,
    )
    prompt: str = Field(min_length=1)
    image_paths: tuple[Path, ...] = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    seed: int
    output_path: Path
    mask: ImageEditMask | None = None


class ImageEditBatchRequest(BatchModel):
    kind: Literal["aigen-image-edit-batch-job"] = IMAGE_EDIT_BATCH_JOB_KIND
    version: Literal[1] = IMAGE_EDIT_BATCH_VERSION
    backend: ImageEditBatchBackend
    cases: tuple[ImageEditBatchCase, ...] = Field(min_length=1)
    loras: tuple[ImageEditBatchLora, ...] = ()
    steps: int | None = Field(default=None, gt=0)
    guidance: float | None = None
    strength: float | None = Field(default=None, gt=0, le=1)
    max_sequence_length: int | None = Field(default=None, gt=0)
    guidance_scale: float | None = None
    sampler: str = Field(min_length=1)
    scheduler: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_job(self) -> ImageEditBatchRequest:
        if any(case.mask is not None for case in self.cases):
            if self.backend != QWEN_2511_LIGHTNING_BACKEND:
                raise ValueError("masked image edits require the explicit Qwen-2511 Lightning backend")
            if not all(case.mask is not None for case in self.cases):
                raise ValueError("masked and whole-image edits belong in separate batches")
        case_ids = [case.id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("image-edit batch case ids must be unique")
        output_paths = [
            case.output_path.expanduser().resolve()
            for case in self.cases
        ]
        if len(set(output_paths)) != len(output_paths):
            raise ValueError("image-edit batch output paths must be unique")
        return self


class ImageEditBatchOutput(BatchModel):
    case_id: str
    path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    seed: int


class ImageEditBatchResult(BatchModel):
    kind: Literal["aigen-image-edit-batch-result"] = IMAGE_EDIT_BATCH_RESULT_KIND
    version: Literal[1] = IMAGE_EDIT_BATCH_VERSION
    status: Literal["completed"] = "completed"
    backend: ImageEditBatchBackend
    outputs: tuple[ImageEditBatchOutput, ...]


class ImageEditBatchFailure(BatchModel):
    kind: Literal["aigen-image-edit-batch-result"] = IMAGE_EDIT_BATCH_RESULT_KIND
    version: Literal[1] = IMAGE_EDIT_BATCH_VERSION
    status: Literal["error"] = "error"
    error: str
    message: str


def run_image_edit_batch(
    request: ImageEditBatchRequest,
    *,
    progress: StatusReporter,
    record_dir: Path | None = None,
    on_output: Callable[[ImageEditBatchOutput], None] | None = None,
) -> ImageEditBatchResult:
    cases = tuple(_resolved_case(case) for case in request.cases)
    for case in cases:
        validate_image_edit_inputs(request.backend, len(case.image_paths))
    if record_dir is None:
        record_dir = cases[0].output_path.parent / f"batch-{uuid4().hex}"
    record_dir = record_dir.expanduser().resolve()
    record_dir.mkdir(parents=True)
    atomic_write_json(record_dir / "request.json", request.model_copy(update={"cases": cases}).model_dump(mode="json"))
    loras = tuple(
        LoraLoadSpec(
            path=lora.path.expanduser().resolve(),
            weight=lora.weight,
        )
        for lora in request.loras
    )
    if request.backend == FLUX2_KLEIN_BACKEND:
        outputs = _run_flux2_klein_batch(
            request,
            cases=cases,
            loras=loras,
            progress=progress, record_dir=record_dir, on_output=on_output,
        )
    elif request.backend in (
        QWEN_2511_LIGHTNING_BACKEND,
        QWEN_2511_BASE_BACKEND,
    ):
        outputs = _run_qwen_2511_batch(
            request,
            cases=cases,
            loras=loras,
            progress=progress, record_dir=record_dir, on_output=on_output,
        )
    else:
        raise ImageEditBatchError(
            f"image-edit batching is not supported by {request.backend}"
        )
    return ImageEditBatchResult(
        backend=request.backend,
        outputs=outputs,
    )


def _resolved_case(case: ImageEditBatchCase) -> ImageEditBatchCase:
    if not case.prompt.strip():
        raise ImageEditBatchError(
            f"image-edit batch case {case.id!r} has an empty prompt"
        )
    image_paths = tuple(
        path.expanduser().resolve()
        for path in case.image_paths
    )
    missing = next((path for path in image_paths if not path.is_file()), None)
    if missing is not None:
        raise ImageEditBatchError(
            f"image-edit batch input does not exist: {missing}"
        )
    output_path = case.output_path.expanduser().resolve()
    mask = case.mask
    if mask is not None:
        if output_path.suffix.lower() != ".png":
            raise ImageEditBatchError("masked edits require lossless PNG output for exact pixel preservation")
        paths = {name: getattr(mask, name).expanduser().resolve() for name in ("source_image", "mask_image")}
        for path in paths.values():
            if not path.is_file():
                raise ImageEditBatchError(f"masked edit input does not exist: {path}")
        mask = mask.model_copy(update=paths)
    if output_path.exists():
        raise ImageEditBatchError(
            f"image-edit batch output already exists: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return case.model_copy(
        update={
            "image_paths": image_paths,
            "output_path": output_path,
            "mask": mask,
        }
    )


def _run_flux2_klein_batch(
    request: ImageEditBatchRequest,
    *,
    cases: tuple[ImageEditBatchCase, ...],
    loras: tuple[LoraLoadSpec, ...],
    progress: StatusReporter,
    record_dir: Path,
    on_output: Callable[[ImageEditBatchOutput], None] | None,
) -> tuple[ImageEditBatchOutput, ...]:
    from aigen.generation.flux2_klein import (
        FLUX2_KLEIN_SAMPLERS,
        Flux2KleinError,
        Flux2KleinSession,
        encode_flux2_klein_prompts,
    )

    if request.steps not in (None, FLUX2_KLEIN_STEPS):
        raise ImageEditBatchError(
            f"{FLUX2_KLEIN_BACKEND} uses its official "
            f"{FLUX2_KLEIN_STEPS}-step schedule"
        )
    if request.guidance is not None:
        raise ImageEditBatchError(
            f"{FLUX2_KLEIN_BACKEND} does not expose CFG guidance"
        )
    if request.max_sequence_length is not None or request.guidance_scale is not None:
        raise ImageEditBatchError("FLUX.2 Klein does not expose Qwen prompt-length or guidance-embedding settings")
    if request.scheduler != FLUX2_KLEIN_SCHEDULER:
        raise ImageEditBatchError(
            f"{FLUX2_KLEIN_BACKEND} requires scheduler "
            f"{FLUX2_KLEIN_SCHEDULER!r}"
        )
    if request.sampler not in FLUX2_KLEIN_SAMPLERS:
        raise ImageEditBatchError(
            f"unsupported FLUX.2 Klein sampler {request.sampler!r}"
        )

    generation_cases = tuple(
        ImageGenerationCaseRequest(
            name=case.id,
            prompt=case.prompt,
            image_paths=case.image_paths,
            width=case.width,
            height=case.height,
            outputs=(
                ImageGenerationOutputRequest(
                    name=case.id,
                    seed=case.seed,
                    path=case.output_path,
                ),
            ),
        )
        for case in cases
    )
    outputs = []

    def completed(output: Any) -> None:
        atomic_write_json(record_dir / "cases" / f"{output.case}.json", output.to_json())
        result = ImageEditBatchOutput(
            case_id=output.case, path=Path(output.output), width=output.width,
            height=output.height, seed=output.seed,
        )
        outputs.append(result)
        if on_output is not None:
            on_output(result)

    try:
        prompt_embeddings, encoding_ms = encode_flux2_klein_prompts(
            prompts=tuple(case.prompt for case in cases),
            progress=progress,
        )
        session = Flux2KleinSession(
            loras=loras,
            sampler=request.sampler,
            strength=request.strength,
            progress=progress,
        )
        try:
            result = session.generate(
                cases=generation_cases,
                prompt_embeddings=prompt_embeddings,
                progress=progress, on_output=completed,
            )
        finally:
            session.close()
    except Flux2KleinError as error:
        raise ImageEditBatchError(str(error)) from error

    atomic_write_json(record_dir / "result.json", {**result.to_json(), "prompt_encoding_ms": encoding_ms})
    return tuple(outputs)


def _run_qwen_2511_batch(
    request: ImageEditBatchRequest,
    *,
    cases: tuple[ImageEditBatchCase, ...],
    loras: tuple[LoraLoadSpec, ...],
    progress: StatusReporter,
    record_dir: Path,
    on_output: Callable[[ImageEditBatchOutput], None] | None,
) -> tuple[ImageEditBatchOutput, ...]:
    if cases[0].mask is not None:
        from aigen.generation.qwen_image_edit_masked import run_qwen_masked_batch

        return run_qwen_masked_batch(request, cases=cases, loras=loras, progress=progress,
                                     record_dir=record_dir, on_output=on_output)
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

    if request.strength is not None:
        raise ImageEditBatchError(
            f"{request.backend} does not expose image-to-image strength"
        )
    canvas_sizes = {(case.width, case.height) for case in cases}
    if len(canvas_sizes) != 1:
        raise ImageEditBatchError(
            "Qwen image-edit batch cases must use the same canvas size"
        )
    canvas_size = next(iter(canvas_sizes))
    source_images: dict[str, Path] = {}
    source_names_by_path: dict[Path, str] = {}
    edit_cases = []
    for case in cases:
        source_names = []
        for image_path in case.image_paths:
            source_name = source_names_by_path.get(image_path)
            if source_name is None:
                source_name = f"image-{len(source_images) + 1}"
                source_names_by_path[image_path] = source_name
                source_images[source_name] = image_path
            source_names.append(source_name)
        edit_cases.append(
            QwenIdentityCase(
                name=case.id,
                source_images=tuple(source_names),
                references=(),
                prompt=case.prompt,
                seeds=(case.seed,),
            )
        )

    profile_name = (
        LIGHTX2V_QWEN_EDIT_2511_PROFILE
        if request.backend == QWEN_2511_LIGHTNING_BACKEND
        else LIGHTX2V_QWEN_EDIT_2511_BASE_PROFILE
    )
    cases_by_id = {case.id: case for case in cases}
    outputs = []

    def completed(raw: dict[str, Any]) -> None:
        case = cases_by_id[raw["case"]]
        shutil.copyfile(Path(raw["path"]), case.output_path)
        atomic_write_json(record_dir / "cases" / f"{case.id}.json", raw)
        output = ImageEditBatchOutput(
            case_id=case.id, path=case.output_path, width=int(raw["width"]),
            height=int(raw["height"]), seed=int(raw["seed"]),
        )
        outputs.append(output)
        if on_output is not None:
            on_output(output)

    try:
        run_qwen_image_edit_cases(
            source_images=source_images, references={}, guides={}, controls={},
            output_dir=record_dir / "qwen",
            profile=qwen_image_edit_identity_profile_for_name(profile_name),
            edit_cases=tuple(edit_cases), max_side=DEFAULT_QWEN_IDENTITY_MAX_SIDE,
            steps=request.steps, true_cfg_scale=request.guidance, guidance_scale=request.guidance_scale,
            seed=cases[0].seed, max_sequence_length=(DEFAULT_QWEN_IDENTITY_MAX_SEQUENCE_LENGTH
                if request.max_sequence_length is None else request.max_sequence_length),
            candidates_per_case=1, overwrite=False, nunchaku_blocks_on_gpu=None,
            aspect_ratio=None, canvas_size=canvas_size,
            upscale_long_side=DEFAULT_QWEN_UPSCALE_LONG_SIDE,
            postprocess="none", result_kind=IMAGE_EDIT_BATCH_RESULT_KIND,
            manifest_context=None, loras=loras, sampler=request.sampler,
            scheduler=request.scheduler, progress=progress, on_raw_output=completed,
        )
    except QwenImageEditIdentityError as error:
        raise ImageEditBatchError(str(error)) from error
    return tuple(outputs)
