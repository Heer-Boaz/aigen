from __future__ import annotations

import sys
from pathlib import Path

from aigen.generation.image_upscale import (
    DEFAULT_UPSCALE_MODEL,
    upscale_model_names,
)
from aigen.generation.vosr_backend import (
    VOSR_MODEL_NAME,
    VOSR_POSTPROCESS_NAME,
)
from aigen.image_tui_model import DropdownOption, FormField


UPSCALE_OPERATION = "upscale"
DOWNSCALE_OPERATION = "downscale"
EXTRACT_VIDEO_FRAMES_OPERATION = "extract-video-frames"
POSTPROCESS_OPERATIONS = (
    UPSCALE_OPERATION,
    DOWNSCALE_OPERATION,
    EXTRACT_VIDEO_FRAMES_OPERATION,
)
POSTPROCESS_OPERATION_LABELS = {
    UPSCALE_OPERATION: "Upscale image",
    DOWNSCALE_OPERATION: "Downscale image",
    EXTRACT_VIDEO_FRAMES_OPERATION: "Extract video frames",
}

VOSR_MODEL = VOSR_POSTPROCESS_NAME
WU_PIXELIZATION_MODEL = "wu-pixelization"
PIXEL_ART_FIXER_MODEL = "pixel-art-fixer"

UPSCALE_MODELS = (VOSR_MODEL, *upscale_model_names())
DOWNSCALE_MODELS = (WU_PIXELIZATION_MODEL, PIXEL_ART_FIXER_MODEL)
POSTPROCESS_MODEL_LABELS = {
    VOSR_MODEL: VOSR_MODEL_NAME,
    "illustrationjanai-dat2": "IllustrationJaNai DAT2",
    "illustrationjanai-esrgan": "IllustrationJaNai ESRGAN",
    "animesharp-x4": "AnimeSharp x4",
    WU_PIXELIZATION_MODEL: "Wu Pixelization",
    PIXEL_ART_FIXER_MODEL: "Pixel Art Fixer",
}


class PostprocessForm:
    def __init__(self) -> None:
        self.slot_move_states: dict[int, tuple[bool, bool]] = {}
        self._fields = {
            "operation": FormField("operation", "Operation", UPSCALE_OPERATION),
            "model": FormField("model", "Model", VOSR_MODEL),
            "input": FormField("input", "Input image", "", "image"),
            "output_dir": FormField(
                "output_dir",
                "Output directory",
                "runs/postprocess",
            ),
            "long_side": FormField("long_side", "Long side", "2048"),
            "cell_size": FormField("cell_size", "Cell size", "16"),
            "mode": FormField("mode", "Detection mode", "full"),
            "low_memory": FormField("low_memory", "Low memory", "false"),
            "force_step": FormField("force_step", "Force cell size", ""),
        }
        self.fields: list[FormField] = []
        self._rebuild_fields()

    def field(self, name: str) -> FormField:
        return self._fields[name]

    def set_value(self, field: FormField, value: str) -> None:
        field.value = value
        if field.name == "operation":
            input_field = self._fields["input"]
            if value == EXTRACT_VIDEO_FRAMES_OPERATION:
                input_field.label = "Input video"
                input_field.slot_kind = "video"
            else:
                input_field.label = "Input image"
                input_field.slot_kind = "image"
                self._fields["model"].value = (
                    VOSR_MODEL
                    if value == UPSCALE_OPERATION
                    else WU_PIXELIZATION_MODEL
                )
        if field.name in {"operation", "model"}:
            self._rebuild_fields()

    def dropdown_options(self, field: FormField) -> tuple[DropdownOption, ...] | None:
        if field.name == "operation":
            return tuple(
                DropdownOption(POSTPROCESS_OPERATION_LABELS[operation], operation)
                for operation in POSTPROCESS_OPERATIONS
            )
        if field.name == "model":
            models = (
                UPSCALE_MODELS
                if self._fields["operation"].value == UPSCALE_OPERATION
                else DOWNSCALE_MODELS
            )
            return tuple(
                DropdownOption(POSTPROCESS_MODEL_LABELS[model], model)
                for model in models
            )
        if field.name == "mode":
            return (
                DropdownOption("Full", "full"),
                DropdownOption("Fast", "fast"),
            )
        if field.name == "low_memory":
            return (
                DropdownOption("Off", "false"),
                DropdownOption("On", "true"),
            )
        return None

    def generation_command(self) -> tuple[list[str], str]:
        input_path = self._fields["input"].value.strip()
        output_dir = self._fields["output_dir"].value.strip()
        if not input_path:
            media_type = (
                "video"
                if self._fields["operation"].value == EXTRACT_VIDEO_FRAMES_OPERATION
                else "image"
            )
            raise ValueError(f"Input {media_type} is required.")
        if not output_dir:
            raise ValueError("Output directory is required.")

        if self._fields["operation"].value == EXTRACT_VIDEO_FRAMES_OPERATION:
            frames_dir = (
                Path(output_dir) / f"{Path(input_path).stem}-frames"
            ).as_posix()
            return (
                [
                    sys.executable,
                    "-m",
                    "aigen.cli",
                    "video-postprocess",
                    "extract-frames",
                    "--input",
                    input_path,
                    "--output-dir",
                    frames_dir,
                ],
                frames_dir,
            )

        model = self._fields["model"].value
        output = (Path(output_dir) / f"{Path(input_path).stem}-{model}.png").as_posix()
        command = [sys.executable, "-m", "aigen.cli"]
        if model == VOSR_MODEL:
            command.extend(
                (
                    "characters",
                    "vosr-upscale",
                    "--input",
                    input_path,
                    "--output",
                    output,
                    "--long-side",
                    self._fields["long_side"].value.strip(),
                )
            )
        elif model in upscale_model_names():
            command.extend(
                (
                    "characters",
                    "postprocess",
                    "--image",
                    input_path,
                    "--output",
                    output,
                    "--long-side",
                    self._fields["long_side"].value.strip(),
                    "--model",
                    model,
                )
            )
        elif model == WU_PIXELIZATION_MODEL:
            command.extend(
                (
                    "pixel-art-wu",
                    "--input",
                    input_path,
                    "--output",
                    output,
                    "--cell-size",
                    self._fields["cell_size"].value.strip(),
                )
            )
        elif model == PIXEL_ART_FIXER_MODEL:
            command.extend(
                (
                    "pixel-art-fixer",
                    "--input",
                    input_path,
                    "--output",
                    output,
                    "--mode",
                    self._fields["mode"].value.strip(),
                )
            )
            if self._fields["low_memory"].value == "true":
                command.append("--low-memory")
            force_step = self._fields["force_step"].value.strip()
            if force_step:
                command.extend(("--force-step", force_step))
        else:
            raise RuntimeError(f"unsupported postprocessing model: {model}")
        return command, output_dir

    def _rebuild_fields(self) -> None:
        if self._fields["operation"].value == EXTRACT_VIDEO_FRAMES_OPERATION:
            self.fields = [
                self._fields[name]
                for name in ("operation", "input", "output_dir")
            ]
            return
        model = self._fields["model"].value
        names = ["operation", "model", "input", "output_dir"]
        if model in UPSCALE_MODELS:
            names.append("long_side")
        elif model == WU_PIXELIZATION_MODEL:
            names.append("cell_size")
        elif model == PIXEL_ART_FIXER_MODEL:
            names.extend(("mode", "low_memory", "force_step"))
        else:
            raise RuntimeError(f"unsupported postprocessing model: {model}")
        self.fields = [self._fields[name] for name in names]
