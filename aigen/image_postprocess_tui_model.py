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
POSTPROCESS_OPERATIONS = (UPSCALE_OPERATION, DOWNSCALE_OPERATION)

VOSR_MODEL = VOSR_POSTPROCESS_NAME
WU_PIXELIZATION_MODEL = "wu-pixelization"
SD_PIXL_MODEL = "sd-pixl"

UPSCALE_MODELS = (VOSR_MODEL, *upscale_model_names())
DOWNSCALE_MODELS = (WU_PIXELIZATION_MODEL, SD_PIXL_MODEL)
POSTPROCESS_MODEL_LABELS = {
    VOSR_MODEL: VOSR_MODEL_NAME,
    "illustrationjanai-dat2": "IllustrationJaNai DAT2",
    "illustrationjanai-esrgan": "IllustrationJaNai ESRGAN",
    "animesharp-x4": "AnimeSharp x4",
    WU_PIXELIZATION_MODEL: "Wu Pixelization",
    SD_PIXL_MODEL: "SD-piXL",
}


class ImagePostprocessForm:
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
            "width": FormField("width", "Width", "128"),
            "height": FormField("height", "Height", "128"),
            "colors": FormField("colors", "Colors", "16"),
            "steps": FormField("steps", "Steps", "10001"),
            "prompt": FormField("prompt", "Prompt", ""),
            "seed": FormField("seed", "Seed", "0"),
        }
        self.fields: list[FormField] = []
        self._rebuild_fields()

    def set_value(self, field: FormField, value: str) -> None:
        field.value = value
        if field.name == "operation":
            self._fields["model"].value = (
                VOSR_MODEL if value == UPSCALE_OPERATION else WU_PIXELIZATION_MODEL
            )
        if field.name in {"operation", "model"}:
            self._rebuild_fields()

    def dropdown_options(self, field: FormField) -> tuple[DropdownOption, ...] | None:
        if field.name == "operation":
            return tuple(
                DropdownOption(operation.title(), operation)
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
        return None

    def generation_command(self) -> tuple[list[str], str]:
        input_path = self._fields["input"].value.strip()
        output_dir = self._fields["output_dir"].value.strip()
        if not input_path:
            raise ValueError("Input image is required.")
        if not output_dir:
            raise ValueError("Output directory is required.")

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
        else:
            command.extend(
                (
                    "pixel-art",
                    "--input",
                    input_path,
                    "--output",
                    output,
                    "--width",
                    self._fields["width"].value.strip(),
                    "--height",
                    self._fields["height"].value.strip(),
                    "--colors",
                    self._fields["colors"].value.strip(),
                    "--steps",
                    self._fields["steps"].value.strip(),
                    "--seed",
                    self._fields["seed"].value.strip(),
                )
            )
            prompt = self._fields["prompt"].value.strip()
            if prompt:
                command.extend(("--prompt", prompt))
        return command, output_dir

    def _rebuild_fields(self) -> None:
        model = self._fields["model"].value
        names = ["operation", "model", "input", "output_dir"]
        if model in UPSCALE_MODELS:
            names.append("long_side")
        elif model == WU_PIXELIZATION_MODEL:
            names.append("cell_size")
        else:
            names.extend(("width", "height", "colors", "steps", "prompt", "seed"))
        self.fields = [self._fields[name] for name in names]
