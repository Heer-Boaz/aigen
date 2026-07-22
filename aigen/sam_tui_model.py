from __future__ import annotations

import sys

from aigen.generation.qwen_image_edit_identity import (
    DEFAULT_QWEN_INPAINT_PROFILE,
    qwen_image_edit_identity_profile_for_name,
    qwen_image_edit_inpaint_model_names,
)
from aigen.image_tui_model import DropdownOption, FormField


SAM_ENGINES = ("sam2", "sam1", "anime")
SAM_ENGINE_LABELS = {
    "sam2": "SAM2",
    "sam1": "SAM",
    "anime": "Anime segmentation",
}
SAM_OUTPUT_MODES = ("all", "mask", "cutout", "preview")
SAM_OUTPUT_MODE_LABELS = {
    "all": "Mask + cutout + preview",
    "mask": "Mask",
    "cutout": "Transparent cutout",
    "preview": "Overlay preview",
}
SAM_PROMPT_MODES = ("auto", "box", "points", "box+points")
SAM_PROMPT_MODE_LABELS = {
    "auto": "Automatic foreground",
    "box": "Box prompt",
    "points": "Point prompts",
    "box+points": "Box + point prompts",
}
SAM_MASK_SOURCES = ("mask", "region-plan")
SAM_MASK_SOURCE_LABELS = {
    "mask": "White-on-black mask",
    "region-plan": "Existing region plan",
}
SAM_OPERATIONS = ("segment", "qwen-edit", "region-plan")
SAM_OPERATION_LABELS = {
    "segment": "Segment image",
    "qwen-edit": "Qwen edit (mask or region plan)",
    "region-plan": "Text regions + SAM2 masks",
}


class SamEditForm:
    def __init__(self) -> None:
        self.slot_move_states: dict[int, tuple[bool, bool]] = {}
        self._fields = {
            "input": FormField("input", "Input image", "", "image"),
            "operation": FormField("operation", "Operation", "segment"),
            "mask_source": FormField("mask_source", "Edit mask source", "mask"),
            "mask": FormField("mask", "White-on-black edit mask", "", "image"),
            "region_plan": FormField("region_plan", "Region plan", "", "config"),
            "region": FormField("region", "Region name", ""),
            "reference_pack": FormField("reference_pack", "Reference pack", "", "reference_pack"),
            "instruction": FormField("instruction", "Edit instruction", ""),
            "profile": FormField("profile", "Qwen profile", DEFAULT_QWEN_INPAINT_PROFILE),
            "max_side": FormField("max_side", "Max source side", ""),
            "steps": FormField("steps", "Steps", ""),
            "true_cfg_scale": FormField("true_cfg_scale", "True CFG", ""),
            "guidance_scale": FormField("guidance_scale", "Guidance", ""),
            "strength": FormField("strength", "Edit strength", "0.6"),
            "padding_mask_crop": FormField("padding_mask_crop", "Mask crop padding", ""),
            "seed": FormField("seed", "Seed", "0"),
            "candidates": FormField("candidates", "Candidates", "2"),
            "max_sequence_length": FormField("max_sequence_length", "Max prompt tokens", "512"),
            "nunchaku_blocks_on_gpu": FormField("nunchaku_blocks_on_gpu", "Nunchaku blocks on GPU", ""),
            "engine": FormField("engine", "Segmenter", "sam2"),
            "device": FormField("device", "Device", "cuda"),
            "prompt_mode": FormField("prompt_mode", "Prompt mode", "auto"),
            "mask_candidate": FormField("mask_candidate", "Mask candidate", "auto"),
            "regions": FormField("regions", "Regions (name=text;...)", ""),
            "box": FormField("box", "Box (x1,y1,x2,y2)", ""),
            "positive_points": FormField("positive_points", "Positive points (x,y;...)", ""),
            "negative_points": FormField("negative_points", "Negative points (x,y;...)", ""),
            "threshold": FormField("threshold", "Auto-box threshold", "28"),
            "grow": FormField("grow", "Mask grow/shrink", "0"),
            "feather": FormField("feather", "Mask feather", "0"),
            "fill_holes": FormField("fill_holes", "Fill holes", "true"),
            "largest_component": FormField("largest_component", "Keep largest component", "true"),
            "invert": FormField("invert", "Invert mask", "false"),
            "output_mode": FormField("output_mode", "Output", "all"),
            "output_dir": FormField("output_dir", "Output directory", "runs/sam-tui"),
        }
        self.fields: list[FormField] = []
        self._rebuild_fields()
        self.set_value(self._fields["profile"], DEFAULT_QWEN_INPAINT_PROFILE)

    def field(self, name: str) -> FormField:
        return self._fields[name]

    def set_value(self, field: FormField, value: str) -> None:
        field.value = value
        if field.name == "profile":
            profile = qwen_image_edit_identity_profile_for_name(value)
            self.field("steps").value = str(profile.default_steps)
            self.field("true_cfg_scale").value = str(profile.default_true_cfg_scale)
            self.field("guidance_scale").value = str(profile.default_guidance_scale)
        if field.name == "engine" and value == "anime":
            self.field("device").value = "cuda"
            self.field("prompt_mode").value = "auto"
        if field.name in {"operation", "prompt_mode", "mask_source", "engine"}:
            self._rebuild_fields()

    def dropdown_options(self, field: FormField) -> tuple[DropdownOption, ...] | None:
        if field.name == "operation":
            return tuple(
                DropdownOption(SAM_OPERATION_LABELS[operation], operation)
                for operation in SAM_OPERATIONS
            )
        if field.name == "engine":
            return tuple(
                DropdownOption(SAM_ENGINE_LABELS[engine], engine)
                for engine in SAM_ENGINES
            )
        if field.name == "mask_candidate":
            return (
                DropdownOption("Highest score", "auto"),
                DropdownOption("Candidate 1", "1"),
                DropdownOption("Candidate 2", "2"),
                DropdownOption("Candidate 3", "3"),
            )
        if field.name == "profile":
            return tuple(
                DropdownOption(profile, profile)
                for profile in qwen_image_edit_inpaint_model_names()
            )
        if field.name == "device":
            if self.field("engine").value == "anime":
                return (DropdownOption("CUDA", "cuda"),)
            return (
                DropdownOption("CUDA", "cuda"),
                DropdownOption("CPU", "cpu"),
            )
        if field.name == "prompt_mode":
            return tuple(
                DropdownOption(SAM_PROMPT_MODE_LABELS[mode], mode)
                for mode in SAM_PROMPT_MODES
            )
        if field.name == "mask_source":
            return tuple(
                DropdownOption(SAM_MASK_SOURCE_LABELS[source], source)
                for source in SAM_MASK_SOURCES
            )
        if field.name in {"fill_holes", "largest_component", "invert"}:
            return (
                DropdownOption("Off", "false"),
                DropdownOption("On", "true"),
            )
        if field.name == "output_mode":
            return tuple(
                DropdownOption(SAM_OUTPUT_MODE_LABELS[mode], mode)
                for mode in SAM_OUTPUT_MODES
            )
        return None

    def generation_command(self) -> tuple[list[str], str]:
        input_path = self.field("input").value.strip()
        output_dir = self.field("output_dir").value.strip()
        if not input_path:
            raise ValueError("Input image is required.")
        if not output_dir:
            raise ValueError("Output directory is required.")

        operation = self.field("operation").value
        if operation not in SAM_OPERATIONS:
            raise ValueError(f"Unknown SAM operation: {operation}")

        if operation == "region-plan":
            regions = [item.strip() for item in self.field("regions").value.split(";") if item.strip()]
            if not regions:
                raise ValueError("At least one region is required.")
            command = [
                sys.executable,
                "-m",
                "aigen.cli",
                "characters",
                "region-plan",
                "--image",
                input_path,
                "--output-dir",
                output_dir,
                "--device",
                self.field("device").value,
                "--overwrite",
            ]
            for region in regions:
                command.extend(("--region", region))
            return command, output_dir

        if operation == "qwen-edit":
            mask_source = self.field("mask_source").value
            if mask_source not in SAM_MASK_SOURCES:
                raise ValueError(f"Unknown edit mask source: {mask_source}")
            pack_path = self.field("reference_pack").value.strip()
            instruction = self.field("instruction").value.strip()
            if not pack_path:
                raise ValueError("Reference pack is required for a Qwen masked edit.")
            if not instruction:
                raise ValueError("Edit instruction is required for a Qwen masked edit.")
            if mask_source == "mask":
                mask_path = self.field("mask").value.strip()
                if not mask_path:
                    raise ValueError("Mask path is required for a mask edit.")
                mask_arguments = ("--mask", mask_path)
            else:
                region_plan_path = self.field("region_plan").value.strip()
                region_name = self.field("region").value.strip()
                if not region_plan_path:
                    raise ValueError("Region plan is required for a region-plan edit.")
                if not region_name:
                    raise ValueError("Region name is required for a region-plan edit.")
                mask_arguments = (
                    "--region-plan",
                    region_plan_path,
                    "--region",
                    region_name,
                )
            command = [
                sys.executable,
                "-m",
                "aigen.cli",
                "characters",
                "qwen-edit-refine",
                "--pack",
                pack_path,
                "--image",
                input_path,
                "--instruction",
                instruction,
                "--output-dir",
                output_dir,
                "--model",
                self.field("profile").value,
                "--strength",
                self.field("strength").value.strip(),
                "--seed",
                self.field("seed").value.strip(),
                "--candidates",
                self.field("candidates").value.strip(),
                "--max-sequence-length",
                self.field("max_sequence_length").value.strip(),
                "--overwrite",
            ]
            command.extend(mask_arguments)
            for name, option in (
                ("max_side", "--max-side"),
                ("steps", "--steps"),
                ("true_cfg_scale", "--true-cfg-scale"),
                ("guidance_scale", "--guidance-scale"),
                ("padding_mask_crop", "--padding-mask-crop"),
                ("nunchaku_blocks_on_gpu", "--nunchaku-blocks-on-gpu"),
            ):
                value = self.field(name).value.strip()
                if value:
                    command.extend((option, value))
            return command, output_dir

        command = [
            sys.executable,
            "-m",
            "aigen.cli",
            "sam-segment",
            "--input",
            input_path,
            "--engine",
            self.field("engine").value,
            "--device",
            self.field("device").value,
            "--prompt-mode",
            self.field("prompt_mode").value,
            "--mask-candidate",
            self.field("mask_candidate").value,
            "--threshold",
            self.field("threshold").value.strip(),
            "--grow",
            self.field("grow").value.strip(),
            "--feather",
            self.field("feather").value.strip(),
            "--fill-holes",
            self.field("fill_holes").value,
            "--largest-component",
            self.field("largest_component").value,
            "--invert",
            self.field("invert").value,
            "--output-mode",
            self.field("output_mode").value,
            "--output-dir",
            output_dir,
            "--overwrite",
        ]
        box = self.field("box").value.strip()
        if box:
            command.extend(("--box", box))
        for name, option in (
            ("positive_points", "--positive-points"),
            ("negative_points", "--negative-points"),
        ):
            points = self.field(name).value.strip()
            if points:
                command.extend((option, points))
        return command, output_dir

    def _rebuild_fields(self) -> None:
        names = ["input", "operation"]
        operation = self._fields["operation"].value
        if operation == "region-plan":
            self.fields = [
                self._fields[name]
                for name in ("input", "operation", "device", "regions", "output_dir")
            ]
            return
        if operation == "qwen-edit":
            names.append("mask_source")
            if self._fields["mask_source"].value == "mask":
                names.append("mask")
            else:
                names.extend(("region_plan", "region"))
            names.extend(
                (
                    "reference_pack",
                    "instruction",
                    "profile",
                    "max_side",
                    "steps",
                    "true_cfg_scale",
                    "guidance_scale",
                    "strength",
                    "padding_mask_crop",
                    "seed",
                    "candidates",
                    "max_sequence_length",
                    "nunchaku_blocks_on_gpu",
                )
            )
        else:
            names.extend(("engine", "device"))
            if self._fields["engine"].value != "anime":
                names.extend(("prompt_mode", "mask_candidate"))
                prompt_mode = self._fields["prompt_mode"].value
                if prompt_mode in {"box", "box+points"}:
                    names.append("box")
                if prompt_mode in {"points", "box+points"}:
                    names.extend(("positive_points", "negative_points"))
                if prompt_mode == "auto":
                    names.append("threshold")
        if operation == "segment":
            names.extend(
                (
                    "grow",
                    "feather",
                    "fill_holes",
                    "largest_component",
                    "invert",
                    "output_mode",
                )
            )
        names.append("output_dir")
        self.fields = [self._fields[name] for name in names]
