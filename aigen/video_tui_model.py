from __future__ import annotations

import sys
from pathlib import Path

from aigen.generation.hunyuanvideo15 import HUNYUANVIDEO15_STEPS
from aigen.generation.ltx23_keyframes import (
    LTX23_DEFAULT_CONDITIONING_STRENGTH,
    LTX23_DEFAULT_FPS,
    LTX23_DEFAULT_MODEL,
    LTX23_MODEL_TYPES,
    LTX23_SOLVERS,
)
from aigen.image_tui_model import DropdownOption, FormField


LTX23_BACKEND = "ltx23-keyframes"
HUNYUANVIDEO15_BACKEND = "hunyuanvideo15-i2v"
VIDEO_BACKENDS = (LTX23_BACKEND, HUNYUANVIDEO15_BACKEND)
VIDEO_BACKEND_LABELS = {
    LTX23_BACKEND: "LTX-2.3 keyframes",
    HUNYUANVIDEO15_BACKEND: "HunyuanVideo-1.5 I2V",
}


class VideoForm:
    def __init__(self) -> None:
        self._fields = {
            "backend": FormField("backend", "Backend", LTX23_BACKEND),
            "prompt": FormField("prompt", "Prompt", ""),
            "resolution": FormField("resolution", "Resolution", "640x640"),
            "frames": FormField("frames", "Frames", "121"),
            "fps": FormField("fps", "FPS", str(LTX23_DEFAULT_FPS)),
            "steps": FormField("steps", "Steps", "15"),
            "solver": FormField("solver", "Solver", "res2s"),
            "conditioning_strength": FormField(
                "conditioning_strength",
                "Conditioning strength",
                str(LTX23_DEFAULT_CONDITIONING_STRENGTH),
            ),
            "model": FormField("model", "Model", LTX23_DEFAULT_MODEL),
            "overlap_group_offloading": FormField(
                "overlap_group_offloading",
                "Overlap group offloading",
                "false",
            ),
            "output_dir": FormField(
                "output_dir",
                "Output directory",
                "runs/video-tui",
            ),
            "filename": FormField("filename", "Filename", "video.mp4"),
            "seed": FormField("seed", "Seed", "42"),
        }
        self._slots: list[tuple[str, int]] = [
            ("keyframe", 2),
            ("keyframe", 3),
            ("seed", 1),
        ]
        self._slot_values: dict[tuple[str, int], dict[str, str]] = {
            ("keyframe", 2): {"keyframe": "", "frame": "0"},
            ("keyframe", 3): {"keyframe": "", "frame": "120"},
            ("seed", 1): {"seed": "42"},
        }
        self.next_slot_id = 4
        self.fields: list[FormField] = []
        self.slot_move_states: dict[int, tuple[bool, bool]] = {}
        self._rebuild_fields()

    def field(self, name: str) -> FormField:
        return self._fields[name]

    def set_value(self, field: FormField, value: str) -> None:
        field.value = value
        if field.slot_kind is not None and field.slot_id is not None:
            self._slot_values[(field.slot_kind, field.slot_id)][field.name] = value
        elif field.name == "seed" and ("seed", 1) in self._slot_values:
            self._slot_values[("seed", 1)]["seed"] = value
        if field.name == "backend":
            self._set_backend_defaults(value)
            self._rebuild_fields()

    def dropdown_options(self, field: FormField) -> tuple[DropdownOption, ...] | None:
        if field.name == "backend":
            return tuple(
                DropdownOption(VIDEO_BACKEND_LABELS[backend], backend)
                for backend in VIDEO_BACKENDS
            )
        backend = self.field("backend").value
        if field.name == "model" and backend == LTX23_BACKEND:
            return tuple(DropdownOption(model, model) for model in LTX23_MODEL_TYPES)
        if field.name == "solver" and backend == LTX23_BACKEND:
            return tuple(DropdownOption(solver, solver) for solver in sorted(LTX23_SOLVERS))
        if field.name == "steps" and backend == HUNYUANVIDEO15_BACKEND:
            return tuple(
                DropdownOption(str(steps), str(steps))
                for steps in sorted(HUNYUANVIDEO15_STEPS)
            )
        if field.name == "overlap_group_offloading":
            return (
                DropdownOption("Off", "false"),
                DropdownOption("On", "true"),
            )
        return None

    def add_slot(self, slot_kind: str) -> int:
        if slot_kind not in {"keyframe", "seed"}:
            raise ValueError(f"Unknown video slot kind: {slot_kind}")
        if slot_kind == "keyframe" and self.field("backend").value != LTX23_BACKEND:
            raise ValueError("HunyuanVideo-1.5 accepts one input image")
        slot_id = self.next_slot_id
        self.next_slot_id += 1
        if slot_kind == "keyframe":
            frames = self.field("frames").value.strip()
            frame = str(max(0, int(frames) - 1)) if frames.isdigit() else "0"
            self._slot_values[(slot_kind, slot_id)] = {
                "keyframe": "",
                "frame": frame,
            }
        else:
            used = {
                self._slot_values[(kind, slot_id)]["seed"]
                for kind, slot_id in self._slots
                if kind == "seed"
            }
            seed = 0
            while str(seed) in used:
                seed += 1
            self._slot_values[(slot_kind, slot_id)] = {"seed": str(seed)}
        self._slots.append((slot_kind, slot_id))
        self._rebuild_fields()
        return slot_id

    def remove_slot(self, slot_id: int) -> None:
        slot = next((slot for slot in self._slots if slot[1] == slot_id), None)
        if slot is None:
            return
        self._slots.remove(slot)
        self._slot_values.pop(slot, None)
        self._rebuild_fields()

    def move_slot(self, slot_id: int, direction: int) -> None:
        slot = next(slot for slot in self._slots if slot[1] == slot_id)
        same_kind = [index for index, candidate in enumerate(self._slots) if candidate[0] == slot[0]]
        source = same_kind.index(self._slots.index(slot))
        target = source + direction
        if not 0 <= target < len(same_kind):
            return
        first_index, second_index = same_kind[source], same_kind[target]
        self._slots[first_index], self._slots[second_index] = (
            self._slots[second_index],
            self._slots[first_index],
        )
        self._rebuild_fields()

    def generation_command(self) -> tuple[list[str], str]:
        backend = self.field("backend").value
        prompt = self.field("prompt").value.strip()
        output_dir = self.field("output_dir").value.strip()
        filename = self.field("filename").value.strip()
        if not prompt:
            raise ValueError("Prompt is required.")
        if not output_dir:
            raise ValueError("Output directory is required.")
        if not filename:
            raise ValueError("Filename is required.")
        output_path = Path(filename)
        if output_path.name != filename:
            raise ValueError("Filename must not contain a directory.")
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".mp4")
        if output_path.suffix.casefold() != ".mp4":
            raise ValueError("Video filename must use the .mp4 extension.")
        output = (Path(output_dir) / output_path).as_posix()
        command = [sys.executable, "-m", "aigen.cli", backend, "--prompt", prompt, "--output", output]
        if backend == LTX23_BACKEND:
            command.extend(
                (
                    "--resolution",
                    self.field("resolution").value.strip(),
                    "--frames",
                    self.field("frames").value.strip(),
                    "--fps",
                    self.field("fps").value.strip(),
                    "--steps",
                    self.field("steps").value.strip(),
                    "--solver",
                    self.field("solver").value.strip(),
                    "--conditioning-strength",
                    self.field("conditioning_strength").value.strip(),
                    "--model",
                    self.field("model").value.strip(),
                )
            )
            seeds = [
                field.value.strip()
                for field in self.fields
                if field.slot_kind == "seed" and field.value.strip()
            ]
            if not seeds:
                raise ValueError("At least one seed is required.")
            for seed in seeds:
                command.extend(("--seed", seed))
            keyframes = [
                (field, next(item for item in self.fields if item.name == "frame" and item.slot_id == field.slot_id))
                for field in self.fields
                if field.slot_kind == "keyframe" and field.name == "keyframe"
            ]
            if len(keyframes) < 2:
                raise ValueError("LTX-2.3 requires at least two keyframes.")
            for image, frame in keyframes:
                if not image.value.strip():
                    raise ValueError(f"{image.label} is required.")
                command.extend(("--keyframe", image.value.strip(), frame.value.strip()))
        else:
            image = next(
                (field for field in self.fields if field.slot_kind == "image"),
                None,
            )
            if image is None or not image.value.strip():
                raise ValueError("Input image is required.")
            command.extend(
                (
                    "--image",
                    image.value.strip(),
                    "--steps",
                    self.field("steps").value.strip(),
                    "--frames",
                    self.field("frames").value.strip(),
                    "--seed",
                    self.field("seed").value.strip(),
                )
            )
            if self.field("overlap_group_offloading").value == "true":
                command.append("--overlap-group-offloading")
        return command, output_dir

    def _set_backend_defaults(self, backend: str) -> None:
        if backend == LTX23_BACKEND:
            self._fields.update(
                {
                    "resolution": FormField("resolution", "Resolution", "640x640"),
                    "frames": FormField("frames", "Frames", "121"),
                    "fps": FormField("fps", "FPS", str(LTX23_DEFAULT_FPS)),
                    "steps": FormField("steps", "Steps", "15"),
                    "solver": FormField("solver", "Solver", "res2s"),
                    "conditioning_strength": FormField(
                        "conditioning_strength",
                        "Conditioning strength",
                        str(LTX23_DEFAULT_CONDITIONING_STRENGTH),
                    ),
                    "model": FormField("model", "Model", LTX23_DEFAULT_MODEL),
                }
            )
        elif backend == HUNYUANVIDEO15_BACKEND:
            self._fields.update(
                {
                    "frames": FormField("frames", "Frames", "49"),
                    "steps": FormField("steps", "Steps", "8"),
                    "seed": FormField("seed", "Seed", "42"),
                    "overlap_group_offloading": FormField(
                        "overlap_group_offloading",
                        "Overlap group offloading",
                        "false",
                    ),
                }
            )
        else:
            raise ValueError(f"Unknown video backend: {backend}")

    def _rebuild_fields(self) -> None:
        backend = self.field("backend").value
        fixed_names = ["backend", "prompt"]
        if backend == LTX23_BACKEND:
            fixed_names.extend(
                (
                    "resolution",
                    "frames",
                    "fps",
                    "steps",
                    "solver",
                    "conditioning_strength",
                    "model",
                )
            )
            slot_kinds = {"keyframe", "seed"}
        else:
            fixed_names.extend(("frames", "steps", "overlap_group_offloading"))
            slot_kinds = {"image"}
        fixed_names.extend(("output_dir", "filename"))
        fields = [self._fields[name] for name in fixed_names]
        visible_slots = [slot for slot in self._slots if slot[0] in slot_kinds]
        if backend == HUNYUANVIDEO15_BACKEND and not any(
            slot[0] == "image" for slot in self._slots
        ):
            slot_id = self.next_slot_id
            self.next_slot_id += 1
            self._slots.append(("image", slot_id))
            self._slot_values[("image", slot_id)] = {"image": ""}
            visible_slots = [("image", slot_id)]
        for kind, slot_id in visible_slots:
            values = self._slot_values[(kind, slot_id)]
            if kind == "keyframe":
                fields.extend(
                    (
                        FormField("keyframe", "", values["keyframe"], kind, slot_id),
                        FormField("frame", "", values["frame"], kind, slot_id),
                    )
                )
            elif kind == "seed":
                fields.append(FormField("seed", "", values["seed"], kind, slot_id))
            else:
                fields.append(FormField("image", "", values["image"], kind, slot_id))
        if backend == HUNYUANVIDEO15_BACKEND:
            fields.append(self._fields["seed"])
        self.fields = fields
        self._renumber_slots()

    def _renumber_slots(self) -> None:
        counts: dict[str, int] = {}
        slot_ids: dict[str, list[int]] = {}
        for field in self.fields:
            if field.slot_kind is None or field.slot_id is None:
                continue
            if field.slot_id not in slot_ids.setdefault(field.slot_kind, []):
                counts[field.slot_kind] = counts.get(field.slot_kind, 0) + 1
                slot_ids[field.slot_kind].append(field.slot_id)
            number = slot_ids[field.slot_kind].index(field.slot_id) + 1
            if field.name == "frame":
                field.label = f"Keyframe {number} frame"
            elif field.slot_kind == "image":
                field.label = f"Input image {number}"
            elif field.slot_kind == "seed":
                field.label = f"Seed {number}"
            else:
                field.label = f"Keyframe {number}"
        self.slot_move_states = {
            slot_id: (index > 0, index + 1 < len(ids))
            for ids in slot_ids.values()
            for index, slot_id in enumerate(ids)
        }
