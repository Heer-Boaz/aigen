from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from aigen.character_reference_pack import build_character_reference_pack
from aigen.image_edit_commands import (
    IMAGE_EDIT_ASPECT_RATIOS,
    IMAGE_EDIT_BACKEND_LORA_ARCHITECTURES,
    IMAGE_EDIT_BACKEND_SETTINGS,
    IMAGE_EDIT_BACKENDS,
)
from aigen.lora_weights import inspect_lora_weights
from aigen.runtime_profiles import (
    PROJECT_ROOT,
    display_project_path,
    resolve_project_path,
)


LORA_ROOT = PROJECT_ROOT / "loras"
REFERENCE_PACK_ROOT = PROJECT_ROOT / "assets" / "reference-packs"


@dataclass
class FormField:
    name: str
    label: str
    value: str
    slot_kind: str | None = None
    slot_id: int | None = None


@dataclass(frozen=True)
class DropdownOption:
    label: str
    value: str


class ImageEditForm:
    def __init__(self) -> None:
        default_model = IMAGE_EDIT_BACKENDS[0]
        defaults = IMAGE_EDIT_BACKEND_SETTINGS[default_model]
        self.fields = [
            FormField("prompt", "Prompt", ""),
            FormField("model", "Model", default_model),
            FormField("sampler", "Sampler", defaults.sampler),
            FormField("scheduler", "Scheduler", defaults.scheduler),
            FormField("aspect_ratio", "Aspect ratio", ""),
            FormField("width", "Width", ""),
            FormField("height", "Height", ""),
            FormField("steps", "Steps", str(defaults.steps)),
            FormField(
                "guidance",
                "Guidance",
                "" if defaults.guidance is None else str(defaults.guidance),
            ),
            FormField("output_dir", "Output directory", "runs/image-tui"),
            FormField("seed", "Seed 1", "0", "seed", 1),
            FormField("image", "Image 1", "", "image", 2),
        ]
        self.next_slot_id = 3
        self.slot_move_states: dict[int, tuple[bool, bool]] = {}
        self._renumber_slots()

    def add_slot(self, slot_kind: str) -> int:
        slot_id = self.next_slot_id
        self.next_slot_id += 1
        if slot_kind == "lora":
            insert_at = next(
                index for index, field in enumerate(self.fields) if field.name == "output_dir"
            )
            new_fields = [
                FormField("lora", "", "", "lora", slot_id),
                FormField("lora_weight", "", "1.0", "lora", slot_id),
            ]
        elif slot_kind == "seed":
            insert_at = next(
                (
                    index
                    for index, field in enumerate(self.fields)
                    if field.slot_kind in ("image", "reference_pack")
                ),
                len(self.fields),
            )
            used_seeds = {
                field.value.strip()
                for field in self.fields
                if field.slot_kind == "seed"
            }
            seed = 0
            while str(seed) in used_seeds:
                seed += 1
            new_fields = [FormField("seed", "", str(seed), "seed", slot_id)]
        elif slot_kind == "image":
            insert_at = next(
                (
                    index
                    for index, field in enumerate(self.fields)
                    if field.slot_kind == "reference_pack"
                ),
                len(self.fields),
            )
            new_fields = [FormField("image", "", "", "image", slot_id)]
        elif slot_kind == "reference_pack":
            insert_at = len(self.fields)
            new_fields = [
                FormField("reference_pack", "", "", "reference_pack", slot_id)
            ]
        else:
            raise ValueError(f"Unknown slot kind: {slot_kind}")
        self.fields[insert_at:insert_at] = new_fields
        self._renumber_slots()
        return slot_id

    def remove_slot(self, slot_id: int) -> None:
        self.fields = [field for field in self.fields if field.slot_id != slot_id]
        self._renumber_slots()

    def move_slot(self, slot_id: int, direction: int) -> None:
        selected = next(field for field in self.fields if field.slot_id == slot_id)
        slot_kind = selected.slot_kind
        assert slot_kind is not None
        slot_ids = list(
            dict.fromkeys(
                field.slot_id for field in self.fields if field.slot_kind == slot_kind
            )
        )
        source = slot_ids.index(slot_id)
        target = source + direction
        if not 0 <= target < len(slot_ids):
            return
        slot_ids[source], slot_ids[target] = slot_ids[target], slot_ids[source]
        positions = [
            index for index, field in enumerate(self.fields) if field.slot_kind == slot_kind
        ]
        ordered_fields = [
            field
            for ordered_slot_id in slot_ids
            for field in self.fields
            if field.slot_kind == slot_kind and field.slot_id == ordered_slot_id
        ]
        for position, field in zip(positions, ordered_fields, strict=True):
            self.fields[position] = field
        self._renumber_slots()

    def field(self, name: str) -> FormField:
        return next(field for field in self.fields if field.name == name)

    def set_value(self, field: FormField, value: str) -> None:
        field.value = value
        if field.name == "model":
            defaults = IMAGE_EDIT_BACKEND_SETTINGS[value]
            self.field("sampler").value = defaults.sampler
            self.field("scheduler").value = defaults.scheduler
            self.field("steps").value = str(defaults.steps)
            self.field("guidance").value = (
                "" if defaults.guidance is None else str(defaults.guidance)
            )

    def dropdown_options(self, field: FormField) -> tuple[DropdownOption, ...] | None:
        if field.name == "model":
            return tuple(DropdownOption(backend, backend) for backend in IMAGE_EDIT_BACKENDS)
        if field.name == "sampler":
            return tuple(
                DropdownOption(sampler, sampler)
                for sampler in IMAGE_EDIT_BACKEND_SETTINGS[
                    self.field("model").value
                ].samplers
            )
        if field.name == "scheduler":
            return tuple(
                DropdownOption(scheduler, scheduler)
                for scheduler in IMAGE_EDIT_BACKEND_SETTINGS[
                    self.field("model").value
                ].schedulers
            )
        if field.name == "aspect_ratio":
            return (
                DropdownOption("Auto", ""),
                *(
                    DropdownOption(f"{width}:{height}", f"{width}:{height}")
                    for width, height in IMAGE_EDIT_ASPECT_RATIOS
                ),
            )
        if field.name == "lora":
            architecture = IMAGE_EDIT_BACKEND_LORA_ARCHITECTURES.get(
                self.field("model").value
            )
            weights = sorted(
                LORA_ROOT.glob("*.safetensors"), key=lambda path: path.name.casefold()
            )
            return (
                DropdownOption("None", ""),
                *(
                    DropdownOption(path.name, path.relative_to(PROJECT_ROOT).as_posix())
                    for path in weights
                    if architecture is not None
                    and inspect_lora_weights(path).architecture == architecture
                ),
            )
        if field.name == "reference_pack":
            packs = sorted(
                REFERENCE_PACK_ROOT.glob("*.json"), key=lambda path: path.stem.casefold()
            )
            return (
                DropdownOption("None", ""),
                *(
                    DropdownOption(path.stem, path.relative_to(PROJECT_ROOT).as_posix())
                    for path in packs
                ),
            )
        return None

    def save_reference_pack(self, pack_id: str) -> Path:
        images = [
            resolve_project_path(field.value)
            for field in self.fields
            if field.slot_kind == "image" and field.value.strip()
        ]
        references = {
            f"reference_{index}": image for index, image in enumerate(images, start=1)
        }
        output = REFERENCE_PACK_ROOT / f"{pack_id}.json"
        build_character_reference_pack(
            character_id=pack_id,
            references=references,
            output=output,
            overwrite=False,
        )
        return output

    def load(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["version"] not in (1, 2, 3):
            raise ValueError(f"Unsupported image TUI configuration version in {path}.")
        records = list(payload["fields"])
        if payload["version"] < 3:
            model_index = next(
                index for index, record in enumerate(records) if record["name"] == "model"
            )
            model = records[model_index]["value"]
            if model not in IMAGE_EDIT_BACKEND_SETTINGS:
                raise ValueError(f"Saved TUI model is unknown in {path}: {model}")
            if payload["version"] == 1:
                records.insert(
                    model_index + 1,
                    {
                        "name": "sampler",
                        "value": IMAGE_EDIT_BACKEND_SETTINGS[model].sampler,
                    },
                )
            records.insert(
                model_index + 2,
                {
                    "name": "scheduler",
                    "value": IMAGE_EDIT_BACKEND_SETTINGS[model].scheduler,
                },
            )
        fixed_labels = {
            field.name: field.label for field in self.fields if field.slot_kind is None
        }
        fields = [
            FormField(
                record["name"],
                fixed_labels.get(record["name"], ""),
                record["value"],
                record.get("slot_kind"),
                record.get("slot_id"),
            )
            for record in records
        ]
        expected_fixed = [field.name for field in self.fields if field.slot_kind is None]
        loaded_fixed = [field.name for field in fields if field.slot_kind is None]
        if loaded_fixed != expected_fixed:
            raise ValueError(f"Saved TUI fields do not match this version in {path}.")
        loaded_model = next(field.value for field in fields if field.name == "model")
        if loaded_model not in IMAGE_EDIT_BACKEND_SETTINGS:
            raise ValueError(f"Saved TUI model is unknown in {path}: {loaded_model}")
        slot_kinds = {"seed", "image", "reference_pack", "lora"}
        if any(
            field.slot_kind not in slot_kinds or not isinstance(field.slot_id, int)
            for field in fields
            if field.slot_kind is not None
        ):
            raise ValueError(f"Saved TUI slots are invalid in {path}.")
        self.fields = fields
        self.next_slot_id = max(
            (field.slot_id for field in fields if field.slot_id is not None), default=0
        ) + 1
        self._renumber_slots()

    def save(self, path: Path) -> None:
        records = []
        for field in self.fields:
            record: dict[str, int | str] = {"name": field.name, "value": field.value}
            if field.slot_kind is not None:
                assert field.slot_id is not None
                record["slot_kind"] = field.slot_kind
                record["slot_id"] = field.slot_id
            records.append(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        temporary_path.write_text(
            json.dumps({"version": 3, "fields": records}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)

    def generation_command(self) -> tuple[list[str], str]:
        prompt = self.field("prompt").value.strip()
        model = self.field("model").value.strip()
        output_dir = self.field("output_dir").value.strip()
        if not prompt:
            raise ValueError("Prompt is required.")
        if not model:
            raise ValueError("Model is required.")
        if not output_dir:
            raise ValueError("Output directory is required.")
        images = [
            field.value.strip()
            for field in self.fields
            if field.slot_kind == "image" and field.value.strip()
        ]
        packs = [
            field.value.strip()
            for field in self.fields
            if field.slot_kind == "reference_pack" and field.value.strip()
        ]
        if not images and not packs:
            raise ValueError("At least one image or reference pack is required.")

        command = [
            sys.executable,
            "-m",
            "aigen.cli",
            "image-edit",
            "--backend",
            model,
            "--prompt",
            prompt,
            "--output-dir",
            output_dir,
            "--overwrite",
        ]
        for field_name, option in (
            ("aspect_ratio", "--aspect-ratio"),
            ("width", "--width"),
            ("height", "--height"),
            ("steps", "--steps"),
            ("guidance", "--guidance"),
            ("sampler", "--sampler"),
            ("scheduler", "--scheduler"),
        ):
            value = self.field(field_name).value.strip()
            if value:
                command.extend((option, value))
        for field in self.fields:
            value = field.value.strip()
            if field.slot_kind == "seed" and value:
                command.extend(("--seed", value))
            elif field.slot_kind == "image" and value:
                command.extend(("--image", value))
            elif field.slot_kind == "reference_pack" and value:
                command.extend(("--reference-pack", value))
            elif field.name == "lora" and value:
                weight = next(
                    item.value.strip()
                    for item in self.fields
                    if item.name == "lora_weight" and item.slot_id == field.slot_id
                )
                command.extend(("--lora", str(Path(value).expanduser())))
                command.extend(("--lora-weight", weight or "1.0"))
        return command, output_dir

    def _renumber_slots(self) -> None:
        counts = {"seed": 0, "lora": 0, "image": 0, "reference_pack": 0}
        slot_numbers: dict[tuple[str, int], int] = {}
        slot_ids_by_kind: dict[str, list[int]] = {kind: [] for kind in counts}
        for field in self.fields:
            if field.slot_kind is None:
                continue
            assert field.slot_id is not None
            key = (field.slot_kind, field.slot_id)
            if key not in slot_numbers:
                counts[field.slot_kind] += 1
                slot_numbers[key] = counts[field.slot_kind]
                slot_ids_by_kind[field.slot_kind].append(field.slot_id)
            number = slot_numbers[key]
            if field.name == "lora_weight":
                field.label = f"LoRA {number} weight"
            elif field.slot_kind == "reference_pack":
                field.label = f"Reference pack {number}"
            else:
                field.label = f"{field.slot_kind.replace('_', ' ').title()} {number}"
        self.slot_move_states = {
            slot_id: (index > 0, index + 1 < len(slot_ids))
            for slot_ids in slot_ids_by_kind.values()
            for index, slot_id in enumerate(slot_ids)
        }
