from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from aigen.image_assets import image_asset_json
from aigen.lora_datasets import ApprovedLoraAnchor, build_lora_anchor_dataset
from aigen.lora_training import (
    LoraLocalTrainConfig,
    build_lora_train_plan,
    materialize_captioned_train_dataset,
)
from aigen.manifest_io import resolve_existing_path, write_json
from aigen.progress import StatusReporter


class LoraSmokeError(RuntimeError):
    pass


def run_lora_smoke(
    *,
    smoke_id: str,
    character_id: str,
    trigger_token: str,
    anchor_specs: list[str],
    identity_caption: str,
    tags: list[str],
    approved_by: str,
    output_dir: Path,
    overwrite: bool,
    trainer_script: Path,
    base_model: Path,
    config: LoraLocalTrainConfig,
    progress: StatusReporter,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise LoraSmokeError(f"Output exists and overwrite=false: {output_dir.as_posix()}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    progress.phase("collect approved anchors")
    anchors = _approved_anchors(
        anchor_specs=anchor_specs,
        identity_caption=identity_caption,
        tags=tags,
        approved_by=approved_by,
    )
    dataset_dir = output_dir / "dataset"
    dataset = build_lora_anchor_dataset(
        dataset_id=smoke_id,
        character_id=character_id,
        trigger_token=trigger_token,
        anchors=anchors,
        output_dir=dataset_dir,
        overwrite=True,
        validation_ratio=0.0,
        write_contact_sheet=True,
        progress=progress,
    )

    progress.step("build training plan")
    plan = build_lora_train_plan(
        dataset_dir,
        output_dir=output_dir,
        trainer_script=trainer_script,
        base_model=base_model,
        config=config,
    )
    write_json(Path(plan["output"]["plan"]), plan)

    progress.step("materialize train dataset")
    materialize_captioned_train_dataset(dataset_dir, Path(plan["dataset"]["train_data_dir"]))

    result = {
        "status": "planned",
        "kind": "lora-smoke-result",
        "id": smoke_id,
        "purpose": "identity_lora_smoke",
        "warning": (
            "Smoke training tests whether approved anchors can move identity out of Kontext; "
            "it is not a production LoRA dataset."
        ),
        "dataset": dataset,
        "train_plan": plan,
        "output": {
            "directory": output_dir.as_posix(),
            "dataset": dataset_dir.as_posix(),
            "train_dataset": plan["dataset"]["train_data_dir"],
            "train_plan": plan["output"]["plan"],
            "result": (output_dir / "smoke_result.json").as_posix(),
            "contact_sheet": dataset["output"]["contact_sheet"],
        },
    }
    progress.step("write smoke result")
    write_json(output_dir / "smoke_result.json", result)
    return result


def _approved_anchors(
    *,
    anchor_specs: list[str],
    identity_caption: str,
    tags: list[str],
    approved_by: str,
) -> list[ApprovedLoraAnchor]:
    if not anchor_specs:
        raise LoraSmokeError("LoRA smoke requires at least one approved anchor")
    anchors = []
    for spec in anchor_specs:
        name, image_path = _parse_anchor_spec(spec)
        anchors.append(
            ApprovedLoraAnchor(
                name=name,
                image_path=image_path,
                caption_parts=[identity_caption, _caption_label(name)],
                tags=tags,
                split="train",
                source_metadata={
                    "approval": {
                        "mode": "human_approved_anchor",
                        "approved_by": approved_by,
                    },
                    "anchor": image_asset_json(image_path),
                    "caption_source": {
                        "kind": "identity_template",
                    },
                },
            )
        )
    return anchors


def _parse_anchor_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise LoraSmokeError(f"Anchor must use NAME=PATH: {value}")
    name, raw_path = value.split("=", 1)
    cleaned_name = name.strip()
    if not cleaned_name:
        raise LoraSmokeError(f"Anchor name is empty: {value}")
    cleaned_path = raw_path.strip()
    if not cleaned_path:
        raise LoraSmokeError(f"Anchor path is empty: {value}")
    return cleaned_name, resolve_existing_path(cleaned_path, Path.cwd())


def _caption_label(name: str) -> str:
    return " ".join(name.replace("_", " ").replace("-", " ").split())
