from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from aigen.manifest_io import read_json, sha256_file, write_json
from aigen.pix2pix.corpus_io import (
    corpus_member,
    require_exact_keys,
    write_json_records,
)
from aigen.pix2pix.dataset import DATASET_FORMAT, audit_dataset
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.flux_source_audit import (
    derive_flux_output_coverage,
    load_flux_output_audit,
)
from aigen.pix2pix.flux_source_corpus import (
    flux_source_result_path,
    load_flux_source_inventory,
)
from aigen.pix2pix.flux_source_set_corpus import (
    flux_source_set_layout,
    flux_source_set_result_path,
    load_flux_source_set_inventory,
)
from aigen.pix2pix.iro_corpus import (
    IRO_SELECTION_V2_FORMAT,
    load_iro_selection,
)
from aigen.pix2pix.qwen_source_corpus import (
    QWEN_SOURCE_DIRECTORY,
    load_frozen_qwen_source_config,
    load_qwen_source_inventory,
    qwen_source_result_path,
)


CORPUS_DATASET_PROVENANCE_FORMAT = "aigen.pix2pix.iro-dataset-provenance.v2"
QWEN_CORPUS_DATASET_PROVENANCE_FORMAT = (
    "aigen.pix2pix.iro-dataset-provenance.v3"
)
QWEN_CORPUS_DATASET_DIRECTORY = "dataset-qwen-2511-lightning-v1"
FLUX_SOURCE_SET_DATASET_PROVENANCE_FORMAT = (
    "aigen.pix2pix.iro-dataset-provenance.v4"
)
FLUX_SOURCE_SET_DATASET_V2_PROVENANCE_FORMAT = (
    "aigen.pix2pix.iro-dataset-provenance.v5"
)
FLUX_SOURCE_SET_DATASET_DIRECTORY_PREFIX = "dataset-flux-source-set-"


def prepare_iro_dataset(root: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    config, selected, selection = load_iro_selection(root)
    inventory = load_flux_source_inventory(root)
    flux_result_path = flux_source_result_path(root)
    flux_result_sha256 = sha256_file(flux_result_path)
    source_raster = config.source_raster.model_dump(mode="json")
    target_raster = _target_raster_contract(config.image_size)
    return _assemble_iro_dataset(
        root,
        selected=selected,
        inventory=inventory,
        dataset_dir=root / "dataset",
        dataset_name=config.name,
        image_size=config.image_size,
        source_raster=source_raster,
        target_raster=target_raster,
        provenance_base={
            "format": CORPUS_DATASET_PROVENANCE_FORMAT,
            "config_fingerprint": selection["config_fingerprint"],
            "selection_sha256": selection["selected_sha256"],
            "flux_result_sha256": flux_result_sha256,
            "pair_count": len(selected),
            "source_raster": source_raster,
            "target_raster": target_raster,
        },
        kind="iRO-pix2pix-dataset",
    )


def prepare_iro_qwen_dataset(root: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    config, selected, selection = load_iro_selection(root)
    qwen_config = load_frozen_qwen_source_config(root)
    inventory = load_qwen_source_inventory(root)
    result_sha256 = sha256_file(qwen_source_result_path(root))
    source_raster = qwen_config.source_raster.model_dump(mode="json")
    target_raster = _target_raster_contract(config.image_size)
    return _assemble_iro_dataset(
        root,
        selected=selected,
        inventory=inventory,
        dataset_dir=root / QWEN_CORPUS_DATASET_DIRECTORY,
        dataset_name=qwen_config.name,
        image_size=config.image_size,
        source_raster=source_raster,
        target_raster=target_raster,
        provenance_base={
            "format": QWEN_CORPUS_DATASET_PROVENANCE_FORMAT,
            "config_fingerprint": selection["config_fingerprint"],
            "selection_sha256": selection["selected_sha256"],
            "source_backend": QWEN_SOURCE_DIRECTORY,
            "source_result_sha256": result_sha256,
            "pair_count": len(selected),
            "source_raster": source_raster,
            "target_raster": target_raster,
        },
        kind="iRO-Qwen-pix2pix-dataset",
    )


def prepare_iro_flux_source_set_dataset(
    root: Path,
    name: str,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    config, selected, selection = load_iro_selection(root)
    inventory, source_set, _, source_plan_sha256 = (
        load_flux_source_set_inventory(root, name)
    )
    result_path = flux_source_set_result_path(root, name)
    result_sha256 = sha256_file(result_path)
    audit = load_flux_output_audit(
        root,
        name,
        inventory=inventory,
        source_set=source_set,
        selected=selected,
        source_plan_sha256=source_plan_sha256,
        source_result_sha256=result_sha256,
    )
    accepted = tuple(
        record
        for record in selected
        if str(record["id"]) in audit.accepted_ids
    )
    accepted_inventory = {
        pair_id: path
        for pair_id, path in inventory.items()
        if pair_id in audit.accepted_ids
    }
    source_raster = config.source_raster.model_dump(mode="json")
    target_raster = _target_raster_contract(config.image_size)
    layout = flux_source_set_layout(name)
    provenance: dict[str, Any] = {
        "format": FLUX_SOURCE_SET_DATASET_PROVENANCE_FORMAT,
        "config_fingerprint": selection["config_fingerprint"],
        "selection_sha256": selection["selected_sha256"],
        "source_backend": layout.directory,
        "source_plan_sha256": source_plan_sha256,
        "source_result_sha256": result_sha256,
        "source_set_fingerprint": source_set.fingerprint,
        "prompt_guide_sha256": source_set.prompt_guide_sha256,
        "output_audit_sha256": audit.manifest_sha256,
        "output_audit_records_sha256": audit.records_sha256,
        "source_pair_count": len(selected),
        "pair_count": len(accepted),
        "rejected_pair_count": len(selected) - len(accepted),
        "source_raster": source_raster,
        "target_raster": target_raster,
    }
    if selection["format"] == IRO_SELECTION_V2_FORMAT:
        coverage = derive_flux_output_coverage(selected, audit)
        provenance.update(
            {
                "format": FLUX_SOURCE_SET_DATASET_V2_PROVENANCE_FORMAT,
                "coverage_report": coverage.report,
                "coverage_report_sha256": coverage.sha256,
            }
        )
    return _assemble_iro_dataset(
        root,
        selected=accepted,
        inventory=accepted_inventory,
        dataset_dir=root / f"{FLUX_SOURCE_SET_DATASET_DIRECTORY_PREFIX}{name}",
        dataset_name=f"{config.name}-{source_set.name}",
        image_size=config.image_size,
        source_raster=source_raster,
        target_raster=target_raster,
        provenance_base=provenance,
        kind="iRO-reviewed-FLUX-pix2pix-dataset",
    )


def _assemble_iro_dataset(
    root: Path,
    *,
    selected: tuple[dict[str, Any], ...],
    inventory: dict[str, Path],
    dataset_dir: Path,
    dataset_name: str,
    image_size: int,
    source_raster: dict[str, Any],
    target_raster: dict[str, object],
    provenance_base: dict[str, Any],
    kind: str,
) -> dict[str, object]:
    if dataset_dir.exists():
        dataset = audit_dataset(dataset_dir)
        _verify_dataset_provenance(
            dataset_dir,
            expected={
                **provenance_base,
                "dataset_fingerprint": dataset.fingerprint,
            },
        )
        return {
            **dataset.to_json(),
            "kind": kind,
            "reused": True,
        }

    with TemporaryDirectory(
        dir=root,
        prefix=f".{dataset_dir.name}.",
        suffix=".incomplete",
    ) as temporary:
        staging = Path(temporary)
        source_dir = staging / "source"
        target_dir = staging / "target"
        source_dir.mkdir()
        target_dir.mkdir()
        pair_records = []
        for record in selected:
            pair_id = str(record["id"])
            source_path = source_dir / f"{pair_id}.png"
            target_path = target_dir / f"{pair_id}.png"
            _prepare_source(
                inventory[pair_id],
                source_path,
                canvas_size=int(source_raster["canvas_size"]),
                inner_size=(
                    int(source_raster["inner_width"]),
                    int(source_raster["inner_height"]),
                ),
                offset=(
                    int(source_raster["offset_x"]),
                    int(source_raster["offset_y"]),
                ),
                background=tuple(source_raster["background_rgb"]),
            )
            selected_target = corpus_member(
                root,
                str(record["target"]),
                label=f"selected target for {pair_id}",
            )
            shutil.copyfile(selected_target, target_path)
            if sha256_file(target_path) != record["target_sha256"]:
                raise Pix2PixError(f"copied target checksum mismatch: {pair_id}")
            pair_records.append(
                {
                    "id": pair_id,
                    "group": str(record["group"]),
                    "split": str(record["split"]),
                    "source": f"source/{pair_id}.png",
                    "target": f"target/{pair_id}.png",
                }
            )

        write_json(
            staging / "dataset.json",
            {
                "format": DATASET_FORMAT,
                "name": dataset_name,
                "image_size": image_size,
                "pairs": "pairs.jsonl",
            },
        )
        write_json_records(staging / "pairs.jsonl", pair_records)
        dataset = audit_dataset(staging)
        write_json(
            staging / "provenance.json",
            {
                **provenance_base,
                "dataset_fingerprint": dataset.fingerprint,
            },
        )
        os.rename(staging, dataset_dir)

    dataset = audit_dataset(dataset_dir)
    _verify_dataset_provenance(
        dataset_dir,
        expected={
            **provenance_base,
            "dataset_fingerprint": dataset.fingerprint,
        },
    )
    return {
        **dataset.to_json(),
        "kind": kind,
        "reused": False,
    }


def _prepare_source(
    input_path: Path,
    output_path: Path,
    *,
    canvas_size: int,
    inner_size: tuple[int, int],
    offset: tuple[int, int],
    background: tuple[int, int, int],
) -> None:
    try:
        with Image.open(input_path) as image:
            image.load()
            rgba = image.convert("RGBA")
            matte = Image.new("RGBA", rgba.size, (*background, 255))
            matte.alpha_composite(rgba)
            resized = matte.convert("RGB").resize(
                inner_size,
                Image.Resampling.LANCZOS,
            )
            canvas = Image.new("RGB", (canvas_size, canvas_size), background)
            canvas.paste(resized, offset)
            canvas.save(output_path, format="PNG", optimize=False)
    except OSError as error:
        raise Pix2PixError(f"cannot prepare FLUX source {input_path}: {error}") from error


def _verify_dataset_provenance(
    dataset_dir: Path,
    *,
    expected: dict[str, Any],
) -> None:
    provenance = read_json(
        dataset_dir / "provenance.json",
        label="iRO dataset provenance",
    )
    require_exact_keys(
        provenance,
        set(expected),
        "iRO dataset provenance",
    )
    for key, value in expected.items():
        if provenance[key] != value:
            raise Pix2PixError(f"iRO dataset provenance mismatch: {key}")


def _target_raster_contract(image_size: int) -> dict[str, object]:
    return {
        "mode": "RGB",
        "width": image_size,
        "height": image_size,
        "source": "native renderer frame composited on white",
        "resampled": False,
    }
