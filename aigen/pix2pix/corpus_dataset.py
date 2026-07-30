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
from aigen.pix2pix.corpus_config import IroCorpusConfigVersion
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
FLUX_SOURCE_SET_LOSSLESS_1024_PROVENANCE_FORMAT = (
    "aigen.pix2pix.iro-dataset-provenance.v6"
)
FLUX_SOURCE_SET_DATASET_DIRECTORY_PREFIX = "dataset-flux-source-set-"
FLUX_SOURCE_SET_TRAINING_RASTERS = frozenset({"native128", "lossless1024"})


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
    *,
    training_raster: str = "native128",
    pair_filter: Path | None = None,
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
    filter_dataset = None
    if pair_filter is not None:
        filter_dataset = audit_dataset(pair_filter.expanduser().resolve())
        try:
            filter_dataset.root.relative_to(root)
        except ValueError as error:
            raise Pix2PixError(
                "FLUX source-set pair filter must belong to the same corpus"
            ) from error
        selected_by_id = {str(record["id"]): record for record in selected}
        for pair in filter_dataset.pairs:
            selected_record = selected_by_id.get(pair.id)
            if (
                selected_record is None
                or pair.id not in audit.accepted_ids
                or pair.group != selected_record["group"]
                or pair.split != selected_record["split"]
                or pair.target_sha256 != selected_record["target_sha256"]
            ):
                raise Pix2PixError(
                    f"FLUX source-set pair filter is incompatible at {pair.id}"
                )
        filtered_ids = {pair.id for pair in filter_dataset.pairs}
        accepted = tuple(
            record
            for record in accepted
            if str(record["id"]) in filtered_ids
        )
    accepted_ids = {str(record["id"]) for record in accepted}
    accepted_inventory = {
        pair_id: path
        for pair_id, path in inventory.items()
        if pair_id in accepted_ids
    }
    (
        image_size,
        source_raster,
        target_raster,
        dataset_suffix,
    ) = _flux_source_set_training_raster(config, training_raster)
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
    if filter_dataset is not None:
        provenance.update(
            {
                "pair_filter_dataset": filter_dataset.root.as_posix(),
                "pair_filter_dataset_fingerprint": filter_dataset.fingerprint,
            }
        )
    if selection["format"] == IRO_SELECTION_V2_FORMAT:
        coverage = derive_flux_output_coverage(selected, audit)
        provenance.update(
            {
                "format": FLUX_SOURCE_SET_DATASET_V2_PROVENANCE_FORMAT,
                "coverage_report": coverage.report,
                "coverage_report_sha256": coverage.sha256,
            }
        )
    if training_raster == "lossless1024":
        provenance.update(
            {
                "format": FLUX_SOURCE_SET_LOSSLESS_1024_PROVENANCE_FORMAT,
                "training_raster": training_raster,
            }
        )
    dataset_directory_name = f"{FLUX_SOURCE_SET_DATASET_DIRECTORY_PREFIX}{name}"
    dataset_name = f"{config.name}-{source_set.name}"
    if filter_dataset is not None:
        dataset_directory_name = filter_dataset.root.name
        dataset_name = filter_dataset.name
    return _assemble_iro_dataset(
        root,
        selected=accepted,
        inventory=accepted_inventory,
        dataset_dir=root / f"{dataset_directory_name}{dataset_suffix}",
        dataset_name=f"{dataset_name}{dataset_suffix}",
        image_size=image_size,
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
                resample=str(source_raster["resample"]),
                background=tuple(source_raster["background_rgb"]),
            )
            selected_target = corpus_member(
                root,
                str(record["target"]),
                label=f"selected target for {pair_id}",
            )
            if sha256_file(selected_target) != record["target_sha256"]:
                raise Pix2PixError(f"selected target checksum mismatch: {pair_id}")
            _prepare_target(
                selected_target,
                target_path,
                raster=target_raster,
            )
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
    resample: str,
    background: tuple[int, int, int],
) -> None:
    try:
        with Image.open(input_path) as image:
            image.load()
            rgba = image.convert("RGBA")
            matte = Image.new("RGBA", rgba.size, (*background, 255))
            matte.alpha_composite(rgba)
            prepared = matte.convert("RGB")
            if resample == "none":
                if prepared.size != inner_size:
                    raise Pix2PixError(
                        "lossless FLUX source raster does not match its input size: "
                        f"{input_path}"
                    )
            else:
                prepared = prepared.resize(
                    inner_size,
                    Image.Resampling.LANCZOS,
                )
            canvas = Image.new("RGB", (canvas_size, canvas_size), background)
            canvas.paste(prepared, offset)
            canvas.save(output_path, format="PNG", optimize=False)
    except OSError as error:
        raise Pix2PixError(f"cannot prepare FLUX source {input_path}: {error}") from error


def _prepare_target(
    input_path: Path,
    output_path: Path,
    *,
    raster: dict[str, object],
) -> None:
    if not raster["resampled"]:
        shutil.copyfile(input_path, output_path)
        return
    try:
        with Image.open(input_path) as image:
            image.load()
            prepared = image.convert("RGB").resize(
                (int(raster["width"]), int(raster["height"])),
                Image.Resampling.NEAREST,
            )
            prepared.save(output_path, format="PNG", optimize=False)
    except OSError as error:
        raise Pix2PixError(f"cannot prepare pixel-art target {input_path}: {error}") from error


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


def _flux_source_set_training_raster(
    config: IroCorpusConfigVersion,
    training_raster: str,
) -> tuple[int, dict[str, object], dict[str, object], str]:
    if training_raster not in FLUX_SOURCE_SET_TRAINING_RASTERS:
        supported = ", ".join(sorted(FLUX_SOURCE_SET_TRAINING_RASTERS))
        raise Pix2PixError(f"training_raster must be one of: {supported}")
    if training_raster == "native128":
        return (
            config.image_size,
            config.source_raster.model_dump(mode="json"),
            _target_raster_contract(config.image_size),
            "",
        )

    image_size = 1024
    source_width = config.flux.width
    source_height = config.flux.height
    source_raster = {
        "canvas_size": image_size,
        "inner_width": source_width,
        "inner_height": source_height,
        "offset_x": (image_size - source_width) // 2,
        "offset_y": (image_size - source_height) // 2,
        "resample": "none",
        "background_rgb": list(config.source_raster.background_rgb),
    }
    target_raster = {
        "mode": "RGB",
        "width": image_size,
        "height": image_size,
        "source_width": config.image_size,
        "source_height": config.image_size,
        "source": "native renderer frame composited on white",
        "resample": "nearest",
        "scale": image_size // config.image_size,
        "resampled": True,
    }
    return image_size, source_raster, target_raster, "-lossless1024-v1"
