from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aigen.generation.flux2_klein_artifacts import (
    FLUX2_KLEIN_IMPLEMENTATION_REVISION,
    flux2_klein_model_provenance,
    flux2_klein_runtime_provenance,
)
from aigen.manifest_io import atomic_write_json, read_json, sha256_file
from aigen.model_artifacts import validate_model_artifact_provenance
from aigen.pix2pix.corpus_config import SAFE_NAME_PATTERN
from aigen.pix2pix.corpus_io import require_exact_keys
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.flux_source_engine import (
    FluxSourceCaseSpec,
    FluxSourceLayout,
    flux_source_corpus_result_path,
    generate_flux_source_corpus,
    load_flux_source_corpus_inventory,
)
from aigen.pix2pix.flux_source_set import (
    LoadedFluxSourceSet,
    load_flux_source_set,
    load_frozen_flux_source_set,
)
from aigen.pix2pix.iro_corpus import load_iro_selection
from aigen.progress import StatusReporter
from aigen.runtime_provenance import validate_python_runtime_provenance


FLUX_SOURCE_SET_PLAN_FORMAT = "aigen.pix2pix.flux-source-set-plan.v1"
FLUX_SOURCE_SET_SHARD_FORMAT = "aigen.pix2pix.flux-source-set-shard.v1"
FLUX_SOURCE_SET_RESULT_FORMAT = "aigen.pix2pix.flux-source-set-corpus.v1"
FLUX_SOURCE_SET_IMPLEMENTATION_REVISION = "1"
FLUX_SOURCE_SET_DIRECTORY_PREFIX = "flux-source-set-"
FLUX_SOURCE_SET_BACKEND = "flux2-klein"
FLUX_SOURCE_SET_MODEL = "black-forest-labs/FLUX.2-klein-9B-scaled-fp8"


def generate_flux_source_set_sources(
    root: Path,
    source_set_path: Path,
    *,
    progress: StatusReporter,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    config, selected, selection = load_iro_selection(root)
    source_set = load_flux_source_set(
        source_set_path,
        selected=selected,
        selection=selection,
    )
    layout = flux_source_set_layout(source_set.name)
    source_plan, source_plan_sha256 = _load_or_create_source_plan(
        root,
        config=config,
        selected=selected,
        selection=selection,
        source_set=source_set,
        layout=layout,
        model_provenance=flux2_klein_model_provenance(),
        runtime_provenance=flux2_klein_runtime_provenance(),
    )
    return generate_flux_source_corpus(
        root,
        selected=selected,
        generation=config.flux,
        cases=_engine_cases(source_set),
        layout=layout,
        source_plan=source_plan,
        source_plan_sha256=source_plan_sha256,
        progress=progress,
    )


def load_flux_source_set_inventory(
    root: Path,
    name: str,
) -> tuple[dict[str, Path], LoadedFluxSourceSet, dict[str, Any], str]:
    root = root.expanduser().resolve()
    config, selected, selection = load_iro_selection(root)
    layout = flux_source_set_layout(name)
    source_plan, source_plan_sha256, source_set = _load_frozen_source_plan(
        root,
        config=config,
        selected=selected,
        selection=selection,
        layout=layout,
    )
    inventory = load_flux_source_corpus_inventory(
        root,
        selected=selected,
        generation=config.flux,
        cases=_engine_cases(source_set),
        layout=layout,
        source_plan_sha256=source_plan_sha256,
    )
    if source_plan["pair_count"] != len(inventory):
        raise Pix2PixError("FLUX source-set plan pair count differs from inventory")
    return inventory, source_set, source_plan, source_plan_sha256


def flux_source_set_result_path(root: Path, name: str) -> Path:
    return flux_source_corpus_result_path(
        root.expanduser().resolve(),
        flux_source_set_layout(name),
    )


def flux_source_set_layout(name: str) -> FluxSourceLayout:
    if not re.fullmatch(SAFE_NAME_PATTERN, name):
        raise Pix2PixError(f"invalid FLUX source-set name: {name!r}")
    return FluxSourceLayout(
        directory=f"{FLUX_SOURCE_SET_DIRECTORY_PREFIX}{name}",
        shard_format=FLUX_SOURCE_SET_SHARD_FORMAT,
        result_format=FLUX_SOURCE_SET_RESULT_FORMAT,
        kind="FLUX-source-set-paired-source-corpus",
        label=f"FLUX source set {name}",
    )


def _load_or_create_source_plan(
    root: Path,
    *,
    config: Any,
    selected: tuple[dict[str, Any], ...],
    selection: dict[str, Any],
    source_set: LoadedFluxSourceSet,
    layout: FluxSourceLayout,
    model_provenance: dict[str, object],
    runtime_provenance: dict[str, object],
) -> tuple[dict[str, Any], str]:
    if source_set.manifest_sha256 is None:
        raise Pix2PixError("external FLUX source set has no manifest checksum")
    source_root = root / layout.directory
    plan_path = source_root / "source-plan.json"
    expected = _source_plan_payload(
        config=config,
        selected=selected,
        selection=selection,
        source_set=source_set,
        source_set_manifest_sha256=source_set.manifest_sha256,
        model_provenance=model_provenance,
        runtime_provenance=runtime_provenance,
    )
    if plan_path.exists():
        plan = read_json(plan_path, label="FLUX source-set plan")
        _verify_source_plan(plan, expected)
    else:
        if source_root.exists() and any(source_root.iterdir()):
            raise Pix2PixError(
                f"non-empty FLUX source-set corpus has no source plan: {source_root}"
            )
        source_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(plan_path, expected)
        plan = read_json(plan_path, label="FLUX source-set plan")
        _verify_source_plan(plan, expected)
    return plan, sha256_file(plan_path)


def _load_frozen_source_plan(
    root: Path,
    *,
    config: Any,
    selected: tuple[dict[str, Any], ...],
    selection: dict[str, Any],
    layout: FluxSourceLayout,
) -> tuple[dict[str, Any], str, LoadedFluxSourceSet]:
    plan_path = root / layout.directory / "source-plan.json"
    plan = read_json(plan_path, label="FLUX source-set plan")
    source_set = load_frozen_flux_source_set(
        plan.get("source_set"),
        selected=selected,
        selection=selection,
    )
    if source_set.name != layout.directory.removeprefix(
        FLUX_SOURCE_SET_DIRECTORY_PREFIX
    ):
        raise Pix2PixError("FLUX source-set plan name differs from its directory")
    source_set_manifest_sha256 = plan.get("source_set_manifest_sha256")
    model_provenance = plan.get("model_provenance")
    runtime_provenance = plan.get("runtime_provenance")
    if (
        not isinstance(source_set_manifest_sha256, str)
        or not isinstance(model_provenance, dict)
        or not isinstance(runtime_provenance, dict)
    ):
        raise Pix2PixError("FLUX source-set plan has invalid frozen records")
    expected = _source_plan_payload(
        config=config,
        selected=selected,
        selection=selection,
        source_set=source_set,
        source_set_manifest_sha256=source_set_manifest_sha256,
        model_provenance=model_provenance,
        runtime_provenance=runtime_provenance,
    )
    _verify_source_plan(plan, expected)
    return plan, sha256_file(plan_path), source_set


def _source_plan_payload(
    *,
    config: Any,
    selected: tuple[dict[str, Any], ...],
    selection: dict[str, Any],
    source_set: LoadedFluxSourceSet,
    source_set_manifest_sha256: str,
    model_provenance: dict[str, object],
    runtime_provenance: dict[str, object],
) -> dict[str, object]:
    return {
        "format": FLUX_SOURCE_SET_PLAN_FORMAT,
        "source_implementation_revision": (
            FLUX_SOURCE_SET_IMPLEMENTATION_REVISION
        ),
        "config_fingerprint": selection["config_fingerprint"],
        "selection_sha256": selection["selected_sha256"],
        "source_set_fingerprint": source_set.fingerprint,
        "source_set_manifest_sha256": source_set_manifest_sha256,
        "source_set": source_set.frozen_payload(),
        "pair_count": len(selected),
        "backend": {
            "name": FLUX_SOURCE_SET_BACKEND,
            "implementation_revision": FLUX2_KLEIN_IMPLEMENTATION_REVISION,
            "model": FLUX_SOURCE_SET_MODEL,
        },
        "model_provenance": model_provenance,
        "runtime_provenance": runtime_provenance,
        "generation": {
            "width": config.flux.width,
            "height": config.flux.height,
            "steps": config.flux.steps,
            "sampler": config.flux.sampler,
            "scheduler": config.flux.scheduler,
            "shard_size": config.flux.shard_size,
            "reference_raster": {
                "mode": "RGB",
                "width": config.image_size,
                "height": config.image_size,
                "resampled": False,
            },
        },
    }


def _verify_source_plan(
    plan: dict[str, Any],
    expected: dict[str, object],
) -> None:
    require_exact_keys(
        plan,
        {
            "format",
            "source_implementation_revision",
            "config_fingerprint",
            "selection_sha256",
            "source_set_fingerprint",
            "source_set_manifest_sha256",
            "source_set",
            "pair_count",
            "backend",
            "model_provenance",
            "runtime_provenance",
            "generation",
        },
        "FLUX source-set plan",
    )
    try:
        model_provenance = plan["model_provenance"]
        runtime_provenance = plan["runtime_provenance"]
        if not isinstance(model_provenance, dict) or not isinstance(
            runtime_provenance,
            dict,
        ):
            raise ValueError("source provenance records must be objects")
        validate_model_artifact_provenance(model_provenance)
        validate_python_runtime_provenance(runtime_provenance)
    except ValueError as error:
        raise Pix2PixError(f"invalid FLUX source-set provenance: {error}") from error
    if plan != expected:
        raise Pix2PixError(
            "FLUX source-set plan differs from the current selection, source "
            "set, model, or runtime"
        )


def _engine_cases(
    source_set: LoadedFluxSourceSet,
) -> tuple[FluxSourceCaseSpec, ...]:
    return tuple(
        FluxSourceCaseSpec(
            id=record.id,
            prompt=record.prompt,
            seed=record.seed,
        )
        for record in source_set.records
    )
