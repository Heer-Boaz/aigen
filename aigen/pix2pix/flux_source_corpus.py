from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from aigen.generation.flux2_klein_artifacts import (
    FLUX2_KLEIN_IMPLEMENTATION_REVISION,
    flux2_klein_model_provenance,
    flux2_klein_runtime_provenance,
)
from aigen.manifest_io import atomic_write_json, read_json, sha256_file
from aigen.model_artifacts import validate_model_artifact_provenance
from aigen.pix2pix.corpus_config import IroCorpusConfigV2
from aigen.pix2pix.corpus_io import require_exact_keys
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.flux_source_engine import (
    FluxSourceCaseSpec,
    FluxSourceLayout,
    flux_source_corpus_result_path,
    generate_flux_source_corpus,
    load_flux_source_corpus_inventory,
)
from aigen.pix2pix.iro_corpus import load_iro_selection
from aigen.progress import StatusReporter
from aigen.runtime_provenance import validate_python_runtime_provenance


FLUX_SOURCE_PLAN_FORMAT = "aigen.pix2pix.flux-source-plan.v3"
FLUX_SOURCE_SHARD_FORMAT = "aigen.pix2pix.flux-source-shard.v3"
FLUX_SOURCE_RESULT_FORMAT = "aigen.pix2pix.flux-source-corpus.v3"
FLUX_SOURCE_IMPLEMENTATION_REVISION = "3"
FLUX_SOURCE_DIRECTORY = "flux-v3"
FLUX_SOURCE_BACKEND = "flux2-klein"
FLUX_SOURCE_MODEL = "black-forest-labs/FLUX.2-klein-9B-scaled-fp8"
FLUX_SOURCE_LAYOUT = FluxSourceLayout(
    directory=FLUX_SOURCE_DIRECTORY,
    shard_format=FLUX_SOURCE_SHARD_FORMAT,
    result_format=FLUX_SOURCE_RESULT_FORMAT,
    kind="FLUX-paired-source-corpus",
    label="FLUX",
)


def _load_or_create_source_plan(
    root: Path,
    *,
    config: Any,
    selected: tuple[dict[str, Any], ...],
    selection: dict[str, Any],
    model_provenance: dict[str, object],
    runtime_provenance: dict[str, object],
) -> tuple[dict[str, Any], str]:
    source_root = root / FLUX_SOURCE_DIRECTORY
    plan_path = source_root / "source-plan.json"
    expected = _source_plan_payload(
        config=config,
        selected=selected,
        selection=selection,
        model_provenance=model_provenance,
        runtime_provenance=runtime_provenance,
    )
    if plan_path.exists():
        plan = read_json(plan_path, label="FLUX source plan")
        _verify_source_plan(plan, expected)
    else:
        if source_root.exists() and any(source_root.iterdir()):
            raise Pix2PixError(
                f"non-empty FLUX source corpus has no source plan: {source_root}"
            )
        source_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(plan_path, expected)
        plan = read_json(plan_path, label="FLUX source plan")
        _verify_source_plan(plan, expected)
    return plan, sha256_file(plan_path)


def _source_plan_payload(
    *,
    config: Any,
    selected: tuple[dict[str, Any], ...],
    selection: dict[str, Any],
    model_provenance: dict[str, object],
    runtime_provenance: dict[str, object],
) -> dict[str, object]:
    return {
        "format": FLUX_SOURCE_PLAN_FORMAT,
        "source_implementation_revision": FLUX_SOURCE_IMPLEMENTATION_REVISION,
        "config_fingerprint": selection["config_fingerprint"],
        "selection_sha256": selection["selected_sha256"],
        "pair_count": len(selected),
        "backend": {
            "name": FLUX_SOURCE_BACKEND,
            "implementation_revision": FLUX2_KLEIN_IMPLEMENTATION_REVISION,
            "model": FLUX_SOURCE_MODEL,
        },
        "model_provenance": model_provenance,
        "runtime_provenance": runtime_provenance,
        "generation": {
            "prompt": config.flux.prompt,
            "prompt_sha256": hashlib.sha256(
                config.flux.prompt.encode("utf-8")
            ).hexdigest(),
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
            "pair_count",
            "backend",
            "model_provenance",
            "runtime_provenance",
            "generation",
        },
        "FLUX source plan",
    )
    try:
        model_provenance = plan["model_provenance"]
        runtime_provenance = plan["runtime_provenance"]
        if not isinstance(model_provenance, dict) or not isinstance(
            runtime_provenance, dict
        ):
            raise ValueError("source provenance records must be objects")
        validate_model_artifact_provenance(model_provenance)
        validate_python_runtime_provenance(runtime_provenance)
    except ValueError as error:
        raise Pix2PixError(f"invalid FLUX source provenance: {error}") from error
    if plan != expected:
        raise Pix2PixError(
            "FLUX source plan differs from the current corpus, model, or runtime"
        )


def _load_frozen_source_plan(
    root: Path,
    *,
    config: Any,
    selected: tuple[dict[str, Any], ...],
    selection: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    plan_path = root / FLUX_SOURCE_DIRECTORY / "source-plan.json"
    plan = read_json(plan_path, label="FLUX source plan")
    model_provenance = plan.get("model_provenance")
    runtime_provenance = plan.get("runtime_provenance")
    if not isinstance(model_provenance, dict) or not isinstance(
        runtime_provenance, dict
    ):
        raise Pix2PixError("FLUX source plan has invalid provenance records")
    expected = _source_plan_payload(
        config=config,
        selected=selected,
        selection=selection,
        model_provenance=model_provenance,
        runtime_provenance=runtime_provenance,
    )
    _verify_source_plan(plan, expected)
    return plan, sha256_file(plan_path)


def generate_flux_sources(
    root: Path,
    *,
    progress: StatusReporter,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    config, selected, selection = load_iro_selection(root)
    _reject_v2_generic_source_route(config)
    source_plan, source_plan_sha256 = _load_or_create_source_plan(
        root,
        config=config,
        selected=selected,
        selection=selection,
        model_provenance=flux2_klein_model_provenance(),
        runtime_provenance=flux2_klein_runtime_provenance(),
    )
    cases = tuple(
        FluxSourceCaseSpec(
            id=str(record["id"]),
            prompt=config.flux.prompt,
            seed=int(record["flux_seed"]),
        )
        for record in selected
    )
    return generate_flux_source_corpus(
        root,
        selected=selected,
        generation=config.flux,
        cases=cases,
        layout=FLUX_SOURCE_LAYOUT,
        source_plan=source_plan,
        source_plan_sha256=source_plan_sha256,
        progress=progress,
    )


def load_flux_source_inventory(root: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    config, selected, selection = load_iro_selection(root)
    _reject_v2_generic_source_route(config)
    source_plan, source_plan_sha256 = _load_frozen_source_plan(
        root,
        config=config,
        selected=selected,
        selection=selection,
    )
    cases = tuple(
        FluxSourceCaseSpec(
            id=str(record["id"]),
            prompt=config.flux.prompt,
            seed=int(record["flux_seed"]),
        )
        for record in selected
    )
    inventory = load_flux_source_corpus_inventory(
        root,
        selected=selected,
        generation=config.flux,
        cases=cases,
        layout=FLUX_SOURCE_LAYOUT,
        source_plan_sha256=source_plan_sha256,
    )
    if source_plan["pair_count"] != len(inventory):
        raise Pix2PixError("FLUX source plan pair count differs from inventory")
    return inventory


def _reject_v2_generic_source_route(config: Any) -> None:
    if isinstance(config, IroCorpusConfigV2):
        raise Pix2PixError(
            "iRO v2 corpora require independently reviewed per-image FLUX "
            "source sets; use iro-generate-flux-source-set"
        )


def flux_source_result_path(root: Path) -> Path:
    return flux_source_corpus_result_path(
        root.expanduser().resolve(),
        FLUX_SOURCE_LAYOUT,
    )
