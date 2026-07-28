from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from aigen.generation.flux2_klein_artifacts import (
    FLUX2_KLEIN_IMPLEMENTATION_REVISION,
    flux2_klein_model_provenance,
    flux2_klein_runtime_provenance,
)
from aigen.generation.image_generation_requests import (
    ImageGenerationCaseRequest,
    ImageGenerationOutputRequest,
)
from aigen.manifest_io import atomic_write_json, read_json, sha256_file, write_json
from aigen.model_artifacts import validate_model_artifact_provenance
from aigen.pix2pix.corpus_io import corpus_member, require_exact_keys
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.iro_corpus import load_iro_selection
from aigen.pix2pix.source_corpus import (
    expected_source_shards,
    inspect_source_image,
    validate_source_output,
)
from aigen.progress import StatusReporter
from aigen.runtime_provenance import validate_python_runtime_provenance


FLUX_SOURCE_PLAN_FORMAT = "aigen.pix2pix.flux-source-plan.v3"
FLUX_SOURCE_SHARD_FORMAT = "aigen.pix2pix.flux-source-shard.v3"
FLUX_SOURCE_RESULT_FORMAT = "aigen.pix2pix.flux-source-corpus.v3"
FLUX_SOURCE_IMPLEMENTATION_REVISION = "3"
FLUX_SOURCE_DIRECTORY = "flux-v3"
FLUX_SOURCE_BACKEND = "flux2-klein"
FLUX_SOURCE_MODEL = "black-forest-labs/FLUX.2-klein-9B-scaled-fp8"


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


def _source_result(
    source_root: Path,
    *,
    shards: tuple[tuple[int, tuple[dict[str, Any], ...]], ...],
    source_plan_sha256: str,
    pair_count: int,
    shard_size: int,
) -> dict[str, object]:
    shard_records = []
    for shard_index, _ in shards:
        manifest_path = (
            source_root / "shards" / f"shard-{shard_index:05d}" / "shard.json"
        )
        if not manifest_path.is_file():
            raise Pix2PixError(f"missing FLUX source shard manifest: {manifest_path}")
        shard_records.append(
            {
                "index": shard_index,
                "path": f"shards/shard-{shard_index:05d}/shard.json",
                "sha256": sha256_file(manifest_path),
            }
        )
    return {
        "format": FLUX_SOURCE_RESULT_FORMAT,
        "status": "completed",
        "source_plan_sha256": source_plan_sha256,
        "pair_count": pair_count,
        "shard_size": shard_size,
        "shard_count": len(shards),
        "shards": shard_records,
    }


def _load_source_result(
    source_root: Path,
    *,
    expected: dict[str, object],
) -> dict[str, Any]:
    result = read_json(
        source_root / "result.json",
        label="FLUX source result",
    )
    require_exact_keys(
        result,
        {
            "format",
            "status",
            "source_plan_sha256",
            "pair_count",
            "shard_size",
            "shard_count",
            "shards",
        },
        "FLUX source result",
    )
    if result != expected:
        raise Pix2PixError(
            "FLUX source result differs from its source plan or shard manifests"
        )
    return result


def generate_flux_sources(
    root: Path,
    *,
    progress: StatusReporter,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    config, selected, selection = load_iro_selection(root)
    source_plan, source_plan_sha256 = _load_or_create_source_plan(
        root,
        config=config,
        selected=selected,
        selection=selection,
        model_provenance=flux2_klein_model_provenance(),
        runtime_provenance=flux2_klein_runtime_provenance(),
    )
    shards = expected_source_shards(selected, config.flux.shard_size)
    source_root = root / FLUX_SOURCE_DIRECTORY
    shards_dir = source_root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    result_path = source_root / "result.json"
    if result_path.exists():
        result = _source_result(
            source_root,
            shards=shards,
            source_plan_sha256=source_plan_sha256,
            pair_count=len(selected),
            shard_size=config.flux.shard_size,
        )
        _load_source_result(source_root, expected=result)
        inventory = _load_source_inventory(
            source_root,
            shards,
            config=config,
            source_plan_sha256=source_plan_sha256,
        )
        return _generation_result(
            root,
            source_plan=source_plan,
            source_plan_sha256=source_plan_sha256,
            pair_count=len(inventory),
            shard_count=len(shards),
            generated_shards=0,
            reused_shards=len(shards),
        )

    missing = []
    reused = 0
    for shard_index, cases in shards:
        shard_dir = shards_dir / f"shard-{shard_index:05d}"
        if shard_dir.exists():
            _load_source_shard(
                shard_dir,
                shard_index=shard_index,
                cases=cases,
                config=config,
                source_plan_sha256=source_plan_sha256,
            )
            reused += 1
        else:
            missing.append((shard_index, cases))

    generated = 0
    if missing:
        from aigen.generation.flux2_klein import (
            Flux2KleinError,
            Flux2KleinSession,
            encode_flux2_klein_prompts,
        )

        try:
            prompt_embeddings, _ = encode_flux2_klein_prompts(
                prompts=(config.flux.prompt,),
                progress=progress,
            )
            session = Flux2KleinSession(
                loras=(),
                sampler=config.flux.sampler,
                strength=None,
                progress=progress,
            )
            progress.begin(len(missing), "generate FLUX source shards")
            try:
                for shard_index, cases in missing:
                    _generate_source_shard(
                        root,
                        shards_dir,
                        shard_index,
                        cases,
                        config=config,
                        source_plan_sha256=source_plan_sha256,
                        prompt_embeddings=prompt_embeddings,
                        session=session,
                        progress=progress,
                    )
                    generated += 1
                    progress.step(f"published FLUX shard {shard_index:05d}")
            finally:
                session.close()
        except Flux2KleinError as error:
            raise Pix2PixError(str(error)) from error

    inventory = _load_source_inventory(
        source_root,
        shards,
        config=config,
        source_plan_sha256=source_plan_sha256,
    )
    result = _source_result(
        source_root,
        shards=shards,
        source_plan_sha256=source_plan_sha256,
        pair_count=len(selected),
        shard_size=config.flux.shard_size,
    )
    atomic_write_json(result_path, result)
    _load_source_result(
        source_root,
        expected=result,
    )
    return _generation_result(
        root,
        source_plan=source_plan,
        source_plan_sha256=source_plan_sha256,
        pair_count=len(inventory),
        shard_count=len(shards),
        generated_shards=generated,
        reused_shards=reused,
    )


def _generation_result(
    root: Path,
    *,
    source_plan: dict[str, Any],
    source_plan_sha256: str,
    pair_count: int,
    shard_count: int,
    generated_shards: int,
    reused_shards: int,
) -> dict[str, object]:
    return {
        "status": "completed",
        "kind": "FLUX-paired-source-corpus",
        "root": root.as_posix(),
        "source_plan_sha256": source_plan_sha256,
        "model_fingerprint": source_plan["model_provenance"]["fingerprint"],
        "runtime_fingerprint": source_plan["runtime_provenance"]["fingerprint"],
        "pair_count": pair_count,
        "shard_count": shard_count,
        "generated_shards": generated_shards,
        "reused_shards": reused_shards,
    }


def load_flux_source_inventory(root: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    config, selected, selection = load_iro_selection(root)
    source_plan, source_plan_sha256 = _load_frozen_source_plan(
        root,
        config=config,
        selected=selected,
        selection=selection,
    )
    shards = expected_source_shards(selected, config.flux.shard_size)
    source_root = root / FLUX_SOURCE_DIRECTORY
    expected_result = _source_result(
        source_root,
        shards=shards,
        source_plan_sha256=source_plan_sha256,
        pair_count=len(selected),
        shard_size=config.flux.shard_size,
    )
    _load_source_result(source_root, expected=expected_result)
    inventory = _load_source_inventory(
        source_root,
        shards,
        config=config,
        source_plan_sha256=source_plan_sha256,
    )
    if source_plan["pair_count"] != len(inventory):
        raise Pix2PixError("FLUX source plan pair count differs from inventory")
    return inventory


def flux_source_result_path(root: Path) -> Path:
    return root.expanduser().resolve() / FLUX_SOURCE_DIRECTORY / "result.json"


def _load_source_inventory(
    source_root: Path,
    shards: tuple[tuple[int, tuple[dict[str, Any], ...]], ...],
    *,
    config: Any,
    source_plan_sha256: str,
) -> dict[str, Path]:
    inventory: dict[str, Path] = {}
    expected_ids = set()
    for shard_index, cases in shards:
        shard_dir = source_root / "shards" / f"shard-{shard_index:05d}"
        manifest = _load_source_shard(
            shard_dir,
            shard_index=shard_index,
            cases=cases,
            config=config,
            source_plan_sha256=source_plan_sha256,
        )
        expected_ids.update(str(case["id"]) for case in cases)
        for output in manifest["outputs"]:
            pair_id = str(output["id"])
            if pair_id in inventory:
                raise Pix2PixError(f"duplicate FLUX source id: {pair_id}")
            inventory[pair_id] = corpus_member(
                shard_dir,
                str(output["path"]),
                label=f"FLUX source for {pair_id}",
            )
    if set(inventory) != expected_ids:
        raise Pix2PixError("FLUX source inventory differs from target selection")
    return inventory


def _generate_source_shard(
    root: Path,
    shards_dir: Path,
    shard_index: int,
    cases: tuple[dict[str, Any], ...],
    *,
    config: Any,
    source_plan_sha256: str,
    prompt_embeddings: Any,
    session: Any,
    progress: StatusReporter,
) -> None:
    destination = shards_dir / f"shard-{shard_index:05d}"
    with TemporaryDirectory(
        dir=shards_dir,
        prefix=f".shard-{shard_index:05d}.",
        suffix=".incomplete",
    ) as temporary:
        staging = Path(temporary)
        raw_dir = staging / "raw"
        raw_dir.mkdir()
        generation_cases = []
        for case in cases:
            pair_id = str(case["id"])
            target_path = corpus_member(
                root,
                str(case["target"]),
                label=f"FLUX reference for {pair_id}",
            )
            generation_cases.append(
                ImageGenerationCaseRequest(
                    name=pair_id,
                    prompt=config.flux.prompt,
                    image_paths=(target_path,),
                    width=config.flux.width,
                    height=config.flux.height,
                    outputs=(
                        ImageGenerationOutputRequest(
                            name=pair_id,
                            seed=int(case["flux_seed"]),
                            path=raw_dir / f"{pair_id}.png",
                        ),
                    ),
                )
            )
        result = session.generate(
            cases=tuple(generation_cases),
            prompt_embeddings=prompt_embeddings,
            progress=progress,
        )
        result_by_case = {output.case: output for output in result.outputs}
        if set(result_by_case) != {str(case["id"]) for case in cases}:
            raise Pix2PixError(f"FLUX shard {shard_index} output inventory mismatch")
        outputs = []
        for case in cases:
            pair_id = str(case["id"])
            output_path = raw_dir / f"{pair_id}.png"
            mode, width, height = inspect_source_image(
                output_path,
                expected_size=(config.flux.width, config.flux.height),
                label="FLUX",
            )
            generated = result_by_case[pair_id]
            if generated.seed != case["flux_seed"]:
                raise Pix2PixError(f"FLUX shard {shard_index} seed mismatch: {pair_id}")
            outputs.append(
                {
                    "id": pair_id,
                    "path": f"raw/{pair_id}.png",
                    "sha256": sha256_file(output_path),
                    "size_bytes": output_path.stat().st_size,
                    "mode": mode,
                    "width": width,
                    "height": height,
                    "seed": int(case["flux_seed"]),
                }
            )
        manifest = {
            "format": FLUX_SOURCE_SHARD_FORMAT,
            "shard_index": shard_index,
            "source_plan_sha256": source_plan_sha256,
            "generation_ms": result.generation_ms,
            "model_load_ms": result.model_load_ms,
            "peak_vram_mb": result.peak_vram_mb,
            "outputs": outputs,
        }
        write_json(staging / "shard.json", manifest)
        _load_source_shard(
            staging,
            shard_index=shard_index,
            cases=cases,
            config=config,
            source_plan_sha256=source_plan_sha256,
        )
        os.rename(staging, destination)


def _load_source_shard(
    shard_dir: Path,
    *,
    shard_index: int,
    cases: tuple[dict[str, Any], ...],
    config: Any,
    source_plan_sha256: str,
) -> dict[str, Any]:
    manifest = read_json(shard_dir / "shard.json", label="FLUX source shard")
    require_exact_keys(
        manifest,
        {
            "format",
            "shard_index",
            "source_plan_sha256",
            "generation_ms",
            "model_load_ms",
            "peak_vram_mb",
            "outputs",
        },
        "FLUX source shard",
    )
    expected_values = {
        "format": FLUX_SOURCE_SHARD_FORMAT,
        "shard_index": shard_index,
        "source_plan_sha256": source_plan_sha256,
    }
    for key, expected in expected_values.items():
        if manifest[key] != expected:
            raise Pix2PixError(
                f"FLUX source shard {shard_index} has incompatible {key}"
            )
    outputs = manifest["outputs"]
    if not isinstance(outputs, list) or len(outputs) != len(cases):
        raise Pix2PixError(f"FLUX source shard {shard_index} has invalid output count")
    expected_ids = [str(case["id"]) for case in cases]
    if [output.get("id") for output in outputs if isinstance(output, dict)] != expected_ids:
        raise Pix2PixError(f"FLUX source shard {shard_index} output order mismatch")
    for case, output in zip(cases, outputs, strict=True):
        if not isinstance(output, dict):
            raise Pix2PixError(f"invalid FLUX output in shard {shard_index}")
        pair_id = str(case["id"])
        validate_source_output(
            shard_dir,
            output,
            pair_id=pair_id,
            expected_seed=int(case["flux_seed"]),
            expected_size=(config.flux.width, config.flux.height),
            label="FLUX",
        )
    return manifest
