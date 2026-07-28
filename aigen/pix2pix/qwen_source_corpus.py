from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pydantic import ValidationError

from aigen.generation.qwen_image_edit_artifacts import (
    QWEN_IMAGE_EDIT_LIGHTX2V_IMPLEMENTATION_REVISION,
    qwen_2511_lightning_model_provenance,
    qwen_2511_lightx2v_runtime_provenance,
)
from aigen.generation.qwen_image_edit_lightx2v import (
    LIGHTX2V_QWEN_EDIT_2511_PROFILE,
    LIGHTX2V_REVISION,
    QWEN_EDIT_2511_LIGHTNING_REVISION,
    QWEN_EDIT_2511_REVISION,
)
from aigen.manifest_io import atomic_write_json, read_json, sha256_file, write_json
from aigen.model_artifacts import validate_model_artifact_provenance
from aigen.pix2pix.corpus_io import corpus_member, require_exact_keys
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.iro_corpus import load_iro_selection
from aigen.pix2pix.qwen_source_config import (
    QwenSourceConfig,
    load_qwen_source_config,
    qwen_source_config_fingerprint,
)
from aigen.pix2pix.source_corpus import (
    expected_source_shards,
    inspect_source_image,
    validate_source_output,
)
from aigen.progress import StatusReporter
from aigen.runtime_provenance import validate_python_runtime_provenance


QWEN_SOURCE_PLAN_FORMAT = "aigen.pix2pix.qwen-source-plan.v1"
QWEN_SOURCE_SHARD_FORMAT = "aigen.pix2pix.qwen-source-shard.v1"
QWEN_SOURCE_RESULT_FORMAT = "aigen.pix2pix.qwen-source-corpus.v1"
QWEN_SOURCE_IMPLEMENTATION_REVISION = "1"
QWEN_SOURCE_DIRECTORY = "qwen-2511-lightning-v1"


def generate_qwen_sources(
    root: Path,
    source_config_path: Path,
    *,
    progress: StatusReporter,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    config = load_qwen_source_config(source_config_path)
    _, selected, selection = load_iro_selection(root)
    source_plan, source_plan_sha256 = _load_or_create_source_plan(
        root,
        config=config,
        selected=selected,
        selection=selection,
        model_provenance=qwen_2511_lightning_model_provenance(),
        runtime_provenance=qwen_2511_lightx2v_runtime_provenance(),
    )
    shards = expected_source_shards(selected, config.shard_size)
    source_root = root / QWEN_SOURCE_DIRECTORY
    shards_dir = source_root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    result_path = source_root / "result.json"
    if result_path.exists():
        expected_result = _source_result(
            source_root,
            shards=shards,
            source_plan_sha256=source_plan_sha256,
            pair_count=len(selected),
            shard_size=config.shard_size,
        )
        _load_source_result(source_root, expected=expected_result)
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
        from aigen.generation.image_edit_batch import ImageEditBatchError

        progress.begin(len(missing), "generate Qwen source shards")
        try:
            for shard_index, cases in missing:
                _generate_source_shard(
                    root,
                    shards_dir,
                    shard_index,
                    cases,
                    config=config,
                    source_plan_sha256=source_plan_sha256,
                    progress=progress,
                )
                generated += 1
                progress.step(f"published Qwen shard {shard_index:05d}")
        except ImageEditBatchError as error:
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
        shard_size=config.shard_size,
    )
    atomic_write_json(result_path, result)
    _load_source_result(source_root, expected=result)
    return _generation_result(
        root,
        source_plan=source_plan,
        source_plan_sha256=source_plan_sha256,
        pair_count=len(inventory),
        shard_count=len(shards),
        generated_shards=generated,
        reused_shards=reused,
    )


def load_qwen_source_inventory(root: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    _, selected, selection = load_iro_selection(root)
    source_plan, source_plan_sha256, config = _load_frozen_source_plan(
        root,
        selected=selected,
        selection=selection,
    )
    shards = expected_source_shards(selected, config.shard_size)
    source_root = root / QWEN_SOURCE_DIRECTORY
    expected_result = _source_result(
        source_root,
        shards=shards,
        source_plan_sha256=source_plan_sha256,
        pair_count=len(selected),
        shard_size=config.shard_size,
    )
    _load_source_result(source_root, expected=expected_result)
    inventory = _load_source_inventory(
        source_root,
        shards,
        config=config,
        source_plan_sha256=source_plan_sha256,
    )
    if source_plan["pair_count"] != len(inventory):
        raise Pix2PixError("Qwen source plan pair count differs from inventory")
    return inventory


def load_frozen_qwen_source_config(root: Path) -> QwenSourceConfig:
    root = root.expanduser().resolve()
    _, selected, selection = load_iro_selection(root)
    _, _, config = _load_frozen_source_plan(
        root,
        selected=selected,
        selection=selection,
    )
    return config


def qwen_source_result_path(root: Path) -> Path:
    return root.expanduser().resolve() / QWEN_SOURCE_DIRECTORY / "result.json"


def _load_or_create_source_plan(
    root: Path,
    *,
    config: QwenSourceConfig,
    selected: tuple[dict[str, Any], ...],
    selection: dict[str, Any],
    model_provenance: dict[str, object],
    runtime_provenance: dict[str, object],
) -> tuple[dict[str, Any], str]:
    source_root = root / QWEN_SOURCE_DIRECTORY
    plan_path = source_root / "source-plan.json"
    expected = _source_plan_payload(
        config=config,
        selected=selected,
        selection=selection,
        model_provenance=model_provenance,
        runtime_provenance=runtime_provenance,
    )
    if plan_path.exists():
        plan = read_json(plan_path, label="Qwen source plan")
        _verify_source_plan(plan, expected)
    else:
        if source_root.exists() and any(source_root.iterdir()):
            raise Pix2PixError(
                f"non-empty Qwen source corpus has no source plan: {source_root}"
            )
        source_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(plan_path, expected)
        plan = read_json(plan_path, label="Qwen source plan")
        _verify_source_plan(plan, expected)
    return plan, sha256_file(plan_path)


def _load_frozen_source_plan(
    root: Path,
    *,
    selected: tuple[dict[str, Any], ...],
    selection: dict[str, Any],
) -> tuple[dict[str, Any], str, QwenSourceConfig]:
    plan_path = root / QWEN_SOURCE_DIRECTORY / "source-plan.json"
    plan = read_json(plan_path, label="Qwen source plan")
    source_config = plan.get("source_config")
    model_provenance = plan.get("model_provenance")
    runtime_provenance = plan.get("runtime_provenance")
    if (
        not isinstance(source_config, dict)
        or not isinstance(model_provenance, dict)
        or not isinstance(runtime_provenance, dict)
    ):
        raise Pix2PixError("Qwen source plan has invalid frozen records")
    try:
        config = QwenSourceConfig.model_validate(source_config)
    except ValidationError as error:
        raise Pix2PixError(f"invalid frozen Qwen source config: {error}") from error
    expected = _source_plan_payload(
        config=config,
        selected=selected,
        selection=selection,
        model_provenance=model_provenance,
        runtime_provenance=runtime_provenance,
    )
    _verify_source_plan(plan, expected)
    return plan, sha256_file(plan_path), config


def _source_plan_payload(
    *,
    config: QwenSourceConfig,
    selected: tuple[dict[str, Any], ...],
    selection: dict[str, Any],
    model_provenance: dict[str, object],
    runtime_provenance: dict[str, object],
) -> dict[str, object]:
    return {
        "format": QWEN_SOURCE_PLAN_FORMAT,
        "source_implementation_revision": QWEN_SOURCE_IMPLEMENTATION_REVISION,
        "config_fingerprint": selection["config_fingerprint"],
        "selection_sha256": selection["selected_sha256"],
        "source_config_fingerprint": qwen_source_config_fingerprint(config),
        "source_config": config.model_dump(mode="json"),
        "pair_count": len(selected),
        "backend": {
            "name": config.backend,
            "profile": LIGHTX2V_QWEN_EDIT_2511_PROFILE,
            "implementation_revision": (
                QWEN_IMAGE_EDIT_LIGHTX2V_IMPLEMENTATION_REVISION
            ),
            "engine_revision": LIGHTX2V_REVISION,
            "base_revision": QWEN_EDIT_2511_REVISION,
            "transformer_revision": QWEN_EDIT_2511_LIGHTNING_REVISION,
        },
        "model_provenance": model_provenance,
        "runtime_provenance": runtime_provenance,
        "generation": {
            "prompt_sha256": hashlib.sha256(
                config.prompt.encode("utf-8")
            ).hexdigest(),
            "input_raster": {
                "mode": "RGB",
                "width": 128,
                "height": 128,
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
            "source_config_fingerprint",
            "source_config",
            "pair_count",
            "backend",
            "model_provenance",
            "runtime_provenance",
            "generation",
        },
        "Qwen source plan",
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
        raise Pix2PixError(f"invalid Qwen source provenance: {error}") from error
    if plan != expected:
        raise Pix2PixError(
            "Qwen source plan differs from the current corpus, config, model, "
            "or runtime"
        )


def _generate_source_shard(
    root: Path,
    shards_dir: Path,
    shard_index: int,
    cases: tuple[dict[str, Any], ...],
    *,
    config: QwenSourceConfig,
    source_plan_sha256: str,
    progress: StatusReporter,
) -> None:
    from aigen.generation.image_edit_batch import (
        ImageEditBatchCase,
        ImageEditBatchRequest,
        run_image_edit_batch,
    )

    destination = shards_dir / f"shard-{shard_index:05d}"
    with TemporaryDirectory(
        dir=shards_dir,
        prefix=f".shard-{shard_index:05d}.",
        suffix=".incomplete",
    ) as temporary:
        staging = Path(temporary)
        raw_dir = staging / "raw"
        raw_dir.mkdir()
        request_cases = []
        seeds = {}
        for case_index, case in enumerate(cases):
            pair_id = str(case["id"])
            target_path = corpus_member(
                root,
                str(case["target"]),
                label=f"Qwen reference for {pair_id}",
            )
            seed = _case_seed(config, shard_index, case_index)
            seeds[pair_id] = seed
            request_cases.append(
                ImageEditBatchCase(
                    id=pair_id,
                    prompt=config.prompt,
                    image_paths=(target_path,),
                    width=config.width,
                    height=config.height,
                    seed=seed,
                    output_path=raw_dir / f"{pair_id}.png",
                )
            )
        request = ImageEditBatchRequest(
            backend=config.backend,
            cases=tuple(request_cases),
            steps=config.steps,
            guidance=config.guidance,
            strength=None,
            sampler=config.sampler,
            scheduler=config.scheduler,
        )
        started = time.monotonic()
        result = run_image_edit_batch(request, progress=progress)
        elapsed_ms = (time.monotonic() - started) * 1000
        outputs_by_case = {output.case_id: output for output in result.outputs}
        expected_ids = {str(case["id"]) for case in cases}
        if set(outputs_by_case) != expected_ids:
            raise Pix2PixError(
                f"Qwen shard {shard_index} output inventory mismatch"
            )
        outputs = []
        for case in cases:
            pair_id = str(case["id"])
            output_path = raw_dir / f"{pair_id}.png"
            mode, width, height = inspect_source_image(
                output_path,
                expected_size=(config.width, config.height),
                label="Qwen",
            )
            generated = outputs_by_case[pair_id]
            if generated.seed != seeds[pair_id]:
                raise Pix2PixError(
                    f"Qwen shard {shard_index} seed mismatch: {pair_id}"
                )
            outputs.append(
                {
                    "id": pair_id,
                    "path": f"raw/{pair_id}.png",
                    "sha256": sha256_file(output_path),
                    "size_bytes": output_path.stat().st_size,
                    "mode": mode,
                    "width": width,
                    "height": height,
                    "seed": seeds[pair_id],
                }
            )
        manifest = {
            "format": QWEN_SOURCE_SHARD_FORMAT,
            "shard_index": shard_index,
            "source_plan_sha256": source_plan_sha256,
            "elapsed_ms": elapsed_ms,
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
    config: QwenSourceConfig,
    source_plan_sha256: str,
) -> dict[str, Any]:
    manifest = read_json(shard_dir / "shard.json", label="Qwen source shard")
    require_exact_keys(
        manifest,
        {
            "format",
            "shard_index",
            "source_plan_sha256",
            "elapsed_ms",
            "outputs",
        },
        "Qwen source shard",
    )
    expected_values = {
        "format": QWEN_SOURCE_SHARD_FORMAT,
        "shard_index": shard_index,
        "source_plan_sha256": source_plan_sha256,
    }
    for key, expected in expected_values.items():
        if manifest[key] != expected:
            raise Pix2PixError(
                f"Qwen source shard {shard_index} has incompatible {key}"
            )
    elapsed_ms = manifest["elapsed_ms"]
    if (
        isinstance(elapsed_ms, bool)
        or not isinstance(elapsed_ms, (int, float))
        or elapsed_ms < 0
    ):
        raise Pix2PixError(
            f"Qwen source shard {shard_index} has invalid elapsed_ms"
        )
    outputs = manifest["outputs"]
    if not isinstance(outputs, list) or len(outputs) != len(cases):
        raise Pix2PixError(
            f"Qwen source shard {shard_index} has invalid output count"
        )
    expected_ids = [str(case["id"]) for case in cases]
    if [
        output.get("id")
        for output in outputs
        if isinstance(output, dict)
    ] != expected_ids:
        raise Pix2PixError(
            f"Qwen source shard {shard_index} output order mismatch"
        )
    for case_index, (case, output) in enumerate(
        zip(cases, outputs, strict=True)
    ):
        validate_source_output(
            shard_dir,
            output,
            pair_id=str(case["id"]),
            expected_seed=_case_seed(config, shard_index, case_index),
            expected_size=(config.width, config.height),
            label="Qwen",
        )
    return manifest


def _load_source_inventory(
    source_root: Path,
    shards: tuple[tuple[int, tuple[dict[str, Any], ...]], ...],
    *,
    config: QwenSourceConfig,
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
                raise Pix2PixError(f"duplicate Qwen source id: {pair_id}")
            inventory[pair_id] = corpus_member(
                shard_dir,
                str(output["path"]),
                label=f"Qwen source for {pair_id}",
            )
    if set(inventory) != expected_ids:
        raise Pix2PixError("Qwen source inventory differs from target selection")
    return inventory


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
            raise Pix2PixError(
                f"missing Qwen source shard manifest: {manifest_path}"
            )
        shard_records.append(
            {
                "index": shard_index,
                "path": f"shards/shard-{shard_index:05d}/shard.json",
                "sha256": sha256_file(manifest_path),
            }
        )
    return {
        "format": QWEN_SOURCE_RESULT_FORMAT,
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
        label="Qwen source result",
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
        "Qwen source result",
    )
    if result != expected:
        raise Pix2PixError(
            "Qwen source result differs from its source plan or shard manifests"
        )
    return result


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
        "kind": "Qwen-paired-source-corpus",
        "root": root.as_posix(),
        "source_plan_sha256": source_plan_sha256,
        "model_fingerprint": source_plan["model_provenance"]["fingerprint"],
        "runtime_fingerprint": source_plan["runtime_provenance"]["fingerprint"],
        "pair_count": pair_count,
        "shard_count": shard_count,
        "generated_shards": generated_shards,
        "reused_shards": reused_shards,
    }


def _case_seed(
    config: QwenSourceConfig,
    shard_index: int,
    case_index: int,
) -> int:
    return config.seed_base + shard_index * config.shard_size + case_index
