from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from aigen.generation.image_generation_requests import (
    ImageGenerationCaseRequest,
    ImageGenerationOutputRequest,
)
from aigen.manifest_io import atomic_write_json, read_json, sha256_file, write_json
from aigen.pix2pix.corpus_config import FluxSourceConfig
from aigen.pix2pix.corpus_io import corpus_member, require_exact_keys
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.source_corpus import (
    expected_source_shards,
    inspect_source_image,
    validate_source_output,
)
from aigen.progress import StatusReporter


@dataclass(frozen=True)
class FluxSourceCaseSpec:
    id: str
    prompt: str
    seed: int


@dataclass(frozen=True)
class FluxSourceLayout:
    directory: str
    shard_format: str
    result_format: str
    kind: str
    label: str


def generate_flux_source_corpus(
    root: Path,
    *,
    selected: tuple[dict[str, Any], ...],
    generation: FluxSourceConfig,
    cases: tuple[FluxSourceCaseSpec, ...],
    layout: FluxSourceLayout,
    source_plan: dict[str, Any],
    source_plan_sha256: str,
    progress: StatusReporter,
) -> dict[str, object]:
    _validate_case_specs(selected, cases)
    cases_by_id = {case.id: case for case in cases}
    shards = expected_source_shards(selected, generation.shard_size)
    source_root = root / layout.directory
    shards_dir = source_root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    result_path = source_root / "result.json"
    if result_path.exists():
        expected_result = _source_result(
            source_root,
            shards=shards,
            source_plan_sha256=source_plan_sha256,
            pair_count=len(selected),
            shard_size=generation.shard_size,
            layout=layout,
        )
        _load_source_result(
            source_root,
            expected=expected_result,
            layout=layout,
        )
        inventory = _load_source_inventory(
            source_root,
            shards,
            generation=generation,
            cases_by_id=cases_by_id,
            source_plan_sha256=source_plan_sha256,
            layout=layout,
        )
        return _generation_result(
            root,
            source_plan=source_plan,
            source_plan_sha256=source_plan_sha256,
            pair_count=len(inventory),
            shard_count=len(shards),
            generated_shards=0,
            reused_shards=len(shards),
            layout=layout,
        )

    missing = []
    reused = 0
    for shard_index, selected_cases in shards:
        shard_dir = shards_dir / f"shard-{shard_index:05d}"
        if shard_dir.exists():
            _load_source_shard(
                shard_dir,
                shard_index=shard_index,
                selected_cases=selected_cases,
                generation=generation,
                cases_by_id=cases_by_id,
                source_plan_sha256=source_plan_sha256,
                shard_format=layout.shard_format,
                label=layout.label,
            )
            reused += 1
        else:
            missing.append((shard_index, selected_cases))

    generated = 0
    if missing:
        from aigen.generation.flux2_klein import (
            Flux2KleinError,
            Flux2KleinSession,
            encode_flux2_klein_prompts,
        )

        missing_prompts = tuple(
            cases_by_id[str(selected_case["id"])].prompt
            for _, selected_cases in missing
            for selected_case in selected_cases
        )
        try:
            prompt_embeddings, _ = encode_flux2_klein_prompts(
                prompts=missing_prompts,
                progress=progress,
            )
            session = Flux2KleinSession(
                loras=(),
                sampler=generation.sampler,
                strength=None,
                progress=progress,
            )
            progress.begin(
                len(missing),
                f"generate {layout.label} source shards",
            )
            try:
                for shard_index, selected_cases in missing:
                    _generate_source_shard(
                        root,
                        shards_dir,
                        shard_index,
                        selected_cases,
                        generation=generation,
                        cases_by_id=cases_by_id,
                        source_plan_sha256=source_plan_sha256,
                        shard_format=layout.shard_format,
                        prompt_embeddings=prompt_embeddings,
                        session=session,
                        progress=progress,
                        label=layout.label,
                    )
                    generated += 1
                    progress.step(
                        f"published {layout.label} shard {shard_index:05d}"
                    )
            finally:
                session.close()
        except Flux2KleinError as error:
            raise Pix2PixError(str(error)) from error

    inventory = _load_source_inventory(
        source_root,
        shards,
        generation=generation,
        cases_by_id=cases_by_id,
        source_plan_sha256=source_plan_sha256,
        layout=layout,
    )
    result = _source_result(
        source_root,
        shards=shards,
        source_plan_sha256=source_plan_sha256,
        pair_count=len(selected),
        shard_size=generation.shard_size,
        layout=layout,
    )
    atomic_write_json(result_path, result)
    _load_source_result(
        source_root,
        expected=result,
        layout=layout,
    )
    return _generation_result(
        root,
        source_plan=source_plan,
        source_plan_sha256=source_plan_sha256,
        pair_count=len(inventory),
        shard_count=len(shards),
        generated_shards=generated,
        reused_shards=reused,
        layout=layout,
    )


def load_flux_source_corpus_inventory(
    root: Path,
    *,
    selected: tuple[dict[str, Any], ...],
    generation: FluxSourceConfig,
    cases: tuple[FluxSourceCaseSpec, ...],
    layout: FluxSourceLayout,
    source_plan_sha256: str,
) -> dict[str, Path]:
    _validate_case_specs(selected, cases)
    cases_by_id = {case.id: case for case in cases}
    shards = expected_source_shards(selected, generation.shard_size)
    source_root = root / layout.directory
    expected_result = _source_result(
        source_root,
        shards=shards,
        source_plan_sha256=source_plan_sha256,
        pair_count=len(selected),
        shard_size=generation.shard_size,
        layout=layout,
    )
    _load_source_result(
        source_root,
        expected=expected_result,
        layout=layout,
    )
    return _load_source_inventory(
        source_root,
        shards,
        generation=generation,
        cases_by_id=cases_by_id,
        source_plan_sha256=source_plan_sha256,
        layout=layout,
    )


def flux_source_corpus_result_path(
    root: Path,
    layout: FluxSourceLayout,
) -> Path:
    return root / layout.directory / "result.json"


def _validate_case_specs(
    selected: tuple[dict[str, Any], ...],
    cases: tuple[FluxSourceCaseSpec, ...],
) -> None:
    expected_ids = [str(record["id"]) for record in selected]
    case_ids = [case.id for case in cases]
    if case_ids != expected_ids:
        raise Pix2PixError("FLUX source-case order differs from target selection")
    if len(set(case_ids)) != len(case_ids):
        raise Pix2PixError("FLUX source cases contain duplicate ids")
    for case in cases:
        if not case.prompt or case.prompt != case.prompt.strip():
            raise Pix2PixError(f"invalid FLUX source prompt: {case.id}")
        if case.seed < 0:
            raise Pix2PixError(f"invalid FLUX source seed: {case.id}")


def _source_result(
    source_root: Path,
    *,
    shards: tuple[tuple[int, tuple[dict[str, Any], ...]], ...],
    source_plan_sha256: str,
    pair_count: int,
    shard_size: int,
    layout: FluxSourceLayout,
) -> dict[str, object]:
    shard_records = []
    for shard_index, _ in shards:
        manifest_path = (
            source_root / "shards" / f"shard-{shard_index:05d}" / "shard.json"
        )
        if not manifest_path.is_file():
            raise Pix2PixError(
                f"missing {layout.label} source shard manifest: {manifest_path}"
            )
        shard_records.append(
            {
                "index": shard_index,
                "path": f"shards/shard-{shard_index:05d}/shard.json",
                "sha256": sha256_file(manifest_path),
            }
        )
    return {
        "format": layout.result_format,
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
    layout: FluxSourceLayout,
) -> dict[str, Any]:
    result = read_json(
        source_root / "result.json",
        label=f"{layout.label} source result",
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
        f"{layout.label} source result",
    )
    if result != expected:
        raise Pix2PixError(
            f"{layout.label} source result differs from its source plan "
            "or shard manifests"
        )
    return result


def _load_source_inventory(
    source_root: Path,
    shards: tuple[tuple[int, tuple[dict[str, Any], ...]], ...],
    *,
    generation: FluxSourceConfig,
    cases_by_id: dict[str, FluxSourceCaseSpec],
    source_plan_sha256: str,
    layout: FluxSourceLayout,
) -> dict[str, Path]:
    inventory: dict[str, Path] = {}
    expected_ids = set()
    for shard_index, selected_cases in shards:
        shard_dir = source_root / "shards" / f"shard-{shard_index:05d}"
        manifest = _load_source_shard(
            shard_dir,
            shard_index=shard_index,
            selected_cases=selected_cases,
            generation=generation,
            cases_by_id=cases_by_id,
            source_plan_sha256=source_plan_sha256,
            shard_format=layout.shard_format,
            label=layout.label,
        )
        expected_ids.update(str(case["id"]) for case in selected_cases)
        for output in manifest["outputs"]:
            pair_id = str(output["id"])
            if pair_id in inventory:
                raise Pix2PixError(
                    f"duplicate {layout.label} source id: {pair_id}"
                )
            inventory[pair_id] = corpus_member(
                shard_dir,
                str(output["path"]),
                label=f"{layout.label} source for {pair_id}",
            )
    if set(inventory) != expected_ids:
        raise Pix2PixError(
            f"{layout.label} source inventory differs from target selection"
        )
    return inventory


def _generate_source_shard(
    root: Path,
    shards_dir: Path,
    shard_index: int,
    selected_cases: tuple[dict[str, Any], ...],
    *,
    generation: FluxSourceConfig,
    cases_by_id: dict[str, FluxSourceCaseSpec],
    source_plan_sha256: str,
    shard_format: str,
    prompt_embeddings: Any,
    session: Any,
    progress: StatusReporter,
    label: str,
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
        for selected_case in selected_cases:
            pair_id = str(selected_case["id"])
            source_case = cases_by_id[pair_id]
            target_path = corpus_member(
                root,
                str(selected_case["target"]),
                label=f"{label} reference for {pair_id}",
            )
            generation_cases.append(
                ImageGenerationCaseRequest(
                    name=pair_id,
                    prompt=source_case.prompt,
                    image_paths=(target_path,),
                    width=generation.width,
                    height=generation.height,
                    outputs=(
                        ImageGenerationOutputRequest(
                            name=pair_id,
                            seed=source_case.seed,
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
        expected_ids = {str(case["id"]) for case in selected_cases}
        if set(result_by_case) != expected_ids:
            raise Pix2PixError(
                f"{label} shard {shard_index} output inventory mismatch"
            )
        outputs = []
        for selected_case in selected_cases:
            pair_id = str(selected_case["id"])
            source_case = cases_by_id[pair_id]
            output_path = raw_dir / f"{pair_id}.png"
            mode, width, height = inspect_source_image(
                output_path,
                expected_size=(generation.width, generation.height),
                label=label,
            )
            generated = result_by_case[pair_id]
            if generated.seed != source_case.seed:
                raise Pix2PixError(
                    f"{label} shard {shard_index} seed mismatch: {pair_id}"
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
                    "seed": source_case.seed,
                }
            )
        manifest = {
            "format": shard_format,
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
            selected_cases=selected_cases,
            generation=generation,
            cases_by_id=cases_by_id,
            source_plan_sha256=source_plan_sha256,
            shard_format=shard_format,
            label=label,
        )
        os.rename(staging, destination)


def _load_source_shard(
    shard_dir: Path,
    *,
    shard_index: int,
    selected_cases: tuple[dict[str, Any], ...],
    generation: FluxSourceConfig,
    cases_by_id: dict[str, FluxSourceCaseSpec],
    source_plan_sha256: str,
    shard_format: str,
    label: str,
) -> dict[str, Any]:
    manifest = read_json(
        shard_dir / "shard.json",
        label=f"{label} source shard",
    )
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
        f"{label} source shard",
    )
    expected_values = {
        "format": shard_format,
        "shard_index": shard_index,
        "source_plan_sha256": source_plan_sha256,
    }
    for key, expected in expected_values.items():
        if manifest[key] != expected:
            raise Pix2PixError(
                f"{label} source shard {shard_index} "
                f"has incompatible {key}"
            )
    outputs = manifest["outputs"]
    if not isinstance(outputs, list) or len(outputs) != len(selected_cases):
        raise Pix2PixError(
            f"{label} source shard {shard_index} "
            "has invalid output count"
        )
    expected_ids = [str(case["id"]) for case in selected_cases]
    actual_ids = [
        output.get("id")
        for output in outputs
        if isinstance(output, dict)
    ]
    if actual_ids != expected_ids:
        raise Pix2PixError(
            f"{label} source shard {shard_index} output order mismatch"
        )
    for selected_case, output in zip(selected_cases, outputs, strict=True):
        pair_id = str(selected_case["id"])
        validate_source_output(
            shard_dir,
            output,
            pair_id=pair_id,
            expected_seed=cases_by_id[pair_id].seed,
            expected_size=(generation.width, generation.height),
            label=label,
        )
    return manifest


def _generation_result(
    root: Path,
    *,
    source_plan: dict[str, Any],
    source_plan_sha256: str,
    pair_count: int,
    shard_count: int,
    generated_shards: int,
    reused_shards: int,
    layout: FluxSourceLayout,
) -> dict[str, object]:
    return {
        "status": "completed",
        "kind": layout.kind,
        "root": root.as_posix(),
        "source_plan_sha256": source_plan_sha256,
        "model_fingerprint": source_plan["model_provenance"]["fingerprint"],
        "runtime_fingerprint": source_plan["runtime_provenance"]["fingerprint"],
        "pair_count": pair_count,
        "shard_count": shard_count,
        "generated_shards": generated_shards,
        "reused_shards": reused_shards,
    }
