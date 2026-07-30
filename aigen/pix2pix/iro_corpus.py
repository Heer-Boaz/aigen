from __future__ import annotations

import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from scipy.sparse import coo_array
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from aigen.manifest_io import read_json, sha256_file, write_json
from aigen.pix2pix.corpus_config import (
    CORPUS_SPLITS,
    IroCorpusConfig,
    IroCorpusConfigV2,
    IroCorpusConfigVersion,
    SplitName,
    corpus_config_fingerprint,
    load_iro_corpus_config,
    safe_slug,
)
from aigen.pix2pix.corpus_io import (
    corpus_member,
    read_json_records,
    require_exact_keys,
    write_json_records,
)
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.iro_coverage import (
    BodyPoseAssignment,
    BodySlot,
    assign_body_pose_cells,
    joint_pose_bounds,
)
from aigen.progress import StatusReporter


IRO_PLAN_FORMAT = "aigen.pix2pix.iro-plan.v1"
IRO_PLAN_V2_FORMAT = "aigen.pix2pix.iro-plan.v2"
IRO_RENDER_FORMAT = "aigen.pix2pix.iro-render.v1"
IRO_SELECTION_FORMAT = "aigen.pix2pix.iro-selection.v1"
IRO_SELECTION_V2_FORMAT = "aigen.pix2pix.iro-selection.v2"
IRO_EXCLUDED_COVERAGE_FORMAT = "aigen.pix2pix.iro-excluded-coverage.v2"
IRO_COVERAGE_PLANNER = "aigen.pix2pix.body-pose-milp.v1"
IRO_TARGET_MATCHER = "aigen.pix2pix.rgb-target-lapjvsp.v1"
RENDER_PAYLOAD_KEYS = {
    "gender",
    "job",
    "head",
    "headPalette",
    "headdir",
    "headgear",
    "garment",
    "bodyPalette",
    "madogearType",
    "action",
    "canvas",
    "outfit",
}
REQUEST_RECORD_KEYS = {
    "id",
    "group",
    "identity",
    "split",
    "lineage",
    "species",
    "job_id",
    "job_name",
    "gender",
    "head",
    "head_palette",
    "action_name",
    "action_base",
    "direction",
    "payload",
}
REQUEST_RECORD_V2_KEYS = (REQUEST_RECORD_KEYS - {"identity"}) | {
    "head_identity",
    "body_variant_id",
    "coverage_id",
    "rig_family",
    "requested_rig_pose_id",
}
SELECTION_RECORD_KEYS = REQUEST_RECORD_KEYS | {
    "renderer_frame_index",
    "duration_ms",
    "renderer_pixel_sha256",
    "flux_seed",
    "target",
    "target_sha256",
}
SELECTION_RECORD_V2_KEYS = REQUEST_RECORD_V2_KEYS | {
    "renderer_frame_index",
    "renderer_frame_sha256",
    "renderer_result_sha256",
    "duration_ms",
    "renderer_pixel_sha256",
    "flux_seed",
    "target",
    "target_sha256",
    "target_instance_id",
    "realized_rig_pose_id",
    "target_pixel_sha256",
}


def _load_excluded_coverage(
    config: IroCorpusConfigVersion,
    roots: tuple[Path, ...],
) -> dict[str, Any]:
    if roots and not isinstance(config, IroCorpusConfigV2):
        raise Pix2PixError(
            "coverage exclusion requires an aigen.pix2pix.iro-corpus.v2 config"
        )
    sources = []
    cells: dict[BodyPoseAssignment, tuple[str, str]] = {}
    source_request_hashes: set[str] = set()
    namespace_sha256 = _body_pose_namespace_sha256(config)
    current_jobs_by_id = {job.id: job for job in config.jobs}
    if isinstance(config, IroCorpusConfigV2):
        current_bodies = {
            (job.id, body.gender)
            for job in config.jobs
            for body in job.bodies
        }
    else:
        current_bodies = {
            (job.id, gender)
            for job in config.jobs
            for gender in job.genders
        }
    for root in roots:
        source_config, requests, manifest = load_iro_plan(root)
        if _body_pose_namespace_sha256(source_config) != namespace_sha256:
            raise Pix2PixError(
                f"excluded iRO plan uses a different renderer/action namespace: "
                f"{root.resolve()}"
            )
        requests_sha256 = _string(manifest, "requests_sha256")
        if requests_sha256 in source_request_hashes:
            raise Pix2PixError(
                f"duplicate excluded iRO coverage source: {root.resolve()}"
            )
        source_request_hashes.add(requests_sha256)
        sources.append(
            {
                "format": _string(manifest, "format"),
                "name": _string(manifest, "name"),
                "config_fingerprint": _string(
                    manifest,
                    "config_fingerprint",
                ),
                "requests_sha256": requests_sha256,
                "request_count": _integer(manifest, "request_count"),
            }
        )
        for record in requests:
            assignment = BodyPoseAssignment(
                job_id=_integer(record, "job_id"),
                gender=_gender(record, "gender"),
                action_base=_integer(record, "action_base"),
                direction=_integer(record, "direction"),
            )
            semantics = (
                _string(record, "job_name"),
                _string(record, "species"),
            )
            previous = cells.setdefault(assignment, semantics)
            if previous != semantics:
                raise Pix2PixError(
                    "excluded iRO plans disagree about renderer-job semantics "
                    f"for job {assignment.job_id}/gender {assignment.gender}"
                )
            current_job = current_jobs_by_id.get(assignment.job_id)
            if (
                current_job is not None
                and (assignment.job_id, assignment.gender) in current_bodies
                and (current_job.name, current_job.species) != semantics
            ):
                raise Pix2PixError(
                    "excluded iRO plan uses different renderer-job semantics "
                    f"for job {assignment.job_id}/gender {assignment.gender}"
                )
    return {
        "format": IRO_EXCLUDED_COVERAGE_FORMAT,
        "namespace_sha256": namespace_sha256,
        "sources": sorted(
            sources,
            key=lambda source: str(source["requests_sha256"]),
        ),
        "cells": [
            {
                "job_id": cell.job_id,
                "gender": cell.gender,
                "action_base": cell.action_base,
                "direction": cell.direction,
                "job_name": cells[cell][0],
                "species": cells[cell][1],
            }
            for cell in sorted(cells)
        ],
    }


def _audit_excluded_coverage(
    manifest: dict[str, Any],
    config: IroCorpusConfigV2,
) -> dict[str, Any]:
    payload = manifest["excluded_coverage"]
    if not isinstance(payload, dict):
        raise Pix2PixError("iRO excluded coverage must be an object")
    require_exact_keys(
        payload,
        {"format", "namespace_sha256", "sources", "cells"},
        "iRO excluded coverage",
    )
    if payload["format"] != IRO_EXCLUDED_COVERAGE_FORMAT:
        raise Pix2PixError(
            f"unsupported iRO excluded-coverage format: {payload['format']!r}"
        )
    if _string(payload, "namespace_sha256") != _body_pose_namespace_sha256(
        config
    ):
        raise Pix2PixError(
            "iRO excluded coverage uses a different renderer/action namespace"
        )
    sources = payload["sources"]
    cells = payload["cells"]
    if not isinstance(sources, list) or not isinstance(cells, list):
        raise Pix2PixError("iRO excluded coverage sources and cells must be arrays")
    current_jobs_by_body = {
        (job.id, body.gender): job
        for job in config.jobs
        for body in job.bodies
    }
    previous_source_hash = ""
    source_hashes: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise Pix2PixError(f"invalid excluded coverage source {index}")
        require_exact_keys(
            source,
            {
                "format",
                "name",
                "config_fingerprint",
                "requests_sha256",
                "request_count",
            },
            f"excluded coverage source {index}",
        )
        request_hash = _string(source, "requests_sha256")
        if request_hash in source_hashes or request_hash < previous_source_hash:
            raise Pix2PixError(
                "excluded coverage sources are duplicated or not canonical"
            )
        source_hashes.add(request_hash)
        previous_source_hash = request_hash
        _string(source, "format")
        _string(source, "name")
        _string(source, "config_fingerprint")
        _integer(source, "request_count")
    normalized_cells = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise Pix2PixError(f"invalid excluded coverage cell {index}")
        require_exact_keys(
            cell,
            {
                "job_id",
                "gender",
                "action_base",
                "direction",
                "job_name",
                "species",
            },
            f"excluded coverage cell {index}",
        )
        normalized_cells.append(
            (
                BodyPoseAssignment(
                    job_id=_integer(cell, "job_id"),
                    gender=_gender(cell, "gender"),
                    action_base=_integer(cell, "action_base"),
                    direction=_integer(cell, "direction"),
                ),
                _string(cell, "job_name"),
                _string(cell, "species"),
            )
        )
        assignment, job_name, species = normalized_cells[-1]
        current_job = current_jobs_by_body.get(
            (assignment.job_id, assignment.gender)
        )
        if (
            current_job is not None
            and (current_job.name, current_job.species)
            != (job_name, species)
        ):
            raise Pix2PixError(
                "excluded iRO coverage uses different renderer-job semantics "
                f"for job {assignment.job_id}/gender {assignment.gender}"
            )
    if tuple(normalized_cells) != tuple(sorted(set(normalized_cells))):
        raise Pix2PixError("excluded coverage cells are duplicated or not canonical")
    if sum(_integer(source, "request_count") for source in sources) != _integer(
        manifest, "excluded_source_request_count"
    ):
        raise Pix2PixError(
            "iRO excluded source-request count does not match manifest"
        )
    if len(cells) != _integer(
        manifest, "excluded_source_body_pose_cell_count"
    ):
        raise Pix2PixError(
            "iRO excluded source body-pose count does not match manifest"
        )
    if _applicable_excluded_body_pose_cell_count(
        config,
        payload,
    ) != _integer(manifest, "applicable_excluded_body_pose_cell_count"):
        raise Pix2PixError(
            "iRO applicable excluded body-pose count does not match manifest"
        )
    if _canonical_sha256(payload) != _string(
        manifest, "excluded_coverage_sha256"
    ):
        raise Pix2PixError("iRO excluded coverage checksum mismatch")
    return payload


def _applicable_excluded_body_pose_cell_count(
    config: IroCorpusConfigV2,
    excluded_coverage: dict[str, Any],
) -> int:
    jobs_by_id = {job.id: job for job in config.jobs}
    configured_bodies = {
        (job.id, body.gender)
        for job in config.jobs
        for body in job.bodies
    }
    count = 0
    for cell in excluded_coverage["cells"]:
        job_id = _integer(cell, "job_id")
        gender = _gender(cell, "gender")
        job = jobs_by_id.get(job_id)
        if job is None or (job_id, gender) not in configured_bodies:
            continue
        if (
            job.name != _string(cell, "job_name")
            or job.species != _string(cell, "species")
        ):
            raise Pix2PixError(
                "excluded iRO coverage uses different renderer-job semantics "
                f"for job {job_id}/gender {gender}"
            )
        split = config.split_group_splits[job.split_group]
        axes = config.split_axis_quotas[split]
        if (
            axes.actions.get(_integer(cell, "action_base"), 0) > 0
            and axes.directions.get(_integer(cell, "direction"), 0) > 0
        ):
            count += 1
    return count


def _audit_coverage_exclusions(
    requests: tuple[dict[str, Any], ...],
    excluded_coverage: dict[str, Any],
) -> None:
    forbidden = {
        (
            _integer(cell, "job_id"),
            _gender(cell, "gender"),
            _integer(cell, "action_base"),
            _integer(cell, "direction"),
        )
        for cell in excluded_coverage["cells"]
    }
    overlap = [
        str(record["coverage_id"])
        for record in requests
        if (
            _integer(record, "job_id"),
            _gender(record, "gender"),
            _integer(record, "action_base"),
            _integer(record, "direction"),
        )
        in forbidden
    ]
    if overlap:
        raise Pix2PixError(
            "iRO corpus repeats excluded coverage cells: "
            + ", ".join(overlap[:8])
        )


def plan_iro_corpus(
    config_path: Path,
    output_dir: Path,
    *,
    exclude_coverage_from: tuple[Path, ...] = (),
) -> dict[str, object]:
    config = load_iro_corpus_config(config_path)
    excluded_coverage = _load_excluded_coverage(config, exclude_coverage_from)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        existing_config, requests, manifest = load_iro_plan(output_dir)
        if corpus_config_fingerprint(existing_config) != corpus_config_fingerprint(config):
            raise Pix2PixError(
                f"existing iRO corpus plan uses a different config: {output_dir}"
            )
        if isinstance(config, IroCorpusConfigV2):
            if manifest["excluded_coverage"] != excluded_coverage:
                raise Pix2PixError(
                    "existing iRO corpus plan uses different excluded coverage"
                )
        return _plan_result(output_dir, requests, manifest, reused=True)

    requests = _planned_requests(config, excluded_coverage)
    _audit_request_records(config, requests)
    _audit_request_quotas(config, requests)
    if isinstance(config, IroCorpusConfigV2):
        _audit_coverage_exclusions(requests, excluded_coverage)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.",
        suffix=".incomplete",
    ) as temporary:
        staging = Path(temporary)
        normalized_config = config.model_dump(mode="json")
        write_json(staging / "config.json", normalized_config)
        write_json_records(staging / "requests.jsonl", requests)
        manifest = _plan_manifest(
            config,
            requests,
            staging,
            excluded_coverage=excluded_coverage,
        )
        write_json(staging / "plan.json", manifest)
        os.rename(staging, output_dir)
    return _plan_result(output_dir, requests, manifest, reused=False)


def load_iro_plan(
    root: Path,
) -> tuple[IroCorpusConfigVersion, tuple[dict[str, Any], ...], dict[str, Any]]:
    root = root.expanduser().resolve()
    manifest = read_json(root / "plan.json", label="iRO corpus plan")
    plan_format = manifest.get("format")
    if plan_format == IRO_PLAN_FORMAT:
        manifest_keys = {
            "format",
            "name",
            "config",
            "config_fingerprint",
            "requests",
            "requests_sha256",
            "request_count",
            "split_counts",
            "lineage_counts",
            "identity_count",
        }
    elif plan_format == IRO_PLAN_V2_FORMAT:
        manifest_keys = {
            "format",
            "name",
            "config",
            "config_fingerprint",
            "requests",
            "requests_sha256",
            "request_count",
            "split_counts",
            "lineage_counts",
            "head_identity_count",
            "head_identity_overlap_counts",
            "split_group_counts",
            "rig_family_counts",
            "joint_pose_counts",
            "body_pose_cell_count",
            "body_pose_ids_sha256",
            "requested_rig_pose_count",
            "requested_rig_pose_ids_sha256",
            "coverage_planner",
            "coverage_planner_runtime",
            "excluded_coverage",
            "excluded_source_request_count",
            "excluded_source_body_pose_cell_count",
            "applicable_excluded_body_pose_cell_count",
            "excluded_coverage_sha256",
        }
    else:
        raise Pix2PixError(f"unsupported iRO corpus plan format: {plan_format!r}")
    require_exact_keys(manifest, manifest_keys, "iRO corpus plan")
    config_path = corpus_member(root, _string(manifest, "config"), label="plan config")
    requests_path = corpus_member(
        root,
        _string(manifest, "requests"),
        label="plan requests",
    )
    if not config_path.is_file() or not requests_path.is_file():
        raise Pix2PixError(f"incomplete iRO corpus plan: {root}")
    config = load_iro_corpus_config(config_path)
    if manifest["name"] != config.name:
        raise Pix2PixError("iRO corpus plan name differs from its config")
    if corpus_config_fingerprint(config) != _string(manifest, "config_fingerprint"):
        raise Pix2PixError(f"iRO corpus config fingerprint mismatch: {config_path}")
    if sha256_file(requests_path) != _string(manifest, "requests_sha256"):
        raise Pix2PixError(f"iRO corpus request manifest checksum mismatch: {requests_path}")
    requests = read_json_records(requests_path, label="iRO corpus request manifest")
    if len(requests) != _integer(manifest, "request_count"):
        raise Pix2PixError("iRO corpus request count does not match plan manifest")
    _audit_request_records(config, requests)
    _audit_request_quotas(config, requests)
    if _split_counts(requests) != manifest["split_counts"]:
        raise Pix2PixError("iRO corpus split counts do not match plan manifest")
    if _lineage_counts(requests) != manifest["lineage_counts"]:
        raise Pix2PixError("iRO corpus lineage counts do not match plan manifest")
    if isinstance(config, IroCorpusConfigV2):
        if plan_format != IRO_PLAN_V2_FORMAT:
            raise Pix2PixError("iRO corpus v2 config requires a v2 plan")
        excluded_coverage = _audit_excluded_coverage(manifest, config)
        if manifest["coverage_planner"] != IRO_COVERAGE_PLANNER:
            raise Pix2PixError(
                f"unsupported iRO coverage planner: "
                f"{manifest['coverage_planner']!r}"
            )
        planner_runtime = manifest["coverage_planner_runtime"]
        if not isinstance(planner_runtime, dict):
            raise Pix2PixError("iRO coverage planner runtime must be an object")
        require_exact_keys(
            planner_runtime,
            {"numpy", "scipy"},
            "iRO coverage planner runtime",
        )
        _string(planner_runtime, "numpy")
        _string(planner_runtime, "scipy")
        _audit_coverage_exclusions(requests, excluded_coverage)
        if _split_group_counts(requests) != manifest["split_group_counts"]:
            raise Pix2PixError(
                "iRO corpus split-group counts do not match plan manifest"
            )
        if _rig_family_counts(requests) != manifest["rig_family_counts"]:
            raise Pix2PixError(
                "iRO corpus rig-family counts do not match plan manifest"
            )
        if _joint_pose_counts(requests) != manifest["joint_pose_counts"]:
            raise Pix2PixError(
                "iRO corpus joint-pose counts do not match plan manifest"
            )
        body_pose_ids = [str(record["coverage_id"]) for record in requests]
        if len(set(body_pose_ids)) != _integer(
            manifest, "body_pose_cell_count"
        ):
            raise Pix2PixError(
                "iRO corpus body-pose count does not match plan manifest"
            )
        if _canonical_sha256(body_pose_ids) != _string(
            manifest, "body_pose_ids_sha256"
        ):
            raise Pix2PixError("iRO corpus body-pose checksum mismatch")
        requested_rig_pose_ids = sorted(
            {str(record["requested_rig_pose_id"]) for record in requests}
        )
        if len(requested_rig_pose_ids) != _integer(
            manifest, "requested_rig_pose_count"
        ):
            raise Pix2PixError(
                "iRO corpus requested-rig-pose count does not match plan manifest"
            )
        if _canonical_sha256(requested_rig_pose_ids) != _string(
            manifest, "requested_rig_pose_ids_sha256"
        ):
            raise Pix2PixError("iRO corpus requested-rig-pose checksum mismatch")
        head_identities = {
            str(record["head_identity"]) for record in requests
        }
        if len(head_identities) != _integer(manifest, "head_identity_count"):
            raise Pix2PixError(
                "iRO corpus head-identity count does not match plan manifest"
            )
        if (
            _head_identity_overlap_counts(requests)
            != manifest["head_identity_overlap_counts"]
        ):
            raise Pix2PixError(
                "iRO corpus head-identity overlaps do not match plan manifest"
            )
    elif plan_format != IRO_PLAN_FORMAT:
        raise Pix2PixError("iRO corpus v1 config requires a v1 plan")
    elif len({record["identity"] for record in requests}) != _integer(
        manifest, "identity_count"
    ):
        raise Pix2PixError("iRO corpus identity count does not match plan manifest")
    return config, requests, manifest


def render_iro_corpus(
    root: Path,
    *,
    progress: StatusReporter,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    config, requests, plan = load_iro_plan(root)
    renders_dir = root / "renders"
    renders_dir.mkdir(exist_ok=True)
    missing = []
    reused = 0
    progress.begin(len(requests), "verify iRO renderer artifacts")
    for record in requests:
        request_dir = renders_dir / str(record["id"])
        if request_dir.exists():
            _load_render_result(config, record, request_dir)
            reused += 1
            progress.step(f"verified {record['id']}")
        else:
            missing.append(record)

    rendered = 0
    if missing:
        progress.phase("render native iRO APNG targets")
        with ThreadPoolExecutor(max_workers=config.render_workers) as executor:
            futures = {
                executor.submit(
                    _render_request,
                    config,
                    record,
                    renders_dir,
                ): record
                for record in missing
            }
            for future in as_completed(futures):
                record = futures[future]
                future.result()
                rendered += 1
                progress.step(f"rendered {record['id']}")

    return {
        "status": "completed",
        "kind": "iRO-native-render-corpus",
        "root": root.as_posix(),
        "config_fingerprint": plan["config_fingerprint"],
        "request_count": len(requests),
        "rendered": rendered,
        "reused": reused,
    }


def select_iro_targets(
    root: Path,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    config, requests, plan = load_iro_plan(root)
    selection_dir = root / "selection"
    if selection_dir.exists():
        _, selected, manifest = load_iro_selection(root)
        return _selection_result(root, selected, manifest, reused=True)

    if isinstance(config, IroCorpusConfigV2):
        selected_frames = _select_unique_v2_target_frames(
            root,
            config,
            requests,
        )
    else:
        selected_frames = []
        seen_pixels: set[str] = set()
        for request_index, record in enumerate(requests):
            request_dir = root / "renders" / str(record["id"])
            result = _load_render_result(config, record, request_dir)
            candidates = [
                frame
                for frame in result["frames"]
                if _selection_frame_safe(request_dir / str(frame["path"]))
                and frame["pixel_sha256"] not in seen_pixels
            ]
            if not candidates:
                raise Pix2PixError(
                    "iRO request has no unique canvas-safe APNG frame: "
                    f"{record['id']}"
                )
            selected = min(
                candidates,
                key=lambda frame: _frame_rank(config, record, frame),
            )
            seen_pixels.add(str(selected["pixel_sha256"]))
            selected_frames.append((record, result, selected, request_index))

    selection_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=selection_dir.parent,
        prefix=".selection.",
        suffix=".incomplete",
    ) as temporary:
        staging = Path(temporary)
        targets_dir = staging / "targets"
        targets_dir.mkdir()
        selected_records = []
        for record, _, frame, request_index in selected_frames:
            pair_id = str(record["id"])
            source_frame = root / "renders" / pair_id / str(frame["path"])
            target_path = targets_dir / f"{pair_id}.png"
            _composite_white_target(source_frame, target_path, config.image_size)
            selected_record = {
                **record,
                "renderer_frame_index": frame["index"],
                "duration_ms": frame["duration_ms"],
                "renderer_pixel_sha256": frame["pixel_sha256"],
                "flux_seed": config.flux.seed_base + request_index,
                "target": f"selection/targets/{pair_id}.png",
                "target_sha256": sha256_file(target_path),
            }
            if isinstance(config, IroCorpusConfigV2):
                renderer_result_sha256 = sha256_file(
                    root / "renders" / pair_id / "result.json"
                )
                selected_record.update(
                    {
                        "renderer_frame_sha256": frame["sha256"],
                        "renderer_result_sha256": renderer_result_sha256,
                        "target_pixel_sha256": frame["target_pixel_sha256"],
                        "target_instance_id": _target_instance_id(
                            record,
                            renderer_result_sha256,
                            int(frame["index"]),
                            str(frame["sha256"]),
                            str(frame["pixel_sha256"]),
                        ),
                        "realized_rig_pose_id": (
                            f"{record['requested_rig_pose_id']}-"
                            f"f{frame['index']:03d}"
                        ),
                    }
                )
            selected_records.append(selected_record)
        write_json_records(staging / "selected.jsonl", selected_records)
        manifest: dict[str, Any] = {
            "format": (
                IRO_SELECTION_V2_FORMAT
                if isinstance(config, IroCorpusConfigV2)
                else IRO_SELECTION_FORMAT
            ),
            "name": config.name,
            "config_fingerprint": plan["config_fingerprint"],
            "plan_requests_sha256": plan["requests_sha256"],
            "selected": "selected.jsonl",
            "selected_sha256": sha256_file(staging / "selected.jsonl"),
            "pair_count": len(selected_records),
            "split_counts": _split_counts(selected_records),
            "lineage_counts": _lineage_counts(selected_records),
        }
        if isinstance(config, IroCorpusConfigV2):
            manifest.update(
                {
                    "head_identity_count": len(
                        {
                            record["head_identity"]
                            for record in selected_records
                        }
                    ),
                    "head_identity_overlap_counts": (
                        _head_identity_overlap_counts(selected_records)
                    ),
                    "split_group_counts": _split_group_counts(selected_records),
                    "rig_family_counts": _rig_family_counts(
                        selected_records
                    ),
                    "joint_pose_counts": _joint_pose_counts(selected_records),
                    "body_pose_cell_count": len(
                        {record["coverage_id"] for record in selected_records}
                    ),
                    "requested_rig_pose_count": len(
                        {
                            record["requested_rig_pose_id"]
                            for record in selected_records
                        }
                    ),
                    "target_instance_count": len(
                        {
                            record["target_instance_id"]
                            for record in selected_records
                        }
                    ),
                    "realized_rig_pose_count": len(
                        {
                            record["realized_rig_pose_id"]
                            for record in selected_records
                        }
                    ),
                    "target_pixel_count": len(
                        {
                            record["target_pixel_sha256"]
                            for record in selected_records
                        }
                    ),
                    "target_matcher": IRO_TARGET_MATCHER,
                    "target_matcher_runtime": {
                        "numpy": version("numpy"),
                        "scipy": version("scipy"),
                    },
                }
            )
        else:
            manifest["identity_count"] = len(
                {record["identity"] for record in selected_records}
            )
        write_json(staging / "selection.json", manifest)
        os.rename(staging, selection_dir)
    _, selected, loaded_manifest = load_iro_selection(root)
    return _selection_result(root, selected, loaded_manifest, reused=False)


def load_iro_selection(
    root: Path,
) -> tuple[IroCorpusConfigVersion, tuple[dict[str, Any], ...], dict[str, Any]]:
    root = root.expanduser().resolve()
    config, requests, plan = load_iro_plan(root)
    manifest_path = root / "selection" / "selection.json"
    manifest = read_json(manifest_path, label="iRO target selection")
    selection_format = manifest.get("format")
    if selection_format == IRO_SELECTION_FORMAT:
        manifest_keys = {
            "format",
            "name",
            "config_fingerprint",
            "plan_requests_sha256",
            "selected",
            "selected_sha256",
            "pair_count",
            "split_counts",
            "lineage_counts",
            "identity_count",
        }
    elif selection_format == IRO_SELECTION_V2_FORMAT:
        manifest_keys = {
            "format",
            "name",
            "config_fingerprint",
            "plan_requests_sha256",
            "selected",
            "selected_sha256",
            "pair_count",
            "split_counts",
            "lineage_counts",
            "head_identity_count",
            "head_identity_overlap_counts",
            "split_group_counts",
            "rig_family_counts",
            "joint_pose_counts",
            "body_pose_cell_count",
            "requested_rig_pose_count",
            "target_instance_count",
            "realized_rig_pose_count",
            "target_pixel_count",
            "target_matcher",
            "target_matcher_runtime",
        }
    else:
        raise Pix2PixError(
            f"unsupported iRO target selection format: {selection_format!r}"
        )
    require_exact_keys(manifest, manifest_keys, "iRO target selection")
    if manifest["name"] != config.name:
        raise Pix2PixError("iRO target selection name differs from its config")
    if isinstance(config, IroCorpusConfigV2):
        if selection_format != IRO_SELECTION_V2_FORMAT:
            raise Pix2PixError("iRO corpus v2 config requires a v2 selection")
        if manifest["target_matcher"] != IRO_TARGET_MATCHER:
            raise Pix2PixError(
                f"unsupported iRO target matcher: "
                f"{manifest['target_matcher']!r}"
            )
        matcher_runtime = manifest["target_matcher_runtime"]
        if not isinstance(matcher_runtime, dict):
            raise Pix2PixError("iRO target matcher runtime must be an object")
        require_exact_keys(
            matcher_runtime,
            {"numpy", "scipy"},
            "iRO target matcher runtime",
        )
        _string(matcher_runtime, "numpy")
        _string(matcher_runtime, "scipy")
        request_keys = REQUEST_RECORD_V2_KEYS
        selection_keys = SELECTION_RECORD_V2_KEYS
    else:
        if selection_format != IRO_SELECTION_FORMAT:
            raise Pix2PixError("iRO corpus v1 config requires a v1 selection")
        request_keys = REQUEST_RECORD_KEYS
        selection_keys = SELECTION_RECORD_KEYS
    if manifest["config_fingerprint"] != plan["config_fingerprint"]:
        raise Pix2PixError("iRO target selection config fingerprint mismatch")
    if manifest["plan_requests_sha256"] != plan["requests_sha256"]:
        raise Pix2PixError("iRO target selection plan fingerprint mismatch")
    selected_path = corpus_member(
        root / "selection",
        _string(manifest, "selected"),
        label="selected target manifest",
    )
    if not selected_path.is_file():
        raise Pix2PixError(f"missing selected target manifest: {selected_path}")
    if sha256_file(selected_path) != _string(manifest, "selected_sha256"):
        raise Pix2PixError("selected target manifest checksum mismatch")
    selected = read_json_records(selected_path, label="selected target manifest")
    if len(selected) != len(requests) or len(selected) != _integer(
        manifest, "pair_count"
    ):
        raise Pix2PixError("selected target count does not match the corpus plan")
    seen_pixels: set[str] = set()
    target_instance_ids: set[str] = set()
    target_pixel_hashes: set[str] = set()
    for index, record in enumerate(selected):
        require_exact_keys(record, selection_keys, f"selected target {index}")
        pair_id = _string(record, "id")
        planned = requests[index]
        if pair_id != planned["id"]:
            raise Pix2PixError(
                "selected target order differs from the corpus plan: "
                f"{pair_id}"
            )
        if {key: record[key] for key in request_keys} != planned:
            raise Pix2PixError(f"selected target metadata differs from plan: {pair_id}")
        renderer_pixel = _string(record, "renderer_pixel_sha256")
        if renderer_pixel in seen_pixels:
            raise Pix2PixError(f"duplicate renderer pixels in selection: {pair_id}")
        seen_pixels.add(renderer_pixel)
        if _integer(record, "flux_seed") != config.flux.seed_base + index:
            raise Pix2PixError(f"selected target has an invalid FLUX seed: {pair_id}")
        target_path = corpus_member(
            root,
            _string(record, "target"),
            label=f"target for {pair_id}",
        )
        target_pixel_hash = _verify_rgb_image(
            target_path,
            config.image_size,
            config.image_size,
        )
        if sha256_file(target_path) != _string(record, "target_sha256"):
            raise Pix2PixError(f"selected target checksum mismatch: {pair_id}")
        if isinstance(config, IroCorpusConfigV2):
            composited_frame_hash = _verify_selected_render_frame(
                root,
                config,
                planned,
                record,
            )
            if (
                target_pixel_hash
                != _string(record, "target_pixel_sha256")
                or target_pixel_hash != composited_frame_hash
            ):
                raise Pix2PixError(
                    f"selected target pixel checksum mismatch: {pair_id}"
                )
            if target_pixel_hash in target_pixel_hashes:
                raise Pix2PixError(
                    f"duplicate composited target pixels in selection: {pair_id}"
                )
            target_pixel_hashes.add(target_pixel_hash)
            expected_target_instance = _target_instance_id(
                record,
                _string(record, "renderer_result_sha256"),
                _integer(record, "renderer_frame_index"),
                _string(record, "renderer_frame_sha256"),
                _string(record, "renderer_pixel_sha256"),
            )
            if record["target_instance_id"] != expected_target_instance:
                raise Pix2PixError(
                    f"selected target instance id mismatch: {pair_id}"
                )
            target_instance = _string(record, "target_instance_id")
            if target_instance in target_instance_ids:
                raise Pix2PixError(
                    f"duplicate target instance id in selection: {target_instance}"
                )
            target_instance_ids.add(target_instance)
            expected_realized = (
                f"{record['requested_rig_pose_id']}-"
                f"f{record['renderer_frame_index']:03d}"
            )
            if record["realized_rig_pose_id"] != expected_realized:
                raise Pix2PixError(
                    f"selected target realized rig pose mismatch: {pair_id}"
                )
    if _split_counts(selected) != manifest["split_counts"]:
        raise Pix2PixError("selected target split counts do not match manifest")
    if _lineage_counts(selected) != manifest["lineage_counts"]:
        raise Pix2PixError("selected target lineage counts do not match manifest")
    if isinstance(config, IroCorpusConfigV2):
        if len(
            {record["head_identity"] for record in selected}
        ) != _integer(manifest, "head_identity_count"):
            raise Pix2PixError(
                "selected head-identity count does not match manifest"
            )
        if (
            _head_identity_overlap_counts(selected)
            != manifest["head_identity_overlap_counts"]
        ):
            raise Pix2PixError(
                "selected head-identity overlaps do not match manifest"
            )
        if _split_group_counts(selected) != manifest["split_group_counts"]:
            raise Pix2PixError(
                "selected target split-group counts do not match manifest"
            )
        if _rig_family_counts(selected) != manifest["rig_family_counts"]:
            raise Pix2PixError(
                "selected target rig-family counts do not match manifest"
            )
        if _joint_pose_counts(selected) != manifest["joint_pose_counts"]:
            raise Pix2PixError(
                "selected target joint-pose counts do not match manifest"
            )
        if len({record["coverage_id"] for record in selected}) != _integer(
            manifest, "body_pose_cell_count"
        ):
            raise Pix2PixError(
                "selected target body-pose count does not match manifest"
            )
        if len(
            {record["requested_rig_pose_id"] for record in selected}
        ) != _integer(manifest, "requested_rig_pose_count"):
            raise Pix2PixError(
                "selected requested-rig-pose count does not match manifest"
            )
        if len(target_instance_ids) != _integer(
            manifest, "target_instance_count"
        ):
            raise Pix2PixError(
                "selected target-instance count does not match manifest"
            )
        if len({record["realized_rig_pose_id"] for record in selected}) != _integer(
            manifest, "realized_rig_pose_count"
        ):
            raise Pix2PixError(
                "selected realized-rig-pose count does not match manifest"
            )
        if len(target_pixel_hashes) != _integer(
            manifest,
            "target_pixel_count",
        ):
            raise Pix2PixError(
                "selected target-pixel count does not match manifest"
            )
    elif len({record["identity"] for record in selected}) != _integer(
        manifest, "identity_count"
    ):
        raise Pix2PixError("selected target identity count does not match manifest")
    return config, selected, manifest


def _planned_requests(
    config: IroCorpusConfigVersion,
    excluded_coverage: dict[str, Any],
) -> tuple[dict[str, object], ...]:
    if isinstance(config, IroCorpusConfigV2):
        return _planned_requests_v2(config, excluded_coverage)
    return _planned_requests_v1(config)


def _planned_requests_v1(
    config: IroCorpusConfig,
) -> tuple[dict[str, object], ...]:
    action_name_by_base = {action.base: action.name for action in config.actions}
    jobs_by_lineage_gender = {
        (lineage, gender): _stable_order(
            [
                job
                for job in config.jobs
                if job.lineage == lineage and gender in job.genders
            ],
            config.identity_seed,
            f"jobs:{lineage}:{gender}",
            key=lambda job: str(job.id),
        )
        for lineage in config.lineage_splits
        for gender in (0, 1)
    }
    body_records = []
    for quota in config.lineage_pair_quotas:
        for gender, count in ((0, quota.female), (1, quota.male)):
            jobs = jobs_by_lineage_gender[(quota.lineage, gender)]
            for occurrence in range(count):
                job = jobs[occurrence % len(jobs)]
                body_records.append(
                    {
                        "group": quota.lineage,
                        "split": quota.split,
                        "lineage": quota.lineage,
                        "species": job.species,
                        "job_id": job.id,
                        "job_name": job.name,
                        "gender": gender,
                    }
                )

    requests = []
    for split in CORPUS_SPLITS:
        split_bodies = _stable_order(
            [record for record in body_records if record["split"] == split],
            config.identity_seed,
            f"bodies:{split}",
            key=lambda record: (
                f"{record['lineage']}:{record['job_id']}:{record['gender']}"
            ),
        )
        axes = config.split_axis_quotas[split]
        directions = _quota_sequence(
            axes.directions,
            config.identity_seed,
            f"directions:{split}",
        )
        palettes = _quota_sequence(
            axes.head_palettes,
            config.identity_seed,
            f"palettes:{split}",
        )
        actions = _quota_sequence(
            axes.actions,
            config.identity_seed,
            f"actions:{split}",
        )
        species_occurrences: Counter[str] = Counter()
        for split_index, (body, direction, palette, action_base) in enumerate(
            zip(split_bodies, directions, palettes, actions, strict=True)
        ):
            species = str(body["species"])
            heads = _stable_order(
                list(axes.heads_by_species[species]),
                config.identity_seed,
                f"heads:{split}:{species}",
                key=str,
            )
            head = heads[species_occurrences[species] % len(heads)]
            species_occurrences[species] += 1
            identity = f"{species}-g{body['gender']}-h{head:02d}"
            pair_id = (
                f"{split}-{split_index:04d}-"
                f"{safe_slug(str(body['job_name']))}-{body['job_id']}-"
                f"g{body['gender']}-h{head:02d}-p{palette:02d}-"
                f"a{action_base:03d}-d{direction}"
            )
            payload = {
                "gender": body["gender"],
                "job": [str(body["job_id"])],
                "head": head,
                "headPalette": palette,
                "headdir": config.defaults.head_direction,
                "headgear": list(config.defaults.headgear),
                "garment": config.defaults.garment,
                "bodyPalette": config.defaults.body_palette,
                "madogearType": config.defaults.madogear_type,
                "action": action_base + direction,
                "canvas": config.canvas,
                "outfit": config.defaults.outfit,
            }
            requests.append(
                {
                    "id": pair_id,
                    **body,
                    "identity": identity,
                    "head": head,
                    "head_palette": palette,
                    "action_name": action_name_by_base[action_base],
                    "action_base": action_base,
                    "direction": direction,
                    "payload": payload,
                }
            )
    return tuple(requests)


def _planned_requests_v2(
    config: IroCorpusConfigV2,
    excluded_coverage: dict[str, Any],
) -> tuple[dict[str, object], ...]:
    action_name_by_base = {action.base: action.name for action in config.actions}
    jobs_by_id = {job.id: job for job in config.jobs}
    bodies_by_job_gender = {
        (job.id, body.gender): body
        for job in config.jobs
        for body in job.bodies
    }
    body_slots_by_split = _body_slots_by_split_v2(config)

    forbidden = tuple(
        BodyPoseAssignment(
            job_id=_integer(cell, "job_id"),
            gender=_gender(cell, "gender"),
            action_base=_integer(cell, "action_base"),
            direction=_integer(cell, "direction"),
        )
        for cell in excluded_coverage["cells"]
    )
    requests: list[dict[str, object]] = []
    species_occurrences: Counter[tuple[str, str]] = Counter()
    for split in CORPUS_SPLITS:
        axes = config.split_axis_quotas[split]
        body_slots = body_slots_by_split[split]
        assignments = assign_body_pose_cells(
            body_slots,
            axes.actions,
            axes.directions,
            forbidden,
            seed=config.identity_seed,
            domain=f"body-pose:v2:{split}",
        ) if body_slots else ()
        palettes = _quota_sequence(
            axes.head_palettes,
            config.identity_seed,
            f"palettes:v2:{split}",
        )
        for split_index, (assignment, palette) in enumerate(
            zip(assignments, palettes, strict=True)
        ):
            job = jobs_by_id[assignment.job_id]
            body = bodies_by_job_gender[(assignment.job_id, assignment.gender)]
            species_key = (split, job.species)
            heads = _stable_order(
                list(axes.heads_by_species[job.species]),
                config.identity_seed,
                f"heads:v2:{split}:{job.species}",
                key=str,
            )
            head = heads[species_occurrences[species_key] % len(heads)]
            species_occurrences[species_key] += 1
            head_identity = (
                f"{job.species}-g{assignment.gender}-h{head:02d}"
            )
            coverage_id = (
                f"j{job.id}-g{assignment.gender}-"
                f"a{assignment.action_base:03d}-d{assignment.direction}"
            )
            requested_rig_pose_id = (
                f"{body.rig_family}-a{assignment.action_base:03d}-"
                f"d{assignment.direction}"
            )
            pair_id = (
                f"{safe_slug(config.name)}-{split}-{split_index:04d}-"
                f"{safe_slug(job.name)}-{coverage_id}-h{head:02d}-p{palette:02d}"
            )
            payload = {
                "gender": assignment.gender,
                "job": [str(job.id)],
                "head": head,
                "headPalette": palette,
                "headdir": config.defaults.head_direction,
                "headgear": list(config.defaults.headgear),
                "garment": config.defaults.garment,
                "bodyPalette": config.defaults.body_palette,
                "madogearType": config.defaults.madogear_type,
                "action": assignment.action_base + assignment.direction,
                "canvas": config.canvas,
                "outfit": config.defaults.outfit,
            }
            requests.append(
                {
                    "id": pair_id,
                    "group": job.split_group,
                    "head_identity": head_identity,
                    "split": split,
                    "lineage": job.lineage,
                    "species": job.species,
                    "job_id": job.id,
                    "job_name": job.name,
                    "gender": assignment.gender,
                    "body_variant_id": (
                        f"{job.split_group}-g{assignment.gender}"
                    ),
                    "rig_family": body.rig_family,
                    "coverage_id": coverage_id,
                    "requested_rig_pose_id": requested_rig_pose_id,
                    "head": head,
                    "head_palette": palette,
                    "action_name": action_name_by_base[assignment.action_base],
                    "action_base": assignment.action_base,
                    "direction": assignment.direction,
                    "payload": payload,
                }
            )
    return tuple(requests)


def _body_slots_by_split_v2(
    config: IroCorpusConfigV2,
) -> dict[str, tuple[BodySlot, ...]]:
    bodies_by_job_gender = {
        (job.id, body.gender)
        for job in config.jobs
        for body in job.bodies
    }
    body_slots_by_split: dict[str, list[BodySlot]] = {
        split: [] for split in CORPUS_SPLITS
    }
    for quota in config.lineage_pair_quotas:
        for gender, count in ((0, quota.female), (1, quota.male)):
            jobs = _stable_order(
                [
                    job
                    for job in config.jobs
                    if job.lineage == quota.lineage
                    and config.split_group_splits[job.split_group] == quota.split
                    and (job.id, gender) in bodies_by_job_gender
                ],
                config.identity_seed,
                f"jobs:v2:{quota.split}:{quota.lineage}:{gender}",
                key=lambda job: str(job.id),
            )
            for occurrence in range(count):
                job = jobs[occurrence % len(jobs)]
                body_slots_by_split[quota.split].append(
                    BodySlot(job_id=job.id, gender=gender)
                )
    return {
        split: tuple(body_slots_by_split[split])
        for split in CORPUS_SPLITS
    }


def _plan_manifest(
    config: IroCorpusConfigVersion,
    requests: tuple[dict[str, object], ...],
    staging: Path,
    *,
    excluded_coverage: dict[str, Any],
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "format": (
            IRO_PLAN_V2_FORMAT
            if isinstance(config, IroCorpusConfigV2)
            else IRO_PLAN_FORMAT
        ),
        "name": config.name,
        "config": "config.json",
        "config_fingerprint": corpus_config_fingerprint(config),
        "requests": "requests.jsonl",
        "requests_sha256": sha256_file(staging / "requests.jsonl"),
        "request_count": len(requests),
        "split_counts": _split_counts(requests),
        "lineage_counts": _lineage_counts(requests),
    }
    if isinstance(config, IroCorpusConfigV2):
        body_pose_ids = [str(record["coverage_id"]) for record in requests]
        requested_rig_pose_ids = sorted(
            {str(record["requested_rig_pose_id"]) for record in requests}
        )
        manifest.update(
            {
                "split_group_counts": _split_group_counts(requests),
                "rig_family_counts": _rig_family_counts(requests),
                "joint_pose_counts": _joint_pose_counts(requests),
                "head_identity_count": len(
                    {record["head_identity"] for record in requests}
                ),
                "head_identity_overlap_counts": (
                    _head_identity_overlap_counts(requests)
                ),
                "body_pose_cell_count": len(set(body_pose_ids)),
                "body_pose_ids_sha256": _canonical_sha256(body_pose_ids),
                "requested_rig_pose_count": len(requested_rig_pose_ids),
                "requested_rig_pose_ids_sha256": _canonical_sha256(
                    requested_rig_pose_ids
                ),
                "coverage_planner": IRO_COVERAGE_PLANNER,
                "coverage_planner_runtime": {
                    "numpy": version("numpy"),
                    "scipy": version("scipy"),
                },
                "excluded_coverage": excluded_coverage,
                "excluded_source_request_count": sum(
                    int(source["request_count"])
                    for source in excluded_coverage["sources"]
                ),
                "excluded_source_body_pose_cell_count": len(
                    excluded_coverage["cells"]
                ),
                "applicable_excluded_body_pose_cell_count": (
                    _applicable_excluded_body_pose_cell_count(
                        config,
                        excluded_coverage,
                    )
                ),
                "excluded_coverage_sha256": _canonical_sha256(
                    excluded_coverage
                ),
            }
        )
    else:
        manifest["identity_count"] = len(
            {record["identity"] for record in requests}
        )
    return manifest


def _audit_request_records(
    config: IroCorpusConfigVersion,
    requests: tuple[dict[str, Any], ...],
) -> None:
    if isinstance(config, IroCorpusConfigV2):
        _audit_request_records_v2(config, requests)
        return
    _audit_request_records_v1(config, requests)


def _audit_request_records_v1(
    config: IroCorpusConfig,
    requests: tuple[dict[str, Any], ...],
) -> None:
    ids: set[str] = set()
    identity_splits: dict[str, str] = {}
    for index, record in enumerate(requests):
        require_exact_keys(record, REQUEST_RECORD_KEYS, f"iRO request {index}")
        request_id = _string(record, "id")
        if request_id in ids:
            raise Pix2PixError(f"duplicate iRO request id: {request_id}")
        ids.add(request_id)
        split = _string(record, "split")
        lineage = _string(record, "lineage")
        if split not in CORPUS_SPLITS:
            raise Pix2PixError(f"unsupported iRO request split: {split}")
        if config.lineage_splits.get(lineage) != split:
            raise Pix2PixError(f"iRO lineage crosses splits: {lineage}")
        if record["group"] != lineage:
            raise Pix2PixError(f"iRO request group must equal its lineage: {request_id}")
        identity = _string(record, "identity")
        previous = identity_splits.setdefault(identity, split)
        if previous != split:
            raise Pix2PixError(f"iRO identity crosses splits: {identity}")
        payload = record["payload"]
        if not isinstance(payload, dict):
            raise Pix2PixError(f"iRO request payload must be an object: {request_id}")
        require_exact_keys(payload, RENDER_PAYLOAD_KEYS, f"payload for {request_id}")
        if payload["action"] != record["action_base"] + record["direction"]:
            raise Pix2PixError(f"iRO request action is inconsistent: {request_id}")
        if payload["canvas"] != config.canvas:
            raise Pix2PixError(f"iRO request canvas is inconsistent: {request_id}")


def _audit_request_records_v2(
    config: IroCorpusConfigV2,
    requests: tuple[dict[str, Any], ...],
) -> None:
    jobs_by_id = {job.id: job for job in config.jobs}
    action_names_by_base = {
        action.base: action.name
        for action in config.actions
    }
    bodies_by_job_gender = {
        (job.id, body.gender): body
        for job in config.jobs
        for body in job.bodies
    }
    ids: set[str] = set()
    coverage_ids: set[str] = set()
    planned_bodies: set[tuple[int, int]] = set()
    split_by_group: dict[str, str] = {}
    for index, record in enumerate(requests):
        require_exact_keys(record, REQUEST_RECORD_V2_KEYS, f"iRO v2 request {index}")
        request_id = _string(record, "id")
        if request_id in ids:
            raise Pix2PixError(f"duplicate iRO request id: {request_id}")
        ids.add(request_id)
        split = _string(record, "split")
        if split not in CORPUS_SPLITS:
            raise Pix2PixError(f"unsupported iRO request split: {split}")
        job_id = _integer(record, "job_id")
        gender = _gender(record, "gender")
        job = jobs_by_id.get(job_id)
        body = bodies_by_job_gender.get((job_id, gender))
        if job is None or body is None:
            raise Pix2PixError(f"iRO request uses an unknown body: {request_id}")
        planned_bodies.add((job_id, gender))
        expected_split = config.split_group_splits[job.split_group]
        if split != expected_split:
            raise Pix2PixError(
                f"iRO split group {job.split_group} crosses splits"
            )
        if record["group"] != job.split_group:
            raise Pix2PixError(
                f"iRO request group differs from split group: {request_id}"
            )
        existing_split = split_by_group.setdefault(job.split_group, split)
        if existing_split != split:
            raise Pix2PixError(
                f"iRO split group crosses splits: {job.split_group}"
            )
        for key, expected in (
            ("lineage", job.lineage),
            ("species", job.species),
            ("job_name", job.name),
            ("rig_family", body.rig_family),
            ("body_variant_id", f"{job.split_group}-g{gender}"),
        ):
            if record[key] != expected:
                raise Pix2PixError(
                    f"iRO request {key} is inconsistent: {request_id}"
                )
        action_base = _integer(record, "action_base")
        direction = _integer(record, "direction")
        action_name = action_names_by_base.get(action_base)
        if action_name is None or record["action_name"] != action_name:
            raise Pix2PixError(
                f"iRO request action name is inconsistent: {request_id}"
            )
        head = _integer(record, "head")
        head_palette = _integer(record, "head_palette")
        axes = config.split_axis_quotas[split]
        if head not in axes.heads_by_species[job.species]:
            raise Pix2PixError(
                f"iRO request head is outside its split/species catalog: {request_id}"
            )
        if axes.head_palettes.get(head_palette, 0) <= 0:
            raise Pix2PixError(
                f"iRO request head palette is outside its split quota: {request_id}"
            )
        expected_head_identity = (
            f"{job.species}-g{gender}-h{head:02d}"
        )
        if record["head_identity"] != expected_head_identity:
            raise Pix2PixError(
                f"iRO request head identity is inconsistent: {request_id}"
            )
        coverage_id = (
            f"j{job_id}-g{gender}-a{action_base:03d}-d{direction}"
        )
        if record["coverage_id"] != coverage_id:
            raise Pix2PixError(
                f"iRO request coverage id is inconsistent: {request_id}"
            )
        if coverage_id in coverage_ids:
            raise Pix2PixError(f"duplicate iRO coverage cell: {coverage_id}")
        coverage_ids.add(coverage_id)
        requested_rig_pose_id = (
            f"{body.rig_family}-a{action_base:03d}-d{direction}"
        )
        if record["requested_rig_pose_id"] != requested_rig_pose_id:
            raise Pix2PixError(
                f"iRO requested rig pose is inconsistent: {request_id}"
            )
        payload = record["payload"]
        if not isinstance(payload, dict):
            raise Pix2PixError(f"iRO request payload must be an object: {request_id}")
        require_exact_keys(payload, RENDER_PAYLOAD_KEYS, f"payload for {request_id}")
        expected_payload = {
            "gender": gender,
            "job": [str(job_id)],
            "head": head,
            "headPalette": head_palette,
            "headdir": config.defaults.head_direction,
            "headgear": list(config.defaults.headgear),
            "garment": config.defaults.garment,
            "bodyPalette": config.defaults.body_palette,
            "madogearType": config.defaults.madogear_type,
            "action": action_base + direction,
            "canvas": config.canvas,
            "outfit": config.defaults.outfit,
        }
        if payload != expected_payload:
            raise Pix2PixError(
                f"iRO request renderer payload is inconsistent: {request_id}"
            )
    expected_bodies = set(bodies_by_job_gender)
    if planned_bodies != expected_bodies:
        missing = sorted(expected_bodies - planned_bodies)
        raise Pix2PixError(
            "iRO v2 plan does not cover every configured body: "
            + ", ".join(f"job {job_id}/gender {gender}" for job_id, gender in missing)
        )


def _audit_request_quotas(
    config: IroCorpusConfigVersion,
    requests: tuple[dict[str, Any], ...] | tuple[dict[str, object], ...],
) -> None:
    if isinstance(config, IroCorpusConfigV2):
        _audit_request_quotas_v2(config, requests)
        return
    _audit_request_quotas_v1(config, requests)


def _audit_request_quotas_v1(
    config: IroCorpusConfig,
    requests: tuple[dict[str, Any], ...] | tuple[dict[str, object], ...],
) -> None:
    for split in CORPUS_SPLITS:
        records = [record for record in requests if record["split"] == split]
        expected_lineages = {
            quota.lineage: quota.count
            for quota in config.lineage_pair_quotas
            if quota.split == split
        }
        expected_genders = {
            gender: sum(
                quota.female if gender == 0 else quota.male
                for quota in config.lineage_pair_quotas
                if quota.split == split
            )
            for gender in (0, 1)
        }
        expected_axes = config.split_axis_quotas[split]
        actual = {
            "lineage": Counter(record["lineage"] for record in records),
            "gender": Counter(record["gender"] for record in records),
            "direction": Counter(record["direction"] for record in records),
            "head_palette": Counter(record["head_palette"] for record in records),
            "action": Counter(record["action_base"] for record in records),
        }
        expected = {
            "lineage": expected_lineages,
            "gender": expected_genders,
            "direction": expected_axes.directions,
            "head_palette": expected_axes.head_palettes,
            "action": expected_axes.actions,
        }
        for axis, expected_counts in expected.items():
            if dict(actual[axis]) != dict(expected_counts):
                raise Pix2PixError(
                    f"{split} {axis} quotas differ from the corpus config"
                )
def _audit_request_quotas_v2(
    config: IroCorpusConfigV2,
    requests: tuple[dict[str, Any], ...] | tuple[dict[str, object], ...],
) -> None:
    expected_body_slots = _body_slots_by_split_v2(config)
    for split in CORPUS_SPLITS:
        records = [record for record in requests if record["split"] == split]
        expected_lineages = {
            quota.lineage: quota.count
            for quota in config.lineage_pair_quotas
            if quota.split == split
        }
        expected_gender_by_lineage = {
            (quota.lineage, gender): (
                quota.female if gender == 0 else quota.male
            )
            for quota in config.lineage_pair_quotas
            if quota.split == split
            for gender in (0, 1)
            if (quota.female if gender == 0 else quota.male) > 0
        }
        expected_axes = config.split_axis_quotas[split]
        actual = {
            "lineage": Counter(record["lineage"] for record in records),
            "lineage_gender": Counter(
                (record["lineage"], record["gender"]) for record in records
            ),
            "direction": Counter(record["direction"] for record in records),
            "head_palette": Counter(
                record["head_palette"] for record in records
            ),
            "action": Counter(record["action_base"] for record in records),
        }
        expected = {
            "lineage": expected_lineages,
            "lineage_gender": expected_gender_by_lineage,
            "direction": {
                value: count
                for value, count in expected_axes.directions.items()
                if count > 0
            },
            "head_palette": {
                value: count
                for value, count in expected_axes.head_palettes.items()
                if count > 0
            },
            "action": {
                value: count
                for value, count in expected_axes.actions.items()
                if count > 0
            },
        }
        for axis, expected_counts in expected.items():
            if dict(actual[axis]) != dict(expected_counts):
                raise Pix2PixError(
                    f"{split} {axis} quotas differ from the corpus config"
                )
        body_counts = Counter(
            (record["job_id"], record["gender"])
            for record in records
        )
        expected_body_counts = Counter(
            (slot.job_id, slot.gender)
            for slot in expected_body_slots[split]
        )
        if body_counts != expected_body_counts:
            raise Pix2PixError(
                f"{split} body multiplicities differ from the coverage schedule"
            )
        body_actions = Counter(
            (record["job_id"], record["gender"], record["action_base"])
            for record in records
        )
        if any(count != 1 for count in body_actions.values()):
            raise Pix2PixError(
                f"{split} repeats an action for the same body"
            )
        body_directions = Counter(
            (record["job_id"], record["gender"], record["direction"])
            for record in records
        )
        if any(count != 1 for count in body_directions.values()):
            raise Pix2PixError(
                f"{split} repeats a direction for the same body"
            )
        joint_counts = Counter(
            (record["action_base"], record["direction"])
            for record in records
        )
        joint_lower, joint_upper = joint_pose_bounds(
            expected_axes.actions,
            expected_axes.directions,
        )
        expected_joint_cells = {
            (action, direction)
            for action, action_count in expected_axes.actions.items()
            if action_count > 0
            for direction, direction_count in expected_axes.directions.items()
            if direction_count > 0
        }
        if joint_lower and set(joint_counts) != expected_joint_cells:
            raise Pix2PixError(
                f"{split} does not cover the full positive action/direction grid"
            )
        if any(
            count < joint_lower or count > joint_upper
            for count in joint_counts.values()
        ):
            raise Pix2PixError(
                f"{split} joint action/direction counts exceed planner bounds"
            )


def _render_request(
    config: IroCorpusConfigVersion,
    record: dict[str, Any],
    renders_dir: Path,
) -> None:
    request_id = str(record["id"])
    destination = renders_dir / request_id
    with TemporaryDirectory(
        dir=renders_dir,
        prefix=f".{request_id}.",
        suffix=".incomplete",
    ) as temporary:
        staging = Path(temporary)
        response_bytes = _download_renderer_png(
            config.endpoint,
            record["payload"],
            timeout_seconds=config.request_timeout_seconds,
        )
        response_path = staging / "response.png"
        response_path.write_bytes(response_bytes)
        frames, default_image = decode_renderer_frames(
            response_bytes,
            expected_size=config.image_size,
        )
        frames_dir = staging / "frames"
        frames_dir.mkdir()
        frame_records = []
        for frame in frames:
            frame_path = frames_dir / f"frame-{frame['index']:03d}.png"
            image = frame.pop("image")
            image.save(frame_path, format="PNG", optimize=False)
            frame_records.append(
                {
                    **frame,
                    "path": f"frames/{frame_path.name}",
                    "sha256": sha256_file(frame_path),
                }
            )
        result = {
            "format": IRO_RENDER_FORMAT,
            "request_id": request_id,
            "group": record["group"],
            "split": record["split"],
            "payload_sha256": _canonical_sha256(record["payload"]),
            "response": {
                "path": "response.png",
                "sha256": sha256_file(response_path),
                "size_bytes": response_path.stat().st_size,
            },
            "default_image": default_image,
            "frames": frame_records,
        }
        write_json(staging / "result.json", result)
        _load_render_result(config, record, staging)
        os.rename(staging, destination)


def _download_renderer_png(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> bytes:
    body = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={
            "Accept": "image/png",
            "Content-Type": "application/vnd.api+json",
            "User-Agent": "aigen-local-pix2pix-corpus/1",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            content_type = response.headers.get_content_type()
            response_bytes = response.read()
    except OSError as error:
        raise Pix2PixError(f"iRO renderer request failed: {error}") from error
    if content_type != "image/png":
        raise Pix2PixError(
            f"iRO renderer returned {content_type!r}, expected image/png"
        )
    if not response_bytes:
        raise Pix2PixError("iRO renderer returned an empty response")
    return response_bytes


def decode_renderer_frames(
    response_bytes: bytes,
    *,
    expected_size: int,
) -> tuple[list[dict[str, Any]], bool]:
    try:
        with Image.open(io.BytesIO(response_bytes)) as image:
            if image.format != "PNG":
                raise Pix2PixError(f"iRO renderer response is {image.format}, not PNG")
            if image.size != (expected_size, expected_size):
                raise Pix2PixError(
                    f"iRO renderer response is {image.width}x{image.height}, "
                    f"expected {expected_size}x{expected_size}"
                )
            default_image = bool(getattr(image, "default_image", False))
            first_frame = 1 if default_image else 0
            frames = []
            for frame_index in range(first_frame, image.n_frames):
                image.seek(frame_index)
                image.load()
                frame = image.convert("RGBA").copy()
                duration = image.info.get("duration")
                if duration is not None and not isinstance(duration, (int, float)):
                    raise Pix2PixError("iRO APNG frame duration is not numeric")
                frames.append(
                    {
                        "index": frame_index,
                        "duration_ms": float(duration) if duration is not None else None,
                        "pixel_sha256": _rgba_pixel_sha256(frame),
                        "edge_clear": _alpha_edge_clear(frame),
                        "image": frame,
                    }
                )
    except OSError as error:
        raise Pix2PixError(f"cannot decode iRO renderer PNG: {error}") from error
    if not frames:
        raise Pix2PixError("iRO renderer PNG contains no animation frames")
    return frames, default_image


def _load_render_result(
    config: IroCorpusConfigVersion,
    request_record: dict[str, Any],
    request_dir: Path,
) -> dict[str, Any]:
    result = read_json(request_dir / "result.json", label="iRO render result")
    require_exact_keys(
        result,
        {
            "format",
            "request_id",
            "group",
            "split",
            "payload_sha256",
            "response",
            "default_image",
            "frames",
        },
        "iRO render result",
    )
    request_id = str(request_record["id"])
    if result["format"] != IRO_RENDER_FORMAT:
        raise Pix2PixError(f"unsupported iRO render format for {request_id}")
    for key in ("request_id", "group", "split"):
        expected_key = "id" if key == "request_id" else key
        if result[key] != request_record[expected_key]:
            raise Pix2PixError(f"iRO render {key} mismatch for {request_id}")
    if result["payload_sha256"] != _canonical_sha256(request_record["payload"]):
        raise Pix2PixError(f"iRO render payload checksum mismatch for {request_id}")
    response = result["response"]
    if not isinstance(response, dict):
        raise Pix2PixError(f"invalid iRO response manifest for {request_id}")
    require_exact_keys(
        response,
        {"path", "sha256", "size_bytes"},
        f"iRO response for {request_id}",
    )
    response_path = corpus_member(
        request_dir,
        _string(response, "path"),
        label=f"iRO response for {request_id}",
    )
    if not response_path.is_file():
        raise Pix2PixError(f"missing iRO response: {response_path}")
    if response_path.stat().st_size != _integer(response, "size_bytes"):
        raise Pix2PixError(f"iRO response size mismatch for {request_id}")
    if sha256_file(response_path) != _string(response, "sha256"):
        raise Pix2PixError(f"iRO response checksum mismatch for {request_id}")
    if not isinstance(result["default_image"], bool):
        raise Pix2PixError(f"invalid APNG default-image flag for {request_id}")
    frames = result["frames"]
    if not isinstance(frames, list) or not frames:
        raise Pix2PixError(f"iRO render has no frame records: {request_id}")
    seen_indices: set[int] = set()
    for frame in frames:
        if not isinstance(frame, dict):
            raise Pix2PixError(f"invalid iRO frame record for {request_id}")
        require_exact_keys(
            frame,
            {
                "index",
                "duration_ms",
                "pixel_sha256",
                "edge_clear",
                "path",
                "sha256",
            },
            f"iRO frame for {request_id}",
        )
        frame_index = _integer(frame, "index")
        if frame_index in seen_indices:
            raise Pix2PixError(f"duplicate APNG frame index for {request_id}")
        seen_indices.add(frame_index)
        if not isinstance(frame["edge_clear"], bool):
            raise Pix2PixError(f"invalid frame edge-clear flag for {request_id}")
        frame_path = corpus_member(
            request_dir,
            _string(frame, "path"),
            label=f"frame for {request_id}",
        )
        if not frame_path.is_file():
            raise Pix2PixError(f"missing iRO frame: {frame_path}")
        if sha256_file(frame_path) != _string(frame, "sha256"):
            raise Pix2PixError(f"iRO frame checksum mismatch for {request_id}")
        try:
            with Image.open(frame_path) as image:
                image.load()
                if image.mode != "RGBA" or image.size != (
                    config.image_size,
                    config.image_size,
                ):
                    raise Pix2PixError(
                        f"invalid iRO frame raster contract: {frame_path}"
                    )
                if _rgba_pixel_sha256(image) != _string(frame, "pixel_sha256"):
                    raise Pix2PixError(
                        f"iRO frame pixel checksum mismatch for {request_id}"
                    )
                if _alpha_edge_clear(image) != frame["edge_clear"]:
                    raise Pix2PixError(
                        f"iRO frame edge-clear audit mismatch for {request_id}"
                    )
        except OSError as error:
            raise Pix2PixError(f"cannot decode iRO frame {frame_path}: {error}") from error
    return result


def _frame_rank(
    config: IroCorpusConfigVersion,
    request_record: dict[str, Any],
    frame: dict[str, Any],
) -> bytes:
    canonical_request = json.dumps(
        request_record["payload"],
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    value = (
        f"{config.identity_seed}\0{canonical_request}\0"
        f"{frame['index']}\0{frame['pixel_sha256']}"
    )
    return hashlib.sha256(value.encode("utf-8")).digest()


def _select_unique_v2_target_frames(
    root: Path,
    config: IroCorpusConfigV2,
    requests: tuple[dict[str, Any], ...],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]]:
    request_candidates: list[
        tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...], int]
    ] = []
    target_hash_cache: dict[str, str] = {}
    for request_index, record in enumerate(requests):
        request_dir = root / "renders" / str(record["id"])
        result = _load_render_result(config, record, request_dir)
        candidates_by_target: dict[str, dict[str, Any]] = {}
        for frame in result["frames"]:
            frame_path = request_dir / str(frame["path"])
            if not _selection_frame_safe(frame_path):
                continue
            renderer_hash = _string(frame, "pixel_sha256")
            target_hash = target_hash_cache.get(renderer_hash)
            if target_hash is None:
                target_hash = _white_composited_pixel_sha256(frame_path)
                target_hash_cache[renderer_hash] = target_hash
            candidate = {
                **frame,
                "target_pixel_sha256": target_hash,
            }
            current = candidates_by_target.get(target_hash)
            if current is None or _frame_rank(
                config,
                record,
                candidate,
            ) < _frame_rank(config, record, current):
                candidates_by_target[target_hash] = candidate
        if not candidates_by_target:
            raise Pix2PixError(
                f"iRO request has no canvas-safe APNG frame: {record['id']}"
            )
        candidates = tuple(
            sorted(
                candidates_by_target.values(),
                key=lambda frame: _frame_rank(config, record, frame),
            )
        )
        request_candidates.append(
            (record, result, candidates, request_index)
        )

    target_hashes = sorted(
        {
            _string(candidate, "target_pixel_sha256")
            for _, _, candidates, _ in request_candidates
            for candidate in candidates
        }
    )
    if len(target_hashes) < len(request_candidates):
        raise Pix2PixError(
            "iRO renders contain fewer unique RGB targets than requests"
        )
    target_index = {
        target_hash: index
        for index, target_hash in enumerate(target_hashes)
    }
    edge_keys = sorted(
        (
            _frame_rank(config, record, candidate),
            str(record["id"]),
            _string(candidate, "target_pixel_sha256"),
        )
        for record, _, candidates, _ in request_candidates
        for candidate in candidates
    )
    edge_tie_rank = {
        key: rank
        for rank, key in enumerate(edge_keys)
    }
    edge_count = len(edge_keys)
    preference_weight = len(request_candidates) * edge_count + 1
    rows: list[int] = []
    columns: list[int] = []
    weights: list[int] = []
    candidates_by_edge: dict[tuple[int, int], dict[str, Any]] = {}
    for request_index, (record, _, candidates, _) in enumerate(
        request_candidates
    ):
        for local_rank, candidate in enumerate(candidates):
            target_hash = _string(candidate, "target_pixel_sha256")
            column = target_index[target_hash]
            edge_key = (
                _frame_rank(config, record, candidate),
                str(record["id"]),
                target_hash,
            )
            rows.append(request_index)
            columns.append(column)
            weights.append(
                local_rank * preference_weight
                + edge_tie_rank[edge_key]
                + 1
            )
            candidates_by_edge[(request_index, column)] = candidate
    graph = coo_array(
        (
            np.asarray(weights, dtype=np.int64),
            (
                np.asarray(rows, dtype=np.int32),
                np.asarray(columns, dtype=np.int32),
            ),
        ),
        shape=(len(request_candidates), len(target_hashes)),
        dtype=np.int64,
    ).tocsr()
    try:
        matched_rows, matched_columns = min_weight_full_bipartite_matching(
            graph
        )
    except ValueError as error:
        raise Pix2PixError(
            "iRO requests have no globally unique RGB-target assignment"
        ) from error
    if not np.array_equal(
        matched_rows,
        np.arange(len(request_candidates), dtype=matched_rows.dtype),
    ):
        raise Pix2PixError(
            "iRO target matching did not cover every corpus request"
        )
    return [
        (
            record,
            result,
            candidates_by_edge[(request_index, int(matched_columns[request_index]))],
            original_index,
        )
        for request_index, (
            record,
            result,
            _,
            original_index,
        ) in enumerate(request_candidates)
    ]


def _verify_selected_render_frame(
    root: Path,
    config: IroCorpusConfigV2,
    planned: dict[str, Any],
    selected: dict[str, Any],
) -> str:
    pair_id = _string(selected, "id")
    request_dir = root / "renders" / pair_id
    result_path = request_dir / "result.json"
    if sha256_file(result_path) != _string(
        selected,
        "renderer_result_sha256",
    ):
        raise Pix2PixError(
            f"selected renderer-result checksum mismatch: {pair_id}"
        )
    result = read_json(result_path, label=f"iRO render result for {pair_id}")
    require_exact_keys(
        result,
        {
            "format",
            "request_id",
            "group",
            "split",
            "payload_sha256",
            "response",
            "default_image",
            "frames",
        },
        f"iRO render result for {pair_id}",
    )
    if result["format"] != IRO_RENDER_FORMAT:
        raise Pix2PixError(f"unsupported iRO render format for {pair_id}")
    if (
        result["request_id"] != pair_id
        or result["group"] != planned["group"]
        or result["split"] != planned["split"]
        or result["payload_sha256"] != _canonical_sha256(planned["payload"])
    ):
        raise Pix2PixError(
            f"selected renderer-result metadata mismatch: {pair_id}"
        )
    frames = result["frames"]
    if not isinstance(frames, list):
        raise Pix2PixError(f"iRO render frames must be an array: {pair_id}")
    selected_index = _integer(selected, "renderer_frame_index")
    matching_frames = []
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise Pix2PixError(
                f"invalid iRO frame record {frame_index}: {pair_id}"
            )
        require_exact_keys(
            frame,
            {
                "index",
                "duration_ms",
                "pixel_sha256",
                "edge_clear",
                "path",
                "sha256",
            },
            f"iRO frame {frame_index} for {pair_id}",
        )
        if _integer(frame, "index") == selected_index:
            matching_frames.append(frame)
    if len(matching_frames) != 1:
        raise Pix2PixError(
            f"selected renderer frame is absent or duplicated: {pair_id}"
        )
    frame = matching_frames[0]
    if (
        frame["duration_ms"] != selected["duration_ms"]
        or frame["pixel_sha256"] != selected["renderer_pixel_sha256"]
        or frame["sha256"] != selected["renderer_frame_sha256"]
    ):
        raise Pix2PixError(
            f"selected renderer-frame metadata mismatch: {pair_id}"
        )
    frame_path = corpus_member(
        request_dir,
        _string(frame, "path"),
        label=f"selected renderer frame for {pair_id}",
    )
    if sha256_file(frame_path) != _string(
        selected,
        "renderer_frame_sha256",
    ):
        raise Pix2PixError(
            f"selected renderer-frame checksum mismatch: {pair_id}"
        )
    try:
        with Image.open(frame_path) as image:
            image.load()
            if image.mode != "RGBA" or image.size != (
                config.image_size,
                config.image_size,
            ):
                raise Pix2PixError(
                    f"invalid selected renderer-frame raster: {pair_id}"
                )
            if _rgba_pixel_sha256(image) != _string(
                selected,
                "renderer_pixel_sha256",
            ):
                raise Pix2PixError(
                    f"selected renderer-frame pixel mismatch: {pair_id}"
                )
            if not _selection_image_safe(image):
                raise Pix2PixError(
                    "selected renderer frame touches a forbidden canvas edge: "
                    f"{pair_id}"
                )
            composited_hash = _rgb_pixel_sha256(
                _white_composite(image.convert("RGBA"))
            )
    except OSError as error:
        raise Pix2PixError(
            f"cannot decode selected renderer frame {frame_path}: {error}"
        ) from error
    return composited_hash


def _composite_white_target(
    rgba_path: Path,
    target_path: Path,
    image_size: int,
) -> None:
    try:
        with Image.open(rgba_path) as image:
            image.load()
            rgba = image.convert("RGBA")
            if rgba.size != (image_size, image_size):
                raise Pix2PixError(f"invalid target frame size: {rgba_path}")
            target = _white_composite(rgba)
            target.save(target_path, format="PNG", optimize=False)
    except OSError as error:
        raise Pix2PixError(f"cannot materialize target {rgba_path}: {error}") from error


def _white_composited_pixel_sha256(rgba_path: Path) -> str:
    try:
        with Image.open(rgba_path) as image:
            image.load()
            target = _white_composite(image.convert("RGBA"))
            return _rgb_pixel_sha256(target)
    except OSError as error:
        raise Pix2PixError(
            f"cannot inspect composited target {rgba_path}: {error}"
        ) from error


def _white_composite(rgba: Image.Image) -> Image.Image:
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def _verify_rgb_image(path: Path, width: int, height: int) -> str:
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (width, height):
                raise Pix2PixError(
                    f"image must be RGB {width}x{height}: {path.as_posix()}"
                )
            return _rgb_pixel_sha256(image)
    except OSError as error:
        raise Pix2PixError(f"cannot decode image {path.as_posix()}: {error}") from error


def _alpha_edge_clear(image: Image.Image) -> bool:
    alpha = image.getchannel("A")
    width, height = alpha.size
    return not any(
        edge.getbbox()
        for edge in (
            alpha.crop((0, 0, width, 1)),
            alpha.crop((0, height - 1, width, height)),
            alpha.crop((0, 0, 1, height)),
            alpha.crop((width - 1, 0, width, height)),
        )
    )


def _selection_frame_safe(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return _selection_image_safe(image)
    except OSError as error:
        raise Pix2PixError(f"cannot inspect iRO frame edges {path}: {error}") from error


def _selection_image_safe(image: Image.Image) -> bool:
    alpha = image.getchannel("A")
    width, height = alpha.size
    return not any(
        edge.getbbox()
        for edge in (
            alpha.crop((0, 0, width, 1)),
            alpha.crop((0, 0, 1, height)),
            alpha.crop((width - 1, 0, width, height)),
        )
    )


def _rgba_pixel_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(b"aigen.pix2pix.rgba-pixels.v1\0")
    digest.update(f"{image.width}x{image.height}\0RGBA\0".encode("ascii"))
    digest.update(image.convert("RGBA").tobytes())
    return digest.hexdigest()


def _rgb_pixel_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(b"aigen.pix2pix.rgb-pixels.v1\0")
    digest.update(f"{image.width}x{image.height}\0RGB\0".encode("ascii"))
    digest.update(image.convert("RGB").tobytes())
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _body_pose_namespace_sha256(
    config: IroCorpusConfigVersion,
) -> str:
    endpoint = urlsplit(config.endpoint)
    return _canonical_sha256(
        {
            "renderer": f"{endpoint.scheme}://{endpoint.netloc}{endpoint.path}",
            "canvas": config.canvas,
            "image_size": config.image_size,
            "actions": [
                {"base": action.base, "name": action.name}
                for action in sorted(config.actions, key=lambda action: action.base)
            ],
            "defaults": config.defaults.model_dump(mode="json"),
        }
    )


def _target_instance_id(
    request_record: dict[str, Any],
    renderer_result_sha256: str,
    renderer_frame_index: int,
    renderer_frame_sha256: str,
    renderer_pixel_sha256: str,
) -> str:
    digest = _canonical_sha256(
        {
            "coverage_id": request_record["coverage_id"],
            "payload": request_record["payload"],
            "renderer_result_sha256": renderer_result_sha256,
            "renderer_frame_index": renderer_frame_index,
            "renderer_frame_sha256": renderer_frame_sha256,
            "renderer_pixel_sha256": renderer_pixel_sha256,
        }
    )
    return f"target-{digest}"


def _stable_order(
    values: list[Any],
    seed: int,
    domain: str,
    *,
    key: Any,
) -> list[Any]:
    return [
        value
        for _, value in sorted(
            enumerate(values),
            key=lambda item: hashlib.sha256(
                f"{seed}\0{domain}\0{item[0]}\0{key(item[1])}".encode("utf-8")
            ).digest(),
        )
    ]


def _quota_sequence(
    quotas: dict[int, int],
    seed: int,
    domain: str,
) -> list[int]:
    values = [
        value
        for value in sorted(quotas)
        for _ in range(quotas[value])
    ]
    return _stable_order(values, seed, domain, key=str)


def _split_counts(records: Any) -> dict[str, int]:
    counts = Counter(str(record["split"]) for record in records)
    return {split: counts[split] for split in CORPUS_SPLITS}


def _lineage_counts(records: Any) -> dict[str, int]:
    counts = Counter(str(record["lineage"]) for record in records)
    return {lineage: counts[lineage] for lineage in sorted(counts)}


def _split_group_counts(records: Any) -> dict[str, int]:
    counts = Counter(str(record["group"]) for record in records)
    return {group: counts[group] for group in sorted(counts)}


def _rig_family_counts(records: Any) -> dict[str, int]:
    counts = Counter(str(record["rig_family"]) for record in records)
    return {family: counts[family] for family in sorted(counts)}


def _joint_pose_counts(records: Any) -> dict[str, int]:
    cells = {
        split: set()
        for split in CORPUS_SPLITS
    }
    for record in records:
        cells[str(record["split"])].add(
            (record["action_base"], record["direction"])
        )
    return {
        split: len(cells[split])
        for split in CORPUS_SPLITS
    }


def _head_identity_overlap_counts(records: Any) -> dict[str, int]:
    identities = {
        split: {
            str(record["head_identity"])
            for record in records
            if record["split"] == split
        }
        for split in CORPUS_SPLITS
    }
    return {
        f"{left}_{right}": len(identities[left] & identities[right])
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    }


def _plan_result(
    root: Path,
    requests: tuple[dict[str, Any], ...] | tuple[dict[str, object], ...],
    manifest: dict[str, Any] | dict[str, object],
    *,
    reused: bool,
) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "completed",
        "kind": "iRO-corpus-plan",
        "root": root.as_posix(),
        "reused": reused,
        "config_fingerprint": manifest["config_fingerprint"],
        "request_count": len(requests),
        "split_counts": manifest["split_counts"],
        "lineage_counts": manifest["lineage_counts"],
    }
    if manifest["format"] == IRO_PLAN_V2_FORMAT:
        result.update(
            {
                "head_identity_count": manifest["head_identity_count"],
                "head_identity_overlap_counts": manifest[
                    "head_identity_overlap_counts"
                ],
                "split_group_counts": manifest["split_group_counts"],
                "rig_family_counts": manifest["rig_family_counts"],
                "joint_pose_counts": manifest["joint_pose_counts"],
                "body_pose_cell_count": manifest["body_pose_cell_count"],
                "requested_rig_pose_count": manifest[
                    "requested_rig_pose_count"
                ],
                "excluded_source_request_count": manifest[
                    "excluded_source_request_count"
                ],
                "excluded_source_body_pose_cell_count": manifest[
                    "excluded_source_body_pose_cell_count"
                ],
                "applicable_excluded_body_pose_cell_count": manifest[
                    "applicable_excluded_body_pose_cell_count"
                ],
            }
        )
    else:
        result["identity_count"] = manifest["identity_count"]
    return result


def _selection_result(
    root: Path,
    selected: tuple[dict[str, Any], ...],
    manifest: dict[str, Any],
    *,
    reused: bool,
) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "completed",
        "kind": "iRO-target-selection",
        "root": root.as_posix(),
        "reused": reused,
        "pair_count": len(selected),
        "split_counts": manifest["split_counts"],
        "lineage_counts": manifest["lineage_counts"],
    }
    if manifest["format"] == IRO_SELECTION_V2_FORMAT:
        result.update(
            {
                "head_identity_count": manifest["head_identity_count"],
                "head_identity_overlap_counts": manifest[
                    "head_identity_overlap_counts"
                ],
                "split_group_counts": manifest["split_group_counts"],
                "rig_family_counts": manifest["rig_family_counts"],
                "joint_pose_counts": manifest["joint_pose_counts"],
                "body_pose_cell_count": manifest["body_pose_cell_count"],
                "requested_rig_pose_count": manifest[
                    "requested_rig_pose_count"
                ],
                "target_instance_count": manifest["target_instance_count"],
                "realized_rig_pose_count": manifest[
                    "realized_rig_pose_count"
                ],
                "target_pixel_count": manifest["target_pixel_count"],
            }
        )
    else:
        result["identity_count"] = manifest["identity_count"]
    return result


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise Pix2PixError(f"{key} must be a non-empty string")
    return value


def _integer(payload: dict[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise Pix2PixError(f"{key} must be an integer")
    return value


def _gender(payload: dict[str, Any], key: str) -> int:
    value = _integer(payload, key)
    if value not in (0, 1):
        raise Pix2PixError(f"{key} must be 0 or 1")
    return value
