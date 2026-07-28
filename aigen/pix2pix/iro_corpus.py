from __future__ import annotations

import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.request import Request, urlopen

from PIL import Image

from aigen.manifest_io import read_json, sha256_file, write_json
from aigen.pix2pix.corpus_config import (
    CORPUS_SPLITS,
    IroCorpusConfig,
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
from aigen.progress import StatusReporter


IRO_PLAN_FORMAT = "aigen.pix2pix.iro-plan.v1"
IRO_RENDER_FORMAT = "aigen.pix2pix.iro-render.v1"
IRO_SELECTION_FORMAT = "aigen.pix2pix.iro-selection.v1"
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
SELECTION_RECORD_KEYS = REQUEST_RECORD_KEYS | {
    "renderer_frame_index",
    "duration_ms",
    "renderer_pixel_sha256",
    "flux_seed",
    "target",
    "target_sha256",
}


def plan_iro_corpus(
    config_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    config = load_iro_corpus_config(config_path)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        existing_config, requests, manifest = load_iro_plan(output_dir)
        if corpus_config_fingerprint(existing_config) != corpus_config_fingerprint(config):
            raise Pix2PixError(
                f"existing iRO corpus plan uses a different config: {output_dir}"
            )
        return _plan_result(output_dir, requests, manifest, reused=True)

    requests = _planned_requests(config)
    _audit_request_quotas(config, requests)
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
        manifest = _plan_manifest(config, requests, staging)
        write_json(staging / "plan.json", manifest)
        os.rename(staging, output_dir)
    return _plan_result(output_dir, requests, manifest, reused=False)


def load_iro_plan(
    root: Path,
) -> tuple[IroCorpusConfig, tuple[dict[str, Any], ...], dict[str, Any]]:
    root = root.expanduser().resolve()
    manifest = read_json(root / "plan.json", label="iRO corpus plan")
    require_exact_keys(
        manifest,
        {
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
        },
        "iRO corpus plan",
    )
    if manifest["format"] != IRO_PLAN_FORMAT:
        raise Pix2PixError(f"unsupported iRO corpus plan format: {manifest['format']!r}")
    config_path = corpus_member(root, _string(manifest, "config"), label="plan config")
    requests_path = corpus_member(
        root,
        _string(manifest, "requests"),
        label="plan requests",
    )
    if not config_path.is_file() or not requests_path.is_file():
        raise Pix2PixError(f"incomplete iRO corpus plan: {root}")
    config = load_iro_corpus_config(config_path)
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
    if len({record["identity"] for record in requests}) != _integer(
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
                f"iRO request has no unique canvas-safe APNG frame: {record['id']}"
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
            selected_records.append(
                {
                    **record,
                    "renderer_frame_index": frame["index"],
                    "duration_ms": frame["duration_ms"],
                    "renderer_pixel_sha256": frame["pixel_sha256"],
                    "flux_seed": config.flux.seed_base + request_index,
                    "target": f"selection/targets/{pair_id}.png",
                    "target_sha256": sha256_file(target_path),
                }
            )
        write_json_records(staging / "selected.jsonl", selected_records)
        manifest = {
            "format": IRO_SELECTION_FORMAT,
            "name": config.name,
            "config_fingerprint": plan["config_fingerprint"],
            "plan_requests_sha256": plan["requests_sha256"],
            "selected": "selected.jsonl",
            "selected_sha256": sha256_file(staging / "selected.jsonl"),
            "pair_count": len(selected_records),
            "split_counts": _split_counts(selected_records),
            "lineage_counts": _lineage_counts(selected_records),
            "identity_count": len(
                {record["identity"] for record in selected_records}
            ),
        }
        write_json(staging / "selection.json", manifest)
        os.rename(staging, selection_dir)
    _, selected, loaded_manifest = load_iro_selection(root)
    return _selection_result(root, selected, loaded_manifest, reused=False)


def load_iro_selection(
    root: Path,
) -> tuple[IroCorpusConfig, tuple[dict[str, Any], ...], dict[str, Any]]:
    root = root.expanduser().resolve()
    config, requests, plan = load_iro_plan(root)
    manifest_path = root / "selection" / "selection.json"
    manifest = read_json(manifest_path, label="iRO target selection")
    require_exact_keys(
        manifest,
        {
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
        },
        "iRO target selection",
    )
    if manifest["format"] != IRO_SELECTION_FORMAT:
        raise Pix2PixError(
            f"unsupported iRO target selection format: {manifest['format']!r}"
        )
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
    request_by_id = {str(record["id"]): record for record in requests}
    seen_pixels: set[str] = set()
    for index, record in enumerate(selected):
        require_exact_keys(record, SELECTION_RECORD_KEYS, f"selected target {index}")
        pair_id = _string(record, "id")
        planned = request_by_id.get(pair_id)
        if planned is None:
            raise Pix2PixError(f"selected target is absent from plan: {pair_id}")
        if {key: record[key] for key in REQUEST_RECORD_KEYS} != planned:
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
        _verify_rgb_image(target_path, config.image_size, config.image_size)
        if sha256_file(target_path) != _string(record, "target_sha256"):
            raise Pix2PixError(f"selected target checksum mismatch: {pair_id}")
    if _split_counts(selected) != manifest["split_counts"]:
        raise Pix2PixError("selected target split counts do not match manifest")
    if _lineage_counts(selected) != manifest["lineage_counts"]:
        raise Pix2PixError("selected target lineage counts do not match manifest")
    if len({record["identity"] for record in selected}) != _integer(
        manifest, "identity_count"
    ):
        raise Pix2PixError("selected target identity count does not match manifest")
    return config, selected, manifest


def _planned_requests(config: IroCorpusConfig) -> tuple[dict[str, object], ...]:
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


def _plan_manifest(
    config: IroCorpusConfig,
    requests: tuple[dict[str, object], ...],
    staging: Path,
) -> dict[str, object]:
    return {
        "format": IRO_PLAN_FORMAT,
        "name": config.name,
        "config": "config.json",
        "config_fingerprint": corpus_config_fingerprint(config),
        "requests": "requests.jsonl",
        "requests_sha256": sha256_file(staging / "requests.jsonl"),
        "request_count": len(requests),
        "split_counts": _split_counts(requests),
        "lineage_counts": _lineage_counts(requests),
        "identity_count": len({record["identity"] for record in requests}),
    }


def _audit_request_records(
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


def _audit_request_quotas(
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


def _render_request(
    config: IroCorpusConfig,
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
    config: IroCorpusConfig,
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
    config: IroCorpusConfig,
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
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            target = background.convert("RGB")
            target.save(target_path, format="PNG", optimize=False)
    except OSError as error:
        raise Pix2PixError(f"cannot materialize target {rgba_path}: {error}") from error


def _verify_rgb_image(path: Path, width: int, height: int) -> None:
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (width, height):
                raise Pix2PixError(
                    f"image must be RGB {width}x{height}: {path.as_posix()}"
                )
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
    except OSError as error:
        raise Pix2PixError(f"cannot inspect iRO frame edges {path}: {error}") from error


def _rgba_pixel_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(b"aigen.pix2pix.rgba-pixels.v1\0")
    digest.update(f"{image.width}x{image.height}\0RGBA\0".encode("ascii"))
    digest.update(image.convert("RGBA").tobytes())
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


def _plan_result(
    root: Path,
    requests: tuple[dict[str, Any], ...] | tuple[dict[str, object], ...],
    manifest: dict[str, Any] | dict[str, object],
    *,
    reused: bool,
) -> dict[str, object]:
    return {
        "status": "completed",
        "kind": "iRO-corpus-plan",
        "root": root.as_posix(),
        "reused": reused,
        "config_fingerprint": manifest["config_fingerprint"],
        "request_count": len(requests),
        "split_counts": manifest["split_counts"],
        "lineage_counts": manifest["lineage_counts"],
        "identity_count": manifest["identity_count"],
    }


def _selection_result(
    root: Path,
    selected: tuple[dict[str, Any], ...],
    manifest: dict[str, Any],
    *,
    reused: bool,
) -> dict[str, object]:
    return {
        "status": "completed",
        "kind": "iRO-target-selection",
        "root": root.as_posix(),
        "reused": reused,
        "pair_count": len(selected),
        "split_counts": manifest["split_counts"],
        "lineage_counts": manifest["lineage_counts"],
        "identity_count": manifest["identity_count"],
    }


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
