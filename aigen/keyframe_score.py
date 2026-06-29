from __future__ import annotations

import shutil
from contextlib import ExitStack, closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from aigen.keyframe_judge import DEFAULT_JUDGE_ID
from aigen.keyframe_pose import (
    DWPoseKeypointExtractor,
    PoseKeypoints,
    PoseScoreConfig,
    extract_target_pose_map_keypoints,
    save_pose_evidence,
    score_pose_match,
)
from aigen.keyframe_segmentation import SamForegroundSegmenter, SamSegmentationConfig
from aigen.manifest_io import read_json, sha256_file, write_json


DEFAULT_SCORER_ID = "condition"
DEGENERATE_POSE_SCORE_THRESHOLD = 0.05
SEMANTIC_SCORE_FLOOR = 8.5
ARTIFACT_SCORE_FLOOR = 0.60
SEMANTIC_SELECTION_SCORE_KEYS = (
    "condition_adherence",
    "pose_match",
    "contour_match",
    "side_profile",
)


class KeyframeScoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class KeyframeScoreConfig:
    scorer_id: str = DEFAULT_SCORER_ID
    contour_radius: int = 8
    distance_scale_px: float = 48.0
    pose: PoseScoreConfig = field(default_factory=PoseScoreConfig)
    segmentation: SamSegmentationConfig = field(default_factory=SamSegmentationConfig)


def score_keyframe_run(
    run_dir: Path,
    config: KeyframeScoreConfig,
    *,
    project_root: Path,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.resolve()
    result = read_json(resolved_run_dir / "result.json", label="keyframe run result")
    assets = result["assets"]
    score_dir = resolved_run_dir / "score" / config.scorer_id
    masks_dir = score_dir / "masks"
    evidence_dir = score_dir / "evidence"
    pose_evidence_dir = score_dir / "pose_evidence"
    if score_dir.exists():
        shutil.rmtree(score_dir)
    masks_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    pose_evidence_dir.mkdir(parents=True)

    target = _load_target(assets)
    pose_target_source = _pose_target_source(assets)
    with ExitStack() as resources:
        pose_extractor = resources.enter_context(closing(DWPoseKeypointExtractor(config.pose)))
        segmenter = resources.enter_context(closing(SamForegroundSegmenter(config.segmentation)))
        pose_target = _load_pose_target(assets, config, pose_extractor, pose_target_source)

        candidates = [
            _score_candidate(
                output,
                target,
                pose_target,
                pose_extractor,
                segmenter,
                config,
                masks_dir=masks_dir,
                evidence_dir=evidence_dir,
                pose_evidence_dir=pose_evidence_dir,
            )
            for output in result["outputs"]
        ]
        ranking = [candidate["candidate"] for candidate in sorted(candidates, key=_score_ranking_key)]
        ordered = {candidate["candidate"]: candidate for candidate in candidates}
        ranked_candidates = [ordered[name] for name in ranking]
        _save_ranked_sheet(ranked_candidates, score_dir / "ranked_contact_sheet.png", image_key="image")
        _save_ranked_sheet(ranked_candidates, score_dir / "condition_evidence_ranked.png", image_key="condition_evidence")
        _save_ranked_sheet(ranked_candidates, score_dir / "pose_evidence_ranked.png", image_key="pose_evidence")

        payload = {
            "status": "completed",
            "run_dir": resolved_run_dir.as_posix(),
            "job_id": result["job_id"],
            "git_commit": _git_commit(project_root),
            "source_result_sha256": sha256_file(resolved_run_dir / "result.json"),
            "scorer": {
                "id": config.scorer_id,
                "method": "sam-box-prompted-foreground-boundary-to-target-contour",
                "contour_radius": config.contour_radius,
                "distance_scale_px": config.distance_scale_px,
                "pose": {
                    "distance_scale": config.pose.distance_scale,
                    "min_common_keypoints": config.pose.min_common_keypoints,
                    "det_model": config.pose.det_model.as_posix(),
                    "pose_model": config.pose.pose_model.as_posix(),
                },
                "segmentation": {
                    "model": "segment-anything",
                    "model_type": config.segmentation.model_type,
                    "checkpoint": config.segmentation.checkpoint.as_posix(),
                    "device": config.segmentation.device,
                },
                "weights": SCORE_WEIGHTS,
            },
            "assets": {
                "contour": assets["contour"],
                "boundary_mask": assets["boundary_mask"],
                "pose": assets["pose"],
            },
            "candidates": candidates,
            "ranking": {
                "final": ranking,
                "best": ranking[0],
            },
            "selection": _selection_status(candidates),
            "outputs": {
                "scores": (score_dir / "scores.json").as_posix(),
                "ranked_contact_sheet": (score_dir / "ranked_contact_sheet.png").as_posix(),
                "condition_evidence_ranked": (score_dir / "condition_evidence_ranked.png").as_posix(),
                "pose_evidence_ranked": (score_dir / "pose_evidence_ranked.png").as_posix(),
            },
        }
        write_json(score_dir / "scores.json", payload)
        return payload


def select_scored_keyframe_run(
    run_dir: Path,
    *,
    scorer_id: str = DEFAULT_SCORER_ID,
    top_k: int = 1,
) -> dict[str, Any]:
    if top_k < 1:
        raise KeyframeScoreError("top_k must be at least 1")
    resolved_run_dir = run_dir.resolve()
    score_dir = resolved_run_dir / "score" / scorer_id
    score_result = read_json(score_dir / "scores.json", label="keyframe score result")
    selection = score_result["selection"]
    if not selection["usable_for_auto_select"]:
        raise KeyframeScoreError(
            "Refusing automatic score selection: scorer evidence is not usable for auto-select "
            f"({', '.join(selection['blockers'])})."
        )

    semantic_gate = _load_semantic_selection_gate(resolved_run_dir, score_result)
    candidates_by_name = {candidate["candidate"]: candidate for candidate in score_result["candidates"]}
    eligible_names = [
        name
        for name in score_result["ranking"]["final"]
        if _score_candidate_selectable(candidates_by_name[name]) and name in semantic_gate["passed"]
    ]
    if not eligible_names:
        raise KeyframeScoreError("Refusing automatic score selection: no candidate passed score and semantic gates.")
    selected_names = eligible_names[:top_k]
    selected_names_set = set(selected_names)
    selected = [candidates_by_name[name] for name in selected_names]
    rejected = [
        candidates_by_name[name]
        for name in score_result["ranking"]["final"]
        if name not in selected_names_set
    ]
    selected_path = score_dir / "selected.json"
    rejected_path = score_dir / "rejected.json"
    selected_contact_sheet_path = resolved_run_dir / "selected_contact_sheet.png"
    _save_ranked_sheet(selected, selected_contact_sheet_path, image_key="image")
    selected_payload = {
        "selection_mode": "condition_score_with_semantic_gate",
        "scorer": scorer_id,
        "semantic_gate": semantic_gate,
        "outputs": {
            "selected_contact_sheet": selected_contact_sheet_path.as_posix(),
        },
        "selected": selected,
    }
    rejected_payload = {
        "selection_mode": "condition_score_with_semantic_gate",
        "scorer": scorer_id,
        "semantic_gate": semantic_gate,
        "rejected": rejected,
    }
    write_json(selected_path, selected_payload)
    write_json(rejected_path, rejected_payload)
    return {
        "status": "completed",
        "scorer": scorer_id,
        "run_dir": resolved_run_dir.as_posix(),
        "selection": selection,
        "semantic_gate": semantic_gate,
        "best": selected[0]["candidate"],
        "selected": [candidate["candidate"] for candidate in selected],
        "rejected": [candidate["candidate"] for candidate in rejected],
        "outputs": {
            "selected": selected_path.as_posix(),
            "rejected": rejected_path.as_posix(),
            "selected_contact_sheet": selected_contact_sheet_path.as_posix(),
            "ranked_contact_sheet": score_result["outputs"]["ranked_contact_sheet"],
            "condition_evidence_ranked": score_result["outputs"]["condition_evidence_ranked"],
            "pose_evidence_ranked": score_result["outputs"]["pose_evidence_ranked"],
        },
    }


def _load_semantic_selection_gate(run_dir: Path, score_result: dict[str, Any]) -> dict[str, Any]:
    judge_path = run_dir / "judge" / DEFAULT_JUDGE_ID / "judge.json"
    if not judge_path.exists():
        raise KeyframeScoreError(
            f"Refusing automatic score selection: missing semantic judge evidence at {judge_path.as_posix()}."
        )
    judge_result = read_json(judge_path, label="keyframe judge result")
    judged = {candidate["candidate"]: candidate["judgment"] for candidate in judge_result["candidates"]}
    score_candidates = [candidate["candidate"] for candidate in score_result["candidates"]]
    missing = [name for name in score_candidates if name not in judged]
    if missing:
        raise KeyframeScoreError(f"Semantic judge evidence is missing candidates: {', '.join(missing)}")

    passed = []
    blocked = []
    for name in score_candidates:
        judgment = judged[name]
        hard_rejects = [key for key, rejected in judgment["hard_rejects"].items() if rejected]
        floor_score = min(float(judgment["scores"][key]) for key in SEMANTIC_SELECTION_SCORE_KEYS)
        blockers = []
        if not judgment["pass"]:
            blockers.append("semantic_pass_false")
        blockers.extend(hard_rejects)
        if floor_score < SEMANTIC_SCORE_FLOOR:
            blockers.append("semantic_score_floor")
        if blockers:
            blocked.append(
                {
                    "candidate": name,
                    "blockers": blockers,
                    "score_floor": floor_score,
                }
            )
        else:
            passed.append(name)
    if not passed:
        raise KeyframeScoreError("Refusing automatic score selection: semantic gate passed no candidates.")
    return {
        "judge": DEFAULT_JUDGE_ID,
        "selection_owner": "condition_score",
        "usable_for_auto_select": True,
        "score_floor": SEMANTIC_SCORE_FLOOR,
        "score_keys": list(SEMANTIC_SELECTION_SCORE_KEYS),
        "passed": passed,
        "blocked": blocked,
    }


def _score_candidate_selectable(candidate: dict[str, Any]) -> bool:
    return not any(candidate["hard_rejects"].values())


def _selection_status(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    blockers = []
    if max(candidate["scores"]["pose"] for candidate in candidates) < DEGENERATE_POSE_SCORE_THRESHOLD:
        blockers.append("pose_score_degenerate_all_candidates")
    if all(any(candidate["hard_rejects"].values()) for candidate in candidates):
        blockers.append("all_candidates_have_hard_rejects")
    return {
        "usable_for_auto_select": len(blockers) == 0,
        "blockers": blockers,
    }


def _load_target(assets: dict[str, Any]) -> dict[str, Any]:
    contour = _load_luma_mask(Path(assets["contour"]["path"]), threshold=32)
    boundary_weight = _load_boundary_weight(Path(assets["boundary_mask"]["path"]))
    target_bbox = _bbox(contour)
    target_silhouette = ndimage.binary_fill_holes(
        ndimage.binary_closing(contour, structure=np.ones((5, 5), dtype=bool), iterations=2)
    )
    return {
        "contour": contour,
        "contour_dilated": _dilate(contour, 8),
        "silhouette": target_silhouette,
        "boundary_weight": boundary_weight,
        "bbox": target_bbox,
        "bbox_ratio": _bbox_ratio(target_bbox),
    }


def _load_pose_target(
    assets: dict[str, Any],
    config: KeyframeScoreConfig,
    pose_extractor: Any,
    pose_target_source: Path | None,
) -> PoseKeypoints:
    pose_asset = assets["pose"]
    if pose_target_source is not None:
        return pose_extractor.extract(pose_target_source)
    return extract_target_pose_map_keypoints(Path(pose_asset["path"]), config.pose)


def _pose_target_source(assets: dict[str, Any]) -> Path | None:
    pose_asset = assets["pose"]
    pose_path = Path(pose_asset["path"])
    metadata_path = pose_path.with_name(f"{pose_path.stem.removesuffix('_pose')}_extraction.json")
    if not metadata_path.is_file():
        return None
    metadata = read_json(metadata_path, label="pose extraction metadata")
    return Path(metadata["assets"]["normalized_source"]["path"])


def _score_candidate(
    output: dict[str, Any],
    target: dict[str, Any],
    pose_target: PoseKeypoints,
    pose_extractor: Any,
    segmenter: Any,
    config: KeyframeScoreConfig,
    *,
    masks_dir: Path,
    evidence_dir: Path,
    pose_evidence_dir: Path,
) -> dict[str, Any]:
    candidate_name = output["name"]
    image_path = Path(output["path"]).resolve()
    image = _load_rgb(image_path)
    foreground = segmenter.segment(image_path)
    foreground_edge = foreground ^ _erode(foreground, 1)
    foreground_bbox = _bbox(foreground)
    candidate_edge_dilated = _dilate(foreground_edge, config.contour_radius)
    target_contour = target["contour"]
    target_contour_dilated = _dilate(target_contour, config.contour_radius)
    boundary_weight = target["boundary_weight"]

    weighted_edge = _weighted_sum(foreground_edge, boundary_weight)
    contour_precision = _weighted_sum(foreground_edge & target_contour_dilated, boundary_weight) / weighted_edge
    weighted_contour = _weighted_sum(target_contour, boundary_weight)
    contour_recall = _weighted_sum(target_contour & candidate_edge_dilated, boundary_weight) / weighted_contour
    contour_f1 = _f1(contour_precision, contour_recall)

    distance = ndimage.distance_transform_edt(~foreground_edge)
    weighted_distance_px = float((distance[target_contour] * boundary_weight[target_contour]).sum() / weighted_contour)
    contour_distance_score = max(0.0, 1.0 - weighted_distance_px / config.distance_scale_px)
    silhouette_intersection = float((foreground & target["silhouette"]).sum())
    silhouette_union = float((foreground | target["silhouette"]).sum() + 1e-9)
    silhouette_iou = silhouette_intersection / silhouette_union
    outside_target_ratio = float((foreground & ~target["silhouette"]).sum() / (foreground.sum() + 1e-9))
    silhouette_fit = silhouette_iou * (1.0 - outside_target_ratio)
    candidate_bbox_ratio = _bbox_ratio(foreground_bbox)
    side_profile_score = max(0.0, 1.0 - abs(candidate_bbox_ratio - target["bbox_ratio"]) / target["bbox_ratio"])
    foreground_coverage = float(foreground.sum() / foreground.size)

    scores = {
        "condition": _weighted_score(
            {
                "contour_f1": contour_f1,
                "contour_distance": contour_distance_score,
                "silhouette_fit": silhouette_fit,
            },
            {
                "contour_f1": 0.30,
                "contour_distance": 0.20,
                "silhouette_fit": 0.50,
            },
        ),
        "contour": contour_f1,
        "pose": 0.0,
        "side_profile": side_profile_score,
        "artifact": _artifact_score(foreground_coverage),
    }
    candidate_pose = pose_extractor.extract(image_path)
    pose_result = score_pose_match(pose_target, candidate_pose, config.pose)
    scores["pose"] = pose_result.score
    pose_metrics = pose_result.to_json()
    pose_evidence_path = pose_evidence_dir / f"{candidate_name}__pose_match.png"
    save_pose_evidence(image_path, pose_target, candidate_pose, pose_evidence_path)

    scores["final"] = _weighted_score(scores, SCORE_WEIGHTS)
    hard_rejects = {
        "missing_foreground": bool(foreground.sum() == 0),
        "weak_condition_match": bool(scores["condition"] < 0.25),
        "weak_side_profile": bool(scores["side_profile"] < 0.45),
        "weak_pose_match": bool(scores["pose"] < 0.20),
        "artifact_quality_failure": bool(scores["artifact"] < ARTIFACT_SCORE_FLOOR),
    }
    mask_path = masks_dir / f"{candidate_name}__foreground.png"
    evidence_path = evidence_dir / f"{candidate_name}__condition_diff.png"
    _save_mask(foreground, mask_path)
    _save_condition_evidence(image, target_contour, foreground_edge, evidence_path)
    return {
        "candidate": candidate_name,
        "image": image_path.as_posix(),
        "foreground_mask": mask_path.as_posix(),
        "condition_evidence": evidence_path.as_posix(),
        "pose_evidence": pose_evidence_path.as_posix(),
        "hard_rejects": hard_rejects,
        "metrics": {
            "contour_precision": contour_precision,
            "contour_recall": contour_recall,
            "contour_f1": contour_f1,
            "silhouette_iou": silhouette_iou,
            "outside_target_ratio": outside_target_ratio,
            "silhouette_fit": silhouette_fit,
            "weighted_target_distance_px": weighted_distance_px,
            "contour_distance_score": contour_distance_score,
            "target_bbox_ratio": target["bbox_ratio"],
            "candidate_bbox_ratio": candidate_bbox_ratio,
            "foreground_coverage": foreground_coverage,
            "pose": pose_metrics,
        },
        "scores": scores,
    }


SCORE_WEIGHTS = {
    "condition": 0.30,
    "contour": 0.05,
    "pose": 0.40,
    "side_profile": 0.20,
    "artifact": 0.05,
}


def _score_ranking_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        sum(1 for rejected in candidate["hard_rejects"].values() if rejected),
        -candidate["scores"]["final"],
        -candidate["scores"]["condition"],
        -candidate["scores"]["contour"],
        -candidate["scores"]["pose"],
        -candidate["scores"]["side_profile"],
    )


def _weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    return float(sum(scores[name] * weight for name, weight in weights.items()))


def _f1(precision: float, recall: float) -> float:
    return float(2.0 * precision * recall / (precision + recall + 1e-9))


def _artifact_score(foreground_coverage: float) -> float:
    if 0.04 <= foreground_coverage <= 0.55:
        return 1.0
    return max(0.0, 1.0 - min(abs(foreground_coverage - 0.18), 0.18) / 0.18)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_luma_mask(path: Path, *, threshold: int) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > threshold


def _load_boundary_weight(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return 0.25 + 0.75 * (np.asarray(image.convert("L"), dtype=np.float32) / 255.0)


def _weighted_sum(mask: np.ndarray, weight: np.ndarray) -> float:
    return float(weight[mask].sum() + 1e-9)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise KeyframeScoreError("Cannot score an empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _bbox_ratio(box: tuple[int, int, int, int]) -> float:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return float(width / height)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=radius)


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    return ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=radius)


def _save_mask(mask: np.ndarray, output_path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(output_path)


def _save_condition_evidence(
    image: np.ndarray,
    target_contour: np.ndarray,
    candidate_edge: np.ndarray,
    output_path: Path,
) -> None:
    canvas = Image.fromarray(image, mode="RGB").convert("RGBA")
    evidence = np.zeros((*target_contour.shape, 4), dtype=np.uint8)
    evidence[target_contour & ~candidate_edge] = (255, 0, 0, 210)
    evidence[candidate_edge & ~target_contour] = (0, 180, 255, 190)
    evidence[target_contour & candidate_edge] = (255, 255, 255, 230)
    composed = Image.alpha_composite(canvas, Image.fromarray(evidence, mode="RGBA")).convert("RGB")
    composed.save(output_path)


def _save_ranked_sheet(candidates: list[dict[str, Any]], output_path: Path, *, image_key: str) -> None:
    images = []
    for rank, candidate in enumerate(candidates, start=1):
        with Image.open(candidate[image_key]) as image:
            images.append((rank, candidate, image.convert("RGB").copy()))
    thumb_w = 256
    label_h = 58
    thumb_h = max(1, int(thumb_w * images[0][2].height / images[0][2].width))
    sheet = Image.new("RGB", (thumb_w * len(images), thumb_h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (rank, candidate, image) in enumerate(images):
        x = index * thumb_w
        sheet.paste(image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, label_h))
        draw.text((x + 8, 6), f"#{rank} {candidate['candidate']}", fill="black")
        draw.text((x + 8, 24), f"final {candidate['scores']['final']:.3f}", fill="black")
        draw.text((x + 8, 40), f"condition {candidate['scores']['condition']:.3f}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _git_commit(project_root: Path) -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
