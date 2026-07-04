from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aigen.character_reference_models import (
    CHARACTER_REFERENCE_ANALYSIS_KIND,
    BodyMeasurementSpec,
    BodyProfileSpec,
    CharacterReferenceAnalysisOutputSpec,
    CharacterReferenceAnalysisSpec,
    CharacterReferenceError,
    CharacterReferencePackSpec,
    ReferenceAnalysisSpec,
    load_completed_character_reference_pack,
)
from aigen.image_assets import image_asset_json
from aigen.keyframe_image_ops import mask_overlay, save_contact_sheet
from aigen.keyframe_pose import (
    BODY_KEYPOINT_COUNT,
    DWPoseKeypointExtractor,
    KeyframePoseError,
    PoseKeypoints,
    PoseScoreConfig,
    save_pose_keypoints_overlay,
)
from aigen.keyframe_segmentation import KeyframeSegmentationError, foreground_box_mask
from aigen.manifest_io import read_json, resolve_existing_path, write_json
from aigen.progress import StatusReporter


REFERENCE_ANALYSIS_FILENAME = "reference_analysis.json"
REFERENCE_ANALYSIS_ARTIFACTS_DIRNAME = "analysis"
BODY_REFERENCE_NAMES = ("front", "side", "back", "body_shape")
REQUIRED_BODY_MEASUREMENTS = (
    "figure_bbox_height",
    "figure_bbox_width_over_height",
    "head_to_body_ratio",
    "shoulder_width_over_height",
    "upper_torso_width_over_shoulder_width",
    "waist_width_over_shoulder_width",
    "hip_or_skirt_width_over_waist_width",
    "side_torso_thickness_over_height",
    "leg_length_over_height",
    "boot_height_over_leg_length",
    "skirt_hem_back_silhouette",
)
BODY_KEYPOINT_NAMES = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)
SILHOUETTE_BANDS = {
    "head": (0.03, 0.18),
    "shoulders": (0.20, 0.32),
    "upper_torso": (0.32, 0.44),
    "waist": (0.45, 0.56),
    "hip_skirt": (0.56, 0.68),
    "legs": (0.70, 0.86),
    "boots": (0.86, 0.98),
}


def analyze_character_reference_pack(
    pack_path: Path,
    *,
    output_path: Path | None,
    artifacts_dir: Path | None,
    overwrite: bool,
    pose_device: str,
    enable_pose: bool,
    progress: StatusReporter,
) -> dict[str, Any]:
    pack_path = pack_path.resolve()
    pack = load_completed_character_reference_pack(
        read_json(pack_path, label="character reference pack"),
        path_label=pack_path.as_posix(),
    )
    target_path = _analysis_output_path(pack_path, output_path)
    target_artifacts_dir = _analysis_artifacts_dir(pack_path, artifacts_dir)
    _prepare_output(target_path, target_artifacts_dir, overwrite=overwrite)

    reference_paths = _reference_paths(pack, pack_path)
    progress.begin(_analysis_progress_total(reference_paths, enable_pose=enable_pose), "analyze reference pack")
    pose_extractor = _load_pose_extractor(pose_device, enable_pose=enable_pose, progress=progress)
    references: dict[str, ReferenceAnalysisSpec] = {}
    reference_payloads: dict[str, dict[str, Any]] = {}
    contact_sheet_items: list[dict[str, Any]] = []
    try:
        for name, image_path in reference_paths.items():
            progress.phase(f"measure {name} reference silhouette")
            reference_payload = _analyze_reference(
                name,
                image_path,
                role=pack.reference_roles[name],
                artifacts_dir=target_artifacts_dir,
                pose_extractor=pose_extractor,
                pose_requested=enable_pose,
                progress=progress,
            )
            reference_payloads[name] = reference_payload
            references[name] = ReferenceAnalysisSpec(**reference_payload["manifest"])
            contact_sheet_items.append({"name": name, "path": reference_payload["manifest"]["artifacts"]["mask_overlay"]})
        if contact_sheet_items:
            progress.phase("write reference analysis contact sheet")
            contact_sheet_path = target_artifacts_dir / "contact_sheet.png"
            save_contact_sheet(contact_sheet_items, contact_sheet_path, thumb_width=192, max_label_chars=24)
            progress.step("wrote reference analysis contact sheet")

        body_profile = _build_body_profile(pack, reference_payloads)
        analysis = CharacterReferenceAnalysisSpec(
            kind=CHARACTER_REFERENCE_ANALYSIS_KIND,
            character_id=pack.character_id,
            source_reference_pack=pack_path.as_posix(),
            references=references,
            body_profile=body_profile,
            output=CharacterReferenceAnalysisOutputSpec(
                reference_analysis=target_path.as_posix(),
                artifacts_directory=target_artifacts_dir.as_posix(),
            ),
        )
        payload = {"status": "completed", **analysis.model_dump(mode="json")}
        progress.phase("write reference analysis manifest")
        write_json(target_path, payload)
        progress.step("wrote reference analysis manifest")
        return payload
    finally:
        if pose_extractor is not None:
            pose_extractor.close()


def _analysis_output_path(pack_path: Path, output_path: Path | None) -> Path:
    if output_path is None:
        return pack_path.parent / REFERENCE_ANALYSIS_FILENAME
    return output_path.resolve()


def _analysis_artifacts_dir(pack_path: Path, artifacts_dir: Path | None) -> Path:
    if artifacts_dir is None:
        return pack_path.parent / REFERENCE_ANALYSIS_ARTIFACTS_DIRNAME
    return artifacts_dir.resolve()


def _prepare_output(target_path: Path, artifacts_dir: Path, *, overwrite: bool) -> None:
    if target_path.exists() and not overwrite:
        raise CharacterReferenceError(f"Reference analysis exists and overwrite=false: {target_path.as_posix()}")
    if artifacts_dir.exists():
        if not overwrite and any(artifacts_dir.iterdir()):
            raise CharacterReferenceError(f"Reference analysis artifacts exist and overwrite=false: {artifacts_dir.as_posix()}")
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)


def _analysis_progress_total(reference_paths: dict[str, Path], *, enable_pose: bool) -> int:
    load_pose_steps = 1 if enable_pose else 0
    per_reference_steps = 2 if enable_pose else 1
    contact_and_manifest_steps = 2
    return load_pose_steps + len(reference_paths) * per_reference_steps + contact_and_manifest_steps


def _reference_paths(pack: CharacterReferencePackSpec, pack_path: Path) -> dict[str, Path]:
    return {
        name: resolve_existing_path(asset.path, pack_path.parent)
        for name, asset in pack.references.items()
    }


def _load_pose_extractor(
    pose_device: str,
    *,
    enable_pose: bool,
    progress: StatusReporter,
) -> DWPoseKeypointExtractor | None:
    if not enable_pose:
        return None
    progress.phase("load DWPose reference analyzer")
    extractor = DWPoseKeypointExtractor(PoseScoreConfig(device=pose_device, min_common_keypoints=4))
    progress.step("loaded DWPose reference analyzer")
    return extractor


def _analyze_reference(
    name: str,
    image_path: Path,
    *,
    role: str,
    artifacts_dir: Path,
    pose_extractor: DWPoseKeypointExtractor | None,
    pose_requested: bool,
    progress: StatusReporter,
) -> dict[str, Any]:
    image = _load_rgb(image_path)
    mask, mask_warnings = _foreground_mask(image_path, image)
    bbox = _mask_bbox(mask)
    row_widths = _bbox_row_widths(mask, bbox)
    mask_path = artifacts_dir / "masks" / f"{name}.png"
    overlay_path = artifacts_dir / "overlays" / f"{name}_mask.png"
    _save_mask(mask, mask_path)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as base_image:
        with Image.open(mask_path) as mask_image:
            mask_overlay(base_image.convert("RGB"), mask_image.convert("L")).save(overlay_path)
    progress.step(f"measured {name} silhouette")

    pose_payload = _pose_payload_unrequested()
    pose_artifact = None
    pose_warnings: list[str] = []
    if pose_requested:
        progress.phase(f"extract {name} DWPose keypoints")
        pose_payload, pose_artifact, pose_warnings = _extract_pose_payload(
            name,
            image_path,
            artifacts_dir,
            pose_extractor,
        )
        progress.step(f"extracted {name} pose evidence")

    artifacts = {
        "mask": mask_path.as_posix(),
        "mask_overlay": overlay_path.as_posix(),
    }
    if pose_artifact is not None:
        artifacts["pose_overlay"] = pose_artifact.as_posix()

    width_profile = _silhouette_width_profile(mask, bbox)
    mask_payload = {
        "bbox": _bbox_json(bbox),
        "area_ratio": float(mask.mean()),
        "width_profile": width_profile,
    }
    warnings = mask_warnings + pose_warnings
    manifest = {
        "role": role,
        "image": image_asset_json(image_path),
        "extractors_used": {
            "segmentation": "foreground_box_mask",
            "pose": "DWPose" if pose_requested else "none",
            "part_masks": "none",
        },
        "artifacts": artifacts,
        "mask": mask_payload,
        "pose": pose_payload,
        "warnings": warnings,
    }
    return {
        "name": name,
        "image_size": (image.shape[1], image.shape[0]),
        "image": image,
        "foreground_mask": mask,
        "bbox": bbox,
        "row_widths": row_widths,
        "mask": mask_payload,
        "pose": pose_payload,
        "warnings": warnings,
        "manifest": manifest,
    }


def _foreground_mask(image_path: Path, image: np.ndarray) -> tuple[np.ndarray, list[str]]:
    try:
        return foreground_box_mask(image), []
    except KeyframeSegmentationError as error:
        raise CharacterReferenceError(f"Could not segment {image_path.as_posix()}: {error}") from error


def _extract_pose_payload(
    name: str,
    image_path: Path,
    artifacts_dir: Path,
    pose_extractor: DWPoseKeypointExtractor | None,
) -> tuple[dict[str, Any], Path | None, list[str]]:
    if pose_extractor is None:
        return _pose_payload_unavailable("DWPose extractor was not loaded"), None, ["DWPose extractor unavailable"]
    try:
        keypoints = pose_extractor.extract(image_path)
    except KeyframePoseError as error:
        message = f"DWPose failed for {name}: {error}"
        return _pose_payload_unavailable(message), None, [message]

    output_path = artifacts_dir / "pose" / f"{name}_dwpose.png"
    save_pose_keypoints_overlay(image_path, keypoints, output_path)
    return _pose_keypoints_payload(keypoints), output_path, []


def _pose_payload_unrequested() -> dict[str, Any]:
    return {
        "extractor": "none",
        "status": "not_requested",
        "visible_body_keypoints": 0,
        "keypoints": {},
    }


def _pose_payload_unavailable(reason: str) -> dict[str, Any]:
    return {
        "extractor": "DWPose",
        "status": "unavailable",
        "reason": reason,
        "visible_body_keypoints": 0,
        "keypoints": {},
    }


def _pose_keypoints_payload(keypoints: PoseKeypoints) -> dict[str, Any]:
    visible: dict[str, dict[str, float]] = {}
    for index in range(BODY_KEYPOINT_COUNT):
        x_norm, y_norm = keypoints.points[index]
        if not np.isfinite(x_norm):
            continue
        visible[BODY_KEYPOINT_NAMES[index]] = {
            "x": float(x_norm),
            "y": float(y_norm),
            "score": float(keypoints.scores[index]),
        }
    return {
        "extractor": "DWPose",
        "status": "completed",
        "visible_body_keypoints": len(visible),
        "image_size": {
            "width": keypoints.image_size[0],
            "height": keypoints.image_size[1],
        },
        "keypoints": visible,
    }


def _build_body_profile(
    pack: CharacterReferencePackSpec,
    references: dict[str, dict[str, Any]],
) -> BodyProfileSpec:
    body_refs = {name: references[name] for name in BODY_REFERENCE_NAMES if name in references}
    if not body_refs:
        raise CharacterReferenceError("Reference analysis needs at least one full-body reference")

    optional_missing_refs = [name for name in ("body_shape",) if name not in pack.references]
    warnings: list[str] = []
    measurements: dict[str, BodyMeasurementSpec] = {}

    primary = _primary_body_reference(body_refs)
    side = body_refs.get("side")
    back = body_refs.get("back")

    _add_measurement(
        measurements,
        "figure_bbox_height",
        value=float(_bbox_height(primary["bbox"])),
        unit="pixels",
        confidence=0.92,
        evidence_refs=[primary["name"]],
        evidence={"bbox": _bbox_json(primary["bbox"])},
    )
    _add_measurement(
        measurements,
        "figure_bbox_width_over_height",
        value=_bbox_width(primary["bbox"]) / _bbox_height(primary["bbox"]),
        unit="ratio",
        confidence=0.88,
        evidence_refs=[primary["name"]],
        evidence={"bbox": _bbox_json(primary["bbox"])},
    )
    _add_pose_measurement(
        measurements,
        warnings,
        "head_to_body_ratio",
        _head_to_body_ratio(primary) or _head_to_body_ratio_from_silhouette(primary),
    )
    _add_pose_or_band_measurement(
        measurements,
        warnings,
        "shoulder_width_over_height",
        primary,
        pose_measurement=_shoulder_width_over_height(primary),
        band="shoulders",
    )
    _add_band_ratio_measurement(
        measurements,
        warnings,
        "upper_torso_width_over_shoulder_width",
        primary,
        numerator="upper_torso",
        denominator="shoulders",
    )
    _add_band_ratio_measurement(
        measurements,
        warnings,
        "waist_width_over_shoulder_width",
        primary,
        numerator="waist",
        denominator="shoulders",
    )
    _add_band_ratio_measurement(
        measurements,
        warnings,
        "hip_or_skirt_width_over_waist_width",
        primary,
        numerator="hip_skirt",
        denominator="waist",
    )
    if side is not None:
        _add_band_measurement(
            measurements,
            warnings,
            "side_torso_thickness_over_height",
            side,
            band="upper_torso",
        )
    else:
        warnings.append("side_torso_thickness_over_height used front/reference fallback because no side reference was supplied")
        _add_band_measurement(
            measurements,
            warnings,
            "side_torso_thickness_over_height",
            primary,
            band="upper_torso",
            confidence=0.38,
        )
    _add_pose_measurement(
        measurements,
        warnings,
        "leg_length_over_height",
        _leg_length_over_height(primary) or _leg_length_over_height_from_silhouette(primary),
    )
    _add_measurement(
        measurements,
        "boot_height_over_leg_length",
        **_boot_height_over_leg_length(primary),
    )
    if back is not None:
        _add_band_measurement(
            measurements,
            warnings,
            "skirt_hem_back_silhouette",
            back,
            band="hip_skirt",
        )
    else:
        warnings.append("skirt_hem_back_silhouette used front/reference fallback because no back reference was supplied")
        _add_band_measurement(
            measurements,
            warnings,
            "skirt_hem_back_silhouette",
            primary,
            band="hip_skirt",
            confidence=0.46,
        )
    _validate_required_measurements(measurements)

    evidence_refs = [name for name in BODY_REFERENCE_NAMES if name in references]
    return BodyProfileSpec(
        source="measured_from_reference_pack",
        extractors={
            "pose": "DWPose",
            "segmentation": "foreground_box_mask",
            "part_masks": "silhouette_color_boundary_heuristics",
            "semantic_summarizer": "identity_profile_only",
        },
        measurements=measurements,
        semantic_summary=_body_semantic_summary(measurements),
        evidence_refs=evidence_refs,
        optional_missing_refs=optional_missing_refs,
        confidence_warnings=list(dict.fromkeys(warnings)),
    )


def _primary_body_reference(body_refs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for name in ("front", "body_shape", "back", "side"):
        if name in body_refs:
            return body_refs[name]
    raise CharacterReferenceError("Reference analysis needs at least one full-body reference")


def _add_measurement(
    measurements: dict[str, BodyMeasurementSpec],
    name: str,
    *,
    value: float,
    unit: str,
    confidence: float,
    evidence_refs: list[str],
    evidence: dict[str, Any],
) -> None:
    measurements[name] = BodyMeasurementSpec(
        value=round(float(value), 6),
        unit=unit,
        confidence=round(float(confidence), 3),
        evidence_refs=evidence_refs,
        evidence=evidence,
    )


def _add_pose_measurement(
    measurements: dict[str, BodyMeasurementSpec],
    warnings: list[str],
    name: str,
    candidate: dict[str, Any] | None,
    *,
    fallback_warning: str,
) -> None:
    if candidate is None:
        warnings.append(fallback_warning)
        return
    _add_measurement(measurements, name, **candidate)


def _add_pose_or_band_measurement(
    measurements: dict[str, BodyMeasurementSpec],
    warnings: list[str],
    name: str,
    reference: dict[str, Any],
    *,
    pose_measurement: dict[str, Any] | None,
    band: str,
) -> None:
    if pose_measurement is not None:
        _add_measurement(measurements, name, **pose_measurement)
        return
    warnings.append(f"{name} used silhouette band fallback because DWPose shoulder evidence was incomplete")
    _add_band_measurement(measurements, warnings, name, reference, band=band, confidence=0.62)


def _add_band_measurement(
    measurements: dict[str, BodyMeasurementSpec],
    warnings: list[str],
    name: str,
    reference: dict[str, Any],
    *,
    band: str,
    confidence: float = 0.72,
) -> None:
    profile = reference["mask"]["width_profile"]
    if band not in profile:
        warnings.append(f"{name} needs silhouette band {band}")
        return
    _add_measurement(
        measurements,
        name,
        value=profile[band]["width_over_height"],
        unit="ratio",
        confidence=confidence,
        evidence_refs=[reference["name"]],
        evidence={"band": band, "width_profile": profile[band]},
    )


def _add_band_ratio_measurement(
    measurements: dict[str, BodyMeasurementSpec],
    warnings: list[str],
    name: str,
    reference: dict[str, Any],
    *,
    numerator: str,
    denominator: str,
) -> None:
    profile = reference["mask"]["width_profile"]
    if numerator not in profile or denominator not in profile:
        warnings.append(f"{name} needs silhouette bands {numerator} and {denominator}")
        return
    denominator_width = profile[denominator]["width_px"]
    if denominator_width <= 0:
        warnings.append(f"{name} has zero-width denominator band {denominator}")
        return
    _add_measurement(
        measurements,
        name,
        value=profile[numerator]["width_px"] / denominator_width,
        unit="ratio",
        confidence=0.72,
        evidence_refs=[reference["name"]],
        evidence={
            "numerator_band": numerator,
            "denominator_band": denominator,
            "numerator": profile[numerator],
            "denominator": profile[denominator],
        },
    )


def _head_to_body_ratio(reference: dict[str, Any]) -> dict[str, Any] | None:
    keypoints = reference["pose"]["keypoints"]
    neck = keypoints.get("neck")
    if neck is None:
        return None
    bbox = reference["bbox"]
    _, image_height = reference["image_size"]
    value = (neck["y"] * image_height - bbox[1]) / _bbox_height(bbox)
    if value <= 0:
        return None
    return {
        "value": value,
        "unit": "ratio",
        "confidence": min(0.82, max(0.35, neck["score"])),
        "evidence_refs": [reference["name"]],
        "evidence": {"keypoint": "neck", "bbox": _bbox_json(bbox)},
    }


def _shoulder_width_over_height(reference: dict[str, Any]) -> dict[str, Any] | None:
    keypoints = reference["pose"]["keypoints"]
    right = keypoints.get("right_shoulder")
    left = keypoints.get("left_shoulder")
    if right is None or left is None:
        return None
    bbox = reference["bbox"]
    image_width, image_height = reference["image_size"]
    dx = (left["x"] - right["x"]) * image_width
    dy = (left["y"] - right["y"]) * image_height
    value = float(np.sqrt(dx * dx + dy * dy) / _bbox_height(bbox))
    return {
        "value": value,
        "unit": "ratio",
        "confidence": min(0.88, max(0.45, (left["score"] + right["score"]) / 2.0)),
        "evidence_refs": [reference["name"]],
        "evidence": {"keypoints": ["right_shoulder", "left_shoulder"], "bbox": _bbox_json(bbox)},
    }


def _leg_length_over_height(reference: dict[str, Any]) -> dict[str, Any] | None:
    keypoints = reference["pose"]["keypoints"]
    segments: list[tuple[dict[str, float], dict[str, float], dict[str, float]]] = []
    for hip, knee, ankle in (
        ("right_hip", "right_knee", "right_ankle"),
        ("left_hip", "left_knee", "left_ankle"),
    ):
        if hip in keypoints and knee in keypoints and ankle in keypoints:
            segments.append((keypoints[hip], keypoints[knee], keypoints[ankle]))
    if not segments:
        return None
    image_width, image_height = reference["image_size"]
    leg_lengths = [
        _point_distance_px(hip, knee, image_width, image_height)
        + _point_distance_px(knee, ankle, image_width, image_height)
        for hip, knee, ankle in segments
    ]
    scores = [min(hip["score"], knee["score"], ankle["score"]) for hip, knee, ankle in segments]
    return {
        "value": float(np.mean(leg_lengths) / _bbox_height(reference["bbox"])),
        "unit": "ratio",
        "confidence": min(0.84, max(0.42, float(np.mean(scores)))),
        "evidence_refs": [reference["name"]],
        "evidence": {"keypoint_chains": len(segments), "bbox": _bbox_json(reference["bbox"])},
    }


def _body_semantic_summary(measurements: dict[str, BodyMeasurementSpec]) -> dict[str, str]:
    summary = {
        "source": "measured masks and keypoints; VLM text is not authoritative for body proportions",
        "style_constraint": "preserve stylized character silhouette from the reference pack",
    }
    upper = measurements.get("upper_torso_width_over_shoulder_width")
    waist = measurements.get("waist_width_over_shoulder_width")
    side = measurements.get("side_torso_thickness_over_height")
    if upper is not None:
        summary["upper_torso"] = f"preserve measured upper torso ratio {upper.value:.3f}"
    if waist is not None:
        summary["waist"] = f"preserve measured waist ratio {waist.value:.3f}"
    if side is not None:
        summary["side_thickness"] = f"preserve measured side thickness ratio {side.value:.3f}"
    return summary


def _silhouette_width_profile(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, dict[str, float]]:
    left, top, right, bottom = bbox
    bbox_height = _bbox_height(bbox)
    bbox_width = _bbox_width(bbox)
    profile: dict[str, dict[str, float]] = {}
    for name, (start, end) in SILHOUETTE_BANDS.items():
        y0 = min(bottom, top + int(round(start * bbox_height)))
        y1 = min(bottom, max(y0 + 1, top + int(round(end * bbox_height))))
        widths = _row_widths(mask[y0:y1, left:right])
        if widths.size == 0:
            continue
        width_px = float(np.median(widths))
        profile[name] = {
            "width_px": round(width_px, 3),
            "width_over_height": round(width_px / bbox_height, 6),
            "width_over_bbox_width": round(width_px / bbox_width, 6),
            "sample_rows": int(widths.size),
        }
    return profile


def _row_widths(mask_slice: np.ndarray) -> np.ndarray:
    widths = []
    for row in mask_slice:
        xs = np.flatnonzero(row)
        if xs.size:
            widths.append(int(xs[-1] - xs[0] + 1))
    return np.asarray(widths, dtype=np.float32)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise CharacterReferenceError("Foreground segmentation returned an empty character mask")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _bbox_json(bbox: tuple[int, int, int, int]) -> dict[str, int]:
    left, top, right, bottom = bbox
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top,
    }


def _bbox_width(bbox: tuple[int, int, int, int]) -> int:
    return max(1, bbox[2] - bbox[0])


def _bbox_height(bbox: tuple[int, int, int, int]) -> int:
    return max(1, bbox[3] - bbox[1])


def _point_distance_px(
    first: dict[str, float],
    second: dict[str, float],
    image_width: int,
    image_height: int,
) -> float:
    dx = (first["x"] - second["x"]) * image_width
    dy = (first["y"] - second["y"]) * image_height
    return float(np.sqrt(dx * dx + dy * dy))


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(path)
