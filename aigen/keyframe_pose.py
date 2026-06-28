from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


OPENPOSE_BODY_COLORS = np.asarray(
    [
        (255, 0, 0),
        (255, 85, 0),
        (255, 170, 0),
        (255, 255, 0),
        (170, 255, 0),
        (85, 255, 0),
        (0, 255, 0),
        (0, 255, 85),
        (0, 255, 170),
        (0, 255, 255),
        (0, 170, 255),
        (0, 85, 255),
        (0, 0, 255),
        (85, 0, 255),
        (170, 0, 255),
        (255, 0, 255),
        (255, 0, 170),
        (255, 0, 85),
    ],
    dtype=np.float32,
)

BODY_KEYPOINT_COUNT = len(OPENPOSE_BODY_COLORS)
DEFAULT_MODELS_ROOT = Path(__file__).resolve().parent / "models"
DEFAULT_DWPOSE_DET_MODEL = DEFAULT_MODELS_ROOT / "annotators/yzd-v/DWPose/yolox_l.onnx"
DEFAULT_DWPOSE_POSE_MODEL = DEFAULT_MODELS_ROOT / "annotators/yzd-v/DWPose/dw-ll_ucoco_384.onnx"


class KeyframePoseError(RuntimeError):
    pass


@dataclass(frozen=True)
class PoseKeypoints:
    points: np.ndarray
    scores: np.ndarray
    image_size: tuple[int, int]


@dataclass(frozen=True)
class PoseScoreConfig:
    distance_scale: float = 0.30
    min_common_keypoints: int = 6
    target_color_tolerance: float = 120.0
    detector_score_threshold: float = 0.30
    device: str = "cpu"
    det_model: Path = DEFAULT_DWPOSE_DET_MODEL
    pose_model: Path = DEFAULT_DWPOSE_POSE_MODEL


@dataclass(frozen=True)
class PoseScoreResult:
    score: float
    common_keypoints: int
    weighted_mean_distance: float
    aligned_mean_distance: float

    def to_json(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "common_keypoints": self.common_keypoints,
            "weighted_mean_distance": self.weighted_mean_distance,
            "aligned_mean_distance": self.aligned_mean_distance,
        }


class DWPoseKeypointExtractor:
    def __init__(self, config: PoseScoreConfig):
        try:
            from controlnet_dwpose import DWposeDetector
        except ImportError as error:
            raise KeyframePoseError("DWPose scoring requires the controlnet-dwpose package.") from error

        _require_model_file(config.det_model)
        _require_model_file(config.pose_model)
        self._config = config
        self._detector = DWposeDetector(
            config.det_model.as_posix(),
            config.pose_model.as_posix(),
            device=config.device,
        )

    def extract(self, image_path: Path) -> PoseKeypoints:
        rgb = _load_rgb(image_path)
        height, width, _ = rgb.shape
        pose = self._detector(rgb)
        subset = pose["bodies"]["subset"]
        candidates = pose["bodies"]["candidate"]
        scores = pose["bodies"]["score"]
        if len(subset) == 0:
            raise KeyframePoseError(f"DWPose found no body in {image_path.as_posix()}")

        person_index = _best_person_index(scores)
        person_scores = scores[person_index, :BODY_KEYPOINT_COUNT].astype(np.float32)
        normalized = np.full((BODY_KEYPOINT_COUNT, 2), np.nan, dtype=np.float32)
        for body_index, candidate_index in enumerate(subset[person_index, :BODY_KEYPOINT_COUNT].astype(int)):
            if candidate_index < 0 or person_scores[body_index] < self._config.detector_score_threshold:
                continue
            normalized[body_index] = candidates[candidate_index]
        return PoseKeypoints(points=normalized, scores=person_scores, image_size=(width, height))


def extract_target_pose_map_keypoints(image_path: Path, config: PoseScoreConfig) -> PoseKeypoints:
    image = _load_rgb(image_path).astype(np.float32)
    height, width, _ = image.shape
    points = np.full((BODY_KEYPOINT_COUNT, 2), np.nan, dtype=np.float32)
    scores = np.zeros(BODY_KEYPOINT_COUNT, dtype=np.float32)
    labels = _body_color_labels(image, config.target_color_tolerance)

    for index in range(BODY_KEYPOINT_COUNT):
        mask = labels == index
        if not mask.any():
            continue
        y_coords, x_coords = np.nonzero(_largest_component(mask))
        points[index] = (float(x_coords.mean()) / float(width), float(y_coords.mean()) / float(height))
        scores[index] = 1.0

    if int((scores > 0).sum()) < config.min_common_keypoints:
        raise KeyframePoseError(f"Target pose map does not contain enough body keypoints: {image_path.as_posix()}")

    return PoseKeypoints(points=points, scores=scores, image_size=(width, height))


def _body_color_labels(image: np.ndarray, tolerance: float) -> np.ndarray:
    chroma = image.max(axis=2) - image.min(axis=2)
    colored = (image.max(axis=2) > 35.0) & (chroma > 20.0)
    distances = np.sqrt(((image[:, :, None, :] - OPENPOSE_BODY_COLORS[None, None, :, :]) ** 2).sum(axis=3))
    nearest = distances.argmin(axis=2).astype(np.int16)
    nearest[~colored | (distances.min(axis=2) > tolerance)] = -1
    return nearest


def score_pose_match(
    target: PoseKeypoints,
    candidate: PoseKeypoints,
    config: PoseScoreConfig,
) -> PoseScoreResult:
    valid = np.isfinite(target.points[:, 0]) & np.isfinite(candidate.points[:, 0])
    common = int(valid.sum())
    if common < config.min_common_keypoints:
        return PoseScoreResult(
            score=0.0,
            common_keypoints=common,
            weighted_mean_distance=1.0,
            aligned_mean_distance=1.0,
        )

    target_points = target.points[valid]
    candidate_points = candidate.points[valid]
    weights = _body_keypoint_weights()[valid]
    distances = np.linalg.norm(target_points - candidate_points, axis=1)
    weighted_distance = _weighted_distance(distances, weights)
    aligned_candidate = _scale_translate_to_target(candidate_points, target_points, weights)
    aligned_distances = np.linalg.norm(target_points - aligned_candidate, axis=1)
    aligned_distance = _weighted_distance(aligned_distances, weights)
    absolute_score = _linear_pose_score(weighted_distance, config.distance_scale)
    aligned_score = _linear_pose_score(aligned_distance, config.distance_scale)
    score = float(0.35 * absolute_score + 0.65 * aligned_score)
    return PoseScoreResult(
        score=score,
        common_keypoints=common,
        weighted_mean_distance=weighted_distance,
        aligned_mean_distance=aligned_distance,
    )


def _weighted_distance(distances: np.ndarray, weights: np.ndarray) -> float:
    return float((distances * weights).sum() / weights.sum())


def _linear_pose_score(distance: float, distance_scale: float) -> float:
    return max(0.0, 1.0 - distance / distance_scale)


def _scale_translate_to_target(candidate: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weight_sum = weights.sum()
    target_center = (target * weights[:, None]).sum(axis=0) / weight_sum
    candidate_center = (candidate * weights[:, None]).sum(axis=0) / weight_sum
    target_centered = target - target_center
    candidate_centered = candidate - candidate_center
    target_spread = np.sqrt(((target_centered**2).sum(axis=1) * weights).sum() / weight_sum)
    candidate_spread = np.sqrt(((candidate_centered**2).sum(axis=1) * weights).sum() / weight_sum)
    if candidate_spread <= 1e-6:
        return candidate
    return candidate_centered * (target_spread / candidate_spread) + target_center


def save_pose_evidence(
    image_path: Path,
    target: PoseKeypoints,
    candidate: PoseKeypoints,
    output_path: Path,
) -> None:
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    for points, color in ((target.points, (255, 32, 32)), (candidate.points, (0, 210, 255))):
        for x_norm, y_norm in points:
            if not np.isfinite(x_norm):
                continue
            x = int(round(float(x_norm) * width))
            y = int(round(float(y_norm) * height))
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _require_model_file(path: Path) -> None:
    if not path.is_file():
        raise KeyframePoseError(f"Missing DWPose model file: {path.as_posix()}")


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def _best_person_index(scores: np.ndarray) -> int:
    visible_scores = np.where(scores >= 0.3, scores, 0.0)
    return int(visible_scores.sum(axis=1).argmax())


def _body_keypoint_weights() -> np.ndarray:
    return np.asarray(
        [
            0.7,
            1.2,
            1.2,
            0.9,
            0.8,
            1.2,
            0.9,
            0.8,
            1.4,
            1.2,
            1.3,
            1.4,
            1.2,
            1.3,
            0.5,
            0.5,
            0.4,
            0.4,
        ],
        dtype=np.float32,
    )
