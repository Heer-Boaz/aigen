from __future__ import annotations

import gc
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

import numpy as np
from PIL import Image

MODELS_ROOT = Path(__file__).resolve().parent / "models"
DEFAULT_DWPOSE_DET_MODEL = MODELS_ROOT / "annotators/yzd-v/DWPose/yolox_l.onnx"
DEFAULT_DWPOSE_POSE_MODEL = MODELS_ROOT / "annotators/yzd-v/DWPose/dw-ll_ucoco_384.onnx"


class DWPoseControlError(RuntimeError):
    pass


@dataclass(frozen=True)
class DWPoseControl:
    image: Image.Image
    content_box: tuple[int, int, int, int]
    metadata: dict[str, int | float]
    device: str
    det_model: Path
    pose_model: Path


def render_dwpose_control(
    image: Image.Image,
    *,
    source_label: str,
    device: str = "cuda",
    det_model: Path = DEFAULT_DWPOSE_DET_MODEL,
    pose_model: Path = DEFAULT_DWPOSE_POSE_MODEL,
) -> DWPoseControl:
    _require_model_file(det_model)
    _require_model_file(pose_model)
    try:
        from controlnet_dwpose.onnxdet import inference_detector
        from controlnet_dwpose.onnxpose import inference_pose
        from controlnet_dwpose.util import draw_pose
        from controlnet_dwpose.wholebody import Wholebody
    except ImportError as error:
        raise DWPoseControlError("DWPose control rendering requires the controlnet-dwpose package") from error

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    estimator = Wholebody(det_model.as_posix(), pose_model.as_posix(), device=device)
    try:
        detections = np.asarray(inference_detector(estimator.session_det, rgb), dtype=np.float32)
        if detections.size == 0:
            raise DWPoseControlError(f"DWPose found no body in {source_label}")
        keypoints, scores = inference_pose(estimator.session_pose, detections, rgb)
        pose = _dwpose_drawing_pose(keypoints, scores, width=image.width, height=image.height)
        body_scores = np.asarray(pose["bodies"]["score"], dtype=np.float32)
        control = draw_pose(pose, image.height, image.width)
    finally:
        del estimator
        gc.collect()

    person_scores = body_scores[0, :18]
    control_image = Image.fromarray(np.transpose(control, (1, 2, 0)).astype(np.uint8), mode="RGB")
    content_box = _detection_content_box(detections, width=image.width, height=image.height)
    return DWPoseControl(
        image=control_image,
        content_box=content_box,
        metadata={
            "body_count": int(len(pose["bodies"]["subset"])),
            "visible_body_keypoints": int((person_scores > 0.30).sum()),
            "mean_body_score": float(person_scores.mean()),
            "content_left": content_box[0],
            "content_top": content_box[1],
            "content_right": content_box[2],
            "content_bottom": content_box[3],
        },
        device=device,
        det_model=det_model.resolve(),
        pose_model=pose_model.resolve(),
    )


def _dwpose_drawing_pose(
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    width: int,
    height: int,
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    neck[:, 2:] = np.logical_and(
        keypoints_info[:, 5, 2:] > 0.30,
        keypoints_info[:, 6, 2:] > 0.30,
    )
    keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
    keypoints_info[:, [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]] = keypoints_info[
        :, [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    ]

    candidate = keypoints_info[..., :2].copy()
    candidate[..., 0] /= float(width)
    candidate[..., 1] /= float(height)
    body = candidate[:, :18].reshape(-1, 2)
    body_scores = keypoints_info[:, :18, 2]
    body_indices = np.arange(body_scores.size, dtype=np.float32).reshape(body_scores.shape)
    subset = np.where(body_scores > 0.30, body_indices, -1.0)
    faces = candidate[:, 24:92]
    hands = np.vstack((candidate[:, 92:113], candidate[:, 113:]))
    faces_score = keypoints_info[:, 24:92, 2]
    hands_score = np.vstack((keypoints_info[:, 92:113, 2], keypoints_info[:, 113:, 2]))
    return {
        "bodies": {"candidate": body, "subset": subset, "score": body_scores},
        "hands": hands,
        "hands_score": hands_score,
        "faces": faces,
        "faces_score": faces_score,
    }


def _detection_content_box(
    detections: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, floor(float(detections[:, 0].min()))),
        max(0, floor(float(detections[:, 1].min()))),
        min(width, ceil(float(detections[:, 2].max()))),
        min(height, ceil(float(detections[:, 3].max()))),
    )


def _require_model_file(path: Path) -> None:
    if not path.is_file():
        raise DWPoseControlError(f"Missing DWPose model file: {path.as_posix()}")
