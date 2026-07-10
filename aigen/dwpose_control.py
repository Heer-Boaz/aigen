from __future__ import annotations

from dataclasses import dataclass
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
        from controlnet_dwpose import DWposeDetector
        from controlnet_dwpose.util import draw_pose
    except ImportError as error:
        raise DWPoseControlError("DWPose control rendering requires the controlnet-dwpose package") from error

    detector = DWposeDetector(det_model.as_posix(), pose_model.as_posix(), device=device)
    try:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        pose = detector(rgb)
        body_scores = np.asarray(pose["bodies"]["score"], dtype=np.float32)
        if len(pose["bodies"]["subset"]) == 0:
            raise DWPoseControlError(f"DWPose found no body in {source_label}")
        control = draw_pose(pose, image.height, image.width)
    finally:
        detector.release_memory()

    person_scores = body_scores[0, :18]
    control_image = Image.fromarray(np.transpose(control, (1, 2, 0)).astype(np.uint8), mode="RGB")
    return DWPoseControl(
        image=control_image,
        metadata={
            "body_count": int(len(pose["bodies"]["subset"])),
            "visible_body_keypoints": int((person_scores > 0.30).sum()),
            "mean_body_score": float(person_scores.mean()),
        },
        device=device,
        det_model=det_model.resolve(),
        pose_model=pose_model.resolve(),
    )


def _require_model_file(path: Path) -> None:
    if not path.is_file():
        raise DWPoseControlError(f"Missing DWPose model file: {path.as_posix()}")
