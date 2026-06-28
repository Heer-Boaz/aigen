from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


DEFAULT_MODELS_ROOT = Path(__file__).resolve().parent / "models"
DEFAULT_GROUNDING_DINO_MODEL = DEFAULT_MODELS_ROOT / "grounding/IDEA-Research/grounding-dino-base"


class KeyframeGroundingError(RuntimeError):
    pass


@dataclass(frozen=True)
class GroundingConfig:
    model: Path = DEFAULT_GROUNDING_DINO_MODEL
    device: str = "cuda"
    threshold: float = 0.25
    text_threshold: float = 0.25
    min_prior_iou: float = 0.02
    max_prior_area_ratio: float = 4.0


@dataclass(frozen=True)
class GroundedRegionBox:
    box: tuple[int, int, int, int]
    label: str
    score: float
    source: str
    prior_iou: float

    def to_json(self) -> dict[str, object]:
        return {
            "box": list(self.box),
            "label": self.label,
            "score": self.score,
            "source": self.source,
            "prior_iou": self.prior_iou,
        }


class GroundingDinoRegionGrounder:
    def __init__(self, config: GroundingConfig):
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as error:
            raise KeyframeGroundingError("GroundingDINO polish grounding requires torch and transformers.") from error

        if not config.model.is_dir():
            raise KeyframeGroundingError(f"Missing GroundingDINO model: {config.model.as_posix()}")

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(config.model, local_files_only=True)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(config.model, local_files_only=True)
        self._model.to(config.device)
        self._model.eval()
        self._config = config

    def ground_region(
        self,
        image: Image.Image,
        prompt: str,
        prior_box: tuple[int, int, int, int],
    ) -> GroundedRegionBox:
        width, height = image.size
        prior = _clip_box(prior_box, width, height)
        boxes = [
            GroundedRegionBox(
                box=box.box,
                label=box.label,
                score=box.score,
                source=box.source,
                prior_iou=_box_iou(box.box, prior),
            )
            for box in self._grounding_dino_boxes(image.convert("RGB"), prompt)
        ]
        candidates = [
            box
            for box in boxes
            if box.prior_iou >= self._config.min_prior_iou
            and _box_area(box.box) <= _box_area(prior) * self._config.max_prior_area_ratio
        ]
        if candidates:
            return max(candidates, key=lambda box: (box.prior_iou, box.score))
        return GroundedRegionBox(
            box=prior,
            label="polish planner region",
            score=1.0,
            source="polish-plan",
            prior_iou=1.0,
        )

    def _grounding_dino_boxes(self, image: Image.Image, prompt: str) -> list[GroundedRegionBox]:
        query = _grounding_query(prompt)
        inputs = self._processor(images=image, text=query, return_tensors="pt").to(self._config.device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        result = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self._config.threshold,
            text_threshold=self._config.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = []
        for raw_box, raw_score, raw_label in zip(result["boxes"], result["scores"], result["text_labels"], strict=True):
            box = _clip_box(tuple(round(float(value)) for value in raw_box.tolist()), image.width, image.height)
            boxes.append(
                GroundedRegionBox(
                    box=box,
                    label=str(raw_label),
                    score=float(raw_score),
                    source="grounding-dino",
                    prior_iou=0.0,
                )
            )
        return boxes


def _grounding_query(prompt: str) -> str:
    text = " ".join(prompt.lower().split())
    if text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    clipped = (
        max(0, min(width - 1, left)),
        max(0, min(height - 1, top)),
        max(1, min(width, right)),
        max(1, min(height, bottom)),
    )
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise KeyframeGroundingError(f"Invalid grounding box {box} for image size {(width, height)}")
    return clipped


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = _box_area(a) + _box_area(b) - intersection
    return intersection / union if union else 0.0


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])
