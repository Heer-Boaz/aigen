from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


DEFAULT_MODELS_ROOT = Path(__file__).resolve().parent / "models"
DEFAULT_GROUNDING_DINO_MODEL = DEFAULT_MODELS_ROOT / "grounding/IDEA-Research/grounding-dino-base"
DEFAULT_FLORENCE2_MODEL = DEFAULT_MODELS_ROOT / "grounding/florence-community/Florence-2-large-ft"
FLORENCE_PHRASE_GROUNDING_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"


class KeyframeGroundingError(RuntimeError):
    pass


@dataclass(frozen=True)
class GroundingConfig:
    dino_model: Path = DEFAULT_GROUNDING_DINO_MODEL
    florence_model: Path = DEFAULT_FLORENCE2_MODEL
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


class KeyframeRegionGrounder:
    def __init__(self, config: GroundingConfig):
        self._config = config
        self._grounders = (
            GroundingDinoRegionGrounder(config),
            Florence2RegionGrounder(config),
        )

    def ground_region(
        self,
        image: Image.Image,
        prompt: str,
        prior_box: tuple[int, int, int, int],
    ) -> GroundedRegionBox:
        width, height = image.size
        prior = _clip_box(prior_box, width, height)
        boxes = []
        for grounder in self._grounders:
            boxes.extend(
                GroundedRegionBox(
                    box=box.box,
                    label=box.label,
                    score=box.score,
                    source=box.source,
                    prior_iou=_box_iou(box.box, prior),
                )
                for box in grounder.ground_boxes(image, prompt)
            )
        return _best_grounded_region(boxes, prior, self._config)


class GroundingDinoRegionGrounder:
    def __init__(self, config: GroundingConfig):
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as error:
            raise KeyframeGroundingError("GroundingDINO polish grounding requires torch and transformers.") from error

        if not config.dino_model.is_dir():
            raise KeyframeGroundingError(f"Missing GroundingDINO model: {config.dino_model.as_posix()}")

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(config.dino_model, local_files_only=True)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(config.dino_model, local_files_only=True)
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
            for box in self.ground_boxes(image, prompt)
        ]
        return _best_grounded_region(boxes, prior, self._config)

    def ground_boxes(self, image: Image.Image, prompt: str) -> list[GroundedRegionBox]:
        image = image.convert("RGB")
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


class Florence2RegionGrounder:
    def __init__(self, config: GroundingConfig):
        try:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
        except ImportError as error:
            raise KeyframeGroundingError("Florence-2 polish grounding requires torch and transformers.") from error

        if not config.florence_model.is_dir():
            raise KeyframeGroundingError(f"Missing Florence-2 model: {config.florence_model.as_posix()}")

        self._torch = torch
        dtype = torch.float16 if config.device.startswith("cuda") else torch.float32
        self._processor = AutoProcessor.from_pretrained(config.florence_model, local_files_only=True)
        self._model = Florence2ForConditionalGeneration.from_pretrained(
            config.florence_model,
            local_files_only=True,
            torch_dtype=dtype,
        )
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
            for box in self.ground_boxes(image, prompt)
        ]
        return _best_grounded_region(boxes, prior, self._config)

    def ground_boxes(self, image: Image.Image, prompt: str) -> list[GroundedRegionBox]:
        image = image.convert("RGB")
        task_prompt = f"{FLORENCE_PHRASE_GROUNDING_TASK}{_florence_phrase(prompt)}"
        inputs = self._processor(text=task_prompt, images=image, return_tensors="pt").to(self._config.device)
        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
                do_sample=False,
            )
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            generated_text,
            task=FLORENCE_PHRASE_GROUNDING_TASK,
            image_size=image.size,
        )
        result = parsed.get(FLORENCE_PHRASE_GROUNDING_TASK, {})
        return [
            GroundedRegionBox(
                box=_clip_box(tuple(round(float(value)) for value in raw_box), image.width, image.height),
                label=str(raw_label),
                score=1.0,
                source="florence2",
                prior_iou=0.0,
            )
            for raw_box, raw_label in zip(result.get("bboxes", ()), result.get("labels", ()), strict=True)
        ]


def _grounding_query(prompt: str) -> str:
    text = " ".join(prompt.lower().split())
    if text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def _florence_phrase(prompt: str) -> str:
    return " ".join(prompt.split())


def _best_grounded_region(
    boxes: list[GroundedRegionBox],
    prior: tuple[int, int, int, int],
    config: GroundingConfig,
) -> GroundedRegionBox:
    candidates = [
        box
        for box in boxes
        if box.prior_iou >= config.min_prior_iou
        and _box_area(box.box) <= _box_area(prior) * config.max_prior_area_ratio
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
