from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aigen.manifest_io import read_json
from aigen.pix2pix.errors import Pix2PixError


TRAIN_CONFIG_FORMAT_V2 = "aigen.pix2pix.training.v2"
TRAIN_CONFIG_FORMAT_V3 = "aigen.pix2pix.training.v3"
TRAIN_CONFIG_FORMAT_V4 = "aigen.pix2pix.training.v4"
TRAIN_CONFIG_FORMAT_V5 = "aigen.pix2pix.training.v5"
TRAIN_CONFIG_FORMAT_V6 = "aigen.pix2pix.training.v6"
TRAIN_CONFIG_FORMAT_V7 = "aigen.pix2pix.training.v7"
TRAIN_CONFIG_FORMAT = "aigen.pix2pix.training.v8"
MODEL_FORMAT = "aigen.pix2pix.generator.v1"
MODEL_IMAGE_SIZE = 256
MODEL_IMAGE_SIZES = frozenset({128, MODEL_IMAGE_SIZE})
MODEL_CHANNELS = 3
DISCRIMINATOR_LAYER_COUNTS = frozenset({1, 3, 4})
DISCRIMINATOR_SCALE_COUNTS = frozenset({1, 2})
TRAIN_OPTIMIZERS = frozenset({"adam", "paged_adam8bit"})
TRAIN_OBJECTIVES = frozenset({"adversarial_l1", "l1_only"})
TRAIN_PRECISIONS = frozenset({"fp32", "bf16"})
LEARNING_RATE_SCHEDULES = frozenset({"constant", "linear_decay"})
RECONSTRUCTION_BALANCES = frozenset({"uniform", "white_canvas_equal_regions"})
ADVERSARIAL_BALANCES = frozenset({"uniform", "white_canvas_equal_regions"})
DISCRIMINATOR_AUGMENTATIONS = frozenset({"none", "translation"})
CONDITIONAL_NEGATIVES = frozenset({"a_contrario", "none"})


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = MODEL_IMAGE_SIZE
    input_channels: int = MODEL_CHANNELS
    output_channels: int = MODEL_CHANNELS
    generator_channels: int = 64
    discriminator_channels: int = 64
    discriminator_layers: int = 3
    generator_dropout: bool = True

    def to_json(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: object) -> ModelConfig:
        values = _exact_object(payload, set(cls.__dataclass_fields__), "model config")
        config = cls(
            image_size=_integer(values, "image_size"),
            input_channels=_integer(values, "input_channels"),
            output_channels=_integer(values, "output_channels"),
            generator_channels=_integer(values, "generator_channels"),
            discriminator_channels=_integer(values, "discriminator_channels"),
            discriminator_layers=_integer(values, "discriminator_layers"),
            generator_dropout=_boolean(values, "generator_dropout"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.image_size not in MODEL_IMAGE_SIZES:
            supported = ", ".join(str(size) for size in sorted(MODEL_IMAGE_SIZES))
            raise Pix2PixError(f"pix2pix v1 image_size must be one of: {supported}")
        if self.input_channels != MODEL_CHANNELS or self.output_channels != MODEL_CHANNELS:
            raise Pix2PixError("pix2pix v1 requires three-channel RGB input and output")
        if self.generator_channels < 1:
            raise Pix2PixError("generator_channels must be positive")
        if self.discriminator_channels < 1:
            raise Pix2PixError("discriminator_channels must be positive")
        if self.discriminator_layers not in DISCRIMINATOR_LAYER_COUNTS:
            supported = ", ".join(
                str(count) for count in sorted(DISCRIMINATOR_LAYER_COUNTS)
            )
            raise Pix2PixError(
                f"pix2pix v1 discriminator_layers must be one of: {supported}"
            )


@dataclass(frozen=True)
class LearningRateScheduleConfig:
    type: str
    decay_start_step: int | None

    def to_json(self) -> dict[str, object]:
        if self.type == "constant":
            return {"type": self.type}
        assert self.decay_start_step is not None
        return {
            "type": self.type,
            "decay_start_step": self.decay_start_step,
        }

    @classmethod
    def constant(cls) -> LearningRateScheduleConfig:
        return cls(type="constant", decay_start_step=None)

    @classmethod
    def from_json(cls, payload: object) -> LearningRateScheduleConfig:
        if not isinstance(payload, dict):
            raise Pix2PixError("learning-rate schedule must be a JSON object")
        schedule_type = payload.get("type")
        if schedule_type == "constant":
            _exact_object(payload, {"type"}, "learning-rate schedule")
            return cls.constant()
        if schedule_type == "linear_decay":
            values = _exact_object(
                payload,
                {"type", "decay_start_step"},
                "learning-rate schedule",
            )
            return cls(
                type=schedule_type,
                decay_start_step=_integer(values, "decay_start_step"),
            )
        supported = ", ".join(sorted(LEARNING_RATE_SCHEDULES))
        raise Pix2PixError(
            f"learning-rate schedule type must be one of: {supported}"
        )

    def validate(self, *, max_steps: int) -> None:
        if self.type not in LEARNING_RATE_SCHEDULES:
            supported = ", ".join(sorted(LEARNING_RATE_SCHEDULES))
            raise Pix2PixError(
                f"learning-rate schedule type must be one of: {supported}"
            )
        if self.type == "constant":
            if self.decay_start_step is not None:
                raise Pix2PixError(
                    "constant learning-rate schedule cannot define decay_start_step"
                )
            return
        if self.decay_start_step is None:
            raise Pix2PixError(
                "linear_decay learning-rate schedule requires decay_start_step"
            )
        if not 0 < self.decay_start_step < max_steps:
            raise Pix2PixError(
                "decay_start_step must be between zero and max_steps"
            )


@dataclass(frozen=True)
class TrainConfig:
    format: str
    objective: str
    model: ModelConfig
    batch_size: int
    max_steps: int
    learning_rate: float
    learning_rate_schedule: LearningRateScheduleConfig
    beta1: float
    beta2: float
    lambda_l1: float
    discriminator_scales: int
    lambda_feature_matching: float
    reconstruction_balance: str
    adversarial_balance: str
    discriminator_augmentation: str
    conditional_negative: str
    lambda_palette_proximity: float
    horizontal_flip: bool
    optimizer: str
    parameter_precision: str
    precision: str
    checkpoint_every: int
    extra_checkpoint_steps: tuple[int, ...]
    log_every: int
    seed: int
    num_workers: int

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "format": self.format,
            "model": self.model.to_json(),
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "lambda_l1": self.lambda_l1,
            "horizontal_flip": self.horizontal_flip,
            "optimizer": self.optimizer,
            "parameter_precision": self.parameter_precision,
            "precision": self.precision,
            "checkpoint_every": self.checkpoint_every,
            "log_every": self.log_every,
            "seed": self.seed,
            "num_workers": self.num_workers,
        }
        if self.format in {
            TRAIN_CONFIG_FORMAT_V3,
            TRAIN_CONFIG_FORMAT_V4,
            TRAIN_CONFIG_FORMAT_V5,
            TRAIN_CONFIG_FORMAT_V6,
            TRAIN_CONFIG_FORMAT_V7,
            TRAIN_CONFIG_FORMAT,
        }:
            payload["objective"] = self.objective
        if self.format in {
            TRAIN_CONFIG_FORMAT_V4,
            TRAIN_CONFIG_FORMAT_V5,
            TRAIN_CONFIG_FORMAT_V6,
            TRAIN_CONFIG_FORMAT_V7,
            TRAIN_CONFIG_FORMAT,
        }:
            payload["learning_rate_schedule"] = (
                self.learning_rate_schedule.to_json()
            )
            payload["extra_checkpoint_steps"] = list(self.extra_checkpoint_steps)
        if self.format in {
            TRAIN_CONFIG_FORMAT_V5,
            TRAIN_CONFIG_FORMAT_V6,
            TRAIN_CONFIG_FORMAT_V7,
            TRAIN_CONFIG_FORMAT,
        }:
            payload["discriminator_scales"] = self.discriminator_scales
            payload["lambda_feature_matching"] = self.lambda_feature_matching
            payload["reconstruction_balance"] = self.reconstruction_balance
        if self.format in {
            TRAIN_CONFIG_FORMAT_V6,
            TRAIN_CONFIG_FORMAT_V7,
            TRAIN_CONFIG_FORMAT,
        }:
            payload["adversarial_balance"] = self.adversarial_balance
            payload["discriminator_augmentation"] = (
                self.discriminator_augmentation
            )
        if self.format in {TRAIN_CONFIG_FORMAT_V7, TRAIN_CONFIG_FORMAT}:
            payload["conditional_negative"] = self.conditional_negative
        if self.format == TRAIN_CONFIG_FORMAT:
            payload["lambda_palette_proximity"] = (
                self.lambda_palette_proximity
            )
        return payload

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        payload = read_json(path, label="pix2pix training config")
        if not isinstance(payload, dict):
            raise Pix2PixError("training config must be a JSON object")
        config_format = payload.get("format")
        expected = {
            "format",
            "model",
            "batch_size",
            "max_steps",
            "learning_rate",
            "beta1",
            "beta2",
            "lambda_l1",
            "horizontal_flip",
            "optimizer",
            "parameter_precision",
            "precision",
            "checkpoint_every",
            "log_every",
            "seed",
            "num_workers",
        }
        supported_formats = {
            TRAIN_CONFIG_FORMAT_V2,
            TRAIN_CONFIG_FORMAT_V3,
            TRAIN_CONFIG_FORMAT_V4,
            TRAIN_CONFIG_FORMAT_V5,
            TRAIN_CONFIG_FORMAT_V6,
            TRAIN_CONFIG_FORMAT_V7,
            TRAIN_CONFIG_FORMAT,
        }
        if config_format not in supported_formats:
            raise Pix2PixError(
                f"unsupported training config format: {config_format!r}"
            )
        if config_format in {
            TRAIN_CONFIG_FORMAT_V3,
            TRAIN_CONFIG_FORMAT_V4,
            TRAIN_CONFIG_FORMAT_V5,
            TRAIN_CONFIG_FORMAT_V6,
            TRAIN_CONFIG_FORMAT_V7,
            TRAIN_CONFIG_FORMAT,
        }:
            expected.add("objective")
        if config_format in {
            TRAIN_CONFIG_FORMAT_V4,
            TRAIN_CONFIG_FORMAT_V5,
            TRAIN_CONFIG_FORMAT_V6,
            TRAIN_CONFIG_FORMAT_V7,
            TRAIN_CONFIG_FORMAT,
        }:
            expected.update(
                {
                    "learning_rate_schedule",
                    "extra_checkpoint_steps",
                }
            )
        if config_format in {
            TRAIN_CONFIG_FORMAT_V5,
            TRAIN_CONFIG_FORMAT_V6,
            TRAIN_CONFIG_FORMAT_V7,
            TRAIN_CONFIG_FORMAT,
        }:
            expected.update(
                {
                    "discriminator_scales",
                    "lambda_feature_matching",
                    "reconstruction_balance",
                }
            )
        if config_format in {
            TRAIN_CONFIG_FORMAT_V6,
            TRAIN_CONFIG_FORMAT_V7,
            TRAIN_CONFIG_FORMAT,
        }:
            expected.update(
                {
                    "adversarial_balance",
                    "discriminator_augmentation",
                }
            )
        if config_format in {TRAIN_CONFIG_FORMAT_V7, TRAIN_CONFIG_FORMAT}:
            expected.add("conditional_negative")
        if config_format == TRAIN_CONFIG_FORMAT:
            expected.add("lambda_palette_proximity")
        values = _exact_object(payload, expected, "training config")
        objective = (
            _string(values, "objective")
            if config_format
            in {
                TRAIN_CONFIG_FORMAT_V3,
                TRAIN_CONFIG_FORMAT_V4,
                TRAIN_CONFIG_FORMAT_V5,
                TRAIN_CONFIG_FORMAT_V6,
                TRAIN_CONFIG_FORMAT_V7,
                TRAIN_CONFIG_FORMAT,
            }
            else "adversarial_l1"
        )
        learning_rate_schedule = (
            LearningRateScheduleConfig.from_json(
                values["learning_rate_schedule"]
            )
            if config_format
            in {
                TRAIN_CONFIG_FORMAT_V4,
                TRAIN_CONFIG_FORMAT_V5,
                TRAIN_CONFIG_FORMAT_V6,
                TRAIN_CONFIG_FORMAT_V7,
                TRAIN_CONFIG_FORMAT,
            }
            else LearningRateScheduleConfig.constant()
        )
        extra_checkpoint_steps = (
            _integer_tuple(values, "extra_checkpoint_steps")
            if config_format
            in {
                TRAIN_CONFIG_FORMAT_V4,
                TRAIN_CONFIG_FORMAT_V5,
                TRAIN_CONFIG_FORMAT_V6,
                TRAIN_CONFIG_FORMAT_V7,
                TRAIN_CONFIG_FORMAT,
            }
            else ()
        )
        config = cls(
            format=config_format,
            objective=objective,
            model=ModelConfig.from_json(values["model"]),
            batch_size=_integer(values, "batch_size"),
            max_steps=_integer(values, "max_steps"),
            learning_rate=_number(values, "learning_rate"),
            learning_rate_schedule=learning_rate_schedule,
            beta1=_number(values, "beta1"),
            beta2=_number(values, "beta2"),
            lambda_l1=_number(values, "lambda_l1"),
            discriminator_scales=(
                _integer(values, "discriminator_scales")
                if config_format
                in {
                    TRAIN_CONFIG_FORMAT_V5,
                    TRAIN_CONFIG_FORMAT_V6,
                    TRAIN_CONFIG_FORMAT_V7,
                    TRAIN_CONFIG_FORMAT,
                }
                else 1
            ),
            lambda_feature_matching=(
                _number(values, "lambda_feature_matching")
                if config_format
                in {
                    TRAIN_CONFIG_FORMAT_V5,
                    TRAIN_CONFIG_FORMAT_V6,
                    TRAIN_CONFIG_FORMAT_V7,
                    TRAIN_CONFIG_FORMAT,
                }
                else 0.0
            ),
            reconstruction_balance=(
                _string(values, "reconstruction_balance")
                if config_format
                in {
                    TRAIN_CONFIG_FORMAT_V5,
                    TRAIN_CONFIG_FORMAT_V6,
                    TRAIN_CONFIG_FORMAT_V7,
                    TRAIN_CONFIG_FORMAT,
                }
                else "uniform"
            ),
            adversarial_balance=(
                _string(values, "adversarial_balance")
                if config_format
                in {
                    TRAIN_CONFIG_FORMAT_V6,
                    TRAIN_CONFIG_FORMAT_V7,
                    TRAIN_CONFIG_FORMAT,
                }
                else "uniform"
            ),
            discriminator_augmentation=(
                _string(values, "discriminator_augmentation")
                if config_format
                in {
                    TRAIN_CONFIG_FORMAT_V6,
                    TRAIN_CONFIG_FORMAT_V7,
                    TRAIN_CONFIG_FORMAT,
                }
                else "none"
            ),
            conditional_negative=(
                _string(values, "conditional_negative")
                if config_format in {TRAIN_CONFIG_FORMAT_V7, TRAIN_CONFIG_FORMAT}
                else "none"
            ),
            lambda_palette_proximity=(
                _number(values, "lambda_palette_proximity")
                if config_format == TRAIN_CONFIG_FORMAT
                else 0.0
            ),
            horizontal_flip=_boolean(values, "horizontal_flip"),
            optimizer=_string(values, "optimizer"),
            parameter_precision=_string(values, "parameter_precision"),
            precision=_string(values, "precision"),
            checkpoint_every=_integer(values, "checkpoint_every"),
            extra_checkpoint_steps=extra_checkpoint_steps,
            log_every=_integer(values, "log_every"),
            seed=_integer(values, "seed"),
            num_workers=_integer(values, "num_workers"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.format not in {
            TRAIN_CONFIG_FORMAT_V2,
            TRAIN_CONFIG_FORMAT_V3,
            TRAIN_CONFIG_FORMAT_V4,
            TRAIN_CONFIG_FORMAT_V5,
            TRAIN_CONFIG_FORMAT_V6,
            TRAIN_CONFIG_FORMAT_V7,
            TRAIN_CONFIG_FORMAT,
        }:
            raise Pix2PixError(f"unsupported training config format: {self.format!r}")
        if self.objective not in TRAIN_OBJECTIVES:
            supported = ", ".join(sorted(TRAIN_OBJECTIVES))
            raise Pix2PixError(f"objective must be one of: {supported}")
        if (
            self.format == TRAIN_CONFIG_FORMAT_V2
            and self.objective != "adversarial_l1"
        ):
            raise Pix2PixError(
                "pix2pix training v2 supports only adversarial_l1"
            )
        self.model.validate()
        if self.batch_size < 1:
            raise Pix2PixError("batch_size must be positive")
        if self.max_steps < 1:
            raise Pix2PixError("max_steps must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise Pix2PixError("learning_rate must be positive")
        self.learning_rate_schedule.validate(max_steps=self.max_steps)
        if (
            not math.isfinite(self.beta1)
            or not math.isfinite(self.beta2)
            or not 0 <= self.beta1 < 1
            or not 0 <= self.beta2 < 1
        ):
            raise Pix2PixError("Adam beta values must be in [0, 1)")
        if not math.isfinite(self.lambda_l1) or self.lambda_l1 <= 0:
            raise Pix2PixError("lambda_l1 must be positive")
        if self.discriminator_scales not in DISCRIMINATOR_SCALE_COUNTS:
            supported = ", ".join(
                str(count) for count in sorted(DISCRIMINATOR_SCALE_COUNTS)
            )
            raise Pix2PixError(
                f"discriminator_scales must be one of: {supported}"
            )
        if (
            not math.isfinite(self.lambda_feature_matching)
            or self.lambda_feature_matching < 0
        ):
            raise Pix2PixError("lambda_feature_matching must be non-negative")
        if self.reconstruction_balance not in RECONSTRUCTION_BALANCES:
            supported = ", ".join(sorted(RECONSTRUCTION_BALANCES))
            raise Pix2PixError(
                f"reconstruction_balance must be one of: {supported}"
            )
        if self.adversarial_balance not in ADVERSARIAL_BALANCES:
            supported = ", ".join(sorted(ADVERSARIAL_BALANCES))
            raise Pix2PixError(
                f"adversarial_balance must be one of: {supported}"
            )
        if self.discriminator_augmentation not in DISCRIMINATOR_AUGMENTATIONS:
            supported = ", ".join(sorted(DISCRIMINATOR_AUGMENTATIONS))
            raise Pix2PixError(
                f"discriminator_augmentation must be one of: {supported}"
            )
        if self.conditional_negative not in CONDITIONAL_NEGATIVES:
            supported = ", ".join(sorted(CONDITIONAL_NEGATIVES))
            raise Pix2PixError(
                f"conditional_negative must be one of: {supported}"
            )
        if (
            self.format not in {TRAIN_CONFIG_FORMAT_V7, TRAIN_CONFIG_FORMAT}
            and self.conditional_negative != "none"
        ):
            raise Pix2PixError(
                "conditional negatives require pix2pix training v7 or newer"
            )
        if (
            not math.isfinite(self.lambda_palette_proximity)
            or self.lambda_palette_proximity < 0
        ):
            raise Pix2PixError(
                "lambda_palette_proximity must be non-negative"
            )
        if (
            self.format != TRAIN_CONFIG_FORMAT
            and self.lambda_palette_proximity != 0
        ):
            raise Pix2PixError(
                "palette proximity requires pix2pix training v8"
            )
        if self.objective == "l1_only" and self.discriminator_scales != 1:
            raise Pix2PixError("l1_only requires discriminator_scales 1")
        if self.objective == "l1_only" and self.lambda_feature_matching != 0:
            raise Pix2PixError("l1_only cannot use discriminator feature matching")
        if self.objective == "l1_only" and self.adversarial_balance != "uniform":
            raise Pix2PixError("l1_only cannot use adversarial region balancing")
        if self.objective == "l1_only" and self.discriminator_augmentation != "none":
            raise Pix2PixError("l1_only cannot use discriminator augmentation")
        if self.objective == "l1_only" and self.conditional_negative != "none":
            raise Pix2PixError("l1_only cannot use a conditional negative")
        if self.objective == "l1_only" and self.lambda_palette_proximity != 0:
            raise Pix2PixError("l1_only cannot use palette proximity")
        if self.format == TRAIN_CONFIG_FORMAT_V5:
            if self.objective != "adversarial_l1":
                raise Pix2PixError(
                    "pix2pix training v5 supports only adversarial_l1"
                )
            if self.discriminator_scales != 2:
                raise Pix2PixError(
                    "pix2pix training v5 requires two discriminator scales"
                )
            if self.lambda_feature_matching <= 0:
                raise Pix2PixError(
                    "pix2pix training v5 requires positive feature matching"
                )
        if self.format == TRAIN_CONFIG_FORMAT_V6:
            if self.objective != "adversarial_l1":
                raise Pix2PixError(
                    "pix2pix training v6 supports only adversarial_l1"
                )
            if self.discriminator_scales != 2:
                raise Pix2PixError(
                    "pix2pix training v6 requires two discriminator scales"
                )
            if self.lambda_feature_matching <= 0:
                raise Pix2PixError(
                    "pix2pix training v6 requires positive feature matching"
                )
            if (
                self.reconstruction_balance != "white_canvas_equal_regions"
                or self.adversarial_balance != "white_canvas_equal_regions"
            ):
                raise Pix2PixError(
                    "pix2pix training v6 requires white-canvas region balancing"
                )
            if self.discriminator_augmentation != "translation":
                raise Pix2PixError(
                    "pix2pix training v6 requires translation augmentation"
                )
        if self.format == TRAIN_CONFIG_FORMAT_V7:
            if self.objective != "adversarial_l1":
                raise Pix2PixError(
                    "pix2pix training v7 supports only adversarial_l1"
                )
            if self.discriminator_scales != 2:
                raise Pix2PixError(
                    "pix2pix training v7 requires two discriminator scales"
                )
            if self.lambda_feature_matching <= 0:
                raise Pix2PixError(
                    "pix2pix training v7 requires positive feature matching"
                )
            if (
                self.reconstruction_balance != "white_canvas_equal_regions"
                or self.adversarial_balance != "white_canvas_equal_regions"
            ):
                raise Pix2PixError(
                    "pix2pix training v7 requires white-canvas region balancing"
                )
            if self.discriminator_augmentation != "translation":
                raise Pix2PixError(
                    "pix2pix training v7 requires translation augmentation"
                )
            if self.conditional_negative != "a_contrario":
                raise Pix2PixError(
                    "pix2pix training v7 requires a-contrario conditional negatives"
                )
        if self.format == TRAIN_CONFIG_FORMAT:
            if self.objective != "adversarial_l1":
                raise Pix2PixError(
                    "pix2pix training v8 supports only adversarial_l1"
                )
            if self.discriminator_scales != 2:
                raise Pix2PixError(
                    "pix2pix training v8 requires two discriminator scales"
                )
            if self.lambda_feature_matching <= 0:
                raise Pix2PixError(
                    "pix2pix training v8 requires positive feature matching"
                )
            if (
                self.reconstruction_balance != "white_canvas_equal_regions"
                or self.adversarial_balance != "white_canvas_equal_regions"
            ):
                raise Pix2PixError(
                    "pix2pix training v8 requires white-canvas region balancing"
                )
            if self.discriminator_augmentation != "translation":
                raise Pix2PixError(
                    "pix2pix training v8 requires translation augmentation"
                )
            if self.conditional_negative != "a_contrario":
                raise Pix2PixError(
                    "pix2pix training v8 requires a-contrario conditional negatives"
                )
            if self.lambda_palette_proximity <= 0:
                raise Pix2PixError(
                    "pix2pix training v8 requires positive palette proximity"
                )
            if self.batch_size != 1:
                raise Pix2PixError(
                    "pix2pix training v8 requires batch_size 1"
                )
            if self.parameter_precision != "fp32" or self.precision != "fp32":
                raise Pix2PixError(
                    "pix2pix training v8 requires FP32 parameters and compute"
                )
        if self.optimizer not in TRAIN_OPTIMIZERS:
            supported = ", ".join(sorted(TRAIN_OPTIMIZERS))
            raise Pix2PixError(f"optimizer must be one of: {supported}")
        if self.parameter_precision not in TRAIN_PRECISIONS:
            raise Pix2PixError("parameter_precision must be fp32 or bf16")
        if self.precision not in TRAIN_PRECISIONS:
            raise Pix2PixError("precision must be fp32 or bf16")
        if self.parameter_precision == "bf16" and self.precision != "bf16":
            raise Pix2PixError(
                "bf16 parameter_precision requires bf16 compute precision"
            )
        if self.checkpoint_every < 1:
            raise Pix2PixError("checkpoint_every must be positive")
        if (
            tuple(sorted(set(self.extra_checkpoint_steps)))
            != self.extra_checkpoint_steps
        ):
            raise Pix2PixError(
                "extra_checkpoint_steps must be strictly increasing and unique"
            )
        if any(
            step < 1 or step >= self.max_steps
            for step in self.extra_checkpoint_steps
        ):
            raise Pix2PixError(
                "extra checkpoint steps must be between one and max_steps"
            )
        if self.log_every < 1:
            raise Pix2PixError("log_every must be positive")
        if self.seed < 0:
            raise Pix2PixError("seed must be non-negative")
        if self.num_workers < 0:
            raise Pix2PixError("num_workers must be non-negative")


def _exact_object(
    payload: object,
    expected: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise Pix2PixError(f"{label} must be a JSON object")
    keys = set(payload)
    if keys != expected:
        missing = sorted(expected - keys)
        unexpected = sorted(keys - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        raise Pix2PixError(f"invalid {label}: {'; '.join(details)}")
    return payload


def _integer(payload: dict[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise Pix2PixError(f"{key} must be an integer")
    return value


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Pix2PixError(f"{key} must be a number")
    return float(value)


def _boolean(payload: dict[str, Any], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise Pix2PixError(f"{key} must be a boolean")
    return value


def _integer_tuple(payload: dict[str, Any], key: str) -> tuple[int, ...]:
    value = payload[key]
    if not isinstance(value, list):
        raise Pix2PixError(f"{key} must be an array")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise Pix2PixError(f"{key} must contain only integers")
    return tuple(value)


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise Pix2PixError(f"{key} must be a string")
    return value
