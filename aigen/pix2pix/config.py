from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aigen.manifest_io import read_json
from aigen.pix2pix.errors import Pix2PixError


TRAIN_CONFIG_FORMAT = "aigen.pix2pix.training.v1"
MODEL_FORMAT = "aigen.pix2pix.generator.v1"
MODEL_IMAGE_SIZE = 256
MODEL_IMAGE_SIZES = frozenset({128, MODEL_IMAGE_SIZE})
MODEL_CHANNELS = 3
DISCRIMINATOR_LAYER_COUNTS = frozenset({1, 3})


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
class TrainConfig:
    model: ModelConfig
    batch_size: int
    max_steps: int
    learning_rate: float
    beta1: float
    beta2: float
    lambda_l1: float
    horizontal_flip: bool
    precision: str
    checkpoint_every: int
    log_every: int
    seed: int
    num_workers: int

    def to_json(self) -> dict[str, object]:
        return {
            "format": TRAIN_CONFIG_FORMAT,
            "model": self.model.to_json(),
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "lambda_l1": self.lambda_l1,
            "horizontal_flip": self.horizontal_flip,
            "precision": self.precision,
            "checkpoint_every": self.checkpoint_every,
            "log_every": self.log_every,
            "seed": self.seed,
            "num_workers": self.num_workers,
        }

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        payload = read_json(path, label="pix2pix training config")
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
            "precision",
            "checkpoint_every",
            "log_every",
            "seed",
            "num_workers",
        }
        values = _exact_object(payload, expected, "training config")
        if values["format"] != TRAIN_CONFIG_FORMAT:
            raise Pix2PixError(
                f"unsupported training config format: {values['format']!r}"
            )
        config = cls(
            model=ModelConfig.from_json(values["model"]),
            batch_size=_integer(values, "batch_size"),
            max_steps=_integer(values, "max_steps"),
            learning_rate=_number(values, "learning_rate"),
            beta1=_number(values, "beta1"),
            beta2=_number(values, "beta2"),
            lambda_l1=_number(values, "lambda_l1"),
            horizontal_flip=_boolean(values, "horizontal_flip"),
            precision=_string(values, "precision"),
            checkpoint_every=_integer(values, "checkpoint_every"),
            log_every=_integer(values, "log_every"),
            seed=_integer(values, "seed"),
            num_workers=_integer(values, "num_workers"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        self.model.validate()
        if self.batch_size < 1:
            raise Pix2PixError("batch_size must be positive")
        if self.max_steps < 1:
            raise Pix2PixError("max_steps must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise Pix2PixError("learning_rate must be positive")
        if (
            not math.isfinite(self.beta1)
            or not math.isfinite(self.beta2)
            or not 0 <= self.beta1 < 1
            or not 0 <= self.beta2 < 1
        ):
            raise Pix2PixError("Adam beta values must be in [0, 1)")
        if not math.isfinite(self.lambda_l1) or self.lambda_l1 <= 0:
            raise Pix2PixError("lambda_l1 must be positive")
        if self.precision not in {"fp32", "bf16"}:
            raise Pix2PixError("precision must be fp32 or bf16")
        if self.checkpoint_every < 1:
            raise Pix2PixError("checkpoint_every must be positive")
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


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise Pix2PixError(f"{key} must be a string")
    return value
