from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from aigen.manifest_io import read_json
from aigen.pix2pix.corpus_config import SAFE_NAME_PATTERN, SourceRasterConfig
from aigen.pix2pix.errors import Pix2PixError


QWEN_SOURCE_CONFIG_FORMAT = "aigen.pix2pix.qwen-source.v1"
QWEN_SOURCE_BACKEND = "qwen-image-edit-2511-lightning"


class QwenSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["aigen.pix2pix.qwen-source.v1"]
    name: str = Field(pattern=SAFE_NAME_PATTERN)
    backend: Literal["qwen-image-edit-2511-lightning"]
    prompt: str = Field(min_length=1)
    seed_base: int = Field(ge=0)
    shard_size: int = Field(gt=0, le=16)
    width: Literal[1328]
    height: Literal[1328]
    steps: Literal[8]
    guidance: Literal[1.0]
    sampler: Literal["flowmatch-euler"]
    scheduler: Literal["flowmatch-dynamic-shift"]
    source_raster: SourceRasterConfig

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        if prompt != prompt.strip():
            raise ValueError(
                "Qwen source prompt must not contain surrounding whitespace"
            )
        return prompt


def load_qwen_source_config(path: Path) -> QwenSourceConfig:
    config_path = path.expanduser().resolve()
    payload = read_json(config_path, label="Qwen pix2pix source config")
    try:
        return QwenSourceConfig.model_validate(payload)
    except ValidationError as error:
        raise Pix2PixError(
            f"invalid Qwen pix2pix source config: {error}"
        ) from error


def qwen_source_config_fingerprint(config: QwenSourceConfig) -> str:
    encoded = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
