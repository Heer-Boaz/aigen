from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from aigen.workflow_graph import ArtifactType


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ImageArtifact(_ArtifactModel):
    type: Literal[ArtifactType.IMAGE] = ArtifactType.IMAGE
    path: str
    identity: str = Field(min_length=1)


class ReferencePackArtifact(_ArtifactModel):
    type: Literal[ArtifactType.REFERENCE_PACK] = ArtifactType.REFERENCE_PACK
    path: str
    references: tuple[str, ...]
    identity: str = Field(min_length=1)


class LoraArtifact(_ArtifactModel):
    type: Literal[ArtifactType.LORA] = ArtifactType.LORA
    path: str
    weight: float
    identity: str = Field(min_length=1)


class VideoArtifact(_ArtifactModel):
    type: Literal[ArtifactType.VIDEO] = ArtifactType.VIDEO
    path: str
    identity: str = Field(min_length=1)


class ImageSequenceArtifact(_ArtifactModel):
    type: Literal[ArtifactType.IMAGE_SEQUENCE] = ArtifactType.IMAGE_SEQUENCE
    paths: tuple[str, ...] = Field(min_length=1)
    identity: str = Field(min_length=1)


WorkflowArtifact: TypeAlias = Annotated[
    ImageArtifact
    | ReferencePackArtifact
    | LoraArtifact
    | VideoArtifact
    | ImageSequenceArtifact,
    Field(discriminator="type"),
]
