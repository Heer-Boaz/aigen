from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from aigen.workflow_graph import ArtifactType, ImageResultReference
from aigen.media_timing import AudioTrack, FrameTimeline, VideoInfo
from aigen.manifest_io import sha256_bytes


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ImageArtifact(_ArtifactModel):
    type: Literal[ArtifactType.IMAGE] = ArtifactType.IMAGE
    path: str
    identity: str = Field(min_length=1)
    content_sha256: str | None = None


class MaskArtifact(_ArtifactModel):
    type: Literal[ArtifactType.MASK] = ArtifactType.MASK
    path: str
    identity: str = Field(min_length=1)
    content_sha256: str
    source_sha256: str
    source_size: tuple[int, int]


def mask_identity(content_identity: str, source_sha256: str, source_size: tuple[int, int]) -> str:
    return sha256_bytes(json.dumps([content_identity, source_sha256, source_size], separators=(",", ":")).encode())


class ReferencePackArtifact(_ArtifactModel):
    type: Literal[ArtifactType.REFERENCE_PACK] = ArtifactType.REFERENCE_PACK
    path: str
    references: tuple[str, ...]
    identity: str = Field(min_length=1)
    reference_sha256s: tuple[str, ...] | None = None


class LoraArtifact(_ArtifactModel):
    type: Literal[ArtifactType.LORA] = ArtifactType.LORA
    path: str
    weight: float
    identity: str = Field(min_length=1)


class VideoArtifact(_ArtifactModel):
    type: Literal[ArtifactType.VIDEO] = ArtifactType.VIDEO
    path: str
    identity: str = Field(min_length=1)
    content_sha256: str | None = None
    info: VideoInfo | None = None


class AudioArtifact(_ArtifactModel):
    type: Literal[ArtifactType.AUDIO] = ArtifactType.AUDIO
    track: AudioTrack
    identity: str = Field(min_length=1)


class KeyframeArtifact(_ArtifactModel):
    type: Literal[ArtifactType.KEYFRAME] = ArtifactType.KEYFRAME
    image: ImageArtifact
    frame: int
    identity: str = Field(min_length=1)


class ImageSequenceArtifact(_ArtifactModel):
    type: Literal[ArtifactType.IMAGE_SEQUENCE] = ArtifactType.IMAGE_SEQUENCE
    paths: tuple[str, ...] = Field(min_length=1)
    identity: str = Field(min_length=1)
    pixel_identity: str | None = None
    frame_sha256s: tuple[str, ...] | None = None
    timeline: FrameTimeline | None = None
    audio: AudioTrack | None = None


def sequence_identity(pixel_identity: str, timeline: FrameTimeline | None, audio: AudioTrack | None) -> str:
    if timeline is None and audio is None:
        return pixel_identity
    payload = {"pixels": pixel_identity, "timeline": timeline.model_dump(mode="json") if timeline else None,
               "audio": audio.model_dump(mode="json", exclude={"path"}) if audio else None}
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def with_sequence_timing(images: ImageSequenceArtifact, *, timeline: FrameTimeline | None,
                         audio: AudioTrack | None) -> ImageSequenceArtifact:
    assert images.pixel_identity is not None
    return images.model_copy(update={
        "identity": sequence_identity(images.pixel_identity, timeline, audio),
        "timeline": timeline, "audio": audio,
    })


class ImageCandidate(_ArtifactModel):
    node_id: str
    title: str
    image: ImageArtifact
    reference: ImageResultReference
    seed: int | None = None

    @property
    def manifest_path(self) -> Path:
        return Path(self.reference.manifest_path)

    @property
    def output_id(self) -> str:
        return f"{self.reference.producer_signature}:{self.reference.output_port}:{self.image.identity}"


class ImageCollectionArtifact(_ArtifactModel):
    type: Literal[ArtifactType.IMAGE_COLLECTION] = ArtifactType.IMAGE_COLLECTION
    candidates: tuple[ImageCandidate, ...] = Field(min_length=1)
    identity: str = Field(min_length=1)


WorkflowArtifact: TypeAlias = Annotated[
    ImageArtifact
    | MaskArtifact
    | ReferencePackArtifact
    | LoraArtifact
    | VideoArtifact
    | ImageSequenceArtifact
    | ImageCollectionArtifact
    | AudioArtifact
    | KeyframeArtifact,
    Field(discriminator="type"),
]


ArtifactT = TypeVar("ArtifactT", bound=_ArtifactModel)


def one_artifact(inputs: Mapping[str, Sequence[WorkflowArtifact]], port: str, artifact_class: type[ArtifactT]) -> ArtifactT:
    artifacts = inputs[port]
    if len(artifacts) != 1 or not isinstance(artifacts[0], artifact_class):
        raise ValueError(f"workflow input {port!r} does not contain one {artifact_class.__name__}")
    return artifacts[0]
