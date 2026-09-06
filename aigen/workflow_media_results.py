from __future__ import annotations

from dataclasses import dataclass
from hashlib import file_digest
import json
import os
from pathlib import Path
import tempfile
import zipfile

import av
from PIL import Image

from aigen.artifact_actions import export_artifact
from aigen.image_io import load_thumbnail
from aigen.manifest_io import sha256_file
from aigen.workflow_artifacts import AudioArtifact, ImageSequenceArtifact, VideoArtifact, MaskArtifact
from aigen.workflow_cache import WorkflowNodeCache
from aigen.workflow_results import load_node_result


MediaArtifact = AudioArtifact | ImageSequenceArtifact | VideoArtifact | MaskArtifact


@dataclass(frozen=True)
class MediaCandidate:
    title: str
    seed: int | None
    manifest_path: Path
    producer_signature: str
    port: str
    artifact: MediaArtifact

    @property
    def output_id(self) -> str:
        return f"{self.producer_signature}:{self.port}:{self.artifact.identity}"


def resolve_media_result(candidate: MediaCandidate) -> MediaArtifact:
    result = load_node_result(candidate.manifest_path)
    artifact = result.outputs[candidate.port]
    if result.signature != candidate.producer_signature or artifact != candidate.artifact:
        raise ValueError("media output no longer matches its saved result")
    if result.cache_manifest is not None:
        cached = WorkflowNodeCache.read_result(Path(result.cache_manifest))
        output = cached.outputs[candidate.port]
        if isinstance(artifact, ImageSequenceArtifact):
            valid = (isinstance(output, ImageSequenceArtifact) and artifact.paths == output.paths
                     and (artifact.pixel_identity or artifact.identity) == output.pixel_identity)
            if valid:
                artifact = artifact.model_copy(update={"pixel_identity": output.pixel_identity, "frame_sha256s": output.frame_sha256s})
        elif isinstance(artifact, VideoArtifact):
            valid = (isinstance(output, VideoArtifact) and artifact.path == output.path and artifact.identity == output.identity
                     and (artifact.content_sha256 is None or artifact.content_sha256 == output.content_sha256))
            if valid:
                artifact = output
        else:
            valid = artifact == output
        if cached.signature != result.signature or not valid:
            raise ValueError("media output no longer matches its immutable cache entry")
    elif isinstance(artifact, (VideoArtifact, MaskArtifact)):
        if sha256_file(Path(artifact.path)) != artifact.content_sha256:
            raise ValueError(f"source video changed: {artifact.path}")
    if isinstance(artifact, AudioArtifact):
        audio = artifact.track
    elif isinstance(artifact, ImageSequenceArtifact):
        audio = artifact.audio
    else:
        audio = None
    if audio is not None and sha256_file(Path(audio.path)) != audio.file_sha256:
        raise ValueError(f"audio source changed: {audio.path}")
    return artifact


def media_path(artifact: MediaArtifact) -> Path:
    if isinstance(artifact, AudioArtifact):
        return Path(artifact.track.path)
    if isinstance(artifact, ImageSequenceArtifact):
        return Path(artifact.paths[0]).parent
    return Path(artifact.path)


def media_preview(artifact: MediaArtifact, size: tuple[int, int]) -> Image.Image | None:
    if isinstance(artifact, MaskArtifact):
        return load_thumbnail(Path(artifact.path), size, expected_sha256=artifact.content_sha256)
    if isinstance(artifact, AudioArtifact):
        return None
    if isinstance(artifact, ImageSequenceArtifact):
        if artifact.frame_sha256s is None:
            raise ValueError("legacy frames have no recorded checksums")
        return load_thumbnail(Path(artifact.paths[0]), size, expected_sha256=artifact.frame_sha256s[0])
    if artifact.content_sha256 is None:
        raise ValueError("legacy video has no recorded checksum")
    # One poster frame only: opening a result never decodes the entire video.
    with Path(artifact.path).open("rb") as source:
        if file_digest(source, "sha256").hexdigest() != artifact.content_sha256:
            raise ValueError(f"video changed: {artifact.path}")
        source.seek(0)
        with av.open(source) as container:
            frame = next(container.decode(video=0))
            image = frame.to_image()
            if frame.rotation:
                image = image.rotate(frame.rotation, expand=True)
            image.thumbnail(size, Image.Resampling.LANCZOS)
            return image


def media_label(artifact: MediaArtifact) -> str:
    if isinstance(artifact, MaskArtifact):
        return f"Repaint mask · {artifact.source_size[0]}×{artifact.source_size[1]} · white is editable"
    if isinstance(artifact, VideoArtifact):
        info = artifact.info
        return f"Video · {info.width}×{info.height} · {info.frames} frames · {info.fps} FPS" if info else "Video"
    if isinstance(artifact, ImageSequenceArtifact):
        timing = f" · {artifact.timeline.fps} FPS" if artifact.timeline else " · no recorded timing"
        return f"{len(artifact.paths)} frames{timing}"
    return f"Audio · stream {artifact.track.stream_index}"


def export_media(artifact: MediaArtifact, destination: Path) -> Path:
    if not isinstance(artifact, ImageSequenceArtifact):
        return export_artifact(media_path(artifact), destination)
    destination = destination.expanduser().resolve()
    if destination.suffix.lower() != ".zip":
        raise ValueError("export frame sequences as a .zip archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".aigen-export-") as temporary:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            names = []
            for index, source in enumerate(artifact.paths):
                name = f"frame-{index:06d}{Path(source).suffix}"
                archive.write(source, name)
                names.append(name)
            audio = artifact.audio
            if audio is not None:
                name = f"audio-source{Path(audio.path).suffix}"
                archive.write(audio.path, name)
                audio = audio.model_copy(update={"path": name})
            archive.writestr("frames.json", json.dumps({
                "paths": names,
                "timeline": artifact.timeline.model_dump(mode="json") if artifact.timeline else None,
                "audio": audio.model_dump(mode="json") if audio else None,
            }, indent=2))
        temporary.flush()
        os.fsync(temporary.fileno())
        os.link(temporary.name, destination)
    return destination
