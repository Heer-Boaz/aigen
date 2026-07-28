from __future__ import annotations

import errno
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from aigen.manifest_io import atomic_write_json, sha256_bytes, sha256_file
from aigen.workflow_artifacts import (
    ImageArtifact,
    ImageSequenceArtifact,
    VideoArtifact,
)
from aigen.workflow_graph import ArtifactType, NodeKind


WORKFLOW_CACHE_VERSION = 2
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_CACHEABLE_ARTIFACT_TYPES = frozenset(
    (
        ArtifactType.IMAGE,
        ArtifactType.VIDEO,
        ArtifactType.IMAGE_SEQUENCE,
    )
)


class WorkflowCacheError(RuntimeError):
    pass


class WorkflowCacheCorruptionError(WorkflowCacheError):
    pass


class _CacheModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RevisionedComponent(_CacheModel):
    name: str = Field(min_length=1, max_length=160)
    revision: str = Field(min_length=1, max_length=256)


class NodeExecutionProvenance(_CacheModel):
    """Revision identity for every implementation component that can change bytes."""

    executor: RevisionedComponent
    backend: RevisionedComponent
    models: tuple[RevisionedComponent, ...] = ()

    @field_validator("models")
    @classmethod
    def normalize_models(
        cls,
        models: tuple[RevisionedComponent, ...],
    ) -> tuple[RevisionedComponent, ...]:
        ordered = tuple(sorted(models, key=lambda component: component.name))
        names = tuple(component.name for component in ordered)
        if len(names) != len(set(names)):
            raise ValueError("model provenance contains duplicate component names")
        return ordered


class NodeInputIdentity(_CacheModel):
    artifact_type: ArtifactType
    identity: str = Field(min_length=1)


class _CachedFile(_CacheModel):
    path: str = Field(min_length=1)
    size: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)


class _CachedArtifact(_CacheModel):
    artifact_type: ArtifactType
    files: tuple[_CachedFile, ...] = Field(min_length=1)
    identity: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_file_count(self) -> _CachedArtifact:
        if self.artifact_type not in _CACHEABLE_ARTIFACT_TYPES:
            raise ValueError(
                f"artifact type {self.artifact_type!r} is not a generated-file artifact"
            )
        if (
            self.artifact_type in (ArtifactType.IMAGE, ArtifactType.VIDEO)
            and len(self.files) != 1
        ):
            raise ValueError(f"{self.artifact_type} cache artifacts require exactly one file")
        return self


class _NodeCacheManifest(_CacheModel):
    version: Literal[2] = WORKFLOW_CACHE_VERSION
    signature: str = Field(pattern=_SHA256_PATTERN)
    node_kind: NodeKind
    provenance: NodeExecutionProvenance
    outputs: dict[str, _CachedArtifact] = Field(min_length=1)


@dataclass(frozen=True)
class GeneratedNodeOutput:
    artifact_type: ArtifactType
    paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if self.artifact_type not in _CACHEABLE_ARTIFACT_TYPES:
            raise ValueError(
                f"artifact type {self.artifact_type!r} is not a generated-file artifact"
            )
        if not self.paths:
            raise ValueError("generated cache output requires at least one file")
        if (
            self.artifact_type in (ArtifactType.IMAGE, ArtifactType.VIDEO)
            and len(self.paths) != 1
        ):
            raise ValueError(
                f"{self.artifact_type} cache outputs require exactly one file"
            )


@dataclass(frozen=True)
class NodeCacheHit:
    signature: str
    node_kind: NodeKind
    outputs: Mapping[
        str,
        ImageArtifact | VideoArtifact | ImageSequenceArtifact,
    ]


def build_node_signature(
    *,
    node_kind: NodeKind,
    execution_config: Mapping[str, object],
    inputs: Mapping[str, Sequence[NodeInputIdentity]],
    source_outputs: Mapping[str, NodeInputIdentity] | None,
    provenance: NodeExecutionProvenance,
) -> str:
    """Build the graph-independent action key for one node invocation."""

    payload = {
        "node_kind": node_kind,
        "execution_config": dict(execution_config),
        "inputs": {
            port: [
                identity.model_dump(mode="json")
                for identity in identities
            ]
            for port, identities in sorted(inputs.items())
        },
        "source_outputs": (
            {
                port: identity.model_dump(mode="json")
                for port, identity in sorted(source_outputs.items())
            }
            if source_outputs is not None
            else None
        ),
        "provenance": provenance.model_dump(mode="json"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(encoded)


class WorkflowNodeCache:
    """Global immutable node-result cache, independent of workflow run folders."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def entry_dir(self, signature: str) -> Path:
        _validate_signature(signature)
        return self.root / "nodes" / signature[:2] / signature

    def lookup(
        self,
        signature: str,
        *,
        node_kind: NodeKind,
        provenance: NodeExecutionProvenance,
    ) -> NodeCacheHit | None:
        entry_dir = self.entry_dir(signature)
        if not entry_dir.exists():
            return None
        if not entry_dir.is_dir():
            raise WorkflowCacheCorruptionError(
                f"workflow cache entry is not a directory: {entry_dir}"
            )
        manifest_path = entry_dir / "result.json"
        if not manifest_path.is_file():
            raise WorkflowCacheCorruptionError(
                f"workflow cache entry has no result manifest: {entry_dir}"
            )
        try:
            manifest = _NodeCacheManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise WorkflowCacheCorruptionError(
                f"invalid workflow cache manifest: {manifest_path}"
            ) from error
        if manifest.signature != signature:
            raise WorkflowCacheCorruptionError(
                f"workflow cache signature mismatch: {manifest_path}"
            )
        if manifest.node_kind != node_kind:
            raise WorkflowCacheCorruptionError(
                f"workflow cache node-kind mismatch: {manifest_path}"
            )
        if manifest.provenance != provenance:
            raise WorkflowCacheCorruptionError(
                f"workflow cache provenance mismatch: {manifest_path}"
            )
        return _cache_hit(entry_dir, manifest)

    def begin(
        self,
        signature: str,
        *,
        node_kind: NodeKind,
        provenance: NodeExecutionProvenance,
    ) -> NodeCacheWrite:
        _validate_signature(signature)
        staging_root = self.root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                dir=staging_root,
                prefix=f"{signature[:16]}.",
            )
        )
        output_dir = staging_dir / "outputs"
        output_dir.mkdir()
        return NodeCacheWrite(
            cache=self,
            signature=signature,
            node_kind=node_kind,
            provenance=provenance,
            staging_dir=staging_dir,
            output_dir=output_dir,
        )


class NodeCacheWrite:
    """Atomic cache-entry publication transaction."""

    def __init__(
        self,
        *,
        cache: WorkflowNodeCache,
        signature: str,
        node_kind: NodeKind,
        provenance: NodeExecutionProvenance,
        staging_dir: Path,
        output_dir: Path,
    ) -> None:
        self.cache = cache
        self.signature = signature
        self.node_kind = node_kind
        self.provenance = provenance
        self.staging_dir = staging_dir
        self.output_dir = output_dir
        self._published = False

    def __enter__(self) -> NodeCacheWrite:
        return self

    def __exit__(self, *_: object) -> None:
        if not self._published and self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)

    def publish(
        self,
        outputs: Mapping[str, GeneratedNodeOutput],
    ) -> NodeCacheHit:
        if self._published:
            raise WorkflowCacheError("workflow cache transaction was already published")
        if not outputs:
            raise WorkflowCacheError("workflow cache result has no outputs")

        cached_outputs = {
            port: _capture_artifact(self.staging_dir, output)
            for port, output in outputs.items()
        }
        manifest = _NodeCacheManifest(
            signature=self.signature,
            node_kind=self.node_kind,
            provenance=self.provenance,
            outputs=cached_outputs,
        )
        atomic_write_json(
            self.staging_dir / "result.json",
            manifest.model_dump(mode="json"),
        )

        entry_dir = self.cache.entry_dir(self.signature)
        entry_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(self.staging_dir, entry_dir)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            winner = self.cache.lookup(
                self.signature,
                node_kind=self.node_kind,
                provenance=self.provenance,
            )
            if winner is None:
                raise WorkflowCacheCorruptionError(
                    f"workflow cache publication race produced no entry: {entry_dir}"
                ) from error
            shutil.rmtree(self.staging_dir)
            self._published = True
            return winner

        self._published = True
        return _cache_hit(entry_dir, manifest)


def _capture_artifact(
    staging_dir: Path,
    output: GeneratedNodeOutput,
) -> _CachedArtifact:
    resolved_staging = staging_dir.resolve()
    files = tuple(
        _capture_file(resolved_staging, path)
        for path in output.paths
    )
    return _CachedArtifact(
        artifact_type=output.artifact_type,
        files=files,
        identity=_artifact_identity(output.artifact_type, files),
    )


def _capture_file(resolved_staging: Path, path: Path) -> _CachedFile:
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(resolved_staging)
    except (OSError, ValueError) as error:
        raise WorkflowCacheError(
            f"workflow cache output must be a file inside {resolved_staging}: {path}"
        ) from error
    file_stat = resolved.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise WorkflowCacheError(f"workflow cache output is not a file: {resolved}")
    return _CachedFile(
        path=relative.as_posix(),
        size=file_stat.st_size,
        mtime_ns=file_stat.st_mtime_ns,
        sha256=sha256_file(resolved),
    )


def _cache_hit(
    entry_dir: Path,
    manifest: _NodeCacheManifest,
) -> NodeCacheHit:
    outputs: dict[str, ImageArtifact | VideoArtifact | ImageSequenceArtifact] = {}
    resolved_entry_dir = entry_dir.resolve(strict=True)
    for port, artifact in manifest.outputs.items():
        paths = tuple(
            _resolve_cached_file(
                resolved_entry_dir,
                file,
            )
            for file in artifact.files
        )
        if _artifact_identity(artifact.artifact_type, artifact.files) != artifact.identity:
            raise WorkflowCacheCorruptionError(
                f"workflow cache artifact identity mismatch: {entry_dir} port {port!r}"
            )
        outputs[port] = _file_artifact(
            artifact.artifact_type,
            paths,
            identity=artifact.identity,
        )
    return NodeCacheHit(
        signature=manifest.signature,
        node_kind=manifest.node_kind,
        outputs=MappingProxyType(outputs),
    )


def _resolve_cached_file(
    resolved_entry_dir: Path,
    cached_file: _CachedFile,
) -> Path:
    relative = Path(cached_file.path)
    if relative.is_absolute():
        raise WorkflowCacheCorruptionError(
            f"workflow cache manifest contains an absolute output path: {relative}"
        )
    try:
        path = (resolved_entry_dir / relative).resolve(strict=True)
        path.relative_to(resolved_entry_dir)
    except (OSError, ValueError) as error:
        raise WorkflowCacheCorruptionError(
            f"workflow cache output escapes or is missing from its entry: {relative}"
        ) from error
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise WorkflowCacheCorruptionError(
            f"workflow cache output is not a file: {path}"
        )
    if file_stat.st_size != cached_file.size:
        raise WorkflowCacheCorruptionError(
            f"workflow cache output size changed: {path}"
        )
    if file_stat.st_mtime_ns != cached_file.mtime_ns:
        raise WorkflowCacheCorruptionError(
            f"workflow cache output modification time changed: {path}"
        )
    if sha256_file(path) != cached_file.sha256:
        raise WorkflowCacheCorruptionError(
            f"workflow cache output content changed: {path}"
        )
    return path


def _artifact_identity(
    artifact_type: ArtifactType,
    files: Sequence[_CachedFile],
) -> str:
    encoded = json.dumps(
        {
            "artifact_type": artifact_type,
            "files": [
                {
                    "size": file.size,
                    "sha256": file.sha256,
                }
                for file in files
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _file_artifact(
    artifact_type: ArtifactType,
    paths: tuple[Path, ...],
    *,
    identity: str,
) -> ImageArtifact | VideoArtifact | ImageSequenceArtifact:
    if artifact_type == ArtifactType.IMAGE:
        return ImageArtifact(path=paths[0].as_posix(), identity=identity)
    if artifact_type == ArtifactType.VIDEO:
        return VideoArtifact(path=paths[0].as_posix(), identity=identity)
    if artifact_type == ArtifactType.IMAGE_SEQUENCE:
        return ImageSequenceArtifact(
            paths=tuple(path.as_posix() for path in paths),
            identity=identity,
        )
    raise WorkflowCacheCorruptionError(
        f"workflow cache contains unsupported artifact type: {artifact_type}"
    )


def _validate_signature(signature: str) -> None:
    if re.fullmatch(_SHA256_PATTERN, signature) is None:
        raise ValueError("workflow cache signature must be a lowercase SHA-256 digest")
