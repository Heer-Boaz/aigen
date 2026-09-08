from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

from pydantic import BaseModel, ConfigDict, Field

from aigen.manifest_io import atomic_write_json, copy_sha256, read_json, sha256_bytes, sha256_file


@dataclass(frozen=True)
class SourceSnapshot:
    path: Path
    sha256: str


class _SourceIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class WorkflowSourceStore:
    """Own stable input bytes independently of editable files and run lifetimes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._captured: dict[str, SourceSnapshot] = {}

    def capture(self, path: Path, *, expected_sha256: str | None = None) -> SourceSnapshot:
        path = path.expanduser().resolve(strict=True)
        with path.open("rb") as source:
            before = _file_revision(os.fstat(source.fileno()))
            key = sha256_bytes(json.dumps((str(path), before), separators=(",", ":")).encode())
            snapshot = self._captured.get(key)
            if snapshot is None:
                index = self.root / "index" / f"{key}.json"
                if index.exists():
                    checksum = _SourceIndex.model_validate(read_json(index, label="source snapshot index")).sha256
                    snapshot = SourceSnapshot(self._object_path(checksum, path.suffix), checksum)
                    if sha256_file(snapshot.path) != checksum:
                        raise ValueError(f"source snapshot contents changed: {snapshot.path}")
                else:
                    self.root.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=self.root, prefix=".capture-") as temporary:
                        checksum = copy_sha256(source, temporary)
                        if _file_revision(os.fstat(source.fileno())) != before:
                            raise ValueError(f"source changed while being captured; run again: {path}")
                        temporary.flush()
                        os.fsync(temporary.fileno())
                        os.fchmod(temporary.fileno(), 0o444)
                        snapshot = SourceSnapshot(self._object_path(checksum, path.suffix), checksum)
                        snapshot.path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.link(temporary.name, snapshot.path)
                        except FileExistsError:
                            if sha256_file(snapshot.path) != checksum:
                                raise ValueError(f"source snapshot contents changed: {snapshot.path}")
                    atomic_write_json(index, {"sha256": checksum})
                self._captured[key] = snapshot
        if expected_sha256 is not None and snapshot.sha256 != expected_sha256:
            raise ValueError(f"source no longer matches its recorded contents: {path}")
        return snapshot

    def _object_path(self, checksum: str, suffix: str) -> Path:
        return self.root / "objects" / checksum[:2] / f"{checksum}{suffix.lower()}"


def _file_revision(info: os.stat_result) -> tuple[int, ...]:
    # ctime detects in-place changes even when the original mtime is restored.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
