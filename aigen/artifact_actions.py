from __future__ import annotations

import base64
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from aigen.manifest_io import copy_sha256


def open_artifact(path: Path) -> None:
    path = path.expanduser().resolve(strict=True)
    if sys.platform == "win32":
        os.startfile(path)
        return
    if "microsoft" in platform.release().lower():
        windows_path = subprocess.run(
            ["wslpath", "-w", str(path)], capture_output=True, text=True, check=True,
        ).stdout.strip()
        # PowerShell single-quoted data literals escape only the quote itself.
        script = "$ErrorActionPreference='Stop'; Start-Process -FilePath '" + windows_path.replace("'", "''") + "'"
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]
    elif sys.platform == "darwin":
        command = ["open", str(path)]
    else:
        command = ["xdg-open", str(path)]
    subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, check=True)


def export_artifact(source: Path, destination: Path, *, expected_sha256: str | None = None) -> Path:
    """Publish original file bytes atomically, without replacing existing exports."""
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().resolve()
    if source.suffix.lower() != destination.suffix.lower():
        raise ValueError(f"export keeps the original format; use the {source.suffix} extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".aigen-export-") as temporary:
        with source.open("rb") as stream:
            if expected_sha256 is None:
                shutil.copyfileobj(stream, temporary)
            elif copy_sha256(stream, temporary) != expected_sha256:
                raise ValueError(f"artifact changed since this result was recorded: {source}")
        temporary.flush()
        os.fsync(temporary.fileno())
        os.link(temporary.name, destination)
    return destination
