from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from hashlib import sha256
from importlib.metadata import distribution
from pathlib import Path


PYTHON_RUNTIME_PROVENANCE_FORMAT = "aigen.python-runtime.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def build_python_runtime_provenance(
    distribution_names: tuple[str, ...],
) -> dict[str, object]:
    records = [_distribution_record(name) for name in distribution_names]
    return _runtime_provenance(platform.python_version(), records)


def build_python_runtime_provenance_for_interpreter(
    interpreter: Path,
    distribution_names: tuple[str, ...],
) -> dict[str, object]:
    executable = Path(
        os.path.abspath(interpreter.expanduser())
    )
    if not executable.is_file():
        raise RuntimeError(
            f"Python runtime interpreter does not exist: {executable.as_posix()}"
        )
    script = "\n".join(
        (
            "import importlib.metadata as metadata",
            "import json",
            "import platform",
            "import sys",
            "records = []",
            "for requested_name in json.loads(sys.argv[1]):",
            "    installed = metadata.distribution(requested_name)",
            "    direct_url_text = installed.read_text('direct_url.json')",
            "    records.append({",
            "        'name': str(installed.metadata['Name']).lower(),",
            "        'version': installed.version,",
            "        'direct_url': json.loads(direct_url_text) if direct_url_text is not None else None,",
            "    })",
            "print(json.dumps({'python': platform.python_version(), 'distributions': records}))",
        )
    )
    try:
        completed = subprocess.run(
            [
                executable.as_posix(),
                "-c",
                script,
                json.dumps(distribution_names),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        stderr = (
            error.stderr.strip()
            if isinstance(error, subprocess.CalledProcessError)
            and isinstance(error.stderr, str)
            else ""
        )
        raise RuntimeError(
            f"cannot inspect Python runtime {executable.as_posix()}: "
            f"{stderr or error}"
        ) from error
    python_version = payload.get("python")
    records = payload.get("distributions")
    if not isinstance(python_version, str) or not isinstance(records, list):
        raise RuntimeError(
            f"Python runtime {executable.as_posix()} returned invalid provenance"
        )
    return _runtime_provenance(python_version, records)


def _distribution_record(requested_name: str) -> dict[str, object]:
    installed = distribution(requested_name)
    direct_url_text = installed.read_text("direct_url.json")
    return {
        "name": str(installed.metadata["Name"]).lower(),
        "version": installed.version,
        "direct_url": (
            json.loads(direct_url_text)
            if direct_url_text is not None
            else None
        ),
    }


def _runtime_provenance(
    python_version: str,
    records: list[dict[str, object]],
) -> dict[str, object]:
    payload = {
        "format": PYTHON_RUNTIME_PROVENANCE_FORMAT,
        "python": python_version,
        "distributions": sorted(records, key=lambda record: record["name"]),
    }
    return {
        **payload,
        "fingerprint": _provenance_fingerprint(payload),
    }


def validate_python_runtime_provenance(payload: dict[str, object]) -> None:
    if set(payload) != {
        "format",
        "python",
        "distributions",
        "fingerprint",
    }:
        raise ValueError("invalid Python runtime provenance keys")
    if payload["format"] != PYTHON_RUNTIME_PROVENANCE_FORMAT:
        raise ValueError("unsupported Python runtime provenance format")
    if not isinstance(payload["python"], str) or not payload["python"]:
        raise ValueError("invalid Python runtime version")
    distributions = payload["distributions"]
    if not isinstance(distributions, list) or not distributions:
        raise ValueError("Python runtime provenance has no distributions")
    names = []
    for record in distributions:
        if not isinstance(record, dict) or set(record) != {
            "name",
            "version",
            "direct_url",
        }:
            raise ValueError("invalid Python distribution provenance")
        name = record["name"]
        version = record["version"]
        direct_url = record["direct_url"]
        if not isinstance(name, str) or not name:
            raise ValueError("invalid Python distribution name")
        if not isinstance(version, str) or not version:
            raise ValueError(f"invalid Python distribution version for {name!r}")
        if direct_url is not None and not isinstance(direct_url, dict):
            raise ValueError(f"invalid direct_url provenance for {name!r}")
        names.append(name)
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("unordered or duplicate Python distributions")
    fingerprint = payload["fingerprint"]
    if not isinstance(fingerprint, str) or not _SHA256_PATTERN.fullmatch(
        fingerprint
    ):
        raise ValueError("invalid Python runtime fingerprint")
    expected_fingerprint = _provenance_fingerprint(
        {
            "format": payload["format"],
            "python": payload["python"],
            "distributions": distributions,
        }
    )
    if fingerprint != expected_fingerprint:
        raise ValueError("Python runtime provenance fingerprint mismatch")


def _provenance_fingerprint(payload: dict[str, object]) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
