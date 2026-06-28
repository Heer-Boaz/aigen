from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from aigen.keyframe_refine import (
    KeyframeRefineError,
    KeyframeRefineProfile,
    run_keyframe_refine_variant,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aigen.keyframe_refine_worker")
    parser.add_argument("job", type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--resolved", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        resolved = _read_json(args.resolved)
        result = run_keyframe_refine_variant(
            args.job,
            _profile_from_resolved(resolved),
            variant_name=args.variant,
            project_root=args.project_root,
            resolved=resolved,
        )
    except KeyframeRefineError as error:
        _write_json(
            sys.stderr,
            {
                "schema_version": 1,
                "status": "error",
                "error": error.__class__.__name__,
                "message": str(error),
            },
        )
        return 1
    _write_json(sys.stdout, result)
    return 0


def _profile_from_resolved(resolved: dict[str, Any]) -> KeyframeRefineProfile:
    profile = resolved["profile"]
    return KeyframeRefineProfile(
        name=profile["name"],
        model=profile["models"]["kontext"]["path"],
        nunchaku_transformer_model=Path(profile["models"]["nunchaku_transformer"]["path"]),
        attention_impl=profile["attention_impl"],
        dtype=profile["dtype"],
        pipeline_cpu_offload=profile["pipeline_cpu_offload"],
        vae_tiling=profile["vae_tiling"],
        model_revisions={
            "kontext": _model_revision(profile["models"]["kontext"]),
            "nunchaku_transformer": _model_revision(profile["models"]["nunchaku_transformer"]),
        },
    )


def _model_revision(model: dict[str, Any]) -> dict[str, str]:
    return {
        "repo_id": model["repo_id"],
        "revision": model["revision"],
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise KeyframeRefineError(f"Missing refine worker artifact: {path.as_posix()}") from error
    except json.JSONDecodeError as error:
        raise KeyframeRefineError(f"Invalid refine worker JSON: {path.as_posix()}: {error}") from error


def _write_json(stream: Any, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
