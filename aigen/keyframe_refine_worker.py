from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from aigen.keyframe_profiles import KeyframeRefineProfile
from aigen.keyframe_refine import (
    run_keyframe_refine_variant,
)
from aigen.keyframe_refine_models import KeyframeRefineError
from aigen.manifest_io import ManifestIOError, read_json, write_json_line


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aigen.keyframe_refine_worker")
    parser.add_argument("job", type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--resolved", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        resolved = read_json(args.resolved, label="refine worker artifact")
        result = run_keyframe_refine_variant(
            args.job,
            _profile_from_resolved(resolved),
            variant_name=args.variant,
            project_root=args.project_root,
            resolved=resolved,
        )
    except (KeyframeRefineError, ManifestIOError) as error:
        write_json_line(
            sys.stderr,
            {
                "schema_version": 1,
                "status": "error",
                "error": error.__class__.__name__,
                "message": str(error),
            },
        )
        return 1
    write_json_line(sys.stdout, result)
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


if __name__ == "__main__":
    raise SystemExit(main())
