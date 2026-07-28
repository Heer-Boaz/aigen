from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from aigen.generation.image_edit_batch import (
    ImageEditBatchFailure,
    ImageEditBatchRequest,
    run_image_edit_batch,
)
from aigen.manifest_io import atomic_write_json
from aigen.progress import open_cli_progress


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: image_edit_batch_worker REQUEST RESPONSE"
        )
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    try:
        request = ImageEditBatchRequest.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
        with open_cli_progress() as progress:
            result = run_image_edit_batch(
                request,
                progress=progress,
            )
        atomic_write_json(
            response_path,
            result.model_dump(mode="json"),
        )
    except Exception as error:
        traceback.print_exc()
        atomic_write_json(
            response_path,
            ImageEditBatchFailure(
                error=error.__class__.__name__,
                message=str(error),
            ).model_dump(mode="json"),
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
