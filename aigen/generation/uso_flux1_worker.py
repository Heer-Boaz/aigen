from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

from aigen.generation.uso_flux1 import (
    USO_CONTENT_REFERENCE_SIZE,
    USO_MODEL_TYPE,
)


def main() -> int:
    requests = [json.loads(line) for line in sys.stdin if line.strip()]
    if not requests:
        raise SystemExit("uso_flux1_worker requires a JSON request on stdin")
    with _open_progress_stream() as stream:
        try:
            return _run_requests(requests, stream)
        except Exception:
            traceback.print_exc()
            return 1


def _run_requests(requests: list[dict[str, Any]], stream: TextIO) -> int:
    import torch
    from PIL import Image
    from transformers import SiglipImageProcessor, SiglipVisionModel
    from uso.flux.pipeline import USOPipeline, preprocess_ref

    device = torch.device("cuda")
    _send(stream, "phase", text="loading official USO FP8 pipeline")
    pipeline = USOPipeline(
        USO_MODEL_TYPE,
        device,
        offload=True,
        only_lora=True,
        lora_rank=128,
        hf_download=False,
    )
    reference_paths = tuple(Path(path) for path in requests[0]["references"])
    references = []
    for path in reference_paths:
        with Image.open(path) as image:
            references.append(image.convert("RGB"))
    content_reference = preprocess_ref(
        references[0],
        USO_CONTENT_REFERENCE_SIZE,
    )
    style_inputs = []
    if len(references) > 1:
        siglip_path = os.environ["SIGLIP_PATH"]
        siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
        siglip_model = SiglipVisionModel.from_pretrained(siglip_path)
        siglip_model.eval()
        siglip_model.to(device)
        pipeline.model.vision_encoder = siglip_model
        pipeline.model.vision_encoder_processor = siglip_processor
        style_inputs = [
            siglip_processor(reference, return_tensors="pt").to(device)
            for reference in references[1:]
        ]
    _send(
        stream,
        "begin",
        total=len(requests),
        text="generating USO seed sweep",
    )
    for index, request in enumerate(requests, start=1):
        started = time.monotonic()
        try:
            image = pipeline(
                prompt=request["prompt"],
                width=request["width"],
                height=request["height"],
                guidance=request["guidance"],
                num_steps=request["steps"],
                seed=request["seed"],
                ref_imgs=[content_reference],
                pe="d",
                siglip_inputs=style_inputs,
            )
            output = Path(request["output"])
            image.save(output)
        except Exception as error:
            _send(
                stream,
                "result",
                response={"status": "error", "message": str(error)},
            )
            return 1
        elapsed = time.monotonic() - started
        _send(
            stream,
            "result",
            response={
                "status": "completed",
                "output": output.as_posix(),
                "elapsed_seconds": round(elapsed, 3),
                "environment": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                    "model_type": USO_MODEL_TYPE,
                    "offload": True,
                    "content_reference_size": USO_CONTENT_REFERENCE_SIZE,
                    "peak_allocated_mib": round(
                        torch.cuda.max_memory_allocated() / 1024**2,
                        1,
                    ),
                    "peak_reserved_mib": round(
                        torch.cuda.max_memory_reserved() / 1024**2,
                        1,
                    ),
                },
            },
        )
        _send(
            stream,
            "step",
            text=f"generated USO seed {request['seed']} ({index}/{len(requests)})",
        )
    return 0


def _open_progress_stream() -> TextIO:
    sys.stdout.flush()
    progress_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return os.fdopen(progress_fd, "w", encoding="utf-8", buffering=1)


def _send(stream: TextIO, kind: str, **values: Any) -> None:
    stream.write(json.dumps({"kind": kind, **values}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
