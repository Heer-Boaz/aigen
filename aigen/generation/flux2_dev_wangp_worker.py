from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO


# Profile 2 pins every WanGP component in host memory.  It uses the enlarged
# WSL memory allocation without increasing the fixed 16 GiB VRAM budget.
RUNTIME_PROFILE = "2"
PRELOAD_MIB = 4000
ATTENTION_MODE = "sdpa"
MODEL_TYPE = "flux2_dev_nvfp4"


def main() -> int:
    requests = [json.loads(line) for line in sys.stdin if line.strip()]
    if not requests:
        raise SystemExit("flux2_dev_wangp_worker requires a JSON request on stdin")
    with _open_progress_stream() as stream:
        try:
            return _run_requests(requests, stream)
        except Exception:
            traceback.print_exc()
            return 1


def _run_requests(requests: list[dict[str, Any]], stream: TextIO) -> int:
    import torch
    from shared.api import init

    runtime_root = Path(os.environ["AIGEN_WANGP_ROOT"]).resolve()
    source_root = runtime_root / "Wan2GP"
    _send(stream, "phase", text="initializing FLUX.2 dev")
    session = init(
        root=source_root,
        config_path=runtime_root / "config/wgp_config.json",
        output_dir=Path(requests[0]["output"]).parent,
        cli_args=[
            "--attention",
            ATTENTION_MODE,
            "--profile",
            RUNTIME_PROFILE,
            "--preload",
            str(PRELOAD_MIB),
        ],
        console_output=False,
    )
    schema = session.get_model_schema(MODEL_TYPE)
    if schema is None:
        raise RuntimeError(f"WanGP has no {MODEL_TYPE} model definition")
    try:
        for index, request in enumerate(requests, start=1):
            started = time.monotonic()
            callbacks = _ProgressCallbacks(stream)
            _send(stream, "phase", text=f"FLUX.2 dev job {index}/{len(requests)}")
            settings = dict(schema["default_settings"])
            settings.update(
                model_type=MODEL_TYPE,
                prompt=request["prompt"],
                resolution=f"{request['width']}x{request['height']}",
                image_mode=1,
                video_prompt_type="KI",
                image_refs=request["references"],
                remove_background_images_ref=0,
                num_inference_steps=request["steps"],
                embedded_guidance_scale=request["guidance"],
                batch_size=1,
                repeat_generation=1,
                seed=request["seed"],
                activated_loras=[lora["path"] for lora in request["loras"]],
                loras_multipliers=" ".join(
                    str(lora["weight"]) for lora in request["loras"]
                ),
            )
            try:
                result = session.run_task(settings, callbacks=callbacks)
            except Exception as error:
                _send(
                    stream,
                    "result",
                    response={"status": "error", "message": str(error)},
                )
                return 1
            if not result.success:
                message = "; ".join(error.message for error in result.errors)
                _send(stream, "result", response={"status": "error", "message": message})
                return 1
            images = [
                Path(path).resolve()
                for path in result.generated_files
                if Path(path).suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
            if len(images) != 1:
                raise RuntimeError(f"WanGP returned {len(images)} images; expected exactly one")
            output = Path(request["output"]).resolve()
            images[0].replace(output)
            _send(
                stream,
                "result",
                response={
                    "status": "completed",
                    "output": output.as_posix(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "environment": {
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0),
                        "runtime_profile": int(RUNTIME_PROFILE),
                        "preload_mib": PRELOAD_MIB,
                        "attention": ATTENTION_MODE,
                        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
                    },
                },
            )
    finally:
        session.close()
    return 0


class _ProgressCallbacks:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.phase = ""
        self.total = 0
        self.current = 0

    def on_status(self, text: str) -> None:
        text = text.strip()
        if text and text != self.phase:
            self.phase = text
            self.total = 0
            self.current = 0
            _send(self.stream, "phase", text=text)

    def on_progress(self, update: Any) -> None:
        phase = str(update.phase or "").strip()
        total = int(update.total_steps or 0)
        current = int(update.current_step or 0)
        if total <= 0:
            if phase and phase != self.phase:
                self.on_status(phase)
            return
        if total != self.total or current < self.current:
            self.phase = phase
            self.total = total
            self.current = 0
            _send(self.stream, "begin", total=total, text=phase or "generating FLUX.2 dev image")
        while self.current < current:
            self.current += 1
            _send(self.stream, "step", text=f"{phase or 'inference'} {self.current}/{total}")


def _open_progress_stream() -> TextIO:
    sys.stdout.flush()
    progress_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return os.fdopen(progress_fd, "w", encoding="utf-8", buffering=1)


def _send(stream: TextIO, kind: str, **values: Any) -> None:
    stream.write(json.dumps({"kind": kind, **values}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
