"""Read-only runtime/source inventory and FFprobe validation of existing videos."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from aigen.cli import build_parser
from aigen.generation.image_edit import IMAGE_EDIT_BACKENDS
from aigen.generation.image_batch_postprocess import image_batch_postprocess_model_names
from aigen.video_tui_model import VIDEO_BACKENDS
from aigen.workflow_graph import ArtifactType, NodeKind

EVIDENCE = Path(__file__).resolve().parent


def command_inventory(parser):
    result = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                result[name] = command_inventory(child)
    return result


def probe(path):
    command = ("ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,codec_type", "-of", "json", str(path))
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    config = path.with_name(path.stem + "_config.json")
    record = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "returncode": result.returncode, "streams": json.loads(result.stdout).get("streams", []) if result.returncode == 0 else [], "stderr": result.stderr.strip(), "config_exists": config.exists()}
    if config.exists():
        data = json.loads(config.read_text())
        record["request"] = data.get("request", data.get("arguments", {}))
        record["runtime"] = data.get("runtime")
        record["environment"] = data.get("environment", {})
    return record


def main():
    paths = {
        "WanGP": Path.home() / ".cache/aigen-wangp/Wan2GP",
        "LightX2V": Path.home() / ".cache/aigen-lightx2v/LightX2V",
        "VOSR": Path.home() / ".cache/aigen-vosr/VOSR",
        "USO": Path.home() / ".local/share/aigen/runtimes/uso/USO",
        "Boogu": Path.home() / ".cache/aigen-boogu/Boogu-Image",
        "HiDream-Comfy": Path.home() / ".cache/aigen-comfy-image/ComfyUI",
        "Hunyuan": Path.home() / ".cache/aigen-hunyuanvideo15/HunyuanVideo-1.5",
    }
    runtimes = {}
    for name, path in paths.items():
        record = {"default_source_path": str(path), "exists": path.exists()}
        if path.exists():
            for label, args in (("revision", ("rev-parse", "HEAD")), ("changes", ("status", "--short"))):
                result = subprocess.run(("git", "-C", str(path), *args), capture_output=True, text=True)
                record[label] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
        runtimes[name] = record
    results = {
        "checked_at": datetime.now(UTC).isoformat(),
        "head": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
        "image_backends": IMAGE_EDIT_BACKENDS,
        "video_backends": VIDEO_BACKENDS,
        "postprocessors": image_batch_postprocess_model_names(),
        "node_kinds": [x.value for x in NodeKind],
        "artifact_types": [x.value for x in ArtifactType],
        "cli": command_inventory(build_parser()),
        "runtime_sources": runtimes,
    }
    (EVIDENCE / "inventory.json").write_text(json.dumps(results, indent=2) + "\n")
    videos = {name: sorted((ROOT / "runs" / name).rglob("*.mp4")) for name in ("animegen", "ltx23", "hunyuanvideo15")}
    evidence = {}
    with ThreadPoolExecutor(4) as pool:
        for name, paths in videos.items():
            records = list(pool.map(probe, paths))
            evidence[name] = {"files": len(records), "ffprobe_errors": sum(r["returncode"] != 0 for r in records), "records": records}
            print(json.dumps({"backend": name, "files": len(records), "ffprobe_errors": evidence[name]["ffprobe_errors"]}), flush=True)
    (EVIDENCE / "existing-video-results.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
