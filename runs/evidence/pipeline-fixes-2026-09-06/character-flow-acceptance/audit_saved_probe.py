"""Real audit-only protocol acceptance on the explicitly reviewed saved raws."""
from contextlib import closing
from pathlib import Path
import time

import torch

from aigen.character_edit_audit import CharacterAuditGroup, audit_character_candidates, default_character_audit_config
from aigen.generation.image_edit_batch import ImageEditBatchOutput
from aigen.gpu_status import nvidia_smi_memory_snapshot
from aigen.manifest_io import atomic_write_json, read_json, sha256_file
from aigen.vlm_qwen import QwenVlm, qwen_vlm_config_json

root = Path(__file__).resolve().parent
request = read_json(root / "audit-v2.json", label="reviewed audit inputs")
original = read_json(root / "manifest.json", label="generation manifest")
config = default_character_audit_config()
assert qwen_vlm_config_json(config) == request["audit"]
for item in request["images"]:
    assert sha256_file(Path(item["path"])) == item["sha256"]
gpu = nvidia_smi_memory_snapshot()
assert gpu["nvidia_smi_device_total_mb"] - gpu["nvidia_smi_used_mb"] >= original["min_free_mib"], gpu
outputs = tuple(ImageEditBatchOutput.model_validate(item) for item in request["candidates"])
group = CharacterAuditGroup("edit", original["prompt"], "unknown_reference_edit",
    (Path(original["source"]["path"]),), (), tuple(item.case_id for item in outputs))
directory = root / "audit-v2-probe"
directory.mkdir()
atomic_write_json(directory / "preflight.json", gpu)
torch.cuda.reset_peak_memory_stats()
started = time.monotonic()
with closing(QwenVlm(config)) as judge:
    result = audit_character_candidates(judge, group, outputs, directory / "audit")
    peak = torch.cuda.max_memory_allocated() / 1024**2
atomic_write_json(directory / "result.json", {
    "elapsed_seconds": time.monotonic() - started, "peak_allocated_mib": peak,
    "allocated_after_close_mib": torch.cuda.memory_allocated() / 1024**2,
    "reserved_after_close_mib": torch.cuda.memory_reserved() / 1024**2,
    "verdict": result.verdict.model_dump(mode="json"), "selected_seed": result.selected.seed,
    "candidate_identity": result.candidate_identity, "report": str(result.report),
})
print((directory / "result.json").read_text(), flush=True)
