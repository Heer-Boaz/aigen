"""Reviewed real VLM check on saved negative and positive edit examples."""
from contextlib import closing
import json
from pathlib import Path
import sys
import time

from aigen.character_edit_audit import (
    AuditContextImage, CharacterAuditGroup, audit_character_candidates,
    character_audit_prompt, default_character_audit_config,
)
from aigen.generation.image_edit_batch import ImageEditBatchOutput
from aigen.gpu_status import nvidia_smi_memory_snapshot
from aigen.manifest_io import atomic_write_json, sha256_file
from aigen.vlm_qwen import QwenVlm, qwen_vlm_config_json

ROOT = Path(__file__).resolve().parent


def group_for(case):
    return CharacterAuditGroup(case["name"], case["instruction"], case["route"],
        tuple(Path(path) for path in case["references"]),
        tuple(AuditContextImage(item["role"], Path(item["path"])) for item in case["contexts"]),
        tuple(item["case_id"] for item in case["candidates"]))


def prepare():
    original = json.loads((ROOT / "manifest.json").read_text())
    result = json.loads((ROOT / "tui-v1/refine-acceptance.json").read_text())
    audit_path = Path(result["result"]["details"]["measured_outputs"]["image"]["audit_report"])
    negative = json.loads(audit_path.with_name("request.json").read_text())
    previous = ROOT.parent / "character-flow-acceptance"
    positive = json.loads((previous / "audit-v2.json").read_text())
    positive_generation = json.loads((previous / "manifest.json").read_text())
    cases = [
        {"name": "missing-background-change", "instruction": original["edit"]["prompt"],
         "route": "local_repair_or_inpaint", "references": [negative["images"][0]["path"]],
         "contexts": [{"role": "repaint mask: white is editable; black is preserved", "path": negative["images"][1]["path"]}],
         "candidates": negative["candidates"], "images": negative["images"], "expected": {"passed": False}},
        {"name": "completed-front-view", "instruction": positive_generation["prompt"],
         "route": "unknown_reference_edit", "references": [positive["images"][0]["path"]], "contexts": [],
         "candidates": positive["candidates"], "images": positive["images"], "expected": {"passed": True, "seed": 91}},
    ]
    for case in cases:
        case["prompt"] = character_audit_prompt(group_for(case), len(case["candidates"]))
    atomic_write_json(ROOT / "audit-v3.json", {
        "audit": qwen_vlm_config_json(default_character_audit_config()), "min_free_mib": 13000,
        "cases": cases,
    })


def run():
    import torch

    request = json.loads((ROOT / "audit-v3.json").read_text())
    assert request["review_status"] == "approved"
    config = default_character_audit_config()
    assert qwen_vlm_config_json(config) == request["audit"]
    for case in request["cases"]:
        assert character_audit_prompt(group_for(case), len(case["candidates"])) == case["prompt"]
        for item in case["images"]:
            assert sha256_file(Path(item["path"])) == item["sha256"]
    gpu = nvidia_smi_memory_snapshot()
    assert gpu["nvidia_smi_device_total_mb"] - gpu["nvidia_smi_used_mb"] >= request["min_free_mib"], gpu
    directory = ROOT / "audit-v3-probe"
    directory.mkdir()
    atomic_write_json(directory / "preflight.json", gpu)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    records = []
    with closing(QwenVlm(config)) as judge:
        for case in request["cases"]:
            outputs = tuple(ImageEditBatchOutput.model_validate(item) for item in case["candidates"])
            result = audit_character_candidates(judge, group_for(case), outputs, directory / case["name"])
            expected = case["expected"]
            record = {"name": case["name"], "verdict": result.verdict.model_dump(mode="json"),
                      "seed": result.selected.seed, "report": str(result.report),
                      "matches_review": result.verdict.passed == expected["passed"] and
                          ("seed" not in expected or result.selected.seed == expected["seed"])}
            records.append(record)
            print(json.dumps(record), flush=True)
        peak = torch.cuda.max_memory_allocated() / 1024**2
    atomic_write_json(directory / "result.json", {
        "elapsed_seconds": time.monotonic() - started, "peak_allocated_mib": peak,
        "allocated_after_close_mib": torch.cuda.memory_allocated() / 1024**2,
        "reserved_after_close_mib": torch.cuda.memory_reserved() / 1024**2, "cases": records,
    })
    assert all(record["matches_review"] for record in records), records


if __name__ == "__main__":
    {"prepare": prepare, "run": run}[sys.argv[1]]()
