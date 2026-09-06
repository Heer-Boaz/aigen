"""Real GPU acceptance probe. Run one named job per process; never patches neural code."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

import psutil

from aigen.gpu_status import nvidia_smi_memory_snapshot
from aigen.image_assets import image_asset_json
from aigen.manifest_io import atomic_write_json, file_manifest
from aigen.progress import RuntimeStatus
from aigen.system_telemetry import SystemTelemetrySampler


def preflight(min_free_mib: int) -> dict:
    snapshot = nvidia_smi_memory_snapshot()
    free = snapshot["nvidia_smi_device_total_mb"] - snapshot["nvidia_smi_used_mb"]
    print(json.dumps({"gpu_preflight": snapshot, "free_mib": free}), flush=True)
    if free < min_free_mib:
        raise RuntimeError(f"GPU occupied: {free} MiB free; this probe needs {min_free_mib} MiB")
    return snapshot


def host_memory() -> dict:
    swap = psutil.swap_memory()
    return {
        "process_rss_mib": psutil.Process().memory_info().rss / 1024**2,
        "host_available_mib": psutil.virtual_memory().available / 1024**2,
        "swap_used_mib": swap.used / 1024**2,
        "swap_in_bytes": swap.sin,
        "swap_out_bytes": swap.sout,
    }


def run_klein(config: dict, job: dict, directory: Path, progress: RuntimeStatus) -> dict:
    from aigen.generation.flux2_klein import Flux2KleinSession, encode_flux2_klein_prompts
    from aigen.generation.flux2_klein_artifacts import (
        flux2_klein_model_artifacts, flux2_klein_runtime_provenance,
    )
    from aigen.generation.image_generation_requests import ImageGenerationCaseRequest, ImageGenerationOutputRequest
    from aigen.model_artifacts import model_artifact_stat_revision

    request = config["klein"]
    prompt = request["prompt"]
    refs = tuple(Path(path) for path in request["references"])
    atomic_write_json(directory / "provenance.json", {
        "inputs": [image_asset_json(path) for path in refs],
        "runtime": flux2_klein_runtime_provenance(),
        "model_stat_inventory": [{"name": component.name, "root": str(component.root),
                                  "revision": model_artifact_stat_revision(component)}
                                 for component in flux2_klein_model_artifacts()],
    })
    cases = tuple(ImageGenerationCaseRequest(
        name=f"{width}x{height}", prompt=prompt, image_paths=refs, width=width, height=height,
        outputs=tuple(ImageGenerationOutputRequest(
            name=f"{width}x{height}-seed{seed}", seed=seed,
            path=directory / f"{width}x{height}-seed{seed}.png",
        ) for seed in request["seeds"]),
    ) for width, height in request["canvases"])
    preflight(job["min_free_mib"])
    embeddings, conditioning_ms = encode_flux2_klein_prompts(prompts=(prompt,), progress=progress)
    preflight(job["min_free_mib"])
    session = Flux2KleinSession(loras=(), sampler=request["sampler"], strength=job["strength"], progress=progress)
    forward_records = []

    def capture_inputs(module, args, kwargs):
        # Observe real transformer inputs without altering tensors or inference.
        record = {"hidden_states": list(kwargs["hidden_states"].shape),
                  "img_ids": list(kwargs["img_ids"].shape)}
        if not forward_records or forward_records[-1]["hidden_states"] != record["hidden_states"]:
            ids = kwargs["img_ids"].detach().cpu()
            groups, counts = ids[..., 0].unique(return_counts=True)
            record["reference_groups"] = dict(zip(map(str, groups.tolist()), counts.tolist()))
            forward_records.append(record)

    handle = session.pipeline.transformer.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    try:
        preflight(job["min_free_mib"])
        result = session.generate(cases=cases, prompt_embeddings=embeddings, progress=progress)
    finally:
        handle.remove()
        session.close()
        atomic_write_json(directory / "transformer-inputs.json", {"records": forward_records})
    return {**result.to_json(), "conditioning_ms": conditioning_ms,
            "images": [image_asset_json(Path(output.output)) for output in result.outputs]}


def run_vosr(config: dict, job: dict, directory: Path, progress: RuntimeStatus) -> dict:
    from aigen.workflow_compilation import compile_workflow_run
    from aigen.workflow_execution import execute_workflow
    from aigen.workflow_graph import (
        ImageSourceNode, ImageSourceConfig, ImagePostprocessNode, NodePortRef,
        WorkflowConnection, WorkflowGraph,
    )
    from aigen.workflow_templates import default_postprocess_config

    request = config["vosr"]
    nodes, wires = [], []
    settings = default_postprocess_config().model_copy(update=request["settings"])
    for name in job["order"]:
        nodes.extend((
            ImageSourceNode(id=f"src-{name}", title=name, config=ImageSourceConfig(path=request["inputs"][name])),
            ImagePostprocessNode(id=name, title=name, config=settings),
        ))
        wires.append(WorkflowConnection(
            id=f"wire-{name}", source=NodePortRef(node_id=f"src-{name}", port="image"),
            target=NodePortRef(node_id=name, port="image"),
        ))
    graph = WorkflowGraph(name="VOSR GPU acceptance", nodes=nodes, connections=wires)
    atomic_write_json(directory / "graph.json", graph.model_dump(mode="json"))
    preflight(job["min_free_mib"])
    with (directory / "workflow-events.jsonl").open("w", buffering=1) as events:
        result = execute_workflow(
            compile_workflow_run(graph), runs_root=directory.parent / job["cache_root"], progress=progress,
            event_sink=lambda event: events.write(json.dumps(event) + "\n"),
        )
    outputs = {}
    for name in job["order"]:
        manifest = json.loads(result.node_manifests[name].read_text())
        outputs[name] = {"image": image_asset_json(Path(result.terminal_outputs[name]["image"].path)),
                         "signature": manifest["signature"], "node_manifest": str(result.node_manifests[name])}
    return {**result.to_json(), "comparisons": outputs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("job")
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    job = config["jobs"][args.job]
    directory = args.manifest.resolve().parent / args.job
    directory.mkdir()  # A failed probe is retained; retries get a new job name.
    atomic_write_json(directory / "request.json", {"configuration": config, "job": args.job,
        "probe": file_manifest(Path(__file__).resolve()),
        "started_utc": datetime.now(timezone.utc).isoformat()})
    initial_gpu = preflight(job["min_free_mib"])
    import torch

    torch.cuda.reset_peak_memory_stats()
    samples = []
    started = time.perf_counter()
    output = {"status": "failed", "initial_gpu": initial_gpu}
    with (directory / "telemetry.jsonl").open("w", buffering=1) as telemetry:
        def record(payload):
            sample = {**payload, **host_memory()}
            if not samples or samples[-1]["phase"] != sample["phase"]:
                print(json.dumps(sample), flush=True)
            samples.append(sample)
            telemetry.write(json.dumps(sample) + "\n")

        try:
            with RuntimeStatus.callback(interval_seconds=1, callback=record, telemetry=SystemTelemetrySampler()) as progress:
                from application_probes import run_anime_tui, run_image_edit, run_ltx, run_upscale
                from segmentation_probe import run_segmentation
                runner = {"klein": run_klein, "vosr": run_vosr, "image-edit": run_image_edit,
                          "upscale": run_upscale, "ltx": run_ltx, "anime-tui": run_anime_tui,
                          "segmentation": run_segmentation}[job["backend"]]
                output["result"] = runner(config, job, directory, progress)
            output["status"] = "completed"
        except BaseException as error:
            output["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            output.update({
                "elapsed_seconds": time.perf_counter() - started,
                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "nvidia_smi_peak_used_mib": max(s["vram_used_mb"] for s in samples if s["vram_used_mb"] is not None),
                "minimum_host_available_mib": min(s["host_available_mib"] for s in samples),
                "swap_increase_mib": max(s["swap_used_mib"] for s in samples) - samples[0]["swap_used_mib"],
                "swap_in_bytes": samples[-1]["swap_in_bytes"] - samples[0]["swap_in_bytes"],
                "swap_out_bytes": samples[-1]["swap_out_bytes"] - samples[0]["swap_out_bytes"],
            })
            atomic_write_json(directory / "result.json", output)
            print(json.dumps({"status": output["status"], "result": str(directory / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
