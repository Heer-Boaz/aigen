from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.generation.flux_components import FLUX_TEXT_COMPONENTS_DISABLED
from aigen.generation.flux_prompt_encoding import encode_flux_prompts
from aigen.gpu_status import GpuStatusError, nvidia_smi_memory_snapshot
from aigen.image_assets import image_asset_json
from aigen.keyframe_image_ops import save_contact_sheet
from aigen.manifest_io import file_manifest
from aigen.manifest_io import resolve_existing_path, write_json
from aigen.generation.runtime_diagnostics import elapsed_ms, synchronized_time
from aigen.generation.runtime_types import resolve_torch_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLUX_BASE_MODEL = PROJECT_ROOT / "aigen" / "models" / "diffusers" / "black-forest-labs" / "FLUX.1-dev-bf16"
DEFAULT_CONTROLNET_MODEL = (
    PROJECT_ROOT / "aigen" / "models" / "diffusers" / "Shakker-Labs" / "FLUX.1-dev-ControlNet-Union-Pro-2.0"
)
DEFAULT_NUNCHAKU_FLUX_TRANSFORMER = (
    PROJECT_ROOT
    / "aigen"
    / "models"
    / "nunchaku"
    / "nunchaku-ai"
    / "nunchaku-flux.1-dev"
    / "svdq-fp4_r32-flux.1-dev.safetensors"
)
DEFAULT_LORA_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"


class LoraControlAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoraControlAuditConfig:
    width: int = 512
    height: int = 768
    steps: int = 20
    max_sequence_length: int = 128
    guidance_scale: float = 2.5
    controlnet_conditioning_scale: float = 0.8
    control_guidance_end: float = 0.65
    lora_strength: float = 1.0
    seed: int = 1


@dataclass(frozen=True)
class LoraControlAuditCase:
    name: str
    control_image: Path
    baseline_image: Path
    prompt: str


def build_lora_control_audit_plan(
    lora_run_dir: Path,
    *,
    case_specs: list[str],
    baseline_specs: list[str],
    case_prompt_specs: list[str],
    output_dir: Path | None = None,
    lora_weights: Path | None = None,
    base_model: Path = DEFAULT_FLUX_BASE_MODEL,
    controlnet_model: Path = DEFAULT_CONTROLNET_MODEL,
    nunchaku_transformer: Path = DEFAULT_NUNCHAKU_FLUX_TRANSFORMER,
    trigger_token: str | None = None,
    config: LoraControlAuditConfig = LoraControlAuditConfig(),
) -> dict[str, Any]:
    resolved_run_dir = lora_run_dir.resolve()
    resolved_output_dir = (output_dir or resolved_run_dir / "control_audit").resolve()
    resolved_lora = (lora_weights or resolved_run_dir / DEFAULT_LORA_WEIGHTS_NAME).resolve()
    resolved_base = base_model.resolve()
    resolved_controlnet = controlnet_model.resolve()
    resolved_nunchaku = nunchaku_transformer.resolve()
    missing = _missing_inputs(
        lora_run_dir=resolved_run_dir,
        lora_weights=resolved_lora,
        base_model=resolved_base,
        controlnet_model=resolved_controlnet,
        nunchaku_transformer=resolved_nunchaku,
    )
    resolved_trigger = _trigger_token(trigger_token, resolved_run_dir)
    cases = _parse_cases(case_specs, baseline_specs, case_prompt_specs, Path.cwd(), trigger_token=resolved_trigger)
    return {
        "status": "ready_to_launch" if not missing else "missing_local_inputs",
        "kind": "lora-control-audit-plan",
        "purpose": "verify trained character LoRA with plain FLUX ControlNet and no Kontext reference",
        "lora_run_dir": resolved_run_dir.as_posix(),
        "output": {
            "directory": resolved_output_dir.as_posix(),
            "plan": (resolved_output_dir / "audit_plan.json").as_posix(),
        },
        "models": _model_manifests(
            lora_weights=resolved_lora,
            base_model=resolved_base,
            controlnet_model=resolved_controlnet,
            nunchaku_transformer=resolved_nunchaku,
        ),
        "missing": missing,
        "local_gpu": _gpu_snapshot(),
        "runtime": {
            "pipeline": "diffusers.FluxControlNetPipeline",
            "transformer": "nunchaku.NunchakuFluxTransformer2dModel",
            "controlnet": "diffusers.FluxControlNetModel",
            "reference_tokens": 0,
            "prompt_encoding": "precomputed_prompt_embeds",
            "uses_kontext_reference": False,
            "lora_loading": "NunchakuFluxTransformer2dModel.update_lora_params",
        },
        "trigger_token": resolved_trigger,
        "audit_cases": [_case_json(case) for case in cases],
        "generation": {
            "width": config.width,
            "height": config.height,
            "steps": config.steps,
            "max_sequence_length": config.max_sequence_length,
            "guidance_scale": config.guidance_scale,
            "controlnet_conditioning_scale": config.controlnet_conditioning_scale,
            "control_guidance_end": config.control_guidance_end,
            "lora_strength": config.lora_strength,
            "seed": config.seed,
        },
    }


def run_lora_control_audit(
    lora_run_dir: Path,
    *,
    case_specs: list[str],
    baseline_specs: list[str],
    case_prompt_specs: list[str],
    output_dir: Path | None = None,
    lora_weights: Path | None = None,
    base_model: Path = DEFAULT_FLUX_BASE_MODEL,
    controlnet_model: Path = DEFAULT_CONTROLNET_MODEL,
    nunchaku_transformer: Path = DEFAULT_NUNCHAKU_FLUX_TRANSFORMER,
    trigger_token: str | None = None,
    config: LoraControlAuditConfig = LoraControlAuditConfig(),
    progress: Any,
) -> dict[str, Any]:
    progress.phase("plan LoRA control audit")
    plan = build_lora_control_audit_plan(
        lora_run_dir,
        case_specs=case_specs,
        baseline_specs=baseline_specs,
        case_prompt_specs=case_prompt_specs,
        output_dir=output_dir,
        lora_weights=lora_weights,
        base_model=base_model,
        controlnet_model=controlnet_model,
        nunchaku_transformer=nunchaku_transformer,
        trigger_token=trigger_token,
        config=config,
    )
    output = Path(plan["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(Path(plan["output"]["plan"]), plan)
    if plan["missing"]:
        raise LoraControlAuditError(f"LoRA control audit inputs are missing: {', '.join(plan['missing'])}")

    progress.phase("encode LoRA audit prompts")
    prompt_embeddings, encode_ms = _encode_prompts(plan)
    progress.phase("load LoRA control audit pipeline")
    pipeline, torch, load_ms = _load_pipeline(plan)
    outputs = []
    total_start = synchronized_time(torch)
    progress.begin(len(plan["audit_cases"]), "LoRA control audit")
    for index, case in enumerate(plan["audit_cases"], start=1):
        progress.phase(f"audit {case['name']} ({index}/{len(plan['audit_cases'])})")
        image, timing = _generate_case(pipeline, torch, plan, case, prompt_embeddings[case["name"]])
        image_path = output / f"{case['name']}.png"
        image.save(image_path)
        outputs.append(
            {
                "name": case["name"],
                "path": image_path.as_posix(),
                "seed": plan["generation"]["seed"],
                "timings_ms": timing,
            }
        )
        progress.step(f"audited {case['name']}")
    contact_sheet = output / "contact_sheet.png"
    lora_contact_sheet = output / "lora_contact_sheet.png"
    save_contact_sheet(outputs, lora_contact_sheet, thumb_width=256, label_x=8)
    save_contact_sheet(_comparison_outputs(plan, outputs), contact_sheet, thumb_width=256, label_x=8)
    memory = _torch_memory(torch)
    result = {
        "status": "completed",
        "kind": "lora-control-audit-result",
        "plan": plan,
        "outputs": outputs,
        "baseline_outputs": _baseline_outputs(plan),
        "output": {
            "directory": output.as_posix(),
            "contact_sheet": contact_sheet.as_posix(),
            "lora_contact_sheet": lora_contact_sheet.as_posix(),
            "result": (output / "result.json").as_posix(),
        },
        "timings_ms": {
            "prompt_encode_ms": encode_ms,
            "model_load_ms": load_ms,
            "total_ms": elapsed_ms(total_start, synchronized_time(torch)),
        },
        "memory": memory,
    }
    write_json(output / "result.json", result)
    return result


def _missing_inputs(
    *,
    lora_run_dir: Path,
    lora_weights: Path,
    base_model: Path,
    controlnet_model: Path,
    nunchaku_transformer: Path,
) -> list[str]:
    missing = []
    for path in (lora_run_dir, lora_weights, base_model, controlnet_model, nunchaku_transformer):
        if not path.exists():
            missing.append(path.as_posix())
    return missing


def _parse_cases(
    case_specs: list[str],
    baseline_specs: list[str],
    case_prompt_specs: list[str],
    base_dir: Path,
    *,
    trigger_token: str,
) -> list[LoraControlAuditCase]:
    if not case_specs:
        raise LoraControlAuditError("LoRA control audit requires at least one --case NAME=CONTROL_IMAGE")
    baselines = _parse_named_images(
        baseline_specs,
        base_dir,
        missing_message="LoRA control audit requires one --baseline NAME=BASELINE_IMAGE for each --case",
        label="baseline",
    )
    prompts = _parse_case_prompts(case_prompt_specs, trigger_token=trigger_token)
    cases = []
    seen = set()
    for spec in case_specs:
        if "=" not in spec:
            raise LoraControlAuditError(f"Audit case must use NAME=CONTROL_IMAGE: {spec}")
        name, raw_path = spec.split("=", 1)
        case_name = _case_name(name)
        if case_name in seen:
            raise LoraControlAuditError(f"Duplicate audit case: {case_name}")
        seen.add(case_name)
        if case_name not in prompts:
            raise LoraControlAuditError(f"Missing --case-prompt for case: {case_name}")
        if case_name not in baselines:
            raise LoraControlAuditError(f"Missing --baseline for case: {case_name}")
        control_image = resolve_existing_path(raw_path.strip(), base_dir)
        cases.append(
            LoraControlAuditCase(
                name=case_name,
                control_image=control_image,
                baseline_image=baselines[case_name],
                prompt=prompts[case_name],
            )
        )
    extra_prompts = sorted(set(prompts) - seen)
    if extra_prompts:
        raise LoraControlAuditError(f"--case-prompt has no matching --case: {', '.join(extra_prompts)}")
    extra_baselines = sorted(set(baselines) - seen)
    if extra_baselines:
        raise LoraControlAuditError(f"--baseline has no matching --case: {', '.join(extra_baselines)}")
    return cases


def _parse_named_images(
    specs: list[str],
    base_dir: Path,
    *,
    missing_message: str,
    label: str,
) -> dict[str, Path]:
    if not specs:
        raise LoraControlAuditError(missing_message)
    images = {}
    for spec in specs:
        if "=" not in spec:
            raise LoraControlAuditError(f"{label} image must use NAME=IMAGE: {spec}")
        raw_name, raw_path = spec.split("=", 1)
        name = _case_name(raw_name)
        if name in images:
            raise LoraControlAuditError(f"Duplicate {label} image: {name}")
        images[name] = resolve_existing_path(raw_path.strip(), base_dir)
    return images


def _parse_case_prompts(case_prompt_specs: list[str], *, trigger_token: str) -> dict[str, str]:
    if not case_prompt_specs:
        raise LoraControlAuditError(
            "LoRA control audit requires one complete --case-prompt NAME=PROMPT for each --case"
        )
    prompts = {}
    for spec in case_prompt_specs:
        if "=" not in spec:
            raise LoraControlAuditError(f"Case prompt must use NAME=PROMPT: {spec}")
        raw_name, raw_prompt = spec.split("=", 1)
        name = _case_name(raw_name)
        if name in prompts:
            raise LoraControlAuditError(f"Duplicate case prompt: {name}")
        prompt = " ".join(raw_prompt.strip().split())
        if not prompt:
            raise LoraControlAuditError(f"Case prompt is empty: {name}")
        if trigger_token not in prompt:
            raise LoraControlAuditError(f"Case prompt must include the LoRA trigger token: {name}")
        prompts[name] = prompt
    return prompts


def _case_name(value: str) -> str:
    cleaned = "_".join(value.strip().replace("-", "_").split())
    if not cleaned:
        raise LoraControlAuditError("Audit case name is empty")
    return cleaned


def _model_manifests(
    *,
    lora_weights: Path,
    base_model: Path,
    controlnet_model: Path,
    nunchaku_transformer: Path,
) -> dict[str, dict[str, Any]]:
    manifests = {
        "base_model": {"path": base_model.as_posix()},
        "controlnet": {"path": controlnet_model.as_posix()},
        "nunchaku_transformer": {"path": nunchaku_transformer.as_posix()},
        "lora_weights": {"path": lora_weights.as_posix()},
    }
    if lora_weights.is_file():
        manifests["lora_weights"].update(file_manifest(lora_weights))
    if nunchaku_transformer.is_file():
        manifests["nunchaku_transformer"]["size_bytes"] = nunchaku_transformer.stat().st_size
    return manifests


def _case_json(case: LoraControlAuditCase) -> dict[str, Any]:
    return {
        "name": case.name,
        "prompt": case.prompt,
        "control_image": image_asset_json(case.control_image),
        "baseline_image": image_asset_json(case.baseline_image),
    }


def _trigger_token(explicit_token: str | None, lora_run_dir: Path) -> str:
    if explicit_token:
        return explicit_token
    report_path = lora_run_dir / "dataset" / "dataset_report.json"
    if not report_path.exists():
        raise LoraControlAuditError("--trigger-token is required when the LoRA run has no dataset report")
    from aigen.manifest_io import read_json

    report = read_json(report_path, label="LoRA dataset report")
    return str(report["character"]["trigger_token"])


def _gpu_snapshot() -> dict[str, int]:
    try:
        return nvidia_smi_memory_snapshot()
    except GpuStatusError as error:
        raise LoraControlAuditError("LoRA control audit requires nvidia-smi VRAM telemetry") from error


def _load_pipeline(plan: dict[str, Any]) -> tuple[Any, Any, float]:
    import torch
    from diffusers import FluxControlNetModel, FluxControlNetPipeline
    from nunchaku import NunchakuFluxTransformer2dModel

    dtype = resolve_torch_dtype(torch, "bfloat16", auto_value=None)
    start = synchronized_time(torch)
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        plan["models"]["nunchaku_transformer"]["path"],
        torch_dtype=dtype,
    )
    transformer.set_attention_impl("nunchaku-fp16")
    transformer.update_lora_params(plan["models"]["lora_weights"]["path"])
    transformer.set_lora_strength(plan["generation"]["lora_strength"])
    controlnet = FluxControlNetModel.from_pretrained(
        plan["models"]["controlnet"]["path"],
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipeline = FluxControlNetPipeline.from_pretrained(
        plan["models"]["base_model"]["path"],
        transformer=transformer,
        controlnet=controlnet,
        **FLUX_TEXT_COMPONENTS_DISABLED,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, torch, elapsed_ms(start, synchronized_time(torch))


def _generate_case(
    pipeline: Any,
    torch: Any,
    plan: dict[str, Any],
    case: dict[str, Any],
    prompt_embeddings: dict[str, Any],
) -> tuple[Any, dict[str, float]]:
    from PIL import Image

    with Image.open(case["control_image"]["path"]) as control_image:
        control = control_image.convert("RGB")
    device = pipeline._execution_device
    prompt_embeds = prompt_embeddings["prompt_embeds"].to(device=device)
    pooled_prompt_embeds = prompt_embeddings["pooled_prompt_embeds"].to(device=device)
    generator = torch.Generator(device="cuda").manual_seed(plan["generation"]["seed"])
    start = synchronized_time(torch)
    result = pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        width=plan["generation"]["width"],
        height=plan["generation"]["height"],
        control_image=control,
        control_mode=None,
        controlnet_conditioning_scale=plan["generation"]["controlnet_conditioning_scale"],
        control_guidance_start=0.0,
        control_guidance_end=plan["generation"]["control_guidance_end"],
        num_inference_steps=plan["generation"]["steps"],
        guidance_scale=plan["generation"]["guidance_scale"],
        true_cfg_scale=1.0,
        generator=generator,
        max_sequence_length=plan["generation"]["max_sequence_length"],
    )
    return result.images[0], {"total_ms": elapsed_ms(start, synchronized_time(torch))}


def _encode_prompts(plan: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], float]:
    prompt_embeddings, elapsed = encode_flux_prompts(
        plan["models"]["base_model"]["path"],
        prompts=[case["prompt"] for case in plan["audit_cases"]],
        dtype="bfloat16",
        max_sequence_length=plan["generation"]["max_sequence_length"],
    )
    embeddings = {}
    for case in plan["audit_cases"]:
        prompt_embedding = prompt_embeddings[case["prompt"]]
        embeddings[case["name"]] = {
            "prompt": prompt_embedding.prompt,
            "prompt_embeds": prompt_embedding.prompt_embeds,
            "pooled_prompt_embeds": prompt_embedding.pooled_prompt_embeds,
        }
    return embeddings, elapsed


def _baseline_outputs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": case["name"],
            "path": case["baseline_image"]["path"],
        }
        for case in plan["audit_cases"]
    ]


def _comparison_outputs(plan: dict[str, Any], outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_by_name = {output["name"]: output for output in outputs}
    comparison = []
    for case in plan["audit_cases"]:
        name = case["name"]
        comparison.append({"name": f"baseline/{name}", "path": case["baseline_image"]["path"]})
        comparison.append({"name": f"lora/{name}", "path": output_by_name[name]["path"]})
    return comparison


def _torch_memory(torch: Any) -> dict[str, int]:
    if not torch.cuda.is_available():
        raise LoraControlAuditError("LoRA control audit requires CUDA")
    return {
        "torch_max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "torch_max_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
    }
