from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.gpu_status import GpuStatusError, nvidia_smi_memory_snapshot
from aigen.keyframe_memory import NvidiaSmiMemorySampler
from aigen.manifest_io import read_json, write_json, write_json_line
from aigen.progress import StatusReporter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINER_SCRIPT = PROJECT_ROOT / "tools" / "diffusers" / "train_dreambooth_lora_flux.py"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "aigen" / "models" / "diffusers" / "black-forest-labs" / "FLUX.1-dev-bnb-4bit"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "lora"
LOCAL_PROFILE_NAME = "flux-lora-local-16gb"
TRAIN_CAPTION_COLUMN = "prompt"
TRAINER_SOURCE_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/v0.38.0/"
    "examples/dreambooth/train_dreambooth_lora_flux.py"
)


class LoraTrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoraLocalTrainConfig:
    resolution: int = 512
    rank: int = 4
    lora_alpha: int = 4
    max_train_steps: int = 800
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    max_sequence_length: int = 128
    checkpointing_steps: int = 200
    checkpoints_total_limit: int = 2
    seed: int = 1
    mixed_precision: str = "bf16"


def lora_training_preflight(dataset_dir: Path) -> dict[str, Any]:
    plan = build_lora_train_plan(dataset_dir)
    return {
        "status": plan["status"],
        "profile": plan["profile"],
        "dataset": plan["dataset"],
        "local_gpu": plan["local_gpu"],
        "missing": plan["missing"],
        "memory_strategy": plan["memory_strategy"],
        "command": plan["command"],
    }


def build_lora_train_plan(
    dataset_dir: Path,
    *,
    output_dir: Path | None = None,
    trainer_script: Path = DEFAULT_TRAINER_SCRIPT,
    base_model: Path = DEFAULT_BASE_MODEL,
    config: LoraLocalTrainConfig = LoraLocalTrainConfig(),
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    report = _dataset_report(dataset_dir)
    dataset_id = str(report["dataset_id"])
    split_counts = report["split_counts"]
    train_count = split_counts["train"]
    validation_count = split_counts["val"]
    output_dir = (output_dir or DEFAULT_OUTPUT_ROOT / _slug(dataset_id)).resolve()
    source_train_dir = dataset_dir / "images" / "train"
    train_data_dir = output_dir / "train_dataset"
    required_instance_prompt = _required_instance_prompt(report)
    command = _train_command(
        trainer_script=trainer_script.resolve(),
        base_model=base_model.resolve(),
        train_data_dir=train_data_dir,
        output_dir=output_dir,
        required_instance_prompt=required_instance_prompt,
        config=config,
    )
    missing = _missing_inputs(
        dataset_dir=dataset_dir,
        source_train_dir=source_train_dir,
        trainer_script=trainer_script,
        base_model=base_model,
        train_count=train_count,
    )
    return {
        "status": "ready_to_launch" if not missing else "missing_local_inputs",
        "profile": LOCAL_PROFILE_NAME,
        "trainer_source": {
            "kind": "official_diffusers_example",
            "url": TRAINER_SOURCE_URL,
            "diffusers_version": "0.38.0",
            "path": trainer_script.resolve().as_posix(),
        },
        "dataset": {
            "directory": dataset_dir.as_posix(),
            "id": dataset_id,
            "source_train_dir": source_train_dir.as_posix(),
            "train_data_dir": train_data_dir.as_posix(),
            "caption_column": TRAIN_CAPTION_COLUMN,
            "accepted_image_count": report["accepted_image_count"],
            "train_image_count": train_count,
            "validation_image_count": validation_count,
        },
        "trainer": {
            "required_instance_prompt": required_instance_prompt,
        },
        "model": {
            "base_model": base_model.resolve().as_posix(),
            "base_model_kind": "local_bnb_4bit_flux_pipeline",
        },
        "output": {"directory": output_dir.as_posix(), "plan": (output_dir / "train_plan.json").as_posix()},
        "local_gpu": _nvidia_smi_snapshot(),
        "profile_parameters": {
            "resolution": config.resolution,
            "rank": config.rank,
            "lora_alpha": config.lora_alpha,
            "max_train_steps": config.max_train_steps,
            "train_batch_size": config.train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "max_sequence_length": config.max_sequence_length,
            "checkpointing_steps": config.checkpointing_steps,
            "checkpoints_total_limit": config.checkpoints_total_limit,
            "seed": config.seed,
            "mixed_precision": config.mixed_precision,
        },
        "memory_strategy": [
            "local 4-bit FLUX base model",
            "BF16 mixed precision avoids fp16 GradScaler unscale failures with local 4-bit FLUX training",
            "train transformer LoRA only; text encoders are frozen",
            "captioned Hugging Face imagefolder dataset materialized from the train split",
            "per-image prompts are read from metadata.jsonl column prompt",
            "batch size 1 with gradient accumulation",
            "rank 4 LoRA on attention projections only",
            "gradient checkpointing",
            "8-bit Adam optimizer",
            "VAE latent caching",
            "512px training resolution for the first measured local profile",
        ],
        "missing": missing,
        "command": command,
    }


def run_lora_training_plan(
    dataset_dir: Path,
    *,
    output_dir: Path | None = None,
    trainer_script: Path = DEFAULT_TRAINER_SCRIPT,
    base_model: Path = DEFAULT_BASE_MODEL,
    config: LoraLocalTrainConfig = LoraLocalTrainConfig(),
    dry_run: bool = False,
    progress: StatusReporter,
) -> dict[str, Any]:
    progress.phase("build training plan")
    plan = build_lora_train_plan(
        dataset_dir,
        output_dir=output_dir,
        trainer_script=trainer_script,
        base_model=base_model,
        config=config,
    )
    progress.step("write training plan")
    plan_path = Path(plan["output"]["plan"])
    write_json(plan_path, plan)
    if plan["missing"]:
        raise LoraTrainingError(f"LoRA training inputs are missing: {', '.join(plan['missing'])}")
    if dry_run:
        return plan | {"run_status": "planned"}
    progress.step("materialize captioned train dataset")
    materialize_captioned_train_dataset(Path(plan["dataset"]["directory"]), Path(plan["dataset"]["train_data_dir"]))
    log_path = Path(plan["output"]["directory"]) / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    memory_sampler = NvidiaSmiMemorySampler(_training_memory_preflight(plan["local_gpu"]))
    memory_sampler.start()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            progress.step("train LoRA")
            completed = subprocess.run(
                plan["command"],
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        finally:
            memory = memory_sampler.stop()
    progress.step("write training result")
    result = plan | {
        "run_status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "log": log_path.as_posix(),
        "memory": memory,
    }
    write_json(Path(plan["output"]["directory"]) / "train_result.json", result)
    if completed.returncode != 0:
        raise LoraTrainingError(f"LoRA training failed with exit code {completed.returncode}; see {log_path.as_posix()}")
    return result


def _train_command(
    *,
    trainer_script: Path,
    base_model: Path,
    train_data_dir: Path,
    output_dir: Path,
    required_instance_prompt: str,
    config: LoraLocalTrainConfig,
) -> list[str]:
    return [
        (PROJECT_ROOT / ".venv" / "bin" / "accelerate").as_posix(),
        "launch",
        "--mixed_precision",
        config.mixed_precision,
        trainer_script.as_posix(),
        "--pretrained_model_name_or_path",
        base_model.as_posix(),
        "--dataset_name",
        train_data_dir.as_posix(),
        "--caption_column",
        TRAIN_CAPTION_COLUMN,
        "--instance_prompt",
        required_instance_prompt,
        "--output_dir",
        output_dir.as_posix(),
        "--resolution",
        str(config.resolution),
        "--center_crop",
        "--train_batch_size",
        str(config.train_batch_size),
        "--gradient_accumulation_steps",
        str(config.gradient_accumulation_steps),
        "--rank",
        str(config.rank),
        "--lora_alpha",
        str(config.lora_alpha),
        "--lora_layers",
        "to_q,to_k,to_v,to_out.0",
        "--learning_rate",
        _float_arg(config.learning_rate),
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "0",
        "--max_train_steps",
        str(config.max_train_steps),
        "--checkpointing_steps",
        str(config.checkpointing_steps),
        "--checkpoints_total_limit",
        str(config.checkpoints_total_limit),
        "--max_sequence_length",
        str(config.max_sequence_length),
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--cache_latents",
        "--allow_tf32",
        "--dataloader_num_workers",
        "0",
        "--seed",
        str(config.seed),
    ]


def _dataset_report(dataset_dir: Path) -> dict[str, Any]:
    report = read_json(dataset_dir / "dataset_report.json", label="LoRA dataset report")
    if report["status"] != "completed":
        raise LoraTrainingError(f"LoRA dataset is not completed: {dataset_dir.as_posix()}")
    if report["accepted_image_count"] <= 0:
        raise LoraTrainingError(f"LoRA dataset has no accepted images: {dataset_dir.as_posix()}")
    return report


def _required_instance_prompt(report: dict[str, Any]) -> str:
    return report["character"]["trigger_token"]


def _missing_inputs(
    *,
    dataset_dir: Path,
    source_train_dir: Path,
    trainer_script: Path,
    base_model: Path,
    train_count: int,
) -> list[str]:
    missing = []
    if not dataset_dir.exists():
        missing.append(dataset_dir.as_posix())
    if not trainer_script.exists():
        missing.append(trainer_script.as_posix())
    if not base_model.exists():
        missing.append(base_model.as_posix())
    if not source_train_dir.exists() or train_count <= 0:
        missing.append(source_train_dir.as_posix())
    return missing


def materialize_captioned_train_dataset(dataset_dir: Path, target_dir: Path) -> None:
    report = _dataset_report(dataset_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    metadata_path = target_dir / "metadata.jsonl"
    train_count = 0
    with metadata_path.open("w", encoding="utf-8") as metadata:
        for record in report["records"]:
            if record["split"] != "train":
                continue
            prompt = record["prompt"]
            if not prompt:
                raise LoraTrainingError(f"LoRA train record has no prompt: {record['file_name']}")
            source = dataset_dir / record["file_name"]
            target_name = source.name
            shutil.copy2(source, target_dir / target_name)
            write_json_line(metadata, {"file_name": target_name, TRAIN_CAPTION_COLUMN: prompt})
            train_count += 1
    if train_count <= 0:
        raise LoraTrainingError(f"LoRA dataset has no train records: {dataset_dir.as_posix()}")


def _nvidia_smi_snapshot() -> dict[str, int]:
    try:
        return nvidia_smi_memory_snapshot()
    except GpuStatusError as error:
        raise LoraTrainingError("LoRA training requires nvidia-smi VRAM telemetry") from error


def _training_memory_preflight(snapshot: dict[str, int]) -> dict[str, int]:
    return {
        "nvidia_smi_preflight_used_mb": snapshot["nvidia_smi_used_mb"],
        "nvidia_smi_device_total_mb": snapshot["nvidia_smi_device_total_mb"],
        "nvidia_smi_preflight_utilization_gpu": snapshot["nvidia_smi_utilization_gpu"],
    }


def _float_arg(value: float) -> str:
    return f"{value:g}"


def _slug(value: str) -> str:
    chars = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    return "".join(chars).strip("-") or "lora"
