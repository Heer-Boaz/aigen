from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from aigen.manifest_io import atomic_write_json, read_json, write_json_line
from aigen.pix2pix.artifacts import (
    AdversarialCheckpointState,
    CheckpointState,
    L1CheckpointState,
    ResumePosition,
    export_generator_bundle,
    load_training_checkpoint,
    prepare_empty_output_dir,
    save_training_checkpoint,
)
from aigen.pix2pix.config import TrainConfig
from aigen.pix2pix.dataset import AuditedDataset, PairedImage, audit_dataset
from aigen.pix2pix.device import autocast_context, resolve_device, validate_precision
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.evaluation import EvaluationResult, evaluate_generator
from aigen.pix2pix.model import (
    ConditionalPatchDiscriminator,
    Pix2PixGenerator,
    discriminator_loss,
    generator_parameter_report,
    generator_loss,
    initialize_pix2pix_weights,
    model_parameter_report,
    reconstruction_loss,
)
from aigen.pix2pix.training_data import create_training_loader
from aigen.progress import StatusReporter


RUN_FORMAT = "aigen.pix2pix.run.v1"
PYTORCH_REFERENCE_REVISION = "2a7afba2895d52556dd5dfe07e8555ef657ced6f"
PYTORCH_REFERENCE_URL = "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix"
TENSORFLOW_REFERENCE_REVISION = "cf2a57e77485c371f04cc486d9d1e632ef552739"
TENSORFLOW_REFERENCE_URL = (
    "https://github.com/tensorflow/docs/blob/"
    f"{TENSORFLOW_REFERENCE_REVISION}/site/en/tutorials/generative/pix2pix.ipynb"
)


def train_pix2pix(
    dataset_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    resume_checkpoint: Path | None,
    device_name: str,
    progress: StatusReporter,
) -> dict[str, object]:
    progress.phase("audit paired dataset")
    dataset = audit_dataset(dataset_dir)
    config = TrainConfig.load(config_path)
    if dataset.image_size != config.model.image_size:
        raise Pix2PixError("dataset image size does not match the model config")
    device = resolve_device(device_name)
    validate_precision(device, config.precision)
    validate_precision(device, config.parameter_precision)
    optimizer_type = _optimizer_type(config.optimizer, device=device)
    output_dir = output_dir.resolve()
    config_fingerprint = _config_fingerprint(config)
    train_pairs = dataset.split("train")
    validation_pairs = dataset.split("validation")
    if resume_checkpoint is None:
        prepare_empty_output_dir(output_dir)
    _configure_runtime(device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.seed)
        torch.cuda.reset_peak_memory_stats(device)

    parameter_dtype = _parameter_dtype(config.parameter_precision)
    generator = Pix2PixGenerator(
        config.model,
        device=device,
        dtype=parameter_dtype,
    )
    generator.apply(initialize_pix2pix_weights)
    generator_optimizer = optimizer_type(
        generator.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    batch_trainer: Callable[
        [dict[str, Tensor | list[str]]],
        tuple[dict[str, Tensor], int],
    ]
    checkpoint_state: CheckpointState
    optimizers: tuple[torch.optim.Optimizer, ...]
    if config.objective == "adversarial_l1":
        rng_devices = (
            [torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=rng_devices):
            discriminator = ConditionalPatchDiscriminator(
                config.model,
                device=device,
                dtype=parameter_dtype,
            )
            discriminator.apply(initialize_pix2pix_weights)
        discriminator_optimizer = optimizer_type(
            discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )
        batch_trainer = partial(
            _train_adversarial_batch,
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            device=device,
            config=config,
        )
        checkpoint_state = AdversarialCheckpointState(
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
        )
        optimizers = (generator_optimizer, discriminator_optimizer)
        parameter_report = model_parameter_report(generator, discriminator)
    else:
        batch_trainer = partial(
            _train_l1_batch,
            generator=generator,
            generator_optimizer=generator_optimizer,
            device=device,
            config=config,
        )
        checkpoint_state = L1CheckpointState(
            generator_optimizer=generator_optimizer,
        )
        optimizers = (generator_optimizer,)
        parameter_report = generator_parameter_report(generator)
    if resume_checkpoint is None:
        position = ResumePosition(step=0, epoch=0, sample_offset=0)
        run = _new_run_manifest(
            dataset,
            config,
            config_path,
            output_dir,
            config_fingerprint,
            device,
            parameter_report,
        )
        atomic_write_json(output_dir / "run.json", run)
    else:
        position, run = _resume_run(
            output_dir,
            resume_checkpoint,
            dataset=dataset,
            config=config,
            config_fingerprint=config_fingerprint,
            generator=generator,
            checkpoint_state=checkpoint_state,
            device=device,
        )
    if position.step > config.max_steps:
        raise Pix2PixError(
            f"checkpoint step {position.step} exceeds max_steps {config.max_steps}"
        )
    metrics_path = output_dir / "metrics.jsonl"
    if position.step == config.max_steps:
        assert resume_checkpoint is not None
        started = time.monotonic()
        progress.phase(f"validate terminal pix2pix step {position.step}")
        preview_path = (
            output_dir / "previews" / f"step-{position.step:08d}.png"
        )
        last_evaluation = _validate(
            generator,
            validation_pairs,
            config=config,
            device=device,
            preview_path=None if preview_path.exists() else preview_path,
        )
        _record_validation_once(metrics_path, position.step, last_evaluation)
        run["latest_checkpoint"] = resume_checkpoint.resolve().as_posix()
        return _complete_run(
            output_dir,
            run=run,
            generator=generator,
            config=config,
            dataset=dataset,
            config_fingerprint=config_fingerprint,
            step=position.step,
            evaluation=last_evaluation,
            device=device,
            started=started,
            progress=progress,
        )
    if position.sample_offset >= len(train_pairs):
        raise Pix2PixError("checkpoint sample offset is outside the training split")

    loader, sampler = create_training_loader(
        train_pairs,
        image_size=config.model.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        horizontal_flip=config.horizontal_flip,
        pin_memory=device.type == "cuda",
    )
    remaining_steps = config.max_steps - position.step
    progress.begin(remaining_steps, "train pix2pix")
    started = time.monotonic()
    step = position.step
    epoch = position.epoch
    sample_offset = position.sample_offset
    next_epoch = epoch
    next_sample_offset = sample_offset
    last_checkpoint_step = position.step if resume_checkpoint is not None else 0
    last_evaluation_step = 0
    last_evaluation: EvaluationResult | None = None
    extra_checkpoint_steps = frozenset(config.extra_checkpoint_steps)
    with metrics_path.open("a", encoding="utf-8", buffering=1) as metrics:
        while step < config.max_steps:
            sampler.set_position(epoch=epoch, start_offset=sample_offset)
            for batch in loader:
                learning_rate = _learning_rate_for_update(
                    config,
                    completed_steps=step,
                )
                _set_learning_rate(optimizers, learning_rate)
                losses, batch_count = batch_trainer(batch)
                step += 1
                sample_offset += batch_count
                if sample_offset == len(train_pairs):
                    next_epoch, next_sample_offset = epoch + 1, 0
                else:
                    next_epoch, next_sample_offset = epoch, sample_offset
                progress.step(f"train pix2pix step {step}/{config.max_steps}")
                if step == 1 or step % config.log_every == 0:
                    logged_losses = {
                        name: loss.item() for name, loss in losses.items()
                    }
                    write_json_line(
                        metrics,
                        {
                            "kind": "train",
                            "step": step,
                            "epoch": epoch,
                            "learning_rate": learning_rate,
                            **logged_losses,
                        },
                    )
                if (
                    step % config.checkpoint_every == 0
                    or step in extra_checkpoint_steps
                ):
                    progress.phase(f"checkpoint pix2pix step {step}")
                    checkpoint_dir = save_training_checkpoint(
                        output_dir / "checkpoints",
                        step=step,
                        next_epoch=next_epoch,
                        next_sample_offset=next_sample_offset,
                        dataset_fingerprint=dataset.fingerprint,
                        config_fingerprint=config_fingerprint,
                        model_config=config.model,
                        generator=generator,
                        objective=config.objective,
                        state=checkpoint_state,
                        optimizer_name=config.optimizer,
                        parameter_precision=config.parameter_precision,
                        device=device,
                    )
                    last_checkpoint_step = step
                    last_evaluation = _validate(
                        generator,
                        validation_pairs,
                        config=config,
                        device=device,
                        preview_path=output_dir
                        / "previews"
                        / f"step-{step:08d}.png",
                    )
                    last_evaluation_step = step
                    write_json_line(
                        metrics,
                        {
                            "kind": "validation",
                            "step": step,
                            **last_evaluation.to_json(),
                        },
                    )
                    run["step"] = step
                    run["latest_checkpoint"] = checkpoint_dir.as_posix()
                    run["validation"] = last_evaluation.to_json()
                    atomic_write_json(output_dir / "run.json", run)
                if step >= config.max_steps:
                    break
            if step < config.max_steps:
                epoch += 1
                sample_offset = 0

        if last_checkpoint_step != step:
            progress.phase(f"checkpoint final pix2pix step {step}")
            checkpoint_dir = save_training_checkpoint(
                output_dir / "checkpoints",
                step=step,
                next_epoch=next_epoch,
                next_sample_offset=next_sample_offset,
                dataset_fingerprint=dataset.fingerprint,
                config_fingerprint=config_fingerprint,
                model_config=config.model,
                generator=generator,
                objective=config.objective,
                state=checkpoint_state,
                optimizer_name=config.optimizer,
                parameter_precision=config.parameter_precision,
                device=device,
            )
            run["latest_checkpoint"] = checkpoint_dir.as_posix()
        if last_evaluation_step != step:
            last_evaluation = _validate(
                generator,
                validation_pairs,
                config=config,
                device=device,
                preview_path=output_dir / "previews" / f"step-{step:08d}.png",
            )
            write_json_line(
                metrics,
                {
                    "kind": "validation",
                    "step": step,
                    **last_evaluation.to_json(),
                },
            )
    assert last_evaluation is not None
    return _complete_run(
        output_dir,
        run=run,
        generator=generator,
        config=config,
        dataset=dataset,
        config_fingerprint=config_fingerprint,
        step=step,
        evaluation=last_evaluation,
        device=device,
        started=started,
        progress=progress,
    )


def _complete_run(
    output_dir: Path,
    *,
    run: dict[str, Any],
    generator: Pix2PixGenerator,
    config: TrainConfig,
    dataset: AuditedDataset,
    config_fingerprint: str,
    step: int,
    evaluation: EvaluationResult,
    device: torch.device,
    started: float,
    progress: StatusReporter,
) -> dict[str, object]:
    progress.phase("export final pix2pix generator")
    model_metadata = export_generator_bundle(
        output_dir / "final",
        generator=generator,
        model_config=config.model,
        step=step,
        dataset_fingerprint=dataset.fingerprint,
        config_fingerprint=config_fingerprint,
    )
    elapsed_seconds = time.monotonic() - started
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        if device.type == "cuda"
        else 0
    )
    run.update(
        {
            "status": "completed",
            "step": step,
            "completed_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "peak_vram_mb": peak_vram_mb,
            "validation": evaluation.to_json(),
            "final_model": (output_dir / "final").as_posix(),
        }
    )
    atomic_write_json(output_dir / "run.json", run)
    return {
        "status": "completed",
        "output": output_dir.as_posix(),
        "step": step,
        "final_model": (output_dir / "final").as_posix(),
        "model": model_metadata,
        "validation": evaluation.to_json(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "peak_vram_mb": peak_vram_mb,
    }


def _record_validation_once(
    metrics_path: Path,
    step: int,
    evaluation: EvaluationResult,
) -> None:
    expected = {
        "kind": "validation",
        "step": step,
        **evaluation.to_json(),
    }
    matching_records = []
    if metrics_path.exists():
        try:
            lines = metrics_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise Pix2PixError(
                f"cannot read pix2pix metrics: {metrics_path.as_posix()}"
            ) from error
        for line_number, text in enumerate(lines, start=1):
            try:
                record = json.loads(text)
            except json.JSONDecodeError as error:
                raise Pix2PixError(
                    f"invalid pix2pix metrics JSON at line {line_number}: {error}"
                ) from error
            if (
                isinstance(record, dict)
                and record.get("kind") == "validation"
                and record.get("step") == step
            ):
                matching_records.append(record)
    if len(matching_records) > 1:
        raise Pix2PixError(f"duplicate validation metrics for terminal step {step}")
    if matching_records:
        if matching_records[0] != expected:
            raise Pix2PixError(
                f"validation metrics conflict for terminal step {step}"
            )
        return
    try:
        with metrics_path.open("a", encoding="utf-8", buffering=1) as metrics:
            write_json_line(metrics, expected)
    except OSError as error:
        raise Pix2PixError(
            f"cannot append pix2pix metrics: {metrics_path.as_posix()}"
        ) from error


def _train_adversarial_batch(
    batch: dict[str, Tensor | list[str]],
    *,
    generator: Pix2PixGenerator,
    discriminator: ConditionalPatchDiscriminator,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainConfig,
) -> tuple[dict[str, Tensor], int]:
    source = _batch_tensor(batch, "source").to(device, non_blocking=True)
    target = _batch_tensor(batch, "target").to(device, non_blocking=True)
    discriminator_optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, config.precision):
        generated = generator(source)
        real_logits = discriminator(source, target)
        fake_logits = discriminator(source, generated.detach())
        d_loss = discriminator_loss(real_logits, fake_logits)
    d_loss.backward()
    discriminator_optimizer.step()
    discriminator_optimizer.zero_grad(set_to_none=True)

    discriminator.requires_grad_(False)
    generator_optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, config.precision):
        fake_logits = discriminator(source, generated)
        g_total, g_adversarial, g_l1 = generator_loss(
            fake_logits,
            generated,
            target,
            lambda_l1=config.lambda_l1,
        )
    g_total.backward()
    generator_optimizer.step()
    discriminator.requires_grad_(True)
    return (
        {
            "generator_total": g_total.detach(),
            "generator_adversarial": g_adversarial.detach(),
            "generator_l1": g_l1.detach(),
            "discriminator": d_loss.detach(),
        },
        source.shape[0],
    )


def _train_l1_batch(
    batch: dict[str, Tensor | list[str]],
    *,
    generator: Pix2PixGenerator,
    generator_optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainConfig,
) -> tuple[dict[str, Tensor], int]:
    source = _batch_tensor(batch, "source").to(device, non_blocking=True)
    target = _batch_tensor(batch, "target").to(device, non_blocking=True)
    generator_optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, config.precision):
        generated = generator(source)
        total, reconstruction = reconstruction_loss(
            generated,
            target,
            lambda_l1=config.lambda_l1,
        )
    total.backward()
    generator_optimizer.step()
    return (
        {
            "generator_total": total.detach(),
            "generator_l1": reconstruction.detach(),
        },
        source.shape[0],
    )


def _validate(
    generator: Pix2PixGenerator,
    validation_pairs: tuple[PairedImage, ...],
    *,
    config: TrainConfig,
    device: torch.device,
    preview_path: Path | None,
) -> EvaluationResult:
    return evaluate_generator(
        generator,
        validation_pairs,
        image_size=config.model.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        precision=config.precision,
        preview_path=preview_path,
    )


def _new_run_manifest(
    dataset: AuditedDataset,
    config: TrainConfig,
    config_path: Path,
    output_dir: Path,
    config_fingerprint: str,
    device: torch.device,
    parameter_report: dict[str, int],
) -> dict[str, Any]:
    runtime: dict[str, object] = {
        "torch": torch.__version__,
        "device": str(device),
        "cudnn_benchmark": device.type == "cuda",
        "float32_matmul_precision": "high",
    }
    if device.type == "cuda":
        runtime["gpu"] = torch.cuda.get_device_name(device)
        runtime["tf32"] = True
    return {
        "format": RUN_FORMAT,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "step": 0,
        "output": output_dir.as_posix(),
        "dataset": dataset.to_json(),
        "config_path": config_path.resolve().as_posix(),
        "config_fingerprint": config_fingerprint,
        "config": config.to_json(),
        "parameters": parameter_report,
        "runtime": runtime,
        "references": {
            "tensorflow_tutorial": {
                "url": TENSORFLOW_REFERENCE_URL,
                "revision": TENSORFLOW_REFERENCE_REVISION,
            },
            "pytorch_pix2pix": {
                "url": PYTORCH_REFERENCE_URL,
                "revision": PYTORCH_REFERENCE_REVISION,
            },
        },
    }


def _resume_run(
    output_dir: Path,
    checkpoint_dir: Path,
    *,
    dataset: AuditedDataset,
    config: TrainConfig,
    config_fingerprint: str,
    generator: Pix2PixGenerator,
    checkpoint_state: CheckpointState,
    device: torch.device,
) -> tuple[ResumePosition, dict[str, Any]]:
    run = read_json(output_dir / "run.json", label="pix2pix run metadata")
    if not isinstance(run, dict):
        raise Pix2PixError("pix2pix run metadata must be a JSON object")
    if run.get("format") != RUN_FORMAT:
        raise Pix2PixError(f"unsupported pix2pix run format: {run.get('format')!r}")
    if run.get("status") == "completed":
        raise Pix2PixError(f"pix2pix run is already completed: {output_dir.as_posix()}")
    if run.get("status") != "running":
        raise Pix2PixError(f"invalid pix2pix run status: {run.get('status')!r}")
    run_dataset = run.get("dataset")
    if not isinstance(run_dataset, dict) or run_dataset.get("fingerprint") != dataset.fingerprint:
        raise Pix2PixError("run dataset fingerprint does not match the audited dataset")
    if run.get("config_fingerprint") != config_fingerprint:
        raise Pix2PixError("run training config does not match the requested config")
    checkpoint_dir = checkpoint_dir.resolve()
    if checkpoint_dir.parent != (output_dir / "checkpoints").resolve():
        raise Pix2PixError("resume checkpoint must belong to the requested output run")
    position = load_training_checkpoint(
        checkpoint_dir,
        expected_dataset_fingerprint=dataset.fingerprint,
        expected_config_fingerprint=config_fingerprint,
        model_config=config.model,
        generator=generator,
        expected_objective=config.objective,
        state=checkpoint_state,
        expected_optimizer_name=config.optimizer,
        expected_parameter_precision=config.parameter_precision,
        device=device,
    )
    return position, run


def _config_fingerprint(config: TrainConfig) -> str:
    encoded = json.dumps(
        config.to_json(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _configure_runtime(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _optimizer_type(
    name: str,
    *,
    device: torch.device,
) -> type[torch.optim.Optimizer]:
    if name == "adam":
        return torch.optim.Adam
    if device.type != "cuda":
        raise Pix2PixError("paged_adam8bit requires a CUDA device")
    try:
        from bitsandbytes.optim import PagedAdam8bit
    except ImportError as error:
        raise Pix2PixError(
            "paged_adam8bit requires the bitsandbytes package"
        ) from error
    return PagedAdam8bit


def _parameter_dtype(precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    raise AssertionError(f"unsupported parameter precision: {precision}")


def _learning_rate_for_update(
    config: TrainConfig,
    *,
    completed_steps: int,
) -> float:
    assert 0 <= completed_steps < config.max_steps
    schedule = config.learning_rate_schedule
    if schedule.type == "constant":
        return config.learning_rate
    assert schedule.type == "linear_decay"
    assert schedule.decay_start_step is not None
    if completed_steps <= schedule.decay_start_step:
        return config.learning_rate
    remaining = config.max_steps - completed_steps
    decay_steps = config.max_steps - schedule.decay_start_step
    return config.learning_rate * remaining / decay_steps


def _set_learning_rate(
    optimizers: tuple[torch.optim.Optimizer, ...],
    learning_rate: float,
) -> None:
    for optimizer in optimizers:
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate


def _batch_tensor(batch: dict[str, Tensor | list[str]], key: str) -> Tensor:
    value = batch[key]
    assert isinstance(value, Tensor)
    return value
