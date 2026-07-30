from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from aigen.manifest_io import atomic_write_json, read_json, sha256_file
from aigen.pix2pix.config import MODEL_FORMAT, TRAIN_OBJECTIVES, ModelConfig
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.model import (
    Pix2PixGenerator,
    generator_architecture_name,
)


CHECKPOINT_FORMAT_V2 = "aigen.pix2pix.checkpoint.v2"
CHECKPOINT_FORMAT = "aigen.pix2pix.checkpoint.v3"
GENERATOR_BUNDLE_FILES = frozenset({"generator.safetensors", "model.json"})
GENERATOR_NORMALIZATION = {
    "input": "RGB uint8 mapped linearly to [-1, 1]",
    "output": "tanh RGB mapped linearly from [-1, 1] to uint8",
}


@dataclass(frozen=True)
class ResumePosition:
    step: int
    epoch: int
    sample_offset: int


@dataclass(frozen=True)
class AdversarialCheckpointState:
    discriminator: nn.Module
    generator_optimizer: torch.optim.Optimizer
    discriminator_optimizer: torch.optim.Optimizer


@dataclass(frozen=True)
class L1CheckpointState:
    generator_optimizer: torch.optim.Optimizer


CheckpointState = AdversarialCheckpointState | L1CheckpointState


def save_training_checkpoint(
    checkpoints_dir: Path,
    *,
    step: int,
    next_epoch: int,
    next_sample_offset: int,
    dataset_fingerprint: str,
    config_fingerprint: str,
    model_config: ModelConfig,
    generator: nn.Module,
    objective: str,
    state: CheckpointState,
    optimizer_name: str,
    parameter_precision: str,
    device: torch.device,
) -> Path:
    _require_checkpoint_state(objective, state)
    checkpoint_dir = checkpoints_dir / f"step-{step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    generator_path = checkpoint_dir / "generator.safetensors"
    training_state_path = checkpoint_dir / "training-state.pt"
    _atomic_save_weights(generator, generator_path)
    training_state = {
        "generator_optimizer": state.generator_optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
    }
    files = {"generator": _file_record(generator_path)}
    if isinstance(state, AdversarialCheckpointState):
        discriminator_path = checkpoint_dir / "discriminator.safetensors"
        _atomic_save_weights(state.discriminator, discriminator_path)
        training_state["discriminator_optimizer"] = (
            state.discriminator_optimizer.state_dict()
        )
        files["discriminator"] = _file_record(discriminator_path)
    if device.type == "cuda":
        training_state["device_rng_state"] = torch.cuda.get_rng_state(device)
    _atomic_torch_save(training_state, training_state_path)
    files["training_state"] = _file_record(training_state_path)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "objective": objective,
        "step": step,
        "next_epoch": next_epoch,
        "next_sample_offset": next_sample_offset,
        "device_type": device.type,
        "optimizer": optimizer_name,
        "parameter_precision": parameter_precision,
        "dataset_fingerprint": dataset_fingerprint,
        "config_fingerprint": config_fingerprint,
        "model": model_config.to_json(),
        "files": files,
    }
    atomic_write_json(checkpoint_dir / "checkpoint.json", metadata)
    return checkpoint_dir


def load_training_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_dataset_fingerprint: str,
    expected_config_fingerprint: str,
    model_config: ModelConfig,
    generator: Pix2PixGenerator,
    expected_objective: str,
    state: CheckpointState,
    expected_optimizer_name: str,
    expected_parameter_precision: str,
    device: torch.device,
) -> ResumePosition:
    checkpoint_dir = checkpoint_dir.resolve()
    metadata = read_json(
        checkpoint_dir / "checkpoint.json",
        label="pix2pix checkpoint metadata",
    )
    if not isinstance(metadata, dict):
        raise Pix2PixError("checkpoint metadata must be a JSON object")
    objective = _checkpoint_objective(metadata)
    if objective != expected_objective:
        raise Pix2PixError(
            f"checkpoint objective {objective!r} does not match "
            f"{expected_objective!r}"
        )
    _require_checkpoint_state(expected_objective, state)
    if metadata["dataset_fingerprint"] != expected_dataset_fingerprint:
        raise Pix2PixError("checkpoint dataset fingerprint does not match the audited dataset")
    if metadata["config_fingerprint"] != expected_config_fingerprint:
        raise Pix2PixError("checkpoint training config does not match the requested config")
    if metadata["device_type"] != device.type:
        raise Pix2PixError(
            f"checkpoint device {metadata['device_type']!r} does not match {device.type!r}"
        )
    if metadata["optimizer"] != expected_optimizer_name:
        raise Pix2PixError("checkpoint optimizer does not match the training config")
    if metadata["parameter_precision"] != expected_parameter_precision:
        raise Pix2PixError(
            "checkpoint parameter precision does not match the training config"
        )
    checkpoint_model = ModelConfig.from_json(metadata["model"])
    if checkpoint_model != model_config:
        raise Pix2PixError("checkpoint model architecture does not match the training config")
    files = metadata["files"]
    assert isinstance(files, dict)
    generator_path = _verified_file(checkpoint_dir, files["generator"], "generator")
    training_state_path = _verified_file(
        checkpoint_dir,
        files["training_state"],
        "training state",
    )
    generator_weights = load_file(generator_path, device=str(device))
    _require_state_precision(
        generator_weights,
        expected_parameter_precision,
        "generator checkpoint",
    )
    generator.load_state_dict(generator_weights, strict=True)
    if isinstance(state, AdversarialCheckpointState):
        discriminator_path = _verified_file(
            checkpoint_dir,
            files["discriminator"],
            "discriminator",
        )
        discriminator_weights = load_file(
            discriminator_path,
            device=str(device),
        )
        _require_state_precision(
            discriminator_weights,
            expected_parameter_precision,
            "discriminator checkpoint",
        )
        state.discriminator.load_state_dict(discriminator_weights, strict=True)
    training_state = torch.load(
        training_state_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(training_state, dict):
        raise Pix2PixError("checkpoint training state must be an object")
    expected_state_keys = {
        "generator_optimizer",
        "cpu_rng_state",
    }
    if isinstance(state, AdversarialCheckpointState):
        expected_state_keys.add("discriminator_optimizer")
    if device.type == "cuda":
        expected_state_keys.add("device_rng_state")
    _require_exact_keys(
        training_state,
        expected_state_keys,
        "checkpoint training state",
    )
    _load_optimizer_state(
        state.generator_optimizer,
        training_state["generator_optimizer"],
        optimizer_name=expected_optimizer_name,
    )
    if isinstance(state, AdversarialCheckpointState):
        _load_optimizer_state(
            state.discriminator_optimizer,
            training_state["discriminator_optimizer"],
            optimizer_name=expected_optimizer_name,
        )
    torch.set_rng_state(training_state["cpu_rng_state"].cpu())
    if device.type == "cuda":
        device_rng_state = training_state.get("device_rng_state")
        if not isinstance(device_rng_state, torch.Tensor):
            raise Pix2PixError("CUDA checkpoint is missing its device RNG state")
        torch.cuda.set_rng_state(device_rng_state.cpu(), device)
    return ResumePosition(
        step=_positive_integer(metadata, "step"),
        epoch=_nonnegative_integer(metadata, "next_epoch"),
        sample_offset=_nonnegative_integer(metadata, "next_sample_offset"),
    )


def export_generator_bundle(
    output_dir: Path,
    *,
    generator: nn.Module,
    model_config: ModelConfig,
    step: int,
    dataset_fingerprint: str,
    config_fingerprint: str,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        return _verify_reusable_generator_bundle(
            output_dir,
            generator=generator,
            model_config=model_config,
            step=step,
            dataset_fingerprint=dataset_fingerprint,
            config_fingerprint=config_fingerprint,
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.",
        suffix=".incomplete",
    ) as temporary:
        staging_dir = Path(temporary)
        weights_path = staging_dir / "generator.safetensors"
        _atomic_save_weights(generator, weights_path)
        metadata = _generator_metadata(
            weights_path,
            model_config=model_config,
            step=step,
            dataset_fingerprint=dataset_fingerprint,
            config_fingerprint=config_fingerprint,
        )
        atomic_write_json(staging_dir / "model.json", metadata)
        _load_generator_metadata(staging_dir)
        os.rename(staging_dir, output_dir)
    return metadata


def load_generator_bundle(
    model_dir: Path,
    *,
    device: torch.device,
) -> tuple[Pix2PixGenerator, dict[str, Any]]:
    model_dir = model_dir.resolve()
    metadata, config, weights_path = _load_generator_metadata(model_dir)
    weights = load_file(weights_path, device=str(device))
    parameter_precision = _state_precision(weights, "generator bundle")
    generator = Pix2PixGenerator(
        config,
        device="meta",
        dtype=_precision_dtype(parameter_precision),
    )
    generator.load_state_dict(weights, strict=True, assign=True)
    generator.eval().requires_grad_(False)
    return generator, metadata


def _generator_metadata(
    weights_path: Path,
    *,
    model_config: ModelConfig,
    step: int,
    dataset_fingerprint: str,
    config_fingerprint: str,
) -> dict[str, Any]:
    return {
        "format": MODEL_FORMAT,
        "architecture": generator_architecture_name(model_config),
        "step": step,
        "dataset_fingerprint": dataset_fingerprint,
        "config_fingerprint": config_fingerprint,
        "model": model_config.to_json(),
        "normalization": GENERATOR_NORMALIZATION,
        "weights": _file_record(weights_path),
    }


def _load_generator_metadata(
    model_dir: Path,
) -> tuple[dict[str, Any], ModelConfig, Path]:
    metadata = read_json(model_dir / "model.json", label="pix2pix generator metadata")
    if not isinstance(metadata, dict):
        raise Pix2PixError("generator metadata must be a JSON object")
    expected = {
        "format",
        "architecture",
        "step",
        "dataset_fingerprint",
        "config_fingerprint",
        "model",
        "normalization",
        "weights",
    }
    _require_exact_keys(metadata, expected, "generator metadata")
    if metadata["format"] != MODEL_FORMAT:
        raise Pix2PixError(f"unsupported generator format: {metadata['format']!r}")
    config = ModelConfig.from_json(metadata["model"])
    expected_architecture = generator_architecture_name(config)
    if metadata["architecture"] != expected_architecture:
        raise Pix2PixError(
            "generator architecture does not match its model config: "
            f"{metadata['architecture']!r}"
        )
    if metadata["normalization"] != GENERATOR_NORMALIZATION:
        raise Pix2PixError("generator normalization contract does not match pix2pix v1")
    weights_path = _verified_file(model_dir, metadata["weights"], "generator weights")
    return metadata, config, weights_path


def _verify_reusable_generator_bundle(
    model_dir: Path,
    *,
    generator: nn.Module,
    model_config: ModelConfig,
    step: int,
    dataset_fingerprint: str,
    config_fingerprint: str,
) -> dict[str, Any]:
    if not model_dir.is_dir():
        raise Pix2PixError(
            f"generator bundle path is not a directory: {model_dir.as_posix()}"
        )
    files = {path.name for path in model_dir.iterdir()}
    if files != GENERATOR_BUNDLE_FILES:
        raise Pix2PixError(
            f"existing generator bundle is incomplete or has unexpected files: "
            f"{model_dir.as_posix()}"
        )
    metadata, existing_config, weights_path = _load_generator_metadata(model_dir)
    if (
        existing_config != model_config
        or metadata["step"] != step
        or metadata["dataset_fingerprint"] != dataset_fingerprint
        or metadata["config_fingerprint"] != config_fingerprint
    ):
        raise Pix2PixError(
            f"existing generator bundle does not belong to this training run: "
            f"{model_dir.as_posix()}"
        )
    weights = load_file(weights_path, device="cpu")
    generator_state = generator.state_dict()
    if set(weights) != set(generator_state):
        raise Pix2PixError(
            f"existing generator weights do not match the terminal checkpoint: "
            f"{model_dir.as_posix()}"
        )
    for name, expected in generator_state.items():
        if not torch.equal(weights[name], expected.detach().to(device="cpu")):
            raise Pix2PixError(
                f"existing generator weights do not match the terminal checkpoint: "
                f"{model_dir.as_posix()}"
            )
    return metadata


def prepare_empty_output_dir(path: Path) -> None:
    path = path.resolve()
    if path.exists():
        if not path.is_dir():
            raise Pix2PixError(f"output path is not a directory: {path.as_posix()}")
        if any(path.iterdir()):
            raise Pix2PixError(f"output directory is not empty: {path.as_posix()}")
    path.mkdir(parents=True, exist_ok=True)


def _atomic_save_weights(module: nn.Module, path: Path) -> None:
    state = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in module.state_dict().items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(path)
    try:
        save_file(state, temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(path)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _temporary_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


def _file_record(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": sha256_file(path)}


def _verified_file(root: Path, payload: object, label: str) -> Path:
    if not isinstance(payload, dict):
        raise Pix2PixError(f"{label} record must be an object")
    _require_exact_keys(payload, {"path", "sha256"}, f"{label} record")
    relative = payload["path"]
    expected_hash = payload["sha256"]
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise Pix2PixError(f"{label} record has invalid values")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Pix2PixError(f"{label} path escapes its model directory") from error
    if not path.is_file():
        raise Pix2PixError(f"missing {label}: {path.as_posix()}")
    if sha256_file(path) != expected_hash:
        raise Pix2PixError(f"{label} hash does not match metadata: {path.as_posix()}")
    return path


def _checkpoint_objective(metadata: dict[str, Any]) -> str:
    expected = {
        "format",
        "step",
        "next_epoch",
        "next_sample_offset",
        "device_type",
        "optimizer",
        "parameter_precision",
        "dataset_fingerprint",
        "config_fingerprint",
        "model",
        "files",
    }
    checkpoint_format = metadata.get("format")
    if checkpoint_format == CHECKPOINT_FORMAT:
        expected.add("objective")
        objective = metadata.get("objective")
        if objective not in TRAIN_OBJECTIVES:
            raise Pix2PixError(f"unsupported checkpoint objective: {objective!r}")
        assert isinstance(objective, str)
    elif checkpoint_format == CHECKPOINT_FORMAT_V2:
        objective = "adversarial_l1"
    else:
        raise Pix2PixError(
            f"unsupported checkpoint format: {checkpoint_format!r}"
        )
    _require_exact_keys(metadata, expected, "checkpoint metadata")
    files = metadata["files"]
    if not isinstance(files, dict):
        raise Pix2PixError("checkpoint files must be an object")
    expected_files = {"generator", "training_state"}
    if objective == "adversarial_l1":
        expected_files.add("discriminator")
    _require_exact_keys(
        files,
        expected_files,
        "checkpoint files",
    )
    return objective


def _require_checkpoint_state(
    objective: str,
    state: CheckpointState,
) -> None:
    if objective == "adversarial_l1":
        if isinstance(state, AdversarialCheckpointState):
            return
        raise Pix2PixError(
            "adversarial_l1 requires an adversarial checkpoint state"
        )
    if objective == "l1_only":
        if isinstance(state, L1CheckpointState):
            return
        raise Pix2PixError("l1_only requires an L1 checkpoint state")
    raise Pix2PixError(f"unsupported checkpoint objective: {objective!r}")


def _load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    payload: object,
    *,
    optimizer_name: str,
) -> None:
    if not isinstance(payload, dict):
        raise Pix2PixError("checkpoint optimizer state must be an object")
    if optimizer_name == "adam":
        optimizer.load_state_dict(payload)
        return
    from bitsandbytes.optim import PagedAdam8bit

    if not isinstance(optimizer, PagedAdam8bit):
        raise Pix2PixError("checkpoint requires a paged_adam8bit optimizer")
    _load_paged_adam8bit_state(optimizer, payload)


def _load_paged_adam8bit_state(
    optimizer: torch.optim.Optimizer,
    payload: dict[str, Any],
) -> None:
    _require_exact_keys(payload, {"state", "param_groups"}, "optimizer state")
    saved_state = payload["state"]
    saved_groups = payload["param_groups"]
    if not isinstance(saved_state, dict) or not isinstance(saved_groups, list):
        raise Pix2PixError("invalid paged_adam8bit checkpoint state")
    if len(saved_groups) != len(optimizer.param_groups):
        raise Pix2PixError("paged_adam8bit parameter-group count does not match")

    for group_index, (group, saved_group) in enumerate(
        zip(optimizer.param_groups, saved_groups, strict=True)
    ):
        if not isinstance(saved_group, dict):
            raise Pix2PixError("invalid paged_adam8bit parameter group")
        saved_parameters = saved_group.get("params")
        parameters = group["params"]
        if not isinstance(saved_parameters, list) or len(saved_parameters) != len(
            parameters
        ):
            raise Pix2PixError("paged_adam8bit parameter count does not match")
        for key, value in saved_group.items():
            if key != "params":
                group[key] = value

        for parameter_index, (parameter, saved_id) in enumerate(
            zip(parameters, saved_parameters, strict=True)
        ):
            source = saved_state.get(saved_id)
            if not isinstance(source, dict):
                raise Pix2PixError("paged_adam8bit parameter state is missing")
            quantized = source.get("__bnb_optimizer_quant_state__")
            if not isinstance(quantized, dict):
                raise Pix2PixError("paged_adam8bit quantized state is missing")
            source_values = {
                key: value
                for key, value in source.items()
                if key != "__bnb_optimizer_quant_state__"
            }
            source_values.update(quantized)
            optimizer.state[parameter] = {}
            optimizer.init_state(
                group,
                parameter,
                group_index,
                parameter_index,
            )
            target = optimizer.state[parameter]
            if set(source_values) != set(target):
                raise Pix2PixError("paged_adam8bit state fields do not match")
            for key, source_value in source_values.items():
                target_value = target[key]
                if isinstance(target_value, torch.Tensor):
                    if (
                        not isinstance(source_value, torch.Tensor)
                        or source_value.shape != target_value.shape
                        or source_value.dtype != target_value.dtype
                    ):
                        raise Pix2PixError(
                            f"paged_adam8bit tensor {key} does not match"
                        )
                    target_value.copy_(source_value)
                elif isinstance(source_value, torch.Tensor):
                    raise Pix2PixError(
                        f"paged_adam8bit scalar {key} does not match"
                    )
                else:
                    target[key] = source_value


def _require_state_precision(
    state: dict[str, torch.Tensor],
    expected: str,
    label: str,
) -> None:
    actual = _state_precision(state, label)
    if actual != expected:
        raise Pix2PixError(
            f"{label} parameter precision {actual} does not match {expected}"
        )


def _state_precision(
    state: dict[str, torch.Tensor],
    label: str,
) -> str:
    dtypes = {
        tensor.dtype
        for tensor in state.values()
        if tensor.is_floating_point()
    }
    if len(dtypes) != 1:
        raise Pix2PixError(f"{label} has inconsistent floating-point dtypes")
    dtype = next(iter(dtypes))
    if dtype == torch.float32:
        return "fp32"
    if dtype == torch.bfloat16:
        return "bf16"
    raise Pix2PixError(f"{label} uses unsupported parameter dtype {dtype}")


def _precision_dtype(precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    raise AssertionError(f"unsupported parameter precision: {precision}")


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    keys = set(payload)
    if keys != expected:
        missing = sorted(expected - keys)
        unexpected = sorted(keys - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        raise Pix2PixError(f"invalid {label}: {'; '.join(details)}")


def _positive_integer(payload: dict[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Pix2PixError(f"{key} must be a positive integer")
    return value


def _nonnegative_integer(payload: dict[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Pix2PixError(f"{key} must be a non-negative integer")
    return value
