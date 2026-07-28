from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from aigen.pix2pix import training as pix2pix_training
from aigen.pix2pix.artifacts import (
    export_generator_bundle,
    load_training_checkpoint,
)
from aigen.pix2pix.config import ModelConfig, TrainConfig
from aigen.pix2pix.dataset import DATASET_FORMAT
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.inference import run_inference
from aigen.pix2pix.model import ConditionalPatchDiscriminator, Pix2PixGenerator
from aigen.pix2pix.training import train_pix2pix
from aigen.progress import SILENT_STATUS


class Pix2PixPipelineTests(unittest.TestCase):
    def test_one_step_cpu_training_exports_an_inference_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            config = root / "train.json"
            output = root / "run"
            _write_dataset(dataset)
            _write_config(config)

            result = train_pix2pix(
                dataset,
                config,
                output,
                resume_checkpoint=None,
                device_name="cpu",
                progress=SILENT_STATUS,
            )
            generated = root / "generated.png"
            inference = run_inference(
                output / "final",
                dataset / "source" / "validation.png",
                generated,
                device_name="cpu",
                precision="fp32",
                output_size=64,
            )

            self.assertEqual(result["status"], "completed")
            checkpoint = (
                output / "checkpoints" / "step-00000001" / "checkpoint.json"
            )
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((output / "previews" / "step-00000001.png").is_file())
            self.assertTrue((output / "final" / "generator.safetensors").is_file())
            self.assertEqual(inference["output_size"], 64)
            with Image.open(generated) as image:
                self.assertEqual(image.size, (64, 64))

            checkpoint_metadata = json.loads(checkpoint.read_text(encoding="utf-8"))
            training_config = TrainConfig.load(config)
            generator = Pix2PixGenerator(training_config.model)
            discriminator = ConditionalPatchDiscriminator(training_config.model)
            generator_optimizer = torch.optim.Adam(generator.parameters())
            discriminator_optimizer = torch.optim.Adam(discriminator.parameters())
            position = load_training_checkpoint(
                checkpoint.parent,
                expected_dataset_fingerprint=checkpoint_metadata[
                    "dataset_fingerprint"
                ],
                expected_config_fingerprint=checkpoint_metadata[
                    "config_fingerprint"
                ],
                model_config=training_config.model,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                device=torch.device("cpu"),
            )
            self.assertEqual(position.step, 1)
            self.assertEqual(position.epoch, 1)
            self.assertEqual(position.sample_offset, 0)


class Pix2PixTerminalResumeTests(unittest.TestCase):
    def test_terminal_resume_finalizes_without_training_or_another_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, config, output, checkpoint = _completed_training_fixture(root)
            _mark_run_running(output)
            shutil.rmtree(output / "final")
            _remove_validation_metric(output / "metrics.jsonl", step=1)
            checkpoint_before = _snapshot_directory(checkpoint)

            with patch(
                "aigen.pix2pix.training.create_training_loader",
                side_effect=AssertionError("terminal resume created a training loader"),
            ) as create_loader, patch(
                "aigen.pix2pix.training._train_batch",
                side_effect=AssertionError("terminal resume trained a batch"),
            ) as train_batch, patch(
                "aigen.pix2pix.training.save_training_checkpoint",
                side_effect=AssertionError("terminal resume saved another checkpoint"),
            ) as save_checkpoint, patch(
                "aigen.pix2pix.training._validate",
                wraps=pix2pix_training._validate,
            ) as validate:
                result = train_pix2pix(
                    dataset,
                    config,
                    output,
                    resume_checkpoint=checkpoint,
                    device_name="cpu",
                    progress=SILENT_STATUS,
                )

            create_loader.assert_not_called()
            train_batch.assert_not_called()
            save_checkpoint.assert_not_called()
            validate.assert_called_once()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(_snapshot_directory(checkpoint), checkpoint_before)
            checkpoint_metadata = _read_json(checkpoint / "checkpoint.json")
            final_metadata = _read_json(output / "final" / "model.json")
            self.assertEqual(
                final_metadata["weights"]["sha256"],
                checkpoint_metadata["files"]["generator"]["sha256"],
            )
            run = _read_json(output / "run.json")
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["step"], 1)
            self.assertEqual(run["latest_checkpoint"], checkpoint.as_posix())
            validation_records = _metric_records(
                output / "metrics.jsonl",
                kind="validation",
                step=1,
            )
            self.assertEqual(len(validation_records), 1)

    def test_terminal_resume_reuses_matching_complete_final_and_metric(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, config, output, checkpoint = _completed_training_fixture(root)
            _mark_run_running(output)
            final_before = _snapshot_directory(output / "final")
            metrics_before = (output / "metrics.jsonl").read_bytes()

            with patch(
                "aigen.pix2pix.training.create_training_loader",
                side_effect=AssertionError("terminal resume created a training loader"),
            ), patch(
                "aigen.pix2pix.training.save_training_checkpoint",
                side_effect=AssertionError("terminal resume saved another checkpoint"),
            ), patch(
                "aigen.pix2pix.artifacts._atomic_save_weights",
                side_effect=AssertionError("matching final was rewritten"),
            ) as save_weights:
                result = train_pix2pix(
                    dataset,
                    config,
                    output,
                    resume_checkpoint=checkpoint,
                    device_name="cpu",
                    progress=SILENT_STATUS,
                )

            save_weights.assert_not_called()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(_snapshot_directory(output / "final"), final_before)
            self.assertEqual((output / "metrics.jsonl").read_bytes(), metrics_before)

    def test_terminal_resume_rejects_partial_final_without_mutating_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, config, output, checkpoint = _completed_training_fixture(root)
            _mark_run_running(output)
            (output / "final" / "model.json").unlink()
            partial_before = _snapshot_directory(output / "final")

            with self.assertRaisesRegex(Pix2PixError, "incomplete"):
                train_pix2pix(
                    dataset,
                    config,
                    output,
                    resume_checkpoint=checkpoint,
                    device_name="cpu",
                    progress=SILENT_STATUS,
                )

            self.assertEqual(_snapshot_directory(output / "final"), partial_before)
            self.assertEqual(_read_json(output / "run.json")["status"], "running")

    def test_terminal_resume_rejects_mismatched_final_without_mutating_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, config, output, checkpoint = _completed_training_fixture(root)
            _mark_run_running(output)
            metadata_path = output / "final" / "model.json"
            metadata = _read_json(metadata_path)
            metadata["step"] = 2
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            mismatched_before = _snapshot_directory(output / "final")

            with self.assertRaisesRegex(Pix2PixError, "does not belong"):
                train_pix2pix(
                    dataset,
                    config,
                    output,
                    resume_checkpoint=checkpoint,
                    device_name="cpu",
                    progress=SILENT_STATUS,
                )

            self.assertEqual(_snapshot_directory(output / "final"), mismatched_before)
            self.assertEqual(_read_json(output / "run.json")["status"], "running")

    def test_resume_rejects_checkpoint_beyond_max_steps_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, config, output, checkpoint = _completed_training_fixture(root)
            _mark_run_running(output)
            checkpoint_metadata_path = checkpoint / "checkpoint.json"
            checkpoint_metadata = _read_json(checkpoint_metadata_path)
            checkpoint_metadata["step"] = 2
            checkpoint_metadata_path.write_text(
                json.dumps(checkpoint_metadata),
                encoding="utf-8",
            )

            with patch(
                "aigen.pix2pix.training.create_training_loader",
                side_effect=AssertionError("invalid resume created a training loader"),
            ), patch(
                "aigen.pix2pix.training._validate",
                side_effect=AssertionError("invalid resume ran validation"),
            ), patch(
                "aigen.pix2pix.training.export_generator_bundle",
                side_effect=AssertionError("invalid resume exported a final bundle"),
            ):
                with self.assertRaisesRegex(Pix2PixError, "exceeds max_steps 1"):
                    train_pix2pix(
                        dataset,
                        config,
                        output,
                        resume_checkpoint=checkpoint,
                        device_name="cpu",
                        progress=SILENT_STATUS,
                    )

    def test_resume_rejects_completed_run_before_loading_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, config, output, checkpoint = _completed_training_fixture(root)

            with patch(
                "aigen.pix2pix.training.load_training_checkpoint",
                side_effect=AssertionError("completed run loaded its checkpoint"),
            ) as load_checkpoint:
                with self.assertRaisesRegex(Pix2PixError, "already completed"):
                    train_pix2pix(
                        dataset,
                        config,
                        output,
                        resume_checkpoint=checkpoint,
                        device_name="cpu",
                        progress=SILENT_STATUS,
                    )

            load_checkpoint.assert_not_called()


class Pix2PixArtifactPublicationTests(unittest.TestCase):
    def test_generator_bundle_is_not_visible_when_publication_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "final"
            model_config = ModelConfig(
                image_size=128,
                generator_channels=1,
                discriminator_channels=1,
            )
            generator = Pix2PixGenerator(model_config)

            with patch(
                "aigen.pix2pix.artifacts.os.rename",
                side_effect=OSError("injected publication failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected publication failure"):
                    export_generator_bundle(
                        output,
                        generator=generator,
                        model_config=model_config,
                        step=1,
                        dataset_fingerprint="dataset",
                        config_fingerprint="config",
                    )

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".final.*.incomplete")), [])
            metadata = export_generator_bundle(
                output,
                generator=generator,
                model_config=model_config,
                step=1,
                dataset_fingerprint="dataset",
                config_fingerprint="config",
            )
            self.assertEqual(set(path.name for path in output.iterdir()), {
                "generator.safetensors",
                "model.json",
            })
            self.assertEqual(metadata, _read_json(output / "model.json"))


def _write_dataset(root: Path, *, image_size: int = 256) -> None:
    (root / "source").mkdir(parents=True)
    (root / "target").mkdir()
    records = []
    for pair_id, split, source_color, target_color in (
        ("train", "train", (32, 64, 96), (96, 32, 64)),
        ("validation", "validation", (128, 96, 64), (64, 128, 96)),
    ):
        source = f"source/{pair_id}.png"
        target = f"target/{pair_id}.png"
        Image.new("RGB", (image_size, image_size), source_color).save(root / source)
        Image.new("RGB", (image_size, image_size), target_color).save(root / target)
        records.append(
            {
                "id": pair_id,
                "group": f"subject-{pair_id}",
                "split": split,
                "source": source,
                "target": target,
            }
        )
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "format": DATASET_FORMAT,
                "name": "pipeline-test",
                "image_size": image_size,
                "pairs": "pairs.jsonl",
            }
        ),
        encoding="utf-8",
    )
    (root / "pairs.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_config(
    path: Path,
    *,
    image_size: int = 256,
    channels: int = 2,
) -> None:
    path.write_text(
        json.dumps(
            {
                "format": "aigen.pix2pix.training.v1",
                "model": {
                    "image_size": image_size,
                    "input_channels": 3,
                    "output_channels": 3,
                    "generator_channels": channels,
                    "discriminator_channels": channels,
                    "discriminator_layers": 3,
                    "generator_dropout": True,
                },
                "batch_size": 1,
                "max_steps": 1,
                "learning_rate": 0.0002,
                "beta1": 0.5,
                "beta2": 0.999,
                "lambda_l1": 100.0,
                "horizontal_flip": False,
                "precision": "fp32",
                "checkpoint_every": 1,
                "log_every": 1,
                "seed": 7,
                "num_workers": 0,
            }
        ),
        encoding="utf-8",
    )


def _completed_training_fixture(
    root: Path,
) -> tuple[Path, Path, Path, Path]:
    dataset = root / "dataset"
    config = root / "train.json"
    output = root / "run"
    _write_dataset(dataset, image_size=128)
    _write_config(config, image_size=128, channels=1)
    train_pix2pix(
        dataset,
        config,
        output,
        resume_checkpoint=None,
        device_name="cpu",
        progress=SILENT_STATUS,
    )
    checkpoint = output / "checkpoints" / "step-00000001"
    return dataset, config, output, checkpoint


def _mark_run_running(output: Path) -> None:
    path = output / "run.json"
    run = _read_json(path)
    run["status"] = "running"
    run["output"] = output.as_posix()
    for key in (
        "completed_at",
        "elapsed_seconds",
        "peak_vram_mb",
        "final_model",
    ):
        run.pop(key)
    path.write_text(json.dumps(run), encoding="utf-8")


def _remove_validation_metric(path: Path, *, step: int) -> None:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    retained = [
        record
        for record in records
        if not (record.get("kind") == "validation" and record.get("step") == step)
    ]
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in retained),
        encoding="utf-8",
    )


def _metric_records(path: Path, *, kind: str, step: int) -> list[dict[str, object]]:
    return [
        record
        for record in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
        if record.get("kind") == kind and record.get("step") == step
    ]


def _snapshot_directory(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
