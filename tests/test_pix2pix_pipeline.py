from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from aigen.pix2pix import training as pix2pix_training
from aigen.pix2pix.artifacts import (
    AdversarialCheckpointState,
    L1CheckpointState,
    export_generator_bundle,
    load_generator_bundle,
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
                expected_objective=training_config.objective,
                state=AdversarialCheckpointState(
                    discriminator=discriminator,
                    generator_optimizer=generator_optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                ),
                expected_optimizer_name=training_config.optimizer,
                expected_parameter_precision=training_config.parameter_precision,
                device=torch.device("cpu"),
            )
            self.assertEqual(position.step, 1)
            self.assertEqual(position.epoch, 1)
            self.assertEqual(position.sample_offset, 0)


class Pix2PixL1ObjectiveTests(unittest.TestCase):
    def test_l1_training_never_constructs_or_executes_a_discriminator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            config = root / "train.json"
            output = root / "run"
            _write_dataset(dataset, image_size=128)
            _write_config(
                config,
                image_size=128,
                channels=1,
                objective="l1_only",
            )

            with patch(
                "aigen.pix2pix.training.ConditionalPatchDiscriminator",
                side_effect=AssertionError("L1 constructed a discriminator"),
            ), patch(
                "aigen.pix2pix.training.discriminator_loss",
                side_effect=AssertionError("L1 executed discriminator loss"),
            ):
                result = train_pix2pix(
                    dataset,
                    config,
                    output,
                    resume_checkpoint=None,
                    device_name="cpu",
                    progress=SILENT_STATUS,
                )

            self.assertEqual(result["status"], "completed")
            checkpoint = output / "checkpoints" / "step-00000001"
            self.assertEqual(
                {path.name for path in checkpoint.iterdir()},
                {
                    "checkpoint.json",
                    "generator.safetensors",
                    "training-state.pt",
                },
            )
            metadata = _read_json(checkpoint / "checkpoint.json")
            self.assertEqual(metadata["objective"], "l1_only")
            self.assertEqual(
                set(metadata["files"]),
                {"generator", "training_state"},
            )
            training_state = torch.load(
                checkpoint / "training-state.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                set(training_state),
                {"generator_optimizer", "cpu_rng_state"},
            )
            train_record = _metric_records(
                output / "metrics.jsonl",
                kind="train",
                step=1,
            )[0]
            self.assertEqual(
                set(train_record),
                {
                    "kind",
                    "step",
                    "epoch",
                    "learning_rate",
                    "generator_total",
                    "generator_l1",
                },
            )
            self.assertEqual(
                set(_read_json(output / "run.json")["parameters"]),
                {"generator", "total"},
            )

    def test_l1_terminal_resume_restores_without_a_discriminator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            config = root / "train.json"
            output = root / "run"
            _write_dataset(dataset, image_size=128)
            _write_config(
                config,
                image_size=128,
                channels=1,
                objective="l1_only",
            )
            train_pix2pix(
                dataset,
                config,
                output,
                resume_checkpoint=None,
                device_name="cpu",
                progress=SILENT_STATUS,
            )
            checkpoint = output / "checkpoints" / "step-00000001"
            _mark_run_running(output)
            shutil.rmtree(output / "final")
            _remove_validation_metric(output / "metrics.jsonl", step=1)

            with patch(
                "aigen.pix2pix.training.ConditionalPatchDiscriminator",
                side_effect=AssertionError("L1 resume constructed a discriminator"),
            ):
                result = train_pix2pix(
                    dataset,
                    config,
                    output,
                    resume_checkpoint=checkpoint,
                    device_name="cpu",
                    progress=SILENT_STATUS,
                )

            self.assertEqual(result["status"], "completed")
            self.assertTrue((output / "final" / "generator.safetensors").is_file())

    def test_l1_loader_rejects_an_adversarial_checkpoint_by_objective(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, config, _, checkpoint = _completed_training_fixture(root)
            training_config = TrainConfig.load(config)
            generator = Pix2PixGenerator(training_config.model)
            generator_optimizer = torch.optim.Adam(generator.parameters())

            with self.assertRaisesRegex(Pix2PixError, "checkpoint objective"):
                load_training_checkpoint(
                    checkpoint,
                    expected_dataset_fingerprint="not inspected",
                    expected_config_fingerprint="not inspected",
                    model_config=training_config.model,
                    generator=generator,
                    expected_objective="l1_only",
                    state=L1CheckpointState(generator_optimizer),
                    expected_optimizer_name=training_config.optimizer,
                    expected_parameter_precision=training_config.parameter_precision,
                    device=torch.device("cpu"),
                )

    def test_v3_config_requires_an_objective(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "train.json"
            _write_config(config, objective="l1_only")
            payload = _read_json(config)
            del payload["objective"]
            config.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(Pix2PixError, "missing objective"):
                TrainConfig.load(config)

    def test_objectives_share_generator_initialization_and_dropout_rng(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            _write_dataset(dataset, image_size=128)
            captures: dict[str, tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]] = {}

            def capture_batch(
                label: str,
                batch: dict[str, torch.Tensor | list[str]],
                *,
                generator: Pix2PixGenerator,
                **_: object,
            ) -> tuple[dict[str, torch.Tensor], int]:
                state = {
                    name: tensor.detach().clone()
                    for name, tensor in generator.state_dict().items()
                }
                rng = torch.get_rng_state().clone()
                source = batch["source"]
                assert isinstance(source, torch.Tensor)
                with torch.no_grad():
                    generated = generator(source)
                captures[label] = (state, rng, generated)
                zero = generated.new_zeros(())
                losses = {
                    "generator_total": zero,
                    "generator_l1": zero,
                }
                if label == "adversarial_l1":
                    losses["generator_adversarial"] = zero
                    losses["discriminator"] = zero
                return losses, source.shape[0]

            for objective, batch_function in (
                ("adversarial_l1", "_train_adversarial_batch"),
                ("l1_only", "_train_l1_batch"),
            ):
                config = root / f"{objective}.json"
                _write_config(
                    config,
                    image_size=128,
                    channels=1,
                    objective=objective,
                )
                with patch(
                    f"aigen.pix2pix.training.{batch_function}",
                    side_effect=partial(capture_batch, objective),
                ):
                    train_pix2pix(
                        dataset,
                        config,
                        root / objective,
                        resume_checkpoint=None,
                        device_name="cpu",
                        progress=SILENT_STATUS,
                    )

            adversarial_state, adversarial_rng, adversarial_output = captures[
                "adversarial_l1"
            ]
            l1_state, l1_rng, l1_output = captures["l1_only"]
            self.assertEqual(set(adversarial_state), set(l1_state))
            for name in adversarial_state:
                self.assertTrue(
                    torch.equal(adversarial_state[name], l1_state[name]),
                    name,
                )
            self.assertTrue(torch.equal(adversarial_rng, l1_rng))
            self.assertTrue(torch.equal(adversarial_output, l1_output))


class Pix2PixLearningRateScheduleTests(unittest.TestCase):
    def test_linear_decay_is_step_derived_and_applied_to_both_optimizers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            config = root / "train.json"
            output = root / "run"
            _write_dataset(dataset, image_size=128)
            _write_config(
                config,
                image_size=128,
                channels=1,
                objective="adversarial_l1",
                max_steps=4,
                checkpoint_every=4,
                extra_checkpoint_steps=(2,),
                learning_rate_schedule={
                    "type": "linear_decay",
                    "decay_start_step": 2,
                },
            )
            observed_learning_rates: list[float] = []

            def capture_batch(
                batch: dict[str, torch.Tensor | list[str]],
                *,
                generator_optimizer: torch.optim.Optimizer,
                discriminator_optimizer: torch.optim.Optimizer,
                **_: object,
            ) -> tuple[dict[str, torch.Tensor], int]:
                generator_lr = generator_optimizer.param_groups[0]["lr"]
                discriminator_lr = discriminator_optimizer.param_groups[0]["lr"]
                self.assertEqual(generator_lr, discriminator_lr)
                observed_learning_rates.append(generator_lr)
                source = batch["source"]
                assert isinstance(source, torch.Tensor)
                zero = source.new_zeros(())
                return (
                    {
                        "generator_total": zero,
                        "generator_adversarial": zero,
                        "generator_l1": zero,
                        "discriminator": zero,
                    },
                    source.shape[0],
                )

            with patch(
                "aigen.pix2pix.training._train_adversarial_batch",
                side_effect=capture_batch,
            ):
                train_pix2pix(
                    dataset,
                    config,
                    output,
                    resume_checkpoint=None,
                    device_name="cpu",
                    progress=SILENT_STATUS,
                )

            self.assertEqual(
                observed_learning_rates,
                [0.0002, 0.0002, 0.0002, 0.0001],
            )
            self.assertTrue(
                (output / "checkpoints" / "step-00000002").is_dir()
            )
            self.assertTrue(
                (output / "checkpoints" / "step-00000004").is_dir()
            )
            train_records = [
                json.loads(line)
                for line in (output / "metrics.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if json.loads(line)["kind"] == "train"
            ]
            self.assertEqual(
                [record["learning_rate"] for record in train_records],
                observed_learning_rates,
            )

    def test_resume_derives_the_next_learning_rate_from_checkpoint_step(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            config = root / "train.json"
            output = root / "run"
            _write_dataset(dataset, image_size=128)
            _write_config(
                config,
                image_size=128,
                channels=1,
                objective="adversarial_l1",
                max_steps=4,
                checkpoint_every=3,
                learning_rate_schedule={
                    "type": "linear_decay",
                    "decay_start_step": 2,
                },
            )
            calls = 0

            def interrupt_after_checkpoint(
                batch: dict[str, torch.Tensor | list[str]],
                **_: object,
            ) -> tuple[dict[str, torch.Tensor], int]:
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError("injected stop")
                source = batch["source"]
                assert isinstance(source, torch.Tensor)
                zero = source.new_zeros(())
                return (
                    {
                        "generator_total": zero,
                        "generator_adversarial": zero,
                        "generator_l1": zero,
                        "discriminator": zero,
                    },
                    source.shape[0],
                )

            with patch(
                "aigen.pix2pix.training._train_adversarial_batch",
                side_effect=interrupt_after_checkpoint,
            ), self.assertRaisesRegex(RuntimeError, "injected stop"):
                train_pix2pix(
                    dataset,
                    config,
                    output,
                    resume_checkpoint=None,
                    device_name="cpu",
                    progress=SILENT_STATUS,
                )

            checkpoint = output / "checkpoints" / "step-00000003"
            resumed_learning_rates: list[float] = []

            def capture_resumed_batch(
                batch: dict[str, torch.Tensor | list[str]],
                *,
                generator_optimizer: torch.optim.Optimizer,
                discriminator_optimizer: torch.optim.Optimizer,
                **_: object,
            ) -> tuple[dict[str, torch.Tensor], int]:
                generator_lr = generator_optimizer.param_groups[0]["lr"]
                self.assertEqual(
                    generator_lr,
                    discriminator_optimizer.param_groups[0]["lr"],
                )
                resumed_learning_rates.append(generator_lr)
                source = batch["source"]
                assert isinstance(source, torch.Tensor)
                zero = source.new_zeros(())
                return (
                    {
                        "generator_total": zero,
                        "generator_adversarial": zero,
                        "generator_l1": zero,
                        "discriminator": zero,
                    },
                    source.shape[0],
                )

            with patch(
                "aigen.pix2pix.training._train_adversarial_batch",
                side_effect=capture_resumed_batch,
            ):
                train_pix2pix(
                    dataset,
                    config,
                    output,
                    resume_checkpoint=checkpoint,
                    device_name="cpu",
                    progress=SILENT_STATUS,
                )

            self.assertEqual(resumed_learning_rates, [0.0001])


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
                "aigen.pix2pix.training._train_adversarial_batch",
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
    def test_generator_bundle_preserves_bf16_parameter_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "final"
            model_config = ModelConfig(
                image_size=128,
                generator_channels=1,
                discriminator_channels=1,
            )
            generator = Pix2PixGenerator(
                model_config,
                dtype=torch.bfloat16,
            )
            export_generator_bundle(
                output,
                generator=generator,
                model_config=model_config,
                step=1,
                dataset_fingerprint="dataset",
                config_fingerprint="config",
            )

            loaded, _ = load_generator_bundle(
                output,
                device=torch.device("cpu"),
            )

            self.assertEqual(next(loaded.parameters()).dtype, torch.bfloat16)

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
    objective: str | None = None,
    max_steps: int = 1,
    checkpoint_every: int = 1,
    extra_checkpoint_steps: tuple[int, ...] = (),
    learning_rate_schedule: dict[str, object] | None = None,
) -> None:
    if learning_rate_schedule is not None:
        config_format = "aigen.pix2pix.training.v4"
        objective = objective or "adversarial_l1"
    elif objective is not None:
        config_format = "aigen.pix2pix.training.v3"
    else:
        config_format = "aigen.pix2pix.training.v2"
    payload = {
        "format": config_format,
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
        "max_steps": max_steps,
        "learning_rate": 0.0002,
        "beta1": 0.5,
        "beta2": 0.999,
        "lambda_l1": 100.0,
        "horizontal_flip": False,
        "optimizer": "adam",
        "parameter_precision": "fp32",
        "precision": "fp32",
        "checkpoint_every": checkpoint_every,
        "log_every": 1,
        "seed": 7,
        "num_workers": 0,
    }
    if objective is not None:
        payload["objective"] = objective
    if learning_rate_schedule is not None:
        payload["learning_rate_schedule"] = learning_rate_schedule
        payload["extra_checkpoint_steps"] = list(extra_checkpoint_steps)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
