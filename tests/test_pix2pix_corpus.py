from __future__ import annotations

import io
import hashlib
import json
import sys
import tempfile
import types
import unittest
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

from aigen.manifest_io import ManifestIOError, read_json, sha256_file, write_json
from aigen.pix2pix.corpus_config import FluxSourceConfig
from aigen.pix2pix.corpus_dataset import (
    FLUX_SOURCE_SET_DATASET_DIRECTORY_PREFIX,
    QWEN_CORPUS_DATASET_DIRECTORY,
    _prepare_source,
    prepare_iro_flux_source_set_dataset,
    prepare_iro_dataset,
    prepare_iro_qwen_dataset,
)
from aigen.pix2pix.corpus_io import write_json_records
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.flux_source_corpus import (
    FLUX_SOURCE_DIRECTORY,
    FLUX_SOURCE_SHARD_FORMAT,
    _load_or_create_source_plan,
    generate_flux_sources,
    load_flux_source_inventory,
)
from aigen.pix2pix.flux_source_set import load_flux_source_set
from aigen.pix2pix.flux_source_set_corpus import (
    flux_source_set_layout,
    flux_source_set_result_path,
    generate_flux_source_set_sources,
    load_flux_source_set_inventory,
)
from aigen.pix2pix.iro_corpus import (
    IRO_SELECTION_FORMAT,
    _lineage_counts,
    _selection_frame_safe,
    _split_counts,
    decode_renderer_frames,
    load_iro_plan,
    load_iro_selection,
    plan_iro_corpus,
)
from aigen.pix2pix.qwen_source_config import QwenSourceConfig
from aigen.pix2pix.qwen_source_corpus import (
    QWEN_SOURCE_DIRECTORY,
    generate_qwen_sources,
    load_qwen_source_inventory,
)
from aigen.pix2pix.source_corpus import expected_source_shards
from aigen.progress import SILENT_STATUS
from aigen.runtime_provenance import (
    build_python_runtime_provenance_for_interpreter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE_CONFIG = PROJECT_ROOT / "configs" / "pix2pix-iro-gate512.json"


class Pix2PixCorpusTests(unittest.TestCase):
    def test_gate_plan_materializes_exact_disjoint_quotas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpus"

            first = plan_iro_corpus(GATE_CONFIG, root)
            second = plan_iro_corpus(GATE_CONFIG, root)
            config, requests, _ = load_iro_plan(root)

            self.assertEqual(len(config.jobs), 79)
            self.assertEqual(first["split_counts"], {"train": 384, "validation": 64, "test": 64})
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(len(requests), 512)
            self.assertEqual(len({record["id"] for record in requests}), 512)
            self.assertEqual(
                len(
                    {
                        json.dumps(
                            record["payload"],
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        for record in requests
                    }
                ),
                512,
            )
            lineage_splits = {}
            identity_splits = {}
            for record in requests:
                lineage_splits.setdefault(record["lineage"], set()).add(record["split"])
                identity_splits.setdefault(record["identity"], set()).add(record["split"])
                self.assertEqual(record["group"], record["lineage"])
            self.assertTrue(all(len(splits) == 1 for splits in lineage_splits.values()))
            self.assertTrue(all(len(splits) == 1 for splits in identity_splits.values()))
            self.assertEqual(
                Counter(record["direction"] for record in requests if record["split"] == "train"),
                Counter({direction: 48 for direction in range(8)}),
            )

    def test_apng_decoder_skips_a_separate_default_image(self) -> None:
        base = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        first = base.copy()
        first.putpixel((2, 2), (255, 0, 0, 255))
        second = base.copy()
        second.putpixel((3, 3), (0, 255, 0, 255))
        encoded = io.BytesIO()
        base.save(
            encoded,
            format="PNG",
            save_all=True,
            append_images=[first, second],
            default_image=True,
            duration=[100, 200],
            loop=0,
            disposal=[0, 0],
            blend=[0, 0],
        )

        frames, default_image = decode_renderer_frames(
            encoded.getvalue(),
            expected_size=8,
        )

        self.assertTrue(default_image)
        self.assertEqual([frame["index"] for frame in frames], [1, 2])
        self.assertEqual([frame["duration_ms"] for frame in frames], [100.0, 200.0])
        self.assertEqual([frame["image"].getbbox() for frame in frames], [(2, 2, 3, 3), (3, 3, 4, 4)])

    def test_selection_allows_ground_baseline_but_rejects_side_clipping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
            baseline.putpixel((4, 7), (255, 0, 0, 255))
            baseline_path = root / "baseline.png"
            baseline.save(baseline_path)
            side = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
            side.putpixel((0, 4), (255, 0, 0, 255))
            side_path = root / "side.png"
            side.save(side_path)

            self.assertTrue(_selection_frame_safe(baseline_path))
            self.assertFalse(_selection_frame_safe(side_path))

    def test_source_rasterization_is_the_only_resampled_half(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.png"
            output_path = root / "output.png"
            Image.new("RGB", (6, 8), (160, 32, 16)).save(input_path)

            _prepare_source(
                input_path,
                output_path,
                canvas_size=8,
                inner_size=(6, 8),
                offset=(1, 0),
                background=(255, 255, 255),
            )

            with Image.open(output_path) as output:
                self.assertEqual(output.mode, "RGB")
                self.assertEqual(output.size, (8, 8))
                self.assertEqual(output.getpixel((0, 0)), (255, 255, 255))
                self.assertEqual(output.getpixel((1, 0)), (160, 32, 16))

    def test_completed_source_shards_resume_without_importing_the_gpu_backend(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _materialize_fake_source_corpus(Path(temporary))

            with _fake_execution_provenance(), patch.dict(
                "sys.modules",
                {"aigen.generation.flux2_klein": None},
            ):
                first = generate_flux_sources(root, progress=SILENT_STATUS)
                completed_checksums = _tree_checksums(
                    root / FLUX_SOURCE_DIRECTORY
                )
                second = generate_flux_sources(root, progress=SILENT_STATUS)
                inventory = load_flux_source_inventory(root)

            self.assertEqual(first["generated_shards"], 0)
            self.assertEqual(first["reused_shards"], 6)
            self.assertEqual(second["generated_shards"], 0)
            self.assertEqual(len(inventory), 24)
            self.assertTrue((root / FLUX_SOURCE_DIRECTORY / "result.json").is_file())
            self.assertEqual(
                _tree_checksums(root / FLUX_SOURCE_DIRECTORY),
                completed_checksums,
            )

    def test_corrupt_source_fails_before_importing_the_gpu_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _materialize_fake_source_corpus(Path(temporary))
            source = next(
                (root / FLUX_SOURCE_DIRECTORY / "shards").glob(
                    "shard-*/raw/*.png"
                )
            )
            Image.new("RGB", (64, 64), (255, 0, 255)).save(source)

            with _fake_execution_provenance(), patch.dict(
                "sys.modules",
                {"aigen.generation.flux2_klein": None},
            ):
                with self.assertRaisesRegex(Pix2PixError, "size mismatch|checksum mismatch"):
                    generate_flux_sources(root, progress=SILENT_STATUS)

    def test_source_plan_rejects_changed_model_artifacts_before_generation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _materialize_fake_source_corpus(Path(temporary))
            changed_model_payload = json.loads(
                json.dumps(
                    {
                        key: value
                        for key, value in FAKE_MODEL_PROVENANCE.items()
                        if key != "fingerprint"
                    }
                )
            )
            changed_model_payload["components"][0]["files"][0]["sha256"] = "f" * 64
            changed_model = _fingerprinted(changed_model_payload)

            with patch(
                "aigen.pix2pix.flux_source_corpus.flux2_klein_model_provenance",
                return_value=changed_model,
            ), patch(
                "aigen.pix2pix.flux_source_corpus.flux2_klein_runtime_provenance",
                return_value=FAKE_RUNTIME_PROVENANCE,
            ), patch.dict(
                "sys.modules",
                {"aigen.generation.flux2_klein": None},
            ):
                with self.assertRaisesRegex(Pix2PixError, "source plan differs"):
                    generate_flux_sources(root, progress=SILENT_STATUS)

            changed_runtime_payload = {
                key: value
                for key, value in FAKE_RUNTIME_PROVENANCE.items()
                if key != "fingerprint"
            }
            changed_runtime_payload["python"] = "3.13.0"
            changed_runtime = _fingerprinted(changed_runtime_payload)
            with patch(
                "aigen.pix2pix.flux_source_corpus.flux2_klein_model_provenance",
                return_value=FAKE_MODEL_PROVENANCE,
            ), patch(
                "aigen.pix2pix.flux_source_corpus.flux2_klein_runtime_provenance",
                return_value=changed_runtime,
            ), patch.dict(
                "sys.modules",
                {"aigen.generation.flux2_klein": None},
            ):
                with self.assertRaisesRegex(Pix2PixError, "source plan differs"):
                    generate_flux_sources(root, progress=SILENT_STATUS)

    def test_completed_source_result_is_immutable_before_gpu_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _materialize_fake_source_corpus(Path(temporary))
            with _fake_execution_provenance():
                generate_flux_sources(root, progress=SILENT_STATUS)
            shard = root / FLUX_SOURCE_DIRECTORY / "shards" / "shard-00000"
            shard.rename(shard.with_name(".shard-00000.missing"))

            with _fake_execution_provenance(), patch.dict(
                "sys.modules",
                {"aigen.generation.flux2_klein": None},
            ):
                with self.assertRaisesRegex(Pix2PixError, "missing.*shard manifest"):
                    generate_flux_sources(root, progress=SILENT_STATUS)

    def test_missing_source_plan_is_not_repaired_before_gpu_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _materialize_fake_source_corpus(Path(temporary))
            plan = root / FLUX_SOURCE_DIRECTORY / "source-plan.json"
            plan.rename(plan.with_name(".source-plan.missing"))

            with _fake_execution_provenance(), patch.dict(
                "sys.modules",
                {"aigen.generation.flux2_klein": None},
            ):
                with self.assertRaisesRegex(
                    Pix2PixError,
                    "non-empty.*has no source plan",
                ):
                    generate_flux_sources(root, progress=SILENT_STATUS)

    def test_source_result_and_dataset_provenance_are_semantic_contracts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _materialize_fake_source_corpus(Path(temporary))
            with _fake_execution_provenance():
                generate_flux_sources(root, progress=SILENT_STATUS)
                first = prepare_iro_dataset(root)
                second = prepare_iro_dataset(root)
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])

            provenance_path = root / "dataset" / "provenance.json"
            provenance = read_json(provenance_path, label="test provenance")
            provenance["source_raster"]["offset_x"] += 1
            write_json(provenance_path, provenance)
            with _fake_execution_provenance():
                with self.assertRaisesRegex(
                    Pix2PixError,
                    "provenance mismatch: source_raster",
                ):
                    prepare_iro_dataset(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = _materialize_fake_source_corpus(Path(temporary))
            with _fake_execution_provenance():
                generate_flux_sources(root, progress=SILENT_STATUS)
            result_path = root / FLUX_SOURCE_DIRECTORY / "result.json"
            result = read_json(result_path, label="test FLUX result")
            result["pair_count"] += 1
            write_json(result_path, result)
            with _fake_execution_provenance():
                with self.assertRaisesRegex(Pix2PixError, "source result differs"):
                    load_flux_source_inventory(root)

    def test_corpus_and_backend_share_the_real_sampler_names(self) -> None:
        flux = json.loads(GATE_CONFIG.read_text(encoding="utf-8"))["flux"]
        flux["sampler"] = "euler-ancestral"
        self.assertEqual(
            FluxSourceConfig.model_validate(flux).sampler,
            "euler-ancestral",
        )
        flux["sampler"] = "flowmatch-euler-ancestral"
        with self.assertRaises(ValidationError):
            FluxSourceConfig.model_validate(flux)

        from aigen.generation.flux2_klein import (
            Flux2KleinError,
            Flux2KleinSession,
        )

        with self.assertRaisesRegex(Flux2KleinError, "unsupported.*sampler"):
            Flux2KleinSession(
                loras=(),
                sampler="flowmatch-euler-ancestral",
                progress=SILENT_STATUS,
            )

    def test_reviewed_flux_source_set_resumes_with_one_model_session(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            root = _materialize_fake_source_corpus(temporary_path)
            source_set_path = _write_fake_flux_source_set(
                temporary_path,
                root,
            )
            legacy_checksums = _tree_checksums(
                root / FLUX_SOURCE_DIRECTORY
            )
            state: dict[str, object] = {}

            with _fake_flux_source_set_execution(state):
                first = generate_flux_source_set_sources(
                    root,
                    source_set_path,
                    progress=SILENT_STATUS,
                )
                second = generate_flux_source_set_sources(
                    root,
                    source_set_path,
                    progress=SILENT_STATUS,
                )
                inventory, source_set, _, _ = (
                    load_flux_source_set_inventory(root, "reviewed-test")
                )

            self.assertEqual(first["generated_shards"], 6)
            self.assertEqual(first["reused_shards"], 0)
            self.assertEqual(second["generated_shards"], 0)
            self.assertEqual(second["reused_shards"], 6)
            self.assertEqual(len(inventory), 24)
            self.assertEqual(len(source_set.records), 24)
            self.assertEqual(state["session_count"], 1)
            self.assertEqual(state["generate_count"], 6)
            self.assertEqual(
                len(state["encoded_prompts"]),
                24,
            )
            self.assertEqual(
                _tree_checksums(root / FLUX_SOURCE_DIRECTORY),
                legacy_checksums,
            )

    def test_reviewed_flux_source_set_failure_publishes_no_partial_shard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            root = _materialize_fake_source_corpus(temporary_path)
            source_set_path = _write_fake_flux_source_set(
                temporary_path,
                root,
            )
            state: dict[str, object] = {}

            with _fake_flux_source_set_execution(state, fail=True):
                with self.assertRaisesRegex(Pix2PixError, "injected FLUX"):
                    generate_flux_source_set_sources(
                        root,
                        source_set_path,
                        progress=SILENT_STATUS,
                    )

            shards_dir = (
                root / flux_source_set_layout("reviewed-test").directory / "shards"
            )
            self.assertFalse((shards_dir / "shard-00000").exists())
            self.assertFalse(
                any(
                    path.name.endswith(".incomplete")
                    for path in shards_dir.iterdir()
                )
            )

    def test_reviewed_flux_source_plan_rejects_changed_case_seed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            root = _materialize_fake_source_corpus(temporary_path)
            source_set_path = _write_fake_flux_source_set(
                temporary_path,
                root,
            )
            state: dict[str, object] = {}
            with _fake_flux_source_set_execution(state):
                generate_flux_source_set_sources(
                    root,
                    source_set_path,
                    progress=SILENT_STATUS,
                )

            manifest = read_json(
                source_set_path,
                label="test FLUX source-set manifest",
            )
            records_path = source_set_path.parent / manifest["records"]
            records = [
                json.loads(line)
                for line in records_path.read_text(encoding="utf-8").splitlines()
            ]
            records[0]["seed"] += 1
            write_json_records(records_path, records)
            manifest["records_sha256"] = sha256_file(records_path)
            write_json(source_set_path, manifest)

            with _fake_flux_source_set_execution({}):
                with self.assertRaisesRegex(Pix2PixError, "plan differs"):
                    generate_flux_source_set_sources(
                        root,
                        source_set_path,
                        progress=SILENT_STATUS,
                    )

    def test_reviewed_flux_source_set_rejects_stale_prompt_review(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            root = _materialize_fake_source_corpus(temporary_path)
            source_set_path = _write_fake_flux_source_set(
                temporary_path,
                root,
            )
            manifest = read_json(
                source_set_path,
                label="test FLUX source-set manifest",
            )
            records_path = source_set_path.parent / manifest["records"]
            records = [
                json.loads(line)
                for line in records_path.read_text(encoding="utf-8").splitlines()
            ]
            records[0]["prompt"] += " Preserve the rear view."
            write_json_records(records_path, records)
            manifest["records_sha256"] = sha256_file(records_path)
            write_json(source_set_path, manifest)

            with self.assertRaisesRegex(
                Pix2PixError,
                "prompt-review prompt checksum mismatch",
            ):
                generate_flux_source_set_sources(
                    root,
                    source_set_path,
                    progress=SILENT_STATUS,
                )

    def test_reviewed_flux_source_set_requires_current_prompt_guide(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            root = _materialize_fake_source_corpus(temporary_path)
            source_set_path = _write_fake_flux_source_set(
                temporary_path,
                root,
            )
            manifest = read_json(
                source_set_path,
                label="test FLUX source-set manifest",
            )
            manifest["prompt_guide_sha256"] = "1" * 64
            write_json(source_set_path, manifest)

            with self.assertRaisesRegex(
                Pix2PixError,
                "different image-edit prompt guide",
            ):
                generate_flux_source_set_sources(
                    root,
                    source_set_path,
                    progress=SILENT_STATUS,
                )

    def test_reviewed_flux_source_set_requires_independent_reviewer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            root = _materialize_fake_source_corpus(temporary_path)
            source_set_path = _write_fake_flux_source_set(
                temporary_path,
                root,
            )
            manifest = read_json(
                source_set_path,
                label="test FLUX source-set manifest",
            )
            reviews_path = source_set_path.parent / manifest["prompt_reviews"]
            reviews = [
                json.loads(line)
                for line in reviews_path.read_text(encoding="utf-8").splitlines()
            ]
            reviews[0]["reviewer"] = "test-author"
            write_json_records(reviews_path, reviews)
            manifest["prompt_reviews_sha256"] = sha256_file(reviews_path)
            write_json(source_set_path, manifest)

            with self.assertRaisesRegex(
                Pix2PixError,
                "author reviewed their own prompt",
            ):
                generate_flux_source_set_sources(
                    root,
                    source_set_path,
                    progress=SILENT_STATUS,
                )

    def test_reviewed_flux_dataset_requires_and_applies_output_audit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            root = _materialize_fake_source_corpus(temporary_path)
            source_set_path = _write_fake_flux_source_set(
                temporary_path,
                root,
            )
            state: dict[str, object] = {}
            with _fake_flux_source_set_execution(state):
                generate_flux_source_set_sources(
                    root,
                    source_set_path,
                    progress=SILENT_STATUS,
                )

            with self.assertRaises(ManifestIOError):
                prepare_iro_flux_source_set_dataset(root, "reviewed-test")

            _, selected, _ = load_iro_selection(root)
            _write_fake_flux_output_audit(
                root,
                "reviewed-test",
                rejected_ids={str(selected[0]["id"])},
            )
            first = prepare_iro_flux_source_set_dataset(
                root,
                "reviewed-test",
            )
            second = prepare_iro_flux_source_set_dataset(
                root,
                "reviewed-test",
            )

            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(first["pair_count"], 23)
            self.assertTrue(
                (
                    root
                    / f"{FLUX_SOURCE_SET_DATASET_DIRECTORY_PREFIX}reviewed-test"
                ).is_dir()
            )

    def test_qwen_source_config_freezes_native_input_and_lightning_schedule(
        self,
    ) -> None:
        payload = json.loads(
            (
                PROJECT_ROOT
                / "configs/pix2pix-iro-qwen2511-lightning-source.json"
            ).read_text(encoding="utf-8")
        )

        config = QwenSourceConfig.model_validate(payload)

        self.assertEqual((config.width, config.height), (1328, 1328))
        self.assertEqual(config.steps, 8)
        self.assertEqual(config.guidance, 1.0)
        self.assertEqual(config.source_raster.inner_width, 128)
        self.assertEqual(config.source_raster.inner_height, 128)

        payload["steps"] = 40
        with self.assertRaises(ValidationError):
            QwenSourceConfig.model_validate(payload)

    def test_runtime_provenance_executes_the_venv_entrypoint(self) -> None:
        provenance = build_python_runtime_provenance_for_interpreter(
            PROJECT_ROOT / ".venv/bin/python",
            ("torch",),
        )

        self.assertEqual(
            [record["name"] for record in provenance["distributions"]],
            ["torch"],
        )

    def test_qwen_shards_resume_atomically_and_preserve_flux_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            root = _materialize_fake_source_corpus(temporary_path)
            config_path = _write_fake_qwen_source_config(temporary_path)
            flux_inventory_before = _tree_checksums(
                root / FLUX_SOURCE_DIRECTORY
            )

            with _fake_qwen_execution_provenance(), patch(
                "aigen.generation.image_edit_batch.run_image_edit_batch",
                side_effect=_fake_qwen_batch,
            ) as run_batch:
                first = generate_qwen_sources(
                    root,
                    config_path,
                    progress=SILENT_STATUS,
                )
                first_call_count = run_batch.call_count
                second = generate_qwen_sources(
                    root,
                    config_path,
                    progress=SILENT_STATUS,
                )
                inventory = load_qwen_source_inventory(root)

            self.assertEqual(first["generated_shards"], 6)
            self.assertEqual(first["reused_shards"], 0)
            self.assertEqual(first_call_count, 6)
            self.assertEqual(run_batch.call_count, first_call_count)
            self.assertEqual(second["generated_shards"], 0)
            self.assertEqual(second["reused_shards"], 6)
            self.assertEqual(len(inventory), 24)
            self.assertEqual(
                _tree_checksums(root / FLUX_SOURCE_DIRECTORY),
                flux_inventory_before,
            )
            self.assertFalse(
                any(
                    path.name.endswith(".incomplete")
                    for path in (
                        root / QWEN_SOURCE_DIRECTORY / "shards"
                    ).iterdir()
                )
            )

    def test_qwen_batch_failure_exposes_no_partial_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            root = _materialize_fake_source_corpus(temporary_path)
            config_path = _write_fake_qwen_source_config(temporary_path)

            with _fake_qwen_execution_provenance(), patch(
                "aigen.generation.image_edit_batch.run_image_edit_batch",
                side_effect=_failing_qwen_batch,
            ):
                with self.assertRaisesRegex(Pix2PixError, "injected Qwen"):
                    generate_qwen_sources(
                        root,
                        config_path,
                        progress=SILENT_STATUS,
                    )

            shards_dir = root / QWEN_SOURCE_DIRECTORY / "shards"
            self.assertFalse((shards_dir / "shard-00000").exists())
            self.assertFalse(
                any(
                    path.name.endswith(".incomplete")
                    for path in shards_dir.iterdir()
                )
            )

    def test_corrupt_qwen_source_fails_before_batch_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            root = _materialize_fake_source_corpus(temporary_path)
            config_path = _write_fake_qwen_source_config(temporary_path)
            with _fake_qwen_execution_provenance(), patch(
                "aigen.generation.image_edit_batch.run_image_edit_batch",
                side_effect=_fake_qwen_batch,
            ):
                generate_qwen_sources(
                    root,
                    config_path,
                    progress=SILENT_STATUS,
                )
            source = next(
                (root / QWEN_SOURCE_DIRECTORY / "shards").glob(
                    "shard-*/raw/*.png"
                )
            )
            Image.new("RGB", (1328, 1328), (255, 0, 255)).save(source)

            with _fake_qwen_execution_provenance(), patch(
                "aigen.generation.image_edit_batch.run_image_edit_batch",
            ) as run_batch:
                with self.assertRaisesRegex(
                    Pix2PixError,
                    "size mismatch|checksum mismatch",
                ):
                    generate_qwen_sources(
                        root,
                        config_path,
                        progress=SILENT_STATUS,
                    )
            run_batch.assert_not_called()

    def test_qwen_dataset_is_separate_and_reuses_validated_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            root = _materialize_fake_source_corpus(temporary_path)
            config_path = _write_fake_qwen_source_config(temporary_path)
            with _fake_execution_provenance():
                generate_flux_sources(root, progress=SILENT_STATUS)
                flux_dataset = prepare_iro_dataset(root)
            with _fake_qwen_execution_provenance(), patch(
                "aigen.generation.image_edit_batch.run_image_edit_batch",
                side_effect=_fake_qwen_batch,
            ):
                generate_qwen_sources(
                    root,
                    config_path,
                    progress=SILENT_STATUS,
                )
            first = prepare_iro_qwen_dataset(root)
            second = prepare_iro_qwen_dataset(root)

            self.assertFalse(flux_dataset["reused"])
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertTrue((root / "dataset").is_dir())
            self.assertTrue(
                (root / QWEN_CORPUS_DATASET_DIRECTORY).is_dir()
            )
            self.assertNotEqual(
                (root / "dataset").resolve(),
                (root / QWEN_CORPUS_DATASET_DIRECTORY).resolve(),
            )

def _fingerprinted(payload: dict[str, object]) -> dict[str, object]:
    return {
        **payload,
        "fingerprint": hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


FAKE_MODEL_PROVENANCE = _fingerprinted({
    "format": "aigen.model-artifacts.v1",
    "components": [
        {
            "name": "test model",
            "files": [
                {
                    "path": "weights.safetensors",
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                }
            ],
        }
    ],
})


FAKE_RUNTIME_PROVENANCE = _fingerprinted(
    {
        "format": "aigen.python-runtime.v1",
        "python": "3.12.0",
        "distributions": [
            {
                "name": "test-runtime",
                "version": "1.0",
                "direct_url": None,
            }
        ],
    }
)


def _fake_execution_provenance(
    *,
    model: dict[str, object] = FAKE_MODEL_PROVENANCE,
    runtime: dict[str, object] = FAKE_RUNTIME_PROVENANCE,
) -> ExitStack:
    stack = ExitStack()
    stack.enter_context(
        patch(
            "aigen.pix2pix.flux_source_corpus.flux2_klein_model_provenance",
            return_value=model,
        )
    )
    stack.enter_context(
        patch(
            "aigen.pix2pix.flux_source_corpus.flux2_klein_runtime_provenance",
            return_value=runtime,
        )
    )
    return stack


def _fake_qwen_execution_provenance(
    *,
    model: dict[str, object] = FAKE_MODEL_PROVENANCE,
    runtime: dict[str, object] = FAKE_RUNTIME_PROVENANCE,
) -> ExitStack:
    stack = ExitStack()
    stack.enter_context(
        patch(
            "aigen.pix2pix.qwen_source_corpus.qwen_2511_lightning_model_provenance",
            return_value=model,
        )
    )
    stack.enter_context(
        patch(
            "aigen.pix2pix.qwen_source_corpus.qwen_2511_lightx2v_runtime_provenance",
            return_value=runtime,
        )
    )
    return stack


def _write_fake_flux_source_set(
    temporary: Path,
    root: Path,
) -> Path:
    _, selected, selection = load_iro_selection(root)
    source_set_dir = temporary / "reviewed-source-set"
    source_set_dir.mkdir()
    records = [
        {
            "id": str(record["id"]),
            "prompt": f"Redraw sprite {record['id']} as a smooth anime illustration.",
            "author": "test-author",
            "seed": int(record["flux_seed"]) + 1000,
            "target_sha256": str(record["target_sha256"]),
        }
        for record in selected
    ]
    records_path = source_set_dir / "records.jsonl"
    write_json_records(records_path, records)
    reviews = [
        {
            "id": str(record["id"]),
            "prompt_sha256": hashlib.sha256(
                source_record["prompt"].encode("utf-8")
            ).hexdigest(),
            "target_sha256": str(record["target_sha256"]),
            "verdict": "pass",
            "reviewer": "test-reviewer",
        }
        for record, source_record in zip(selected, records, strict=True)
    ]
    reviews_path = source_set_dir / "prompt-reviews.jsonl"
    write_json_records(reviews_path, reviews)
    manifest_path = source_set_dir / "source-set.json"
    write_json(
        manifest_path,
        {
            "format": "aigen.pix2pix.flux-source-set.v1",
            "name": "reviewed-test",
            "selection_sha256": selection["selected_sha256"],
            "prompt_guide_sha256": sha256_file(
                PROJECT_ROOT / "docs" / "image-edit-prompting.md"
            ),
            "records": records_path.name,
            "records_sha256": sha256_file(records_path),
            "record_count": len(records),
            "prompt_reviews": reviews_path.name,
            "prompt_reviews_sha256": sha256_file(reviews_path),
        },
    )
    loaded = load_flux_source_set(
        manifest_path,
        selected=selected,
        selection=selection,
    )
    if len(loaded.records) != len(selected):
        raise AssertionError("fake FLUX source set did not round-trip")
    return manifest_path


def _fake_flux_source_set_execution(
    state: dict[str, object],
    *,
    fail: bool = False,
) -> ExitStack:
    module = types.ModuleType("aigen.generation.flux2_klein")

    class FakeFlux2KleinError(RuntimeError):
        pass

    def encode_prompts(*, prompts: object, progress: object) -> tuple[object, float]:
        del progress
        ordered = tuple(dict.fromkeys(prompts))
        state["encoded_prompts"] = ordered
        return {prompt: prompt for prompt in ordered}, 0.0

    class FakeFlux2KleinSession:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            state["session_count"] = int(state.get("session_count", 0)) + 1

        def generate(
            self,
            *,
            cases: object,
            prompt_embeddings: object,
            progress: object,
        ) -> object:
            del prompt_embeddings, progress
            state["generate_count"] = int(state.get("generate_count", 0)) + 1
            outputs = []
            for index, case in enumerate(cases):
                request = case.outputs[0]
                Image.new(
                    "RGB",
                    (case.width, case.height),
                    (index, request.seed % 256, 96),
                ).save(request.path)
                if fail:
                    raise FakeFlux2KleinError("injected FLUX source failure")
                outputs.append(
                    SimpleNamespace(
                        case=case.name,
                        seed=request.seed,
                    )
                )
            return SimpleNamespace(
                outputs=tuple(outputs),
                generation_ms=0.0,
                model_load_ms=0.0,
                peak_vram_mb=0,
            )

        def close(self) -> None:
            return None

    module.Flux2KleinError = FakeFlux2KleinError
    module.Flux2KleinSession = FakeFlux2KleinSession
    module.encode_flux2_klein_prompts = encode_prompts
    stack = ExitStack()
    stack.enter_context(
        patch(
            "aigen.pix2pix.flux_source_set_corpus.flux2_klein_model_provenance",
            return_value=FAKE_MODEL_PROVENANCE,
        )
    )
    stack.enter_context(
        patch(
            "aigen.pix2pix.flux_source_set_corpus.flux2_klein_runtime_provenance",
            return_value=FAKE_RUNTIME_PROVENANCE,
        )
    )
    stack.enter_context(
        patch.dict(
            sys.modules,
            {"aigen.generation.flux2_klein": module},
        )
    )
    return stack


def _write_fake_flux_output_audit(
    root: Path,
    name: str,
    *,
    rejected_ids: set[str],
) -> None:
    _, selected, _ = load_iro_selection(root)
    inventory, source_set, _, source_plan_sha256 = (
        load_flux_source_set_inventory(root, name)
    )
    result_sha256 = sha256_file(flux_source_set_result_path(root, name))
    source_root = root / flux_source_set_layout(name).directory
    records = []
    for selected_record, source_record in zip(
        selected,
        source_set.records,
        strict=True,
    ):
        pair_id = str(selected_record["id"])
        rejected = pair_id in rejected_ids
        records.append(
            {
                "id": pair_id,
                "source_sha256": sha256_file(inventory[pair_id]),
                "target_sha256": str(selected_record["target_sha256"]),
                "prompt_sha256": hashlib.sha256(
                    source_record.prompt.encode("utf-8")
                ).hexdigest(),
                "chosen_seed": source_record.seed,
                "tested_seeds": [source_record.seed],
                "verdict": "reject" if rejected else "pass",
                "reviewer": "test-output-reviewer",
                "notes": "pose mismatch" if rejected else "pose and view aligned",
            }
        )
    records_path = source_root / "output-audit-records.jsonl"
    write_json_records(records_path, records)
    write_json(
        source_root / "output-audit.json",
        {
            "format": "aigen.pix2pix.flux-output-audit.v1",
            "source_plan_sha256": source_plan_sha256,
            "source_result_sha256": result_sha256,
            "records": records_path.name,
            "records_sha256": sha256_file(records_path),
            "record_count": len(records),
        },
    )


def _write_fake_qwen_source_config(temporary: Path) -> Path:
    source = (
        PROJECT_ROOT
        / "configs/pix2pix-iro-qwen2511-lightning-source.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["name"] = "pix2pix-test-qwen-source"
    payload["seed_base"] = 9000
    payload["shard_size"] = 4
    path = temporary / "qwen-source.json"
    write_json(path, payload)
    return path


def _fake_qwen_batch(request: object, *, progress: object) -> object:
    del progress
    from aigen.generation.image_edit_batch import (
        ImageEditBatchOutput,
        ImageEditBatchResult,
    )

    outputs = []
    for index, case in enumerate(request.cases):
        Image.new(
            "RGB",
            (case.width, case.height),
            (index, case.seed % 256, 64),
        ).save(case.output_path)
        outputs.append(
            ImageEditBatchOutput(
                case_id=case.id,
                path=case.output_path,
                width=case.width,
                height=case.height,
                seed=case.seed,
            )
        )
    return ImageEditBatchResult(
        backend=request.backend,
        outputs=tuple(outputs),
    )


def _failing_qwen_batch(request: object, *, progress: object) -> object:
    del progress
    from aigen.generation.image_edit_batch import ImageEditBatchError

    first = request.cases[0]
    Image.new(
        "RGB",
        (first.width, first.height),
        (255, 0, 0),
    ).save(first.output_path)
    raise ImageEditBatchError("injected Qwen batch failure")


def _tree_checksums(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _materialize_fake_source_corpus(temporary: Path) -> Path:
    config_payload = json.loads(GATE_CONFIG.read_text(encoding="utf-8"))
    lineages = {
        "novice": "train",
        "taekwon": "validation",
        "ninja": "test",
    }
    job_ids = {0, 4046, 25}
    config_payload["name"] = "pix2pix-test-corpus"
    config_payload["lineage_splits"] = lineages
    config_payload["lineage_pair_quotas"] = [
        {
            "lineage": lineage,
            "split": split,
            "female": 4,
            "male": 4,
        }
        for lineage, split in lineages.items()
    ]
    config_payload["jobs"] = [
        job for job in config_payload["jobs"] if job["id"] in job_ids
    ]
    config_payload["actions"] = config_payload["actions"][:8]
    actions = [action["base"] for action in config_payload["actions"]]
    head_ranges = {
        "train": range(1, 9),
        "validation": range(9, 17),
        "test": range(17, 25),
    }
    config_payload["split_axis_quotas"] = {
        split: {
            "directions": {str(direction): 1 for direction in range(8)},
            "head_palettes": {
                str(palette): 1 for palette in config_payload["head_palettes"]
            },
            "actions": {
                str(action): 1 for action in actions
            },
            "heads_by_species": {
                "human": list(head_ranges[split]),
            },
        }
        for split in lineages.values()
    }
    config_payload["flux"]["width"] = 64
    config_payload["flux"]["height"] = 64
    config_payload["flux"]["shard_size"] = 4

    config_path = temporary / "config.json"
    write_json(config_path, config_payload)
    root = temporary / "corpus"
    plan_iro_corpus(config_path, root)
    config, requests, plan = load_iro_plan(root)

    selection_dir = root / "selection"
    targets_dir = selection_dir / "targets"
    targets_dir.mkdir(parents=True)
    selected = []
    for index, request in enumerate(requests):
        pair_id = str(request["id"])
        target_path = targets_dir / f"{pair_id}.png"
        Image.new("RGB", (128, 128), (index, index, index)).save(target_path)
        selected.append(
            {
                **request,
                "renderer_frame_index": 0,
                "duration_ms": 100.0,
                "renderer_pixel_sha256": hashlib.sha256(
                    str(index).encode("ascii")
                ).hexdigest(),
                "flux_seed": config.flux.seed_base + index,
                "target": f"selection/targets/{pair_id}.png",
                "target_sha256": sha256_file(target_path),
            }
        )
    write_json_records(selection_dir / "selected.jsonl", selected)
    write_json(
        selection_dir / "selection.json",
        {
            "format": IRO_SELECTION_FORMAT,
            "name": config.name,
            "config_fingerprint": plan["config_fingerprint"],
            "plan_requests_sha256": plan["requests_sha256"],
            "selected": "selected.jsonl",
            "selected_sha256": sha256_file(selection_dir / "selected.jsonl"),
            "pair_count": len(selected),
            "split_counts": _split_counts(selected),
            "lineage_counts": _lineage_counts(selected),
            "identity_count": len({record["identity"] for record in selected}),
        },
    )
    config, loaded, selection = load_iro_selection(root)
    _, source_plan_sha256 = _load_or_create_source_plan(
        root,
        config=config,
        selected=loaded,
        selection=selection,
        model_provenance=FAKE_MODEL_PROVENANCE,
        runtime_provenance=FAKE_RUNTIME_PROVENANCE,
    )
    source_root = root / FLUX_SOURCE_DIRECTORY
    for shard_index, cases in expected_source_shards(
        loaded,
        config.flux.shard_size,
    ):
        shard_dir = source_root / "shards" / f"shard-{shard_index:05d}"
        raw_dir = shard_dir / "raw"
        raw_dir.mkdir(parents=True)
        outputs = []
        for case_index, case in enumerate(cases):
            pair_id = str(case["id"])
            output_path = raw_dir / f"{pair_id}.png"
            Image.new(
                "RGB",
                (config.flux.width, config.flux.height),
                (shard_index, case_index, 32),
            ).save(output_path)
            outputs.append(
                {
                    "id": pair_id,
                    "path": f"raw/{pair_id}.png",
                    "sha256": sha256_file(output_path),
                    "size_bytes": output_path.stat().st_size,
                    "mode": "RGB",
                    "width": config.flux.width,
                    "height": config.flux.height,
                    "seed": int(case["flux_seed"]),
                }
            )
        write_json(
            shard_dir / "shard.json",
            {
                "format": FLUX_SOURCE_SHARD_FORMAT,
                "shard_index": shard_index,
                "source_plan_sha256": source_plan_sha256,
                "generation_ms": 0.0,
                "model_load_ms": 0.0,
                "peak_vram_mb": 0,
                "outputs": outputs,
            },
        )
    return root


if __name__ == "__main__":
    unittest.main()
