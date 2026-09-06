import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from aigen.generation import flux2_klein_artifacts
from aigen.generation.uso_flux1 import uso_flux1_model_paths
from aigen.model_artifacts import ModelArtifactComponent, model_artifact_stat_revision
from aigen.workflow_cache import build_node_signature
from aigen.workflow_graph import ImageEditConfig, ImageEditNode
from aigen.workflow_provenance import _path_inventory_revision, _postprocess_provenance, workflow_node_provenance


class ModelArtifactIdentityTests(unittest.TestCase):
    def test_segmentation_cache_uses_only_the_selected_engine_runtime(self):
        from importlib.metadata import PackageNotFoundError
        from aigen import keyframe_segmentation
        from aigen.workflow_graph import SamSegmentConfig, SamSegmentNode

        versions = dict.fromkeys(("torch", "torchvision", "segment-anything", "transformers",
                                  "safetensors", "opencv-python-headless", "numpy", "scipy", "pillow"), "1")

        def distribution_record(name):
            if name not in versions:
                raise PackageNotFoundError(name)
            return {"name": name, "version": versions[name], "direct_url": None}

        with TemporaryDirectory() as directory:
            model = Path(directory) / "weights"
            model.write_bytes(b"checkpoint")
            nodes = tuple(SamSegmentNode(id=engine, title=engine, config=SamSegmentConfig(
                engine=engine, device="cuda" if engine == "anime" else "cpu"))
                for engine in ("sam1", "sam2", "anime"))
            with patch.multiple(keyframe_segmentation, DEFAULT_SAM_CHECKPOINT=model,
                                DEFAULT_SAM2_MODEL=model, DEFAULT_ANIME_SEGMENTATION_MODEL=model), patch(
                "aigen.runtime_provenance._distribution_record", side_effect=distribution_record,
            ):
                # Native SAM works without installing the unrelated ONNX engine.
                workflow_node_provenance(nodes[0])
                workflow_node_provenance(nodes[1])
                versions["onnxruntime-gpu"] = "1"
                before = tuple(map(workflow_node_provenance, nodes))
                versions["segment-anything"] = "2"
                changed = tuple(map(workflow_node_provenance, nodes))
                self.assertNotEqual(before[0], changed[0])
                self.assertEqual(before[1:], changed[1:])
                versions["opencv-python-headless"] = "2"
                current = tuple(map(workflow_node_provenance, nodes))
                self.assertEqual(changed[:2], current[:2])
                self.assertNotEqual(changed[2], current[2])

    def test_qwen_and_vosr_cache_tracks_active_local_checkpoints(self):
        from aigen.generation.qwen_image_edit_lightx2v import (
            QWEN_IMAGE_EDIT_LIGHTX2V_PROFILES, LIGHTX2V_QWEN_EDIT_2511_PROFILE,
        )
        from aigen.generation import vosr_backend

        with TemporaryDirectory() as directory:
            root = Path(directory)
            for part in ("vae", "scheduler", "text_encoder", "processor", "tokenizer", "transformer"):
                (root / part).mkdir()
                (root / part / "config.json").write_text("{}")
            (root / "model_index.json").write_text("{}")
            weights = root / "weights.safetensors"
            weights.write_bytes(b"first checkpoint")
            profile = replace(QWEN_IMAGE_EDIT_LIGHTX2V_PROFILES[LIGHTX2V_QWEN_EDIT_2511_PROFILE],
                              base_model=str(root), transformer_model=weights, conditioner_model=root / "text_encoder")
            node = ImageEditNode(id="edit", title="Edit", config=ImageEditConfig(backend="qwen-image-edit-2511-lightning"))
            with patch.dict(QWEN_IMAGE_EDIT_LIGHTX2V_PROFILES, {LIGHTX2V_QWEN_EDIT_2511_PROFILE: profile}), patch.multiple(
                vosr_backend, VOSR_CHECKPOINT=root / "transformer", VOSR_VAE=root / "vae", VOSR_DINOV2_WEIGHTS=weights,
            ):
                def revisions():
                    return workflow_node_provenance(node), _postprocess_provenance(vosr_backend.VOSR_POSTPROCESS_NAME)

                before = revisions()
                self.assertEqual(revisions(), before)
                weights.write_bytes(b"replacement checkpoint")
                self.assertTrue(all(old != new for old, new in zip(before, revisions(), strict=True)))
                before = revisions()
                (root / "vae" / "config.json").write_text('{"changed":true}')
                self.assertTrue(all(old != new for old, new in zip(before, revisions(), strict=True)))
                before = workflow_node_provenance(node)
                (root / "text_encoder" / "added-token.json").write_text("{}")
                self.assertNotEqual(workflow_node_provenance(node), before)

    def test_klein_inventory_detects_added_conditioner_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            transformer = root / "transformer.safetensors"
            transformer.write_bytes(b"transformer")
            for part in ("vae", "scheduler", "conditioner"):
                (root / part).mkdir()
                (root / part / "config.json").write_text("{}")
            with patch.multiple(
                flux2_klein_artifacts,
                FLUX2_KLEIN_TRANSFORMER=transformer,
                FLUX2_KLEIN_MODEL_ROOT=root,
                FLUX2_KLEIN_TEXT_ENCODER=root / "conditioner",
            ):
                node = ImageEditNode(id="edit", title="Edit", config=ImageEditConfig(backend="flux2-klein"))
                before = workflow_node_provenance(node)
                self.assertEqual(workflow_node_provenance(node), before)
                added = root / "conditioner" / "special_tokens_map.json"
                added.write_text('{"eos_token": "<end>"}')
                self.assertNotEqual(workflow_node_provenance(node), before)
                self.assertIn(added, flux2_klein_artifacts.flux2_klein_model_artifacts()[2].files)

    def test_existing_model_replacement_is_visible_in_the_same_process(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "model.safetensors"
            weights.write_bytes(b"first weights")
            component = ModelArtifactComponent("model", root, (weights,))

            def revisions():
                return (
                    _path_inventory_revision(weights),
                    _path_inventory_revision(root),
                    model_artifact_stat_revision(component),
                )

            before = revisions()
            self.assertEqual(revisions(), before)
            weights.write_bytes(b"replacement weights")
            for old, new in zip(before, revisions(), strict=True):
                self.assertNotEqual(old, new)

    def test_uso_cache_tracks_all_worker_model_paths_and_processor_files(self):
        node = ImageEditNode(id="edit", title="Edit", config=ImageEditConfig(backend="uso-flux1-dev-fp8"))

        def signature():
            return build_node_signature(
                node_kind=node.kind,
                execution_config=node.config.model_dump(mode="json"),
                inputs={}, source_outputs=None, provenance=workflow_node_provenance(node),
            )

        with TemporaryDirectory() as directory, patch.dict(os.environ, {"AIGEN_MODELS_ROOT": directory}):
            files = []
            for path in uso_flux1_model_paths().values():
                targets = (path,) if path.suffix else (path / "model.safetensors", path / "config.json")
                for target in targets:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"initial artifact")
                    files.append(target)
            before = signature()
            self.assertEqual(signature(), before)
            for path in files:
                with self.subTest(path=path.relative_to(directory)):
                    path.write_bytes(b"changed artifact contents")
                    after = signature()
                    self.assertNotEqual(after, before)
                    self.assertEqual(signature(), after)
                    before = after


if __name__ == "__main__":
    unittest.main()
