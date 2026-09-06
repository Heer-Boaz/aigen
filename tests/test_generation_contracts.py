from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from aigen.generation import flux2_dev_wangp, flux2_klein_artifacts, qwen_image_edit_lightx2v, vosr_backend, image_upscale, uso_flux1
from aigen.generation.image_edit import IMAGE_EDIT_BACKENDS, ImageEditError, ImageEditRequest, resolve_image_edit_request
from aigen.progress import SILENT_STATUS
from aigen.workflow_cache import build_node_signature
from aigen.workflow_graph import ImageEditConfig, ImageEditNode, ImagePostprocessNode
from aigen.workflow_provenance import workflow_node_provenance
from aigen.workflow_templates import postprocess_config_for_model


class GenerationContractTests(unittest.TestCase):
    def test_duplicate_seeds_are_rejected_before_creating_outputs(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "input.png"
            Image.new("RGB", (64, 64)).save(path)
            output = directory / "results"
            for backend in IMAGE_EDIT_BACKENDS:
                with self.subTest(backend=backend), self.assertRaisesRegex(ImageEditError, "duplicate seeds"):
                    resolve_image_edit_request(ImageEditRequest(
                        backend=backend, prompt="Keep the image unchanged.", images=(path,),
                        output_dir=output, seeds=(7, 7), width=512, height=512,
                    ))
            with patch.object(flux2_dev_wangp, "_run_worker") as worker:
                with self.assertRaisesRegex(flux2_dev_wangp.Flux2DevError, "duplicate seeds"):
                    flux2_dev_wangp.generate_flux2_dev_seed_sweep(
                        prompt="Keep the image unchanged.", references=(path,), output=output / "edit.png",
                        width=512, height=512, seeds=(7, 7), steps=4, guidance=4, loras=(), progress=SILENT_STATUS,
                    )
                worker.assert_not_called()
            self.assertFalse(output.exists())

    def test_changed_backend_implementation_invalidates_cache_signature(self):
        cases = (
            (ImageEditNode(id="edit", title="Edit", config=ImageEditConfig(backend="flux2-klein")), flux2_klein_artifacts, "FLUX2_KLEIN_IMPLEMENTATION_REVISION"),
            (ImageEditNode(id="edit", title="Edit", config=ImageEditConfig(backend="qwen-image-edit-2511-base")), qwen_image_edit_lightx2v, "QWEN_EDIT_IMPLEMENTATION_REVISION"),
            (ImageEditNode(id="edit", title="Edit", config=ImageEditConfig(backend="uso-flux1-dev-fp8")), uso_flux1, "USO_IMPLEMENTATION_REVISION"),
            (ImagePostprocessNode(id="upscale", title="Upscale", config=postprocess_config_for_model(vosr_backend.VOSR_POSTPROCESS_NAME)), vosr_backend, "VOSR_IMPLEMENTATION_REVISION"),
            (ImagePostprocessNode(id="upscale", title="Upscale", config=postprocess_config_for_model("illustrationjanai-dat2")), image_upscale, "IMAGE_UPSCALE_IMPLEMENTATION_REVISION"),
        )
        with patch.object(flux2_klein_artifacts, "flux2_klein_model_artifacts", return_value=()), patch("aigen.workflow_provenance._path_inventory_revision", return_value="model-bytes"):
            for node, module, revision in cases:
                with self.subTest(revision=revision):
                    def signature():
                        return build_node_signature(node_kind=node.kind, execution_config=node.config.model_dump(mode="json"),
                                                    inputs={}, source_outputs=None, provenance=workflow_node_provenance(node))
                    before = signature()
                    with patch.object(module, revision, "changed"):
                        self.assertNotEqual(signature(), before)
                    self.assertEqual(signature(), before)


if __name__ == "__main__":
    unittest.main()
