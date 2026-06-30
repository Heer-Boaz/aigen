from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from aigen.cli import main
from aigen.progress import SILENT_STATUS

from aigen.models.downloads import (
    ModelDownloadDependencyError,
    ModelDownloadProcessError,
    download_models,
    load_download_manifest,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def download_manifest_payload() -> dict[str, object]:
    return {
        "target_hardware": {
            "gpu": "RTX 5070 Ti",
            "vram_gb": 16,
        },
        "downloads": [
            {
                "name": "flux-kontext-4bit-fp4",
                "repo_id": "eramth/flux-kontext-4bit-fp4",
                "local_path": "diffusers/eramth/flux-kontext-4bit-fp4",
                "revision": "499964b43d54eda6ca7e21c346efe18e2a1cdad8",
                "include": ["*.json", "*.safetensors"],
                "exclude": ["*.md"],
            }
        ],
    }


class ModelDownloadTests(unittest.TestCase):
    def test_keyframe_generation_manifest_downloads_kontext_and_controlnet(self) -> None:
        manifest = load_download_manifest(Path("model_sources/keyframe_generation_kontext_controlnet.json"))
        kontext, controlnet = manifest.downloads

        self.assertEqual(kontext.name, "flux-kontext-4bit-fp4")
        self.assertEqual(kontext.repo_id, "eramth/flux-kontext-4bit-fp4")
        self.assertEqual(kontext.local_path, "diffusers/eramth/flux-kontext-4bit-fp4")
        self.assertEqual(controlnet.repo_id, "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0")
        self.assertIn("diffusion_pytorch_model.safetensors", controlnet.include)

    def test_keyframe_generation_manifest_downloads_blackwell_fp4_transformer(self) -> None:
        manifest = load_download_manifest(Path("model_sources/keyframe_generation_nunchaku_transformer.json"))
        download = manifest.downloads[0]

        self.assertEqual(download.name, "nunchaku-flux1-kontext-dev-fp4-blackwell")
        self.assertEqual(download.repo_id, "nunchaku-tech/nunchaku-flux.1-kontext-dev")
        self.assertEqual(
            download.local_path,
            "nunchaku/nunchaku-tech/nunchaku-flux.1-kontext-dev",
        )
        self.assertEqual(download.revision, "70dff7728491f3016e256137e8f7d87812af0b4f")
        self.assertIn("svdq-fp4_r32-flux.1-kontext-dev.safetensors", download.include)

    def test_lora_control_audit_manifest_downloads_plain_flux_transformer(self) -> None:
        manifest = load_download_manifest(Path("model_sources/lora_control_audit_nunchaku_transformer.json"))
        download = manifest.downloads[0]

        self.assertEqual(download.name, "nunchaku-flux1-dev-fp4-blackwell")
        self.assertEqual(download.repo_id, "nunchaku-ai/nunchaku-flux.1-dev")
        self.assertEqual(download.local_path, "nunchaku/nunchaku-ai/nunchaku-flux.1-dev")
        self.assertEqual(download.revision, "1a3d3f78b545e33a0897da2101150292ebbd158a")
        self.assertIn("svdq-fp4_r32-flux.1-dev.safetensors", download.include)

    def test_keyframe_judge_manifest_downloads_qwen_vl_7b(self) -> None:
        manifest = load_download_manifest(Path("model_sources/keyframe_judge_qwen2_5_vl_7b.json"))
        download = manifest.downloads[0]

        self.assertEqual(download.name, "qwen2.5-vl-7b-instruct-keyframe-judge")
        self.assertEqual(download.repo_id, "Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(download.local_path, "vlm/Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(download.revision, "cc594898137f460bfe9f0759e9844b3ce807cfb5")
        self.assertIn("*.safetensors", download.include)

    def test_keyframe_pose_manifest_downloads_dwpose_onnx_models(self) -> None:
        manifest = load_download_manifest(Path("model_sources/keyframe_pose_dwpose_onnx.json"))
        download = manifest.downloads[0]

        self.assertEqual(download.name, "dwpose-onnx-keyframe-scorer")
        self.assertEqual(download.repo_id, "yzd-v/DWPose")
        self.assertEqual(download.local_path, "annotators/yzd-v/DWPose")
        self.assertEqual(download.revision, "1a7144101628d69ee7a3768d1ee3a094070dc388")
        self.assertIn("yolox_l.onnx", download.include)
        self.assertIn("dw-ll_ucoco_384.onnx", download.include)

    def test_keyframe_segmentation_manifest_downloads_sam_checkpoint(self) -> None:
        manifest = load_download_manifest(Path("model_sources/keyframe_segmentation_sam_vit_b.json"))
        download = manifest.downloads[0]

        self.assertEqual(download.name, "sam-vit-b-keyframe-segmentation")
        self.assertEqual(download.repo_id, "ybelkada/segment-anything")
        self.assertEqual(download.local_path, "segmentation/ybelkada/segment-anything")
        self.assertEqual(download.revision, "7790786db131bcdc639f24a915d9f2c331d843ee")
        self.assertIn("checkpoints/sam_vit_b_01ec64.pth", download.include)

    def test_keyframe_grounding_manifest_downloads_grounding_dino(self) -> None:
        manifest = load_download_manifest(Path("model_sources/keyframe_grounding_dino.json"))
        download = manifest.downloads[0]

        self.assertEqual(download.name, "grounding-dino-base-keyframe-polish-grounding")
        self.assertEqual(download.repo_id, "IDEA-Research/grounding-dino-base")
        self.assertEqual(download.revision, "12bdfa3120f3e7ec7b434d90674b3396eccf88eb")
        self.assertEqual(download.local_path, "grounding/IDEA-Research/grounding-dino-base")
        self.assertIn("*.safetensors", download.include)
        self.assertIn("tokenizer*", download.include)

    def test_keyframe_grounding_manifest_downloads_florence2(self) -> None:
        manifest = load_download_manifest(Path("model_sources/keyframe_grounding_florence2.json"))
        download = manifest.downloads[0]

        self.assertEqual(download.name, "florence-2-large-ft-keyframe-polish-grounding")
        self.assertEqual(download.repo_id, "florence-community/Florence-2-large-ft")
        self.assertEqual(download.revision, "26b734a54fdfbf9c398351eedfabb7f27fc470b7")
        self.assertEqual(download.local_path, "grounding/florence-community/Florence-2-large-ft")
        self.assertIn("*.json", download.include)
        self.assertIn("*.safetensors", download.include)

    def test_dry_run_plans_hub_download_without_network_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "downloads.json"
            models_root = root / "models"
            write_json(manifest_path, download_manifest_payload())

            with patch.dict(sys.modules, {"huggingface_hub": None}):
                result = download_models(
                    load_download_manifest(manifest_path),
                    models_root,
                    dry_run=True,
                    progress=SILENT_STATUS,
                ).to_json()

        self.assertTrue(Path(result["models_root"]).is_absolute())
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["target_hardware"]["vram_gb"], 16)
        self.assertEqual(result["downloads"][0]["status"], "planned")
        self.assertEqual(result["downloads"][0]["repo_id"], "eramth/flux-kontext-4bit-fp4")
        self.assertEqual(result["downloads"][0]["revision"], "499964b43d54eda6ca7e21c346efe18e2a1cdad8")
        self.assertEqual(
            result["downloads"][0]["local_dir"],
            str(models_root.resolve() / "diffusers" / "eramth" / "flux-kontext-4bit-fp4"),
        )

    def test_download_uses_hugging_face_snapshot_download(self) -> None:
        calls: list[dict[str, object]] = []
        hub = types.ModuleType("huggingface_hub")

        def snapshot_download(**kwargs: object) -> str:
            calls.append(kwargs)
            return str(kwargs["local_dir"])

        hub.snapshot_download = snapshot_download

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "downloads.json"
            models_root = root / "models"
            write_json(manifest_path, download_manifest_payload())

            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                result = download_models(
                    load_download_manifest(manifest_path),
                    models_root,
                    progress=SILENT_STATUS,
                ).to_json()

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["repo_id"], "eramth/flux-kontext-4bit-fp4")
        self.assertEqual(calls[0]["repo_type"], "model")
        self.assertEqual(calls[0]["revision"], "499964b43d54eda6ca7e21c346efe18e2a1cdad8")
        self.assertEqual(calls[0]["allow_patterns"], ["*.json", "*.safetensors"])
        self.assertEqual(calls[0]["ignore_patterns"], ["*.md"])
        self.assertEqual(result["downloads"][0]["status"], "downloaded")

    def test_rejects_download_path_outside_models_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "downloads.json"
            payload = download_manifest_payload()
            payload["downloads"][0]["local_path"] = "../outside"
            write_json(manifest_path, payload)

            with self.assertRaisesRegex(ValueError, "local_path must stay inside models_root"):
                load_download_manifest(manifest_path)

    def test_reports_missing_hugging_face_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "downloads.json"
            write_json(manifest_path, download_manifest_payload())

            with patch.dict(sys.modules, {"huggingface_hub": None}):
                with self.assertRaises(ModelDownloadDependencyError):
                    download_models(
                        load_download_manifest(manifest_path),
                        root / "models",
                        progress=SILENT_STATUS,
                    )

    def test_reports_hub_download_failure(self) -> None:
        hub = types.ModuleType("huggingface_hub")

        def snapshot_download(**kwargs: object) -> str:
            raise RuntimeError("no access")

        hub.snapshot_download = snapshot_download

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "downloads.json"
            write_json(manifest_path, download_manifest_payload())

            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                with self.assertRaises(ModelDownloadProcessError) as error:
                    download_models(
                        load_download_manifest(manifest_path),
                        root / "models",
                        progress=SILENT_STATUS,
                    )

        payload = error.exception.to_json()
        self.assertEqual(payload["repo_id"], "eramth/flux-kontext-4bit-fp4")
        self.assertIn("no access", payload["message"])

    def test_cli_reports_hub_download_failure(self) -> None:
        hub = types.ModuleType("huggingface_hub")

        def snapshot_download(**kwargs: object) -> str:
            raise RuntimeError("no access")

        hub.snapshot_download = snapshot_download

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "downloads.json"
            write_json(manifest_path, download_manifest_payload())

            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                with contextlib.redirect_stderr(io.StringIO()):
                    exit_code = main(
                        [
                            "models",
                            "download",
                            "--manifest",
                            str(manifest_path),
                            "--models-root",
                            str(root / "models"),
                            "--compact",
                        ]
                    )

        self.assertEqual(exit_code, 1)

    def test_reports_missing_hugging_face_dependency_from_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "downloads.json"
            write_json(manifest_path, download_manifest_payload())

            with patch.dict(sys.modules, {"huggingface_hub": None}):
                with contextlib.redirect_stderr(io.StringIO()):
                    exit_code = main(
                        [
                            "models",
                            "download",
                            "--manifest",
                            str(manifest_path),
                            "--models-root",
                            str(root / "models"),
                            "--compact",
                        ]
                    )

        self.assertEqual(exit_code, 1)

    def test_download_result_keeps_hub_coordinates(self) -> None:
        hub = types.ModuleType("huggingface_hub")

        def snapshot_download(**kwargs: object) -> str:
            return str(kwargs["local_dir"])

        hub.snapshot_download = snapshot_download

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "downloads.json"
            write_json(manifest_path, download_manifest_payload())

            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                result = download_models(
                    load_download_manifest(manifest_path),
                    root / "models",
                    progress=SILENT_STATUS,
                ).to_json()

        self.assertEqual(result["downloads"][0]["name"], "flux-kontext-4bit-fp4")
        self.assertEqual(result["downloads"][0]["repo_type"], "model")
        self.assertEqual(result["downloads"][0]["local_path"], "diffusers/eramth/flux-kontext-4bit-fp4")

    def test_download_result_uses_absolute_models_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "downloads.json"
            write_json(manifest_path, download_manifest_payload())

            result = download_models(
                load_download_manifest(manifest_path),
                root / "models",
                dry_run=True,
                progress=SILENT_STATUS,
            ).to_json()

        self.assertEqual(result["models_root"], str((root / "models").resolve()))

    def test_rejects_unsupported_repo_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "downloads.json"
            payload = download_manifest_payload()
            payload["downloads"][0]["repo_type"] = "dataset"
            write_json(manifest_path, payload)

            with self.assertRaisesRegex(ValueError, "unsupported download.repo_type"):
                load_download_manifest(manifest_path)

    def test_cli_models_download_dry_run_writes_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "downloads.json"
            output_path = root / "downloads.locked.json"
            write_json(manifest_path, download_manifest_payload())

            exit_code = main(
                [
                    "models",
                    "download",
                    "--manifest",
                    str(manifest_path),
                    "--models-root",
                    str(root / "models"),
                    "--output",
                    str(output_path),
                    "--dry-run",
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["downloads"][0]["repo_id"], "eramth/flux-kontext-4bit-fp4")


if __name__ == "__main__":
    unittest.main()
