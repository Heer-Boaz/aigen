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
        "schema_version": 1,
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
    def test_pose_control_manifest_downloads_base_and_controlnet(self) -> None:
        manifest = load_download_manifest(Path("model_sources/pose_control_pipeline.json"))
        base, controlnet = manifest.downloads

        self.assertEqual(base.name, "flux1-dev-bf16-diffusers")
        self.assertEqual(base.repo_id, "black-forest-labs/FLUX.1-dev")
        self.assertEqual(base.local_path, "diffusers/black-forest-labs/FLUX.1-dev-bf16")
        self.assertIn("transformer/*", base.include)
        self.assertIn("text_encoder_2/*", base.include)
        self.assertNotIn("flux1-dev.safetensors", base.include)
        self.assertEqual(controlnet.name, "flux1-dev-controlnet-union-pro-2.0")
        self.assertEqual(controlnet.repo_id, "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0")
        self.assertEqual(
            controlnet.local_path,
            "diffusers/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        )
        self.assertIn("diffusion_pytorch_model.safetensors", controlnet.include)

    def test_pose_control_4bit_manifest_downloads_base_and_controlnet(self) -> None:
        manifest = load_download_manifest(Path("model_sources/pose_control_pipeline_4bit.json"))
        base, controlnet = manifest.downloads

        self.assertEqual(base.name, "flux1-dev-bnb-4bit-diffusers")
        self.assertEqual(base.repo_id, "diffusers/FLUX.1-dev-bnb-4bit")
        self.assertEqual(base.local_path, "diffusers/black-forest-labs/FLUX.1-dev-bnb-4bit")
        self.assertIn("transformer/*", base.include)
        self.assertIn("text_encoder_2/*", base.include)
        self.assertEqual(controlnet.name, "flux1-dev-controlnet-union-pro-2.0")
        self.assertEqual(controlnet.repo_id, "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0")
        self.assertIn("diffusion_pytorch_model.safetensors", controlnet.include)

    def test_kontext_pose_control_manifest_downloads_kontext_and_controlnet(self) -> None:
        manifest = load_download_manifest(Path("model_sources/kontext_pose_control_pipeline_4bit.json"))
        kontext, controlnet = manifest.downloads

        self.assertEqual(kontext.name, "flux-kontext-4bit-fp4")
        self.assertEqual(kontext.repo_id, "eramth/flux-kontext-4bit-fp4")
        self.assertEqual(kontext.local_path, "diffusers/eramth/flux-kontext-4bit-fp4")
        self.assertEqual(controlnet.repo_id, "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0")
        self.assertIn("diffusion_pytorch_model.safetensors", controlnet.include)

    def test_full_bf16_offload_manifest_downloads_only_diffusers_tree(self) -> None:
        manifest = load_download_manifest(Path("model_sources/character_pipeline_full_bf16_offload.json"))
        download = manifest.downloads[0]

        self.assertEqual(download.name, "flux1-kontext-dev-bf16-diffusers")
        self.assertEqual(download.repo_id, "black-forest-labs/FLUX.1-Kontext-dev")
        self.assertEqual(download.local_path, "diffusers/black-forest-labs/FLUX.1-Kontext-dev-bf16")
        self.assertIn("transformer/*", download.include)
        self.assertIn("text_encoder_2/*", download.include)
        self.assertNotIn("flux1-kontext-dev.safetensors", download.include)

    def test_nunchaku_kontext_manifest_downloads_blackwell_fp4_transformer(self) -> None:
        manifest = load_download_manifest(Path("model_sources/nunchaku_kontext_pipeline_fp4.json"))
        download = manifest.downloads[0]

        self.assertEqual(download.name, "nunchaku-flux1-kontext-dev-fp4-blackwell")
        self.assertEqual(download.repo_id, "nunchaku-tech/nunchaku-flux.1-kontext-dev")
        self.assertEqual(
            download.local_path,
            "nunchaku/nunchaku-tech/nunchaku-flux.1-kontext-dev",
        )
        self.assertEqual(download.revision, "70dff7728491f3016e256137e8f7d87812af0b4f")
        self.assertIn("svdq-fp4_r32-flux.1-kontext-dev.safetensors", download.include)

    def test_keyframe_judge_manifest_downloads_qwen_vl_7b(self) -> None:
        manifest = load_download_manifest(Path("model_sources/keyframe_judge_qwen2_5_vl_7b.json"))
        download = manifest.downloads[0]

        self.assertEqual(download.name, "qwen2.5-vl-7b-instruct-keyframe-judge")
        self.assertEqual(download.repo_id, "Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(download.local_path, "vlm/Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(download.revision, "cc594898137f460bfe9f0759e9844b3ce807cfb5")
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
