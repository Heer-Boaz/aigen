from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from aigen.character_qwen_refine import QwenCharacterRefineError, plan_qwen_character_refine, run_qwen_character_refine
from aigen.character_reference_models import CHARACTER_BODY_PROPORTION_SOURCE
from aigen.character_reference_pack import build_character_reference_pack
from aigen.cli import build_parser
from aigen.generation.qwen_image_edit_identity import QwenIdentityPromptConditioningStep, _run_qwen_inpaint_denoise_step
from aigen.progress import SILENT_STATUS


def write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def write_mask(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", size, 0)
    for y in range(size[1] // 4, size[1] // 2):
        for x in range(size[0] // 4, size[0] // 2):
            image.putpixel((x, y), 255)
    image.save(path)


def write_identity_profile(
    path: Path,
    *,
    pack_path: Path,
    reference_roles: dict[str, str],
) -> None:
    payload = {
        "status": "completed",
        "kind": "character-identity-profile",
        "character_id": "subject",
        "source_reference_pack": pack_path.as_posix(),
        "identity": {
            "hair": "model-extracted hair fact",
            "eyes": "model-extracted eye fact",
            "style": "model-extracted style fact",
        },
        "body_proportion": {
            "chest_size": "model-extracted chest-size fact",
            "build": "model-extracted build fact",
            "shoulder_width": "model-extracted shoulder-width fact",
            "waist": "model-extracted waist fact",
            "hip_skirt_silhouette": "model-extracted hip/skirt silhouette fact",
            "side_body_thickness": "model-extracted side-body-thickness fact",
            "leg_proportion": "model-extracted leg-proportion fact",
            "skirt_back_shape": "model-extracted back-shape fact",
            "do_not_change": ["model-extracted body invariant"],
            "evidence_refs": list(reference_roles),
        },
        "body_proportion_source": CHARACTER_BODY_PROPORTION_SOURCE,
        "reference_roles": reference_roles,
        "optional_missing_refs": ["body_shape"],
        "must_preserve": ["model-extracted visual invariant"],
        "avoid": ["model-extracted visual drift"],
        "parser": {"id": "fake-parser"},
        "output": {"identity_profile": path.as_posix()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_region_plan(path: Path, *, source: Path, mask: Path, region_name: str) -> None:
    payload = {
        "status": "completed",
        "kind": "character-region-plan",
        "image": {"path": source.as_posix(), "sha256": "unused", "mode": "RGB", "width": 64, "height": 96},
        "regions": [
            {
                "name": region_name,
                "prompt": "visible local detail",
                "grounding": {"source": "florence2", "box": [1, 2, 3, 4], "score": 0.9},
                "segmentation": {
                    "method": "florence2-box-to-sam2-mask",
                    "model": "sam2",
                    "mask": {"path": mask.as_posix(), "sha256": "unused", "mode": "L", "width": 64, "height": 96},
                    "overlay": {"path": mask.as_posix(), "sha256": "unused", "mode": "RGB", "width": 64, "height": 96},
                },
            }
        ],
        "models": {},
        "output": {"result": path.as_posix()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class CharacterQwenRefineTests(unittest.TestCase):
    def test_qwen_edit_refine_cli_accepts_mask_and_model_alias(self) -> None:
        args = build_parser().parse_args(
            [
                "characters",
                "qwen-edit-refine",
                "--pack",
                "references/reference_pack.json",
                "--image",
                "runs/candidate.png",
                "--mask",
                "runs/mask.png",
                "--instruction",
                "repair the local detail",
                "--model",
                "nunchaku-qwen-edit-2509-r32-4step",
                "--output-dir",
                "runs/refine",
            ]
        )

        self.assertEqual(args.characters_command, "qwen-edit-refine")
        self.assertEqual(args.profile, "nunchaku-qwen-edit-2509-r32-4step")

    def test_qwen_refine_plan_uses_region_mask_and_identity_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs = {
                "side_ref": root / "side.png",
                "portrait_ref": root / "portrait.png",
                "front_ref": root / "front.png",
            }
            write_image(refs["side_ref"], (64, 96), (50, 50, 200))
            write_image(refs["portrait_ref"], (64, 64), (50, 200, 50))
            write_image(refs["front_ref"], (64, 96), (200, 50, 50))
            build_character_reference_pack(
                character_id="subject",
                references=refs,
                output_dir=root / "references",
                overwrite=False,
            )
            pack_path = root / "references" / "reference_pack.json"
            write_identity_profile(
                root / "references" / "identity_profile.json",
                pack_path=pack_path,
                reference_roles={
                    "side_ref": "side",
                    "portrait_ref": "portrait",
                    "front_ref": "front",
                },
            )
            source = root / "candidate.png"
            mask = root / "mask.png"
            region_plan = root / "region_plan" / "result.json"
            write_image(source, (64, 96), (20, 20, 20))
            write_mask(mask, (64, 96))
            region_plan.parent.mkdir()
            write_region_plan(region_plan, source=source, mask=mask, region_name="face")

            plan = plan_qwen_character_refine(
                pack_path=pack_path,
                identity_profile_path=None,
                source_image_path=source,
                mask_path=None,
                region_plan_path=region_plan,
                region_name="face",
                instruction="repair the selected local detail",
                candidates=2,
                progress=SILENT_STATUS,
            )

            self.assertEqual(plan["kind"], "qwen-character-refine-plan")
            self.assertEqual(plan["reference_selector"], "static_repair_role_routing_1_to_4_refs_v1")
            self.assertEqual(plan["routed_reference_limit"], 4)
            self.assertEqual(
                plan["available_reference_roles"],
                {
                    "side_ref": "side",
                    "portrait_ref": "portrait",
                    "front_ref": "front",
                },
            )
            self.assertEqual(
                [reference["reference_id"] for reference in plan["references"]],
                ["front_ref", "portrait_ref", "side_ref"],
            )
            self.assertEqual([reference["input_index"] for reference in plan["references"]], [2, 3, 4])
            self.assertTrue(all(reference["purpose"] for reference in plan["references"]))
            self.assertEqual(plan["mask_source"]["type"], "region-plan")
            normalized = plan["normalized_instruction"]
            self.assertEqual(normalized["task"], "identity_refine")
            self.assertEqual(normalized["refs_used"], ["front_ref", "portrait_ref", "side_ref"])
            self.assertEqual(normalized["prompt_image_order"][0]["source"], "selected_image")
            self.assertEqual(normalized["body_proportion_source"], CHARACTER_BODY_PROPORTION_SOURCE)
            self.assertTrue(plan["prompt"].startswith("Apply this structured character refine instruction exactly:\n"))
            prompt_contract = json.loads(plan["prompt"].split("\n", 1)[1])
            self.assertEqual(prompt_contract["repair_instruction"], "repair the selected local detail")
            self.assertEqual(prompt_contract["mask"], {"white": "repainted", "black": "preserved"})
            self.assertEqual(
                [reference["reference_id"] for reference in prompt_contract["reference_routing"][1:]],
                ["front_ref", "portrait_ref", "side_ref"],
            )
            self.assertTrue(all(reference["purpose"] for reference in prompt_contract["reference_routing"][1:]))
            self.assertEqual(
                prompt_contract["identity_profile"]["body_proportion"]["do_not_change"],
                ["model-extracted body invariant"],
            )

    def test_qwen_refine_plan_rejects_ambiguous_mask_source(self) -> None:
        with self.assertRaisesRegex(QwenCharacterRefineError, "either --mask or --region-plan"):
            plan_qwen_character_refine(
                pack_path=Path("references/reference_pack.json"),
                identity_profile_path=None,
                source_image_path=Path("candidate.png"),
                mask_path=Path("mask.png"),
                region_plan_path=Path("region_plan/result.json"),
                region_name="face",
                instruction="repair",
                candidates=1,
                progress=SILENT_STATUS,
            )

    def test_qwen_refine_run_forwards_routed_refs_to_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs = {
                "front_ref": root / "front.png",
                "portrait_ref": root / "portrait.png",
                "side_ref": root / "side.png",
                "back_ref": root / "back.png",
            }
            write_image(refs["front_ref"], (64, 96), (200, 50, 50))
            write_image(refs["portrait_ref"], (64, 64), (50, 200, 50))
            write_image(refs["side_ref"], (64, 96), (50, 50, 200))
            write_image(refs["back_ref"], (64, 96), (120, 120, 120))
            build_character_reference_pack(
                character_id="subject",
                references=refs,
                output_dir=root / "references",
                overwrite=False,
            )
            pack_path = root / "references" / "reference_pack.json"
            write_identity_profile(
                root / "references" / "identity_profile.json",
                pack_path=pack_path,
                reference_roles={
                    "front_ref": "front",
                    "portrait_ref": "portrait",
                    "side_ref": "side",
                    "back_ref": "back",
                },
            )
            source = root / "candidate.png"
            mask = root / "mask.png"
            output_dir = root / "refine"
            write_image(source, (64, 96), (20, 20, 20))
            write_mask(mask, (64, 96))
            captured = {}

            def fake_run(**kwargs):
                captured.update(kwargs)
                return {"status": "completed", "output": {}}

            with patch("aigen.character_qwen_refine.run_qwen_image_edit_inpaint_candidates", fake_run):
                result = run_qwen_character_refine(
                    pack_path=pack_path,
                    identity_profile_path=None,
                    source_image_path=source,
                    mask_path=mask,
                    region_plan_path=None,
                    region_name=None,
                    instruction="repair the selected local detail",
                    output_dir=output_dir,
                    profile=object(),
                    max_side=640,
                    steps=4,
                    true_cfg_scale=1.0,
                    guidance_scale=1.0,
                    strength=0.6,
                    padding_mask_crop=None,
                    seed=0,
                    max_sequence_length=512,
                    candidates=2,
                    overwrite=False,
                    nunchaku_blocks_on_gpu=None,
                    progress=SILENT_STATUS,
                )

            routed_refs = captured["reference_images"]
            self.assertEqual([reference.name for reference in routed_refs], ["front_ref", "portrait_ref", "side_ref", "back_ref"])
            self.assertEqual([reference.role for reference in routed_refs], ["front", "portrait", "side", "back"])
            self.assertTrue(all(reference.purpose for reference in routed_refs))
            plan = captured["manifest_context"]
            self.assertTrue(plan["identity_profile_used"])
            self.assertEqual(plan["refs_used"], ["front_ref", "portrait_ref", "side_ref", "back_ref"])
            self.assertEqual([reference["input_index"] for reference in plan["references"]], [2, 3, 4, 5])
            self.assertEqual(result["output"]["refine_plan"], (output_dir / "refine_plan.json").as_posix())

    def test_qwen_refine_padding_crop_stays_on_latent_decode_path(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class FakeTorch:
            cuda = FakeCuda()

        class FakeImageProcessor:
            def __init__(self) -> None:
                self.overlay_args = {}

            def resize(self, image: Image.Image, height: int, width: int) -> Image.Image:
                return image.resize((width, height), Image.Resampling.NEAREST)

            def apply_overlay(
                self,
                mask: Image.Image,
                init_image: Image.Image,
                image: Image.Image,
                crop_coords: tuple[int, int, int, int],
            ) -> Image.Image:
                self.overlay_args = {
                    "mask_size": mask.size,
                    "init_size": init_image.size,
                    "image_size": image.size,
                    "crop_coords": crop_coords,
                }
                return Image.new("RGB", init_image.size, (9, 9, 9))

        class FakeMaskProcessor:
            def __init__(self) -> None:
                self.crop_args = {}

            def get_crop_region(self, mask: Image.Image, width: int, height: int, pad: int) -> tuple[int, int, int, int]:
                self.crop_args = {"mask_size": mask.size, "width": width, "height": height, "pad": pad}
                return (16, 24, 48, 72)

        class FakePipeline:
            vae_scale_factor = 8

            def __init__(self) -> None:
                self.image_processor = FakeImageProcessor()
                self.mask_processor = FakeMaskProcessor()

        class FakeSession:
            def __init__(self) -> None:
                self.torch = FakeTorch()
                self.pipeline = FakePipeline()
                self.denoise_kwargs = {}
                self.decode_kwargs = {}
                self.released_for_decode = False
                self.decode_saw_release = False

            def denoise_to_latents(self, **kwargs):
                self.denoise_kwargs = dict(kwargs)
                return object(), {"denoise_ms": 1.0}

            def release_denoise_models_for_decode(self, _progress) -> None:
                self.released_for_decode = True

            def decode_latents(self, _latents, **kwargs) -> tuple[Image.Image, float]:
                self.decode_kwargs = dict(kwargs)
                self.decode_saw_release = self.released_for_decode
                return Image.new("RGB", (kwargs["width"], kwargs["height"]), (1, 2, 3)), 2.0

        class FakePromptEmbedding:
            prompt = "repair local detail"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = Image.new("RGB", (64, 96), (20, 20, 20))
            mask = Image.new("L", (64, 96), 0)
            for y in range(32, 56):
                for x in range(24, 40):
                    mask.putpixel((x, y), 255)
            session = FakeSession()
            prompt_step = QwenIdentityPromptConditioningStep(
                embeddings={"refine": FakePromptEmbedding()},
                elapsed_ms=0,
            )

            with patch(
                "aigen.generation.qwen_image_edit_identity._qwen_inpaint_canvas_size",
                lambda _pipeline, image: image.size,
            ):
                result = _run_qwen_inpaint_denoise_step(
                    session=session,
                    source=source,
                    mask=mask,
                    prompt_step=prompt_step,
                    images_dir=root,
                    steps=4,
                    true_cfg_scale=1.0,
                    guidance_scale=1.0,
                    strength=0.6,
                    padding_mask_crop=8,
                    seed=11,
                    max_sequence_length=512,
                    candidates=1,
                    progress=SILENT_STATUS,
                )

            self.assertNotIn("padding_mask_crop", session.denoise_kwargs)
            self.assertEqual(session.denoise_kwargs["source_image"].size, (32, 48))
            self.assertEqual(session.denoise_kwargs["mask_image"].size, (32, 48))
            self.assertTrue(session.decode_saw_release)
            self.assertEqual(session.decode_kwargs["width"], 32)
            self.assertEqual(session.decode_kwargs["height"], 48)
            self.assertEqual(
                session.pipeline.mask_processor.crop_args,
                {"mask_size": (64, 96), "width": 64, "height": 96, "pad": 8},
            )
            self.assertEqual(
                session.pipeline.image_processor.overlay_args,
                {
                    "mask_size": (64, 96),
                    "init_size": (64, 96),
                    "image_size": (32, 48),
                    "crop_coords": (16, 24, 48, 72),
                },
            )
            self.assertEqual(result.outputs[0]["image"]["width"], 64)
            self.assertEqual(result.outputs[0]["image"]["height"], 96)


if __name__ == "__main__":
    unittest.main()
