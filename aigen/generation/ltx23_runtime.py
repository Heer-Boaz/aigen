from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess

from aigen.generation.ltx23_settings import LTX23_MODEL_TYPES, Ltx23KeyframesError, ResolvedLtx23Settings
from aigen.manifest_io import sha256_bytes, sha256_file
from aigen.media_timing import ffmpeg_runtime_revision
from aigen.model_artifacts import ModelArtifactComponent, model_artifact_stat_revision
from aigen.runtime_profiles import PROJECT_ROOT
from aigen.runtime_provenance import build_python_runtime_provenance_for_interpreter


LTX23_WANGP_REVISION = "5582327dc25e45fec6cda0f27144d4dcf7ed104b"
_GEMMA = "gemma-3-12b-it-qat-q4_0-unquantized"
_PATCHES = (
    ("0001-skip-unused-negative-text-conditioning.patch", "models/ltx2/ltx_pipelines/ti2vid_two_stages.py"),
    ("0002-use-guiding-latents-for-start-end-interpolation.patch", "models/ltx2/ltx2.py"),
)


@dataclass(frozen=True)
class Ltx23Installation:
    root: Path
    python: Path
    source: Path
    provenance: dict[str, str]


def resolve_ltx23_installation(settings: ResolvedLtx23Settings) -> Ltx23Installation:
    root = Path(os.environ.get("AIGEN_LTX23_ROOT", Path.home() / ".cache/aigen-wangp")).expanduser().resolve()
    source = root / "Wan2GP"
    interpreter = root / "venv/bin/python"
    model_type = LTX23_MODEL_TYPES[settings.model]
    config_path = root / "config/wgp_config.json"
    for path in (interpreter, source / "wgp.py", source / "shared/api.py", config_path):
        if not path.is_file():
            raise Ltx23KeyframesError(f"LTX-2.3 runtime file is missing: {path}")
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if revision.returncode or revision.stdout.strip() != LTX23_WANGP_REVISION:
        raise Ltx23KeyframesError(f"WanGP source must be pinned to {LTX23_WANGP_REVISION}")
    provenance = {"WanGP": LTX23_WANGP_REVISION}
    for name, target in _PATCHES:
        patch = PROJECT_ROOT / "patches/ltx23" / name
        applied = subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)], capture_output=True, check=False)
        if applied.returncode:
            raise Ltx23KeyframesError(f"LTX-2.3 runtime patch is not installed: {patch}")
        provenance[f"LTX patch {name}"] = sha256_bytes(patch.read_bytes() + (source / target).read_bytes())

    config = json.loads(config_path.read_text())
    if "P" in config.get("preload_model_policy", ()):
        raise Ltx23KeyframesError("WanGP preload_model_policy loads another model during startup; disable P for this explicit LTX route")
    if config.get("installed_remote_plugins"):
        raise Ltx23KeyframesError("the LTX worker requires a WanGP runtime without remote plugins")
    if config.get("video_container", "mp4") != "mp4":
        raise Ltx23KeyframesError("the LTX route requires WanGP video_container=mp4")
    if config.get("text_encoder_quantization", "int8") != "int8" or (
            settings.model == "int8" and config.get("transformer_quantization", "int8") != "int8"):
        raise Ltx23KeyframesError("this LTX route requires WanGP int8 text encoding and int8 transformer quantization for the int8 model")
    roots = tuple(source / str(path).strip() for path in config.get("checkpoints_paths", ("ckpts", ".")) if str(path).strip())
    if not roots:
        raise Ltx23KeyframesError("WanGP checkpoint search paths are empty")

    def locate(*names: str) -> Path:
        for name in names:
            for search_root in roots:
                path = search_root / name
                if path.is_file():
                    return path.resolve()
        raise Ltx23KeyframesError(f"LTX-2.3 local model asset is missing: {names[0]} (searched {', '.join(map(str, roots))})")

    transformer = ("ltx-2.3-22b-dev-nvfp4_diffusion_model.safetensors" if settings.model == "nvfp4"
                   else "ltx-2.3-22b-dev_diffusion_model_quanto_int8.safetensors")
    model_files = [locate(name) for name in (transformer,
        "ltx-2.3-22b_vae.safetensors", "ltx-2.3-22b_audio_vae.safetensors", "ltx-2.3-22b_vocoder.safetensors",
        "ltx-2.3-22b_text_embedding_projection.safetensors", "ltx-2.3-22b_embeddings_connector.safetensors",
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")]
    weight = f"{_GEMMA}_quanto_bf16_int8.safetensors"
    model_files.append(locate(f"{_GEMMA}/{weight}", weight))
    model_files.append(locate(f"{_GEMMA}/config_light.json"))
    gemma_folder = next((search_root / _GEMMA for search_root in roots if (search_root / _GEMMA).is_dir()), None)
    if gemma_folder is None:
        raise Ltx23KeyframesError(f"LTX-2.3 tokenizer directory is missing: {_GEMMA}")
    model_files.extend(gemma_folder / name for name in (
        "added_tokens.json", "chat_template.json", "generation_config.json",
        "preprocessor_config.json", "processor_config.json", "special_tokens_map.json",
        "tokenizer.json", "tokenizer.model", "tokenizer_config.json"))
    model_files.append(source / "models/ltx2/configs/ltx2_22b_config.json")
    if settings.phases > 1 or settings.solver in ("distilled_8_steps", "res2s"):
        model_files.append(source / config.get("loras_root", "loras") / "ltx2/ltx-2.3-22b-distilled-lora-384.safetensors")
    for path in model_files:
        if not path.is_file():
            raise Ltx23KeyframesError(f"LTX-2.3 local model asset is missing: {path}")
        provenance[path.name] = model_artifact_stat_revision(ModelArtifactComponent(path.name, path.parent, (path,)))

    # Native download_models checks these even when their features are unused.
    # Admission checks presence; their bytes do not participate in rendering.
    for name in _native_download_assets(config):
        locate(name)
    for name in ("ffmpeg", "ffprobe"):
        if (source / name).is_file():
            raise Ltx23KeyframesError(f"WanGP would quarantine {source / name}; use its ffmpeg_bins directory or PATH")
        path = source / "ffmpeg_bins" / name
        if not path.is_file():
            path = None
        if path is None:
            found = shutil.which(name)
            if found is None:
                raise Ltx23KeyframesError(f"WanGP requires installed {name}")
            path = Path(found)
        if path.parent == source / "ffmpeg_bins" and not any(path.parent.glob("libavdevice.so*")):
            raise Ltx23KeyframesError(f"WanGP bundled FFmpeg requires libavdevice.so in {path.parent}")
        provenance[name] = sha256_file(path)
        provenance[f"{name} linked libraries"] = ffmpeg_runtime_revision(path)

    json_paths = [source / "models/_settings.json", source / f"settings/{model_type}_settings.json",
                  source / f"defaults/{model_type}.json"]
    if settings.model == "nvfp4":
        json_paths.append(source / "defaults/ltx2_22B.json")
    override = source / f"finetunes/{model_type}.json"
    if override.is_file():
        raise Ltx23KeyframesError(f"the explicit LTX route does not accept a native model override: {override}")
    for path in json_paths:
        if not path.is_file():
            raise Ltx23KeyframesError(f"LTX-2.3 configuration is missing: {path}")
        payload = json.loads(path.read_text())
        if path.parent.name in ("settings", "models"):
            for field in ("activated_loras", "custom_settings", "image_guide", "image_mask", "video_guide", "video_mask",
                          "video_source", "audio_guide", "audio_guide2", "audio_source", "custom_guide"):
                if payload.get(field):
                    raise Ltx23KeyframesError(f"LTX route does not accept native {field} from {path}; use explicit workflow inputs")
        provenance[path.relative_to(source).as_posix()] = sha256_file(path)
    runtime_keys = ("transformer_quantization", "text_encoder_quantization", "compile", "boost", "enable_int8_kernels",
                    "vae_config", "max_reserved_loras", "checkpoints_paths", "loras_root", "video_output_codec", "video_container")
    provenance["WanGP execution config"] = sha256_bytes(json.dumps({key: config.get(key) for key in runtime_keys}, sort_keys=True).encode())
    runtime = build_python_runtime_provenance_for_interpreter(interpreter, (
        "torch", "mmgp", "transformers", "numpy", "safetensors", "imageio", "imageio-ffmpeg",
        "triton", "lightx2v-kernel"))
    provenance["WanGP Python runtime"] = str(runtime["fingerprint"])
    return Ltx23Installation(root, interpreter, source, provenance)


def _native_download_assets(config: dict) -> tuple[str, ...]:
    files = ["ltx-2.3-temporal-upscaler-x2-1.0.safetensors", "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"]
    folders = {
        "pose": ("dw-ll_ucoco_384.onnx", "yolox_l.onnx"), "scribble": ("netG_A_latest.pth",), "flow": ("raft-things.pth",),
        "wav2vec": ("config.json", "feature_extractor_config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"),
        "chinese-wav2vec2-base": ("config.json", "pytorch_model.bin", "preprocessor_config.json"),
        "roformer": ("model_bs_roformer_ep_317_sdr_12.9755.ckpt", "model_bs_roformer_ep_317_sdr_12.9755.yaml", "download_checks.json"),
        "pyannote": ("pyannote_model_wespeaker-voxceleb-resnet34-LM.bin", "pytorch_model_segmentation-3.0.bin"),
        "det_align": ("detface.pt",),
    }
    depth = {"vitb": "depth_anything_v2_vitb.pth", "da3_metric_large": "depth_anything_v3_metric_large_bf16.safetensors"}.get(
        config.get("depth_anything_v2_variant"), "depth_anything_v2_vitl.pth")
    files.append(f"depth/{depth}")
    mask = str(config.get("matanyone_version", "v1")).strip().lower()
    if mask in ("sam3", "sam", "sam3.1"):
        folders["sam3"] = ("sam3.1_multiplex_bf16.safetensors", "bpe_simple_vocab_16e6.txt.gz")
    else:
        folders["mask"] = ("sam_vit_h_4b8939_fp16.safetensors", "matanyone2.safetensors" if mask in ("2", "v2", "matanyone2") else "matanyone.safetensors", "config.json")
    files.extend(f"{folder}/{name}" for folder, names in folders.items() for name in names)
    return tuple(files)
