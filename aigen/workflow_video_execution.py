from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import cast

from aigen.generation.animegen_i2v import AnimeGenI2VResult, generate_animegen_i2v_seed_sweep
from aigen.generation.hunyuanvideo15 import HunyuanVideo15Result, generate_hunyuanvideo15_i2v, HUNYUANVIDEO15_FPS
from aigen.generation.ltx23_keyframes import Ltx23Keyframe, Ltx23KeyframesResult, generate_ltx23_keyframes_seed_sweep
from aigen.generation.video_postprocess import assemble_video_frames, create_video_contact_sheet, extract_video_frames
from aigen.manifest_io import atomic_write_json
from aigen.media_timing import MediaError, probe_video, verify_video
from aigen.progress import StatusReporter
from aigen.workflow_artifacts import (AudioArtifact, ImageArtifact, ImageSequenceArtifact, KeyframeArtifact,
    VideoArtifact, WorkflowArtifact, one_artifact)
from aigen.workflow_cache import GeneratedNodeOutput
from aigen.workflow_compilation import CompiledAnimeGenConfig, CompiledHunyuanConfig, CompiledLtx23Config, CompiledNode
from aigen.workflow_graph import (AnimeGenI2VNode, ArtifactType, AssembleVideoNode, ExtractVideoFramesNode,
    HunyuanI2VNode, Ltx23Node, VideoContactSheetNode)


VIDEO_EXECUTION_NODES = (AnimeGenI2VNode, Ltx23Node, HunyuanI2VNode, VideoContactSheetNode, ExtractVideoFramesNode, AssembleVideoNode)
VIDEO_SEED_SWEEP_NODES = (AnimeGenI2VNode, Ltx23Node)


def execute_video_seed_sweep(
    compiled: CompiledNode,
    inputs: Mapping[str, Sequence[WorkflowArtifact]],
    seeds: Sequence[int],
    directory: Path,
    *,
    progress: StatusReporter,
    on_output: Callable[[int, dict[str, GeneratedNodeOutput]], None],
) -> None:
    node = compiled.node
    output = directory / "video.mp4"

    def completed(result: AnimeGenI2VResult | Ltx23KeyframesResult) -> None:
        expected_size = ((result.width, result.height) if isinstance(result, AnimeGenI2VResult)
                         else (config.settings.width, config.settings.height))
        generated = _video_result(result, directory / f"seed-{result.seed}", expected_size=expected_size,
                                  frames=settings.frames, fps=settings.fps)
        on_output(result.seed, generated)

    if isinstance(node, AnimeGenI2VNode):
        config = cast(CompiledAnimeGenConfig, compiled.config)
        settings = config.settings
        start = one_artifact(inputs, "start", ImageArtifact)
        end = one_artifact(inputs, "end", ImageArtifact) if "end" in inputs else None
        results = generate_animegen_i2v_seed_sweep(
            prompt=config.prompt, image=Path(start.path), last_image=Path(end.path) if end else None,
            output=output, frames=settings.frames, fps=settings.fps, sampling=settings.sampling,
            steps=settings.steps, precision=settings.precision, seeds=seeds, keyframe_fit=settings.keyframe_fit,
            installation=config.installation, progress=progress, on_output=completed)
    elif isinstance(node, Ltx23Node):
        config = cast(CompiledLtx23Config, compiled.config)
        settings = config.settings
        keyframes = cast(Sequence[KeyframeArtifact], inputs["keyframes"])
        results = generate_ltx23_keyframes_seed_sweep(
            prompt=config.prompt, negative_prompt=config.negative_prompt,
            keyframes=tuple(Ltx23Keyframe(Path(frame.image.path), frame.frame) for frame in keyframes),
            output=output, resolution=settings.resolution, frames=settings.frames, fps=settings.fps,
            steps=settings.steps, phases=settings.phases, solver=settings.solver,
            conditioning_strength=settings.conditioning_strength, model=settings.model,
            keyframe_fit=settings.keyframe_fit, seeds=seeds, installation=config.installation,
            progress=progress, on_output=completed)
    else:
        raise TypeError(f"not a video seed-sweep node: {node.kind}")
    if tuple(result.seed for result in results) != tuple(seeds):
        raise MediaError("video seed sweep did not return every requested seed in order")


def execute_video_node(compiled: CompiledNode, inputs: Mapping[str, Sequence[WorkflowArtifact]],
                       directory: Path, *, progress: StatusReporter) -> dict[str, GeneratedNodeOutput]:
    node = compiled.node
    output = directory / "video.mp4"
    if isinstance(node, HunyuanI2VNode):
        config = cast(CompiledHunyuanConfig, compiled.config)
        settings = config.settings
        start = one_artifact(inputs, "start", ImageArtifact)
        result = generate_hunyuanvideo15_i2v(
            prompt=settings.prompt, image=Path(start.path), output=output, frames=settings.frames,
            steps=settings.steps, seed=settings.seed, overlap_group_offloading=settings.overlap_group_offloading,
            installation=config.installation, progress=progress)
        return _video_result(result, directory, expected_size=None, frames=settings.frames, fps=HUNYUANVIDEO15_FPS)
    elif isinstance(node, VideoContactSheetNode):
        video = one_artifact(inputs, "video", VideoArtifact)
        progress.phase("create video contact sheet")
        image = create_video_contact_sheet(Path(video.path), directory / "contact-sheet.png", info=video.info)
        progress.phase("video contact sheet completed")
        return {"image": GeneratedNodeOutput(ArtifactType.IMAGE, (image,))}
    elif isinstance(node, ExtractVideoFramesNode):
        video = one_artifact(inputs, "video", VideoArtifact)
        extracted = extract_video_frames(Path(video.path), directory / "frames", progress=progress,
                                         info=video.info, source_sha256=video.content_sha256)
        paths = tuple(extracted.output_dir / f"frame-{index:06d}.png" for index in range(extracted.frames))
        return {"images": GeneratedNodeOutput(ArtifactType.IMAGE_SEQUENCE, paths, timeline=extracted.timeline, audio=extracted.audio)}
    elif isinstance(node, AssembleVideoNode):
        images = one_artifact(inputs, "images", ImageSequenceArtifact)
        if images.timeline is None:
            raise MediaError("frame sequence has no recorded timing; extract it again from its source video")
        audio = (one_artifact(inputs, "audio", AudioArtifact).track if node.config.audio_policy == "replace"
                 else images.audio if node.config.audio_policy == "preserve" else None)
        info = assemble_video_frames(tuple(Path(path) for path in images.paths), output, timeline=images.timeline,
                                     audio=audio, background=node.config.background, progress=progress)
        return {"video": GeneratedNodeOutput(ArtifactType.VIDEO, (output,), video_info=info)}
    else:
        raise TypeError(f"not a video execution node: {node.kind}")


def _video_result(
    result: AnimeGenI2VResult | Ltx23KeyframesResult | HunyuanVideo15Result,
    directory: Path,
    *,
    expected_size: tuple[int, int] | None,
    frames: int,
    fps: int,
) -> dict[str, GeneratedNodeOutput]:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / "backend-result.json", result.to_json())
    info = probe_video(result.output)
    verify_video(info, frames=frames, fps=Fraction(fps), audio=False)
    if expected_size is not None and (info.width, info.height) != expected_size:
        raise MediaError(f"video canvas is {info.width}x{info.height}; requested {expected_size[0]}x{expected_size[1]}")
    return {"video": GeneratedNodeOutput(ArtifactType.VIDEO, (result.output,), video_info=info)}
