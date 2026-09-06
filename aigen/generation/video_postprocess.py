from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image

from aigen.image_io import image_alpha, open_image
from aigen.manifest_io import atomic_write_json, sha256_file
from aigen.media_timing import AudioTrack, FrameTimeline, MediaError, VideoInfo, probe_video, verify_video, video_audio_track
from aigen.progress import StatusReporter


CONTACT_SHEET_CELL_WIDTH = 416


VideoPostprocessError = MediaError


@dataclass(frozen=True)
class VideoFrameExtraction:
    video: Path
    output_dir: Path
    frames: int
    timeline: FrameTimeline
    audio: AudioTrack | None

    def to_json(self) -> dict[str, object]:
        return {
            "status": "completed",
            "kind": "video-frame-extraction",
            "video": self.video.as_posix(),
            "output_dir": self.output_dir.as_posix(),
            "frames": self.frames,
            "pattern": "frame-%06d.png",
            "manifest": (self.output_dir / "frames.json").as_posix(),
            "paths": [(self.output_dir / f"frame-{index:06d}.png").as_posix() for index in range(self.frames)],
            "timeline": self.timeline.model_dump(mode="json"),
            "audio": self.audio.model_dump(mode="json") if self.audio else None,
        }


def contact_sheet_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}-contact.png")


def create_video_contact_sheet(video: Path, output: Path | None = None, *, info: VideoInfo | None = None) -> Path:
    video = _video_path(video)
    output = (output or contact_sheet_path(video)).expanduser().resolve()
    info = probe_video(video) if info is None else info
    columns = math.ceil(math.sqrt(info.frames))
    rows = math.ceil(info.frames / columns)
    cell_width = min(info.width, CONTACT_SHEET_CELL_WIDTH)
    cell_height = max(2, round(info.height * cell_width / info.width / 2) * 2)
    video_filter = (
        f"scale={cell_width}:{cell_height}:flags=lanczos,"
        "drawtext=text='%{n}':x=6:y=6:fontsize=18:"
        "fontcolor=white:box=1:boxcolor=black@0.7,"
        f"tile={columns}x{rows}:nb_frames={info.frames}:padding=2:margin=2:color=black"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent,
        prefix=f".{output.stem}-",
    ) as staging_dir:
        staged_output = Path(staging_dir) / output.name
        _run(
            (
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                video.as_posix(),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-vf",
                video_filter,
                "-frames:v",
                "1",
                staged_output.as_posix(),
            ),
            "FFmpeg could not create the video contact sheet",
        )
        staged_output.replace(output)
    return output


def extract_video_frames(
    video: Path,
    output_dir: Path,
    *,
    progress: StatusReporter,
    info: VideoInfo | None = None,
    source_sha256: str | None = None,
) -> VideoFrameExtraction:
    video = _video_path(video)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise VideoPostprocessError(f"output directory already exists: {output_dir}")
    info = probe_video(video) if info is None else info
    audio = video_audio_track(video, info, source_sha256 or sha256_file(video)) if info.audio else None
    result = VideoFrameExtraction(video=video, output_dir=output_dir, frames=info.frames,
                                  timeline=info.timeline, audio=audio)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    progress.begin(info.frames, "extract video frames")
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}-",
    ) as staging_dir:
        staging = Path(staging_dir)
        _run_with_frame_progress(
            (
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-progress",
                "pipe:1",
                "-nostats",
                "-i",
                video.as_posix(),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-fps_mode",
                "passthrough",
                "-start_number",
                "0",
                (staging / "frame-%06d.png").as_posix(),
            ),
            total_frames=info.frames,
            progress=progress,
        )
        extracted = len(tuple(staging.glob("frame-*.png")))
        if extracted != info.frames:
            raise VideoPostprocessError(
                f"FFmpeg extracted {extracted} frames; expected {info.frames}"
            )
        atomic_write_json(staging / "frames.json", result.to_json())
        staging.replace(output_dir)
    progress.phase("video frame extraction completed")
    return result


def assemble_video_frames(
    paths: tuple[Path, ...], output: Path, *, timeline: FrameTimeline,
    audio: AudioTrack | None, progress: StatusReporter, background: str = "black",
) -> VideoInfo:
    """Encode one frame at a time with its recorded presentation time and duration."""
    if len(paths) != len(timeline.pts):
        raise MediaError("frame files and timeline have different frame counts")
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise MediaError("video assembly output must use the .mp4 extension")
    if output.exists():
        raise MediaError(f"output already exists: {output}")
    if audio is not None and sha256_file(Path(audio.path)) != audio.file_sha256:
        raise MediaError(f"audio source changed: {audio.path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    progress.begin(len(paths), "assemble video frames")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.stem}-") as directory:
        staging = Path(directory)
        silent = staging / "video.mp4"
        durations = dict(zip(timeline.pts, timeline.durations, strict=True))
        try:
            with av.open(str(silent), "w") as container:
                stream = container.add_stream("libx264", rate=timeline.fps)
                stream.time_base = stream.codec_context.time_base = timeline.time_base
                stream.codec_context.max_b_frames = 0
                stream.options = {"crf": "18", "preset": "medium"}
                stream.pix_fmt = "yuv420p"

                def mux(packets):
                    for packet in packets:
                        packet.duration = durations[packet.pts]
                        container.mux(packet)

                for index, (path, pts) in enumerate(zip(paths, timeline.pts, strict=True)):
                    with open_image(path) as source:
                        if index == 0:
                            width, height = source.size
                            if width % 2 or height % 2:
                                raise MediaError("H264 yuv420p assembly requires even frame dimensions")
                            stream.width, stream.height = width, height
                        elif source.size != (width, height):
                            raise MediaError(f"frame dimensions changed at {path}")
                        alpha = image_alpha(source)
                        if alpha is not None:
                            with Image.new("RGB", source.size, background) as rgb:
                                rgb.paste(source, mask=alpha)
                                frame = av.VideoFrame.from_image(rgb)
                        else:
                            frame = av.VideoFrame.from_image(source)
                    frame.pts, frame.time_base = pts, timeline.time_base
                    mux(stream.encode(frame))
                    progress.step(f"assembled frame {index + 1}/{len(paths)}")
                mux(stream.encode(None))
        except (av.FFmpegError, OSError, ValueError) as error:
            raise MediaError(f"video frame encoding failed: {error}") from error
        candidate = silent
        if audio is not None:
            progress.phase("mux audio on the video timeline")
            candidate = staging / "with-audio.mp4"
            _run((
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-copyts",
                "-i", str(silent), "-itsoffset", str(float(-audio.timeline_origin)), "-i", audio.path,
                "-map", "0:v:0", "-map", f"1:{audio.stream_index}", "-c:v", "copy", "-c:a", "aac",
                "-af", f"atrim=start=0:end={float(timeline.duration):.12f}",
                "-t", f"{float(timeline.duration):.12f}", "-avoid_negative_ts", "disabled",
                "-movflags", "+faststart", str(candidate),
            ), "FFmpeg could not mux the selected audio")
        info = probe_video(candidate)
        verify_video(info, timeline=timeline, audio=audio is not None)
        candidate.replace(output)
    progress.phase("video assembly completed")
    return info


def _video_path(video: Path) -> Path:
    path = video.expanduser().resolve()
    if not path.is_file():
        raise VideoPostprocessError(f"video does not exist: {path}")
    return path


def _run(command: tuple[str, ...], error_message: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise VideoPostprocessError(f"required executable is unavailable: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        raise VideoPostprocessError(f"{error_message}: {detail}") from error


def _run_with_frame_progress(
    command: tuple[str, ...],
    *,
    total_frames: int,
    progress: StatusReporter,
) -> None:
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as error:
        raise VideoPostprocessError(f"required executable is unavailable: {command[0]}") from error
    assert process.stdout is not None
    reported = 0
    output: list[str] = []
    with process.stdout:
        for line in process.stdout:
            line = line.strip()
            if not line.startswith("frame="):
                if line:
                    output.append(line)
                continue
            frame = min(int(line.partition("=")[2]), total_frames)
            while reported < frame:
                reported += 1
                progress.step(f"extract video frame {reported}/{total_frames}")
    returncode = process.wait()
    if returncode != 0:
        raise VideoPostprocessError(
            "FFmpeg could not extract the video frames: " + "\n".join(output[-20:])
        )
    while reported < total_frames:
        reported += 1
        progress.step(f"extract video frame {reported}/{total_frames}")
