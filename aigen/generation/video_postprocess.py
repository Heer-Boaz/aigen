from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from aigen.progress import StatusReporter


CONTACT_SHEET_CELL_WIDTH = 416


class VideoPostprocessError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frames: int


@dataclass(frozen=True)
class VideoFrameExtraction:
    video: Path
    output_dir: Path
    frames: int

    def to_json(self) -> dict[str, object]:
        return {
            "status": "completed",
            "kind": "video-frame-extraction",
            "video": self.video.as_posix(),
            "output_dir": self.output_dir.as_posix(),
            "frames": self.frames,
            "pattern": "frame-%06d.png",
        }


def contact_sheet_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}-contact.png")


def create_video_contact_sheet(video: Path, output: Path | None = None) -> Path:
    video = _video_path(video)
    output = (output or contact_sheet_path(video)).expanduser().resolve()
    info = probe_video(video)
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
) -> VideoFrameExtraction:
    video = _video_path(video)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise VideoPostprocessError(f"output directory already exists: {output_dir}")
    info = probe_video(video)
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
        staging.replace(output_dir)
    progress.phase("video frame extraction completed")
    return VideoFrameExtraction(
        video=video,
        output_dir=output_dir,
        frames=info.frames,
    )


def probe_video(video: Path) -> VideoInfo:
    video = _video_path(video)
    completed = _run(
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            video.as_posix(),
        ),
        "FFprobe could not inspect the video",
    )
    try:
        stream = json.loads(completed.stdout)["streams"][0]
        return VideoInfo(
            width=int(stream["width"]),
            height=int(stream["height"]),
            frames=int(stream["nb_read_frames"]),
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise VideoPostprocessError(
            f"FFprobe returned invalid video metadata for {video}"
        ) from error


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
