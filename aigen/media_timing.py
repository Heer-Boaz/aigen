from __future__ import annotations

from fractions import Fraction
import json
import subprocess
from pathlib import Path

import av
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aigen.manifest_io import sha256_bytes, sha256_file


MEDIA_TIMING_REVISION = "2"


def media_runtime_revision() -> str:
    return sha256_bytes(json.dumps({
        "measurement": MEDIA_TIMING_REVISION,
        "pyav": av.__version__,
        "libraries": av.library_versions,
    }, sort_keys=True, separators=(",", ":")).encode())


def ffmpeg_runtime_revision(executable: Path) -> str:
    version = subprocess.run([str(executable), "-version"], check=True, capture_output=True).stdout
    return sha256_bytes(version)


class MediaError(RuntimeError):
    pass


class MediaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FrameTimeline(MediaModel):
    time_base: Fraction
    pts: tuple[int, ...] = Field(min_length=1)
    durations: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_timeline(self):
        if self.time_base <= 0 or self.pts[0] != 0:
            raise ValueError("frame timeline requires a positive time base and a zero origin")
        if len(self.pts) != len(self.durations) or any(duration <= 0 for duration in self.durations):
            raise ValueError("every frame must have a positive recorded duration")
        if any(following != start + duration for start, duration, following
               in zip(self.pts, self.durations, self.pts[1:])):
            raise ValueError("frame timestamps and durations do not form one ordered timeline")
        return self

    @property
    def duration(self) -> Fraction:
        return (self.pts[-1] + self.durations[-1]) * self.time_base

    @property
    def fps(self) -> Fraction:
        return len(self.pts) / self.duration


class AudioStreamInfo(MediaModel):
    index: int
    codec: str
    sample_rate: int
    channels: int
    start_time: Fraction


class AudioTrack(MediaModel):
    path: str
    file_sha256: str
    stream_index: int
    # Subtract this source-media time to place audio on the output timeline.
    timeline_origin: Fraction


class VideoInfo(MediaModel):
    width: int
    height: int
    timeline: FrameTimeline
    start_time: Fraction
    audio: tuple[AudioStreamInfo, ...] = ()

    @property
    def frames(self) -> int:
        return len(self.timeline.pts)

    @property
    def fps(self) -> Fraction:
        return self.timeline.fps


def video_audio_track(path: Path, info: VideoInfo, content_sha256: str) -> AudioTrack | None:
    if not info.audio:
        return None
    return AudioTrack(path=path.as_posix(), file_sha256=content_sha256,
                      stream_index=info.audio[0].index, timeline_origin=info.start_time)


def audio_streams(container) -> tuple[AudioStreamInfo, ...]:
    return tuple(AudioStreamInfo(
        index=stream.index, codec=stream.codec_context.name,
        sample_rate=stream.codec_context.sample_rate,
        channels=stream.codec_context.channels,
        start_time=(stream.start_time or 0) * stream.time_base,
    ) for stream in container.streams.audio)


def load_audio_track(path: Path, *, stream_index: int | None = None, content_sha256: str | None = None) -> AudioTrack:
    path = path.expanduser().resolve()
    try:
        with av.open(str(path)) as container:
            tracks = audio_streams(container)
            selected = next((track for track in tracks if stream_index is None or track.index == stream_index), None)
            if selected is None:
                raise MediaError(f"media has no selected audio stream ({stream_index}): {path}")
            return AudioTrack(path=path.as_posix(), file_sha256=sha256_file(path) if content_sha256 is None else content_sha256,
                              stream_index=selected.index, timeline_origin=selected.start_time)
    except (av.FFmpegError, OSError) as error:
        raise MediaError(f"cannot read audio from {path}: {error}") from error


def probe_video(video: Path) -> VideoInfo:
    """Measure displayed frame order/timing once at the media artifact boundary."""
    try:
        with av.open(str(video)) as container:
            if not container.streams.video:
                raise MediaError(f"media has no video stream: {video}")
            stream = container.streams.video[0]
            tracks = audio_streams(container)
            time_base = stream.time_base
            pts = []
            last_duration = 0
            width = height = 0
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise MediaError(f"video frame has no presentation timestamp: {video}")
                pts.append(frame.pts)
                last_duration = frame.duration
                width, height = frame.width, frame.height
                if frame.rotation % 180:
                    width, height = height, width
            if not pts:
                raise MediaError(f"video contains no decoded frames: {video}")
            # FFmpeg can copy the container's (longer audio) duration into a
            # video stream. Only the decoded final frame supplies our endpoint.
            if last_duration <= 0:
                raise MediaError(f"video has no recorded final-frame duration: {video}")
            end = pts[-1] + last_duration
            timeline = FrameTimeline(
                time_base=time_base, pts=tuple(value - pts[0] for value in pts),
                durations=tuple(following - current for current, following in zip(pts, pts[1:])) + (end - pts[-1],),
            )
            return VideoInfo(width=width, height=height, timeline=timeline,
                             start_time=pts[0] * time_base, audio=tracks)
    except (av.FFmpegError, OSError, ValueError) as error:
        raise MediaError(f"cannot read video timing from {video}: {error}") from error


def verify_video(info: VideoInfo, *, timeline: FrameTimeline | None = None,
                 frames: int | None = None, fps: Fraction | None = None,
                 audio: bool | None = None) -> None:
    if frames is not None and info.frames != frames:
        raise MediaError(f"video has {info.frames} frames; requested {frames}")
    if fps is not None and any(duration * info.timeline.time_base != Fraction(1) / fps for duration in info.timeline.durations):
        raise MediaError(f"video frame timing does not match the requested {fps} FPS")
    if timeline is not None:
        actual = info.timeline
        if (tuple(value * actual.time_base for value in actual.pts) != tuple(value * timeline.time_base for value in timeline.pts)
                or tuple(value * actual.time_base for value in actual.durations) != tuple(value * timeline.time_base for value in timeline.durations)):
            raise MediaError("assembled video does not retain the source frame timeline")
    if audio is not None and bool(info.audio) != audio:
        raise MediaError(f"video audio presence is {bool(info.audio)}; requested {audio}")
