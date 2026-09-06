from fractions import Fraction
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

import av
import numpy as np
from PIL import Image

from aigen.generation.video_postprocess import assemble_video_frames, extract_video_frames
from aigen.media_timing import FrameTimeline, load_audio_track, probe_video, verify_video
from aigen.progress import SILENT_STATUS


class VideoTimingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_vfr_extract_assemble_retains_every_frame_time_and_order(self):
        source = self.root / "source.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "nullsrc=size=64x80:rate=25:duration=0.2,geq=r='N*50':g=100:b='200-N*20'", "-vf", "select='not(eq(n,2))'",
            "-fps_mode", "vfr", "-c:v", "libx264", "-bf", "0", str(source),
        ], check=True, capture_output=True)
        expected = probe_video(source)
        self.assertGreater(len(set(expected.timeline.durations)), 1)
        frames = extract_video_frames(source, self.root / "frames", progress=SILENT_STATUS, info=expected)
        manifest = json.loads((frames.output_dir / "frames.json").read_text())
        paths = tuple(Path(path) for path in manifest["paths"])
        output = self.root / "assembled.mp4"
        measured = assemble_video_frames(paths, output, timeline=frames.timeline, audio=frames.audio, progress=SILENT_STATUS)
        verify_video(measured, timeline=expected.timeline, frames=expected.frames, audio=False)
        with av.open(str(output)) as video:
            decoded = list(video.decode(video=0))
        for path, actual in zip(paths, decoded, strict=True):
            with Image.open(path) as image:
                error = np.abs(np.asarray(image, dtype=np.int16) - actual.to_ndarray(format="rgb24"))
            self.assertLess(float(error.mean()), 4.0)

    def test_preserve_audio_origin_and_replace_with_second_stream_without_shortening_video(self):
        source = self.root / "offset.mkv"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "color=c=red:s=64x80:r=25:d=0.4", "-itsoffset", "0.08", "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.16", "-itsoffset", "0.04", "-f", "lavfi", "-i",
            "sine=frequency=880:sample_rate=48000:duration=0.8", "-map", "0:v", "-map", "1:a", "-map", "2:a",
            "-c:v", "libx264", "-bf", "0", "-c:a", "pcm_s16le", "-output_ts_offset", "2", str(source),
        ], check=True, capture_output=True)
        info = probe_video(source)
        self.assertEqual([track.index for track in info.audio], [1, 2])
        self.assertEqual(info.start_time, 2)
        frames = extract_video_frames(source, self.root / "frames", progress=SILENT_STATUS, info=info)
        paths = tuple(frames.output_dir / f"frame-{index:06d}.png" for index in range(frames.frames))
        retained = self.root / "retained.mp4"
        retained_info = assemble_video_frames(paths, retained, timeline=frames.timeline, audio=frames.audio, progress=SILENT_STATUS)
        verify_video(retained_info, timeline=info.timeline, audio=True)
        frequency, onset = self.audio_signal(retained)
        self.assertAlmostEqual(frequency, 440, delta=15)
        self.assertAlmostEqual(onset, 0.08, delta=0.008)

        replacement = load_audio_track(source, stream_index=2)
        replaced = self.root / "replaced.mp4"
        replaced_info = assemble_video_frames(paths, replaced, timeline=frames.timeline, audio=replacement, progress=SILENT_STATUS)
        verify_video(replaced_info, timeline=info.timeline, audio=True)
        frequency, onset = self.audio_signal(replaced)
        self.assertAlmostEqual(frequency, 880, delta=15)
        self.assertAlmostEqual(onset, 0, delta=0.008)

        removed_info = assemble_video_frames(paths, self.root / "removed.mp4", timeline=frames.timeline, audio=None, progress=SILENT_STATUS)
        verify_video(removed_info, timeline=info.timeline, audio=False)

    @staticmethod
    def audio_signal(path):
        with av.open(str(path)) as container:
            frames = list(container.decode(audio=0))
        samples = np.concatenate([frame.to_ndarray()[0] for frame in frames])
        sample_rate = frames[0].sample_rate
        onset = float(frames[0].pts * frames[0].time_base) + np.flatnonzero(np.abs(samples) > 0.04)[0] / sample_rate
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        frequency = np.fft.rfftfreq(len(samples), 1 / sample_rate)[spectrum.argmax()]
        return frequency, onset

    def test_fractional_cfr_and_transparent_frames(self):
        paths = []
        for index in range(3):
            path = self.root / f"frame{index}.png"
            Image.new("RGBA", (16, 20), (255, 0, 0, 0)).save(path)
            paths.append(path)
        timeline = FrameTimeline(time_base=Fraction(1, 30000), pts=(0, 1001, 2002), durations=(1001, 1001, 1001))
        output = self.root / "fractional.mp4"
        info = assemble_video_frames(tuple(paths), output, timeline=timeline, audio=None, background="white", progress=SILENT_STATUS)
        verify_video(info, timeline=timeline, fps=Fraction(30000, 1001))
        with av.open(str(output)) as container:
            frame = next(container.decode(video=0)).to_ndarray(format="rgb24")
        self.assertGreater(frame.min(), 250)


if __name__ == "__main__":
    unittest.main()
