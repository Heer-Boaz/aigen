"""Real FFmpeg/PyAV, CPU frame processing, compiler, executor and durable cache."""
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from aigen.manifest_io import sha256_file
from aigen.media_timing import probe_video, verify_video
from aigen.progress import SILENT_STATUS
from aigen.workflow_cache import WorkflowNodeCache, WorkflowCacheCorruptionError
from aigen.workflow_compilation import compile_workflow_run
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_execution import execute_workflow
from aigen.workflow_graph import (
    AssembleVideoConfig, AssembleVideoNode, AudioSourceConfig, AudioSourceNode,
    ExtractVideoFramesNode, FramePostprocessNode, NodePortRef, PixelArtFixerConfig,
    VideoSourceConfig, VideoSourceNode, WorkflowConnection, WorkflowGraph,
)
from aigen.workflow_results import load_node_result


class WorkflowVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.make_source("source.mkv", 25)

    def make_source(self, name, fps):
        path = self.root / name
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            f"color=c=red:s=64x80:r={fps}:d={4/fps}", "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.4", "-c:v", "libx264",
            "-bf", "0", "-c:a", "pcm_s16le", str(path),
        ], check=True, capture_output=True)
        return path

    def graph(self):
        nodes = (
            VideoSourceNode(id="source", title="Video", config=VideoSourceConfig(path=str(self.source))),
            ExtractVideoFramesNode(id="extract", title="Frames"),
            FramePostprocessNode(id="process", title="Process", config=PixelArtFixerConfig(
                mode="fast", low_memory=False, force_step=2)),
            AudioSourceNode(id="audio", title="Unused replacement", config=AudioSourceConfig(path="absent.wav")),
            AssembleVideoNode(id="assemble", title="Assemble", config=AssembleVideoConfig()),
        )
        wires = tuple(WorkflowConnection(id=f"wire{index}", source=NodePortRef(node_id=source, port=out),
                      target=NodePortRef(node_id=target, port=port))
                      for index, (source, out, target, port) in enumerate((
                          ("source", "video", "extract", "video"), ("extract", "images", "process", "images"),
                          ("process", "images", "assemble", "images"), ("audio", "audio", "assemble", "audio"))))
        return WorkflowEditBuffer(WorkflowGraph(name="Video roundtrip", nodes=nodes, connections=wires))

    def execute(self, buffer):
        with patch("aigen.workflow_execution._node_progress", return_value=SILENT_STATUS):
            return execute_workflow(compile_workflow_run(buffer.document), runs_root=self.root / "workflows", progress=SILENT_STATUS)

    def test_import_process_assemble_reuses_pixels_but_carries_current_timing_and_audio(self):
        buffer = self.graph()
        first = self.execute(buffer)
        output = first.terminal_outputs["assemble"]["video"]
        verify_video(output.info, frames=4, fps=Fraction(25), audio=True)
        self.assertEqual((output.info.width, output.info.height), (32, 40))
        second = self.execute(buffer)
        for node in ("extract", "process", "assemble"):
            self.assertEqual(load_node_result(second.node_manifests[node]).status, "reused")

        different_timing = self.make_source("faster.mkv", 50)
        buffer.update_node_config("source", "path", str(different_timing))
        third = self.execute(buffer)
        self.assertEqual(load_node_result(third.node_manifests["process"]).status, "reused")
        frames = load_node_result(third.node_manifests["process"]).outputs["images"]
        self.assertEqual(frames.timeline.fps, 50)
        self.assertEqual(frames.audio.path, str(different_timing))
        verify_video(third.terminal_outputs["assemble"]["video"].info, frames=4, fps=Fraction(50), audio=True)
        buffer.update_node_config("assemble", "audio_policy", "remove")
        fourth = self.execute(buffer)
        self.assertEqual(load_node_result(fourth.node_manifests["process"]).status, "reused")
        verify_video(fourth.terminal_outputs["assemble"]["video"].info, fps=Fraction(50), audio=False)

    def test_moved_source_reuses_extraction_with_current_audio_location(self):
        buffer = self.graph()
        first = self.execute(buffer)
        moved = self.root / "moved.mkv"
        self.source.rename(moved)
        buffer.update_node_config("source", "path", str(moved))
        # Different mux setting forces assembly to read the current audio path.
        buffer.update_node_config("assemble", "background", "white")
        second = self.execute(buffer)
        for node in ("extract", "process"):
            manifest = load_node_result(second.node_manifests[node])
            self.assertEqual(manifest.status, "reused")
            self.assertEqual(manifest.outputs["images"].audio.path, str(moved))
            self.assertEqual(manifest.details, load_node_result(first.node_manifests[node]).details)
        verify_video(second.terminal_outputs["assemble"]["video"].info, frames=4, audio=True)

    def test_metadata_cache_is_content_and_decoder_revision_addressed(self):
        cache = WorkflowNodeCache(self.root / "metadata")
        identity = sha256_file(self.source)
        copy = self.root / "copy.mkv"
        shutil.copyfile(self.source, copy)
        with patch("aigen.workflow_cache.probe_video", wraps=probe_video) as probe:
            info = cache.video_info(self.source, identity)
            self.assertEqual(cache.video_info(copy, identity), info)
            self.assertEqual(probe.call_count, 1)
            other = self.make_source("other.mkv", 50)
            cache.video_info(other, sha256_file(other))
            self.assertEqual(probe.call_count, 2)
            with patch("aigen.media_timing.MEDIA_TIMING_REVISION", "changed"):
                cache.video_info(self.source, identity)
            self.assertEqual(probe.call_count, 3)
        for manifest in cache.root.glob("video-info/*/*.json"):
            manifest.write_text('{"broken": true}')
        with self.assertRaisesRegex(WorkflowCacheCorruptionError, "invalid cached video metadata"):
            cache.video_info(self.source, identity)


if __name__ == "__main__":
    unittest.main()
