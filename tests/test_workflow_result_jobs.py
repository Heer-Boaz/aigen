import asyncio
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from textual.widgets import Select

from aigen.artifact_actions import export_artifact
from aigen.manifest_io import atomic_write_json, sha256_file
from aigen.workflow_artifacts import ImageArtifact, ImageCollectionArtifact, ImageSequenceArtifact
from aigen.workflow_cache import GeneratedNodeOutput, NodeExecutionProvenance, RevisionedComponent, WorkflowNodeCache
from aigen.workflow_graph import (ImageCollectionNode, ImageSelectionConfig, ImageSelectionNode,
    ArtifactType, NodeKind, NodePortRef, WorkflowConnection, WorkflowGraph)
from aigen.workflow_media_results import MediaCandidate, export_media
from aigen.workflow_results import NodeResultManifest, resolve_image_result
from aigen.workflow_results_tui import WorkflowResults
from test_workflow_properties import open_editor


class ExportIntegrityTests(unittest.TestCase):
    def test_changed_image_after_resolution_is_not_published(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (8, 8), "red").save(source)
            checksum = sha256_file(source)
            manifest = NodeResultManifest(node_id="source", node_kind=NodeKind.IMAGE_SOURCE,
                signature="a" * 64, outputs={"image": ImageArtifact(path=str(source), identity=checksum, content_sha256=checksum)})
            path = root / "result.json"
            atomic_write_json(path, manifest.model_dump(mode="json"))
            resolved = resolve_image_result(manifest.candidate(path, "image").reference)
            Image.new("RGB", (8, 8), "green").save(source)
            target = root / "export.png"
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                export_artifact(Path(resolved.path), target, expected_sha256=resolved.content_sha256)
            self.assertFalse(target.exists())
            self.assertFalse(list(root.glob(".aigen-export-*")))

    def test_changed_frame_does_not_publish_partial_archive(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = tuple(root / f"frame-{index}.png" for index in range(2))
            for path in paths:
                Image.new("RGB", (8, 8), "red").save(path)
            sequence = ImageSequenceArtifact(paths=tuple(str(path) for path in paths), identity="frames",
                                             frame_sha256s=tuple(sha256_file(path) for path in paths))
            Image.new("RGB", (8, 8), "green").save(paths[1])
            target = root / "frames.zip"
            with self.assertRaisesRegex(ValueError, "frame changed"):
                export_media(sequence, target)
            self.assertFalse(target.exists())
            self.assertFalse(list(root.glob(".aigen-export-*")))


class ResultJobUITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.candidates = []
        self.histories = []
        for index in range(2):
            source = self.root / f"source-{index}.png"
            Image.new("RGB", (8, 8), (index * 100, 50, 100)).save(source)
            checksum = sha256_file(source)
            directory = self.root / f"attempt-{index}" / "nodes"
            path = directory / "source" / "result.json"
            result = NodeResultManifest(node_id="source", node_kind=NodeKind.IMAGE_SOURCE, signature=str(index) * 64,
                                       outputs={"image": ImageArtifact(path=str(source), identity=checksum, content_sha256=checksum)})
            atomic_write_json(path, result.model_dump(mode="json"))
            candidate = result.candidate(path, "image")
            self.candidates.append(candidate)
            collection = NodeResultManifest(node_id="collection", node_kind=NodeKind.IMAGE_COLLECTION,
                signature=str(index + 2) * 64, outputs={"collection": ImageCollectionArtifact(candidates=(candidate,), identity=checksum)})
            path = directory / "collection" / "result.json"
            atomic_write_json(path, collection.model_dump(mode="json"))
            self.histories.insert(0, path)
        self.graph = WorkflowGraph(name="Saved choice", nodes=(
            ImageCollectionNode(id="collection", title="Collection"),
            ImageSelectionNode(id="choice", title="Choice", config=ImageSelectionConfig(selected=self.candidates[0].reference)),
        ), connections=(WorkflowConnection(id="wire", source=NodePortRef(node_id="collection", port="collection"),
                                           target=NodePortRef(node_id="choice", port="collection")),))

    async def test_export_reads_source_and_cached_payloads_once_and_checks_copied_bytes(self):
        revision = RevisionedComponent(name="export-test", revision="1")
        provenance = NodeExecutionProvenance(executor=revision, backend=revision)
        cache = WorkflowNodeCache(self.root / "cache")
        cases = [(self.candidates[0], (Path(self.candidates[0].image.path),), self.root / "source-export.png")]
        for index, (kind, artifact_type, port, count, suffix) in enumerate((
            (NodeKind.IMAGE_POSTPROCESS, ArtifactType.IMAGE, "image", 1, ".png"),
            (NodeKind.FRAME_POSTPROCESS, ArtifactType.IMAGE_SEQUENCE, "frames", 2, ".zip"),
        )):
            with cache.begin(str(index + 4) * 64, node_kind=kind, provenance=provenance) as write:
                paths = tuple(write.output_dir / f"{frame}.png" for frame in range(count))
                for frame, path in enumerate(paths):
                    Image.new("RGB", (16, 16), (frame * 50, 20, 30)).save(path)
                cached = write.publish({port: GeneratedNodeOutput(artifact_type=artifact_type, paths=paths)})
            result = NodeResultManifest(node_id=port, node_kind=kind, signature=cached.signature,
                                       outputs=dict(cached.outputs), cache_manifest=str(cached.manifest_path))
            manifest_path = self.root / f"{port}-result.json"
            atomic_write_json(manifest_path, result.model_dump(mode="json"))
            artifact = cached.outputs[port]
            if artifact_type == ArtifactType.IMAGE:
                candidate = result.candidate(manifest_path, port)
                paths = (Path(artifact.path),)
            else:
                candidate = MediaCandidate("Frames", None, manifest_path, cached.signature, port, artifact)
                paths = tuple(Path(path) for path in artifact.paths)
            cases.append((candidate, paths, self.root / f"{port}-export{suffix}"))

        original_open = Path.open
        reads = {}

        class CountedReader(io.BufferedReader):
            def readinto(self, buffer):
                size = super().readinto(buffer)
                reads[Path(self.name)] += size
                return size

        def count_payload_reads(path, mode="r", *args, **kwargs):
            if path in reads and mode == "rb":
                return CountedReader(io.FileIO(path, "r"))
            return original_open(path, mode, *args, **kwargs)

        results = WorkflowResults(self.graph, "choice", self.root)
        app = Mock()
        for candidate, paths, destination in cases:
            reads = dict.fromkeys(paths, 0)
            with patch.object(Path, "open", count_payload_reads):
                await results._export_result(app, candidate, destination)
            self.assertTrue(destination.is_file())
            self.assertEqual(reads, {path: path.stat().st_size for path in paths})
            self.assertEqual(app.notify.call_args.kwargs["title"], "Export completed")

        candidate, paths, _ = cases[1]
        original = paths[0].stat()
        changed = bytearray(paths[0].read_bytes())
        changed[-1] ^= 1
        paths[0].write_bytes(changed)
        os.utime(paths[0], ns=(original.st_atime_ns, original.st_mtime_ns))
        destination = self.root / "changed.png"
        await results._export_result(app, candidate, destination)
        self.assertEqual(app.notify.call_args.kwargs["title"], "Export failed")
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.root.glob(".aigen-export-*")))

    async def test_saved_selection_opens_before_newer_collection(self):
        async with open_editor(self.graph, self.root) as (app, editor, pilot, path):
            results = WorkflowResults(self.graph, "choice", self.root)
            with patch("aigen.workflow_results_tui.node_result_history", return_value=tuple(self.histories)):
                app.push_screen(results)
                await pilot.pause()
                await results.workers.wait_for_complete()
            self.assertEqual(results.manifest_path, self.candidates[0].manifest_path)
            self.assertEqual(results.candidates[results.highlighted_index].reference, self.candidates[0].reference)
            results.query_one("#result-history", Select).value = str(self.histories[0])
            await pilot.pause()
            await results.workers.wait_for_complete()
            self.assertEqual(results.candidates[results.highlighted_index].reference, self.candidates[1].reference)
            self.assertEqual(app.workflow_buffer.document.node("choice").config.selected, self.candidates[0].reference)

    async def test_export_survives_open_and_closing_results_with_completion_notice(self):
        started, release = Event(), Event()

        def delayed_export(*args, **kwargs):
            started.set()
            if not release.wait(10):
                raise TimeoutError("test did not release export")
            return export_artifact(*args, **kwargs)

        async with open_editor(self.graph, self.root) as (app, editor, pilot, path):
            results = WorkflowResults(self.graph, "choice", self.root)
            with patch("aigen.workflow_results_tui.node_result_history", return_value=tuple(self.histories)):
                app.push_screen(results)
                await pilot.pause()
                await results.workers.wait_for_complete()
            target = self.root / "export.png"
            with patch("aigen.workflow_results_tui.export_artifact", side_effect=delayed_export), \
                 patch("aigen.workflow_results_tui.open_artifact") as open_file, patch.object(app, "notify") as notify:
                try:
                    results._export_destination(self.candidates[0], str(target))
                    self.assertTrue(await asyncio.to_thread(started.wait, 5))
                    exports = tuple(worker for worker in app.workers if worker.group == "artifact-export")
                    await results.apply_action("result-open", self.candidates[0]).wait()
                    open_file.assert_called_once()
                    results.dismiss(None)
                    await pilot.pause()
                finally:
                    release.set()
                await app.workers.wait_for_complete(exports)
                self.assertEqual(sha256_file(target), self.candidates[0].image.content_sha256)
                self.assertEqual(notify.call_args.kwargs["title"], "Export completed")
                self.assertFalse(exports[0].is_cancelled)


if __name__ == "__main__":
    unittest.main()
