import os
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from aigen.character_reference_pack import build_character_reference_pack, load_character_reference_pack
from aigen.manifest_io import copy_sha256, sha256_file
from aigen.progress import SILENT_STATUS
from aigen.workflow_compilation import compile_workflow_run
from aigen.workflow_execution import execute_workflow, WorkflowExecutionError
from aigen.workflow_graph import LoraSourceConfig, LoraSourceNode, ReferencePackConfig, ReferencePackNode, WorkflowGraph
from aigen.workflow_sources import WorkflowSourceStore


class WorkflowSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_capture_reuses_storage_without_recopy_and_detects_preserved_mtime_change(self):
        source = self.root / "input.bin"
        original = b"before" * 100_000
        source.write_bytes(original)
        stat = source.stat()
        root = self.root / "snapshots"
        with patch("aigen.workflow_sources.copy_sha256", wraps=copy_sha256) as copy:
            first = WorkflowSourceStore(root).capture(source)
            second = WorkflowSourceStore(root).capture(source)
            self.assertEqual(copy.call_count, 1)
            self.assertEqual(first, second)
            source.write_bytes(b"after!" * 100_000)
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            changed = WorkflowSourceStore(root).capture(source)
            self.assertEqual(copy.call_count, 2)
        self.assertNotEqual(first.sha256, changed.sha256)
        self.assertEqual(first.path.read_bytes(), original)
        self.assertNotEqual(first.path.stat().st_ino, source.stat().st_ino)
        self.assertEqual(changed.path.read_bytes(), source.read_bytes())
        with self.assertRaisesRegex(ValueError, "recorded contents"):
            WorkflowSourceStore(root).capture(source, expected_sha256=first.sha256)

    def test_corrupt_snapshot_fails_instead_of_recapturing(self):
        source = self.root / "input.bin"
        source.write_bytes(b"original")
        root = self.root / "snapshots"
        captured = WorkflowSourceStore(root).capture(source)
        captured.path.chmod(0o644)
        captured.path.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "snapshot contents changed"):
            WorkflowSourceStore(root).capture(source)

    def test_reference_pack_keeps_order_and_remains_loadable_without_originals(self):
        paths = (self.root / "second.png", self.root / "first.png")
        for index, path in enumerate(paths):
            Image.new("RGB", (8, 8), (index * 80, 10, 20)).save(path)
        pack_path = self.root / "pack.json"
        build_character_reference_pack(character_id="test-subject", references=dict(zip(("b", "a"), paths)),
                                       output=pack_path, overwrite=False)
        graph = WorkflowGraph(name="Pack capture", nodes=(ReferencePackNode(
            id="pack", title="Pack", config=ReferencePackConfig(path=str(pack_path))),), connections=())
        compiled = compile_workflow_run(graph)
        original_pack = pack_path.read_bytes()
        build_character_reference_pack(character_id="test-subject", references=dict(zip(("a", "b"), reversed(paths))),
                                       output=pack_path, overwrite=True)
        with self.assertRaisesRegex(WorkflowExecutionError, "pack changed after compilation"):
            execute_workflow(compiled, runs_root=self.root / "reordered", progress=SILENT_STATUS)
        pack_path.write_bytes(original_pack)
        result = execute_workflow(compiled, runs_root=self.root / "runs", progress=SILENT_STATUS)
        artifact = result.terminal_outputs["pack"]["pack"]
        self.assertEqual(artifact.reference_sha256s, tuple(sha256_file(path) for path in paths))
        for path in (*paths, pack_path):
            path.unlink()
        saved = load_character_reference_pack(Path(artifact.path))
        self.assertEqual(tuple(saved.references), ("b", "a"))
        self.assertEqual(tuple(str(path) for path in saved.references.values()), artifact.references)

    def test_pack_changes_after_compilation_fail_before_generation(self):
        source = self.root / "source.png"
        Image.new("RGB", (8, 8)).save(source)
        path = self.root / "pack.json"
        build_character_reference_pack(character_id="test-subject", references={"a": source}, output=path, overwrite=False)
        graph = WorkflowGraph(name="Pack revision", nodes=(ReferencePackNode(
            id="pack", title="Pack", config=ReferencePackConfig(path=str(path))),), connections=())
        compiled = compile_workflow_run(graph)
        build_character_reference_pack(character_id="test-subject", references={"b": source}, output=path, overwrite=True)
        with self.assertRaisesRegex(WorkflowExecutionError, "pack changed after compilation"):
            execute_workflow(compiled, runs_root=self.root / "runs", progress=SILENT_STATUS)

    def test_replaced_lora_architecture_fails_and_fresh_compilation_reads_new_header(self):
        path = self.root / "adapter.safetensors"

        def write_lora(target):
            header = json.dumps({
                f"{target}.lora_A.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]},
                f"{target}.lora_B.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [4, 8]},
            }).encode().ljust(1024, b" ")
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)

        write_lora("diffusion_model.double_blocks.0.img_attn.qkv")
        graph = WorkflowGraph(name="LoRA intake", nodes=(LoraSourceNode(
            id="adapter", title="Adapter", config=LoraSourceConfig(path=str(path))),), connections=())
        compiled = compile_workflow_run(graph)
        self.assertEqual(compiled.node("adapter").config.architecture, "flux2-klein")
        previous = path.stat()
        write_lora("diffusion_model.transformer_blocks.0.attn.to_q")
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        with self.assertRaisesRegex(WorkflowExecutionError, "LoRA architecture changed after compilation"):
            execute_workflow(compiled, runs_root=self.root / "runs", progress=SILENT_STATUS)
        self.assertFalse(list(self.root.glob("runs/runs/*/*/nodes/adapter/result.json")))
        fresh = compile_workflow_run(graph)
        self.assertEqual(fresh.node("adapter").config.architecture, "qwen-image")
        execute_workflow(fresh, runs_root=self.root / "runs", progress=SILENT_STATUS)


if __name__ == "__main__":
    unittest.main()
