import argparse
import io
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from aigen.progress import SILENT_STATUS
from aigen.workflow_commands import run_workflow_command
from aigen.workflow_compilation import compile_workflow_run
from aigen.workflow_document_io import save_workflow_document
from aigen.workflow_execution import _write_node_manifest, execute_workflow
from aigen.workflow_graph import ImageSourceConfig, ImageSourceNode, WorkflowGraph
from aigen.workflow_run_records import finalize_terminated_workflow, new_workflow_run_id, workflow_run_path


class WorkflowInterruptionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = self.root / "source.png"
        Image.new("RGB", (16, 16)).save(source)
        self.graph = WorkflowGraph(name="Interruption", nodes=(
            ImageSourceNode(id="source", title="Source", config=ImageSourceConfig(path=str(source))),
        ), connections=())
        self.document = self.root / "workflow.json"
        save_workflow_document(self.graph, self.document)

    def assert_interrupted(self, root):
        states = list(root.glob("runs/*/attempt-*/run.json"))
        self.assertEqual(len(states), 1)
        state = json.loads(states[0].read_text())
        self.assertEqual(state["status"], "interrupted")
        interruption = json.loads((states[0].parent / "interrupted.json").read_text())
        self.assertEqual(interruption["completed_nodes"], ["source"])
        self.assertEqual(interruption["node_ids"], [])
        self.assertTrue(interruption["message"])

    def test_direct_ctrl_c_persists_terminal_state_and_completed_result(self):
        def interrupt(event):
            if event.get("status") == "completed":
                raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            execute_workflow(compile_workflow_run(self.graph), runs_root=self.root,
                             progress=SILENT_STATUS, event_sink=interrupt)
        self.assert_interrupted(self.root)

    def test_ctrl_c_after_manifest_publication_before_in_memory_registration(self):
        def interrupt_after_write(*args, **kwargs):
            _write_node_manifest(*args, **kwargs)
            raise KeyboardInterrupt()

        with patch("aigen.workflow_execution._write_node_manifest", side_effect=interrupt_after_write):
            with self.assertRaises(KeyboardInterrupt):
                execute_workflow(compile_workflow_run(self.graph), runs_root=self.root, progress=SILENT_STATUS)
        self.assert_interrupted(self.root)

    def test_cli_signals_share_interruption_and_restore_handlers(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signum):
                root = self.root / signum.name
                previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

                def run(workflow, **kwargs):
                    def interrupt(event):
                        if event.get("status") == "completed":
                            try:
                                signal.raise_signal(signum)
                            except Exception as error:
                                raise RuntimeError("backend failure translation") from error
                    kwargs["event_sink"] = interrupt
                    return execute_workflow(workflow, **kwargs)

                args = argparse.Namespace(workflow_operation="run", input=self.document,
                                          targets=None, runs_root=root, run_id=None)
                stderr = io.StringIO()
                with patch("aigen.workflow_commands.execute_workflow", side_effect=run):
                    code = run_workflow_command(args, io.StringIO(), stderr, progress=SILENT_STATUS)
                self.assertEqual(code, 130)
                self.assertEqual(json.loads(stderr.getvalue())["error"], "WorkflowInterrupted")
                self.assertEqual({sig: signal.getsignal(sig) for sig in previous}, previous)
                self.assert_interrupted(root)

    def test_supervisor_retains_result_published_before_run_index_update(self):
        graph = self.graph.model_copy(update={"nodes": (*self.graph.nodes,
            self.graph.nodes[0].model_copy(update={"id": "unstarted"}))})
        run_id = new_workflow_run_id()
        run_dir = workflow_run_path(self.root, graph.workflow_id, run_id)

        def exit_after_publication(event):
            if event.get("node_id") == "source" and event.get("status") == "completed":
                raise SystemExit(137)

        with self.assertRaises(SystemExit):
            execute_workflow(compile_workflow_run(graph), runs_root=self.root, run_id=run_id,
                             progress=SILENT_STATUS, event_sink=exit_after_publication)
        state_path = run_dir / "run.json"
        before = json.loads(state_path.read_text())
        self.assertEqual(before["status"], "running")
        self.assertEqual(before["nodes"], {})
        manifest = run_dir / "nodes" / "source" / "result.json"
        published = manifest.read_bytes()

        finalize_terminated_workflow(run_dir, "Worker terminated")

        after = json.loads(state_path.read_text())
        self.assertEqual(after["status"], "interrupted")
        self.assertEqual(after["nodes"], {"source": str(manifest)})
        interruption = json.loads((run_dir / "interrupted.json").read_text())
        self.assertEqual(interruption["completed_nodes"], ["source"])
        self.assertEqual(interruption["node_ids"], ["unstarted"])
        self.assertEqual(manifest.read_bytes(), published)
        finalized = state_path.read_bytes()
        finalize_terminated_workflow(run_dir, "A repeated notification")
        self.assertEqual(state_path.read_bytes(), finalized)

    def test_assigned_run_id_cannot_escape_or_overwrite_an_attempt(self):
        compiled = compile_workflow_run(self.graph)
        with self.assertRaisesRegex(ValueError, "invalid workflow run id"):
            execute_workflow(compiled, runs_root=self.root, run_id="../../outside", progress=SILENT_STATUS)
        run_id = new_workflow_run_id()
        execute_workflow(compiled, runs_root=self.root, run_id=run_id, progress=SILENT_STATUS)
        run_dir = workflow_run_path(self.root, self.graph.workflow_id, run_id)
        state_path = run_dir / "run.json"
        completed = state_path.read_bytes()
        with self.assertRaises(FileExistsError):
            execute_workflow(compiled, runs_root=self.root, run_id=run_id, progress=SILENT_STATUS)
        finalize_terminated_workflow(run_dir, "Already finished")
        self.assertEqual(state_path.read_bytes(), completed)
        finalize_terminated_workflow(self.root / "never-started", "Cancelled during startup")
        self.assertFalse((self.root / "never-started").exists())

if __name__ == "__main__":
    unittest.main()
