from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from aigen.manifest_io import atomic_write_json
from aigen.workflow_document_io import save_workflow_document
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_graph import (
    ImageCollectionNode, ImageEditConfig, ImageEditNode, ImageResultReference, ImageSelectionNode,
    ImageSourceConfig, ImageSourceNode, NodePortRef, WorkflowConnection, WorkflowGraph,
)
from aigen.workflow_run_state import WorkflowExecutionSnapshot
from aigen.workflow_task import RecordedWorkflowRun, choice_frontier, next_workflow_step, project_results, read_workflow_runs


def choice_graph(*, nested=False, parallel=False):
    nodes = [
        ImageSourceNode(id="source", title="Source", config=ImageSourceConfig(path="source.png")),
        ImageEditNode(id="edit", title="Edit", config=ImageEditConfig(backend="flux2-klein", seed_mode="random")),
        ImageCollectionNode(id="collection", title="Candidates"),
        ImageSelectionNode(id="choice", title="Choice"),
        ImageEditNode(id="end", title="Final edit", config=ImageEditConfig(backend="flux2-klein")),
    ]
    routes = [("source", "image", "edit", "references", 0), ("edit", "image", "collection", "images", 0),
              ("collection", "collection", "choice", "collection", 0)]
    if nested or parallel:
        nodes.extend((ImageCollectionNode(id="collection2", title="Second candidates"),
                      ImageSelectionNode(id="choice2", title="Second choice")))
        routes.extend((("choice" if nested else "source", "image", "collection2", "images", 0),
                       ("collection2", "collection", "choice2", "collection", 0),
                       ("choice2", "image", "end", "references", 0)))
        if parallel:
            routes.append(("choice", "image", "end", "references", 1))
    else:
        routes.append(("choice", "image", "end", "references", 0))
    return WorkflowGraph(name="Choices", nodes=tuple(nodes), connections=tuple(
        WorkflowConnection(id=f"wire{index}", source=NodePortRef(node_id=source, port=output),
                           target=NodePortRef(node_id=target, port=input_port), order=order)
        for index, (source, output, target, input_port, order) in enumerate(routes)
    ))


def chosen_reference():
    return ImageResultReference(manifest_path="saved/result.json", producer_signature="a" * 64,
                                output_port="image", artifact_identity="b" * 64)


class WorkflowTaskTests(unittest.TestCase):
    def test_nested_choices_advance_one_dependency_stage_at_a_time(self):
        graph = choice_graph(nested=True)
        self.assertEqual(choice_frontier(graph), (("choice", "collection"),))
        step = next_workflow_step(graph, {})
        self.assertEqual((step.action, step.targets), ("run", ("collection",)))
        step = next_workflow_step(graph, {"collection": Path("first.json")})
        self.assertEqual((step.action, step.selection_id, step.result_path), ("choose", "choice", Path("first.json")))
        buffer = WorkflowEditBuffer(graph)
        buffer.select_image("choice", chosen_reference())
        self.assertEqual(choice_frontier(buffer.document), (("choice2", "collection2"),))
        buffer.select_image("choice2", chosen_reference())
        self.assertEqual(choice_frontier(buffer.document), ())
        self.assertEqual(next_workflow_step(buffer.document, {}).label, "Continue")
        step = next_workflow_step(buffer.document, {"end": Path("final.json")})
        self.assertEqual((step.action, step.targets), ("results", ("end",)))

    def test_parallel_choices_share_a_run_and_each_remains_explicit(self):
        graph = choice_graph(parallel=True)
        step = next_workflow_step(graph, {})
        self.assertEqual(set(step.targets), {"collection", "collection2"})
        step = next_workflow_step(graph, {"collection2": Path("second.json")})
        self.assertEqual((step.action, step.selection_id), ("choose", "choice2"))
        buffer = WorkflowEditBuffer(graph)
        buffer.select_image("choice2", chosen_reference())
        self.assertEqual(next_workflow_step(buffer.document, {}).targets, ("collection",))

    def test_missing_input_and_explicit_scope_do_not_trigger_the_compiler(self):
        graph = choice_graph()
        disconnected = graph.model_copy(update={"connections": tuple(wire for wire in graph.connections
                                                                     if wire.target.node_id != "choice")})
        step = next_workflow_step(disconnected, {})
        self.assertEqual((step.action, step.selection_id), ("connect", "choice"))
        self.assertEqual(choice_frontier(graph, ("collection",)), ())
        self.assertEqual(next_workflow_step(graph, {}, ("collection",)).targets, ("collection",))
        with self.assertRaisesRegex(ValueError, "unknown workflow targets"):
            next_workflow_step(graph, {}, ("absent",))

    def test_history_matches_execution_settings_and_pinned_choices_cut_upstream_changes(self):
        graph = choice_graph()
        first = RecordedWorkflowRun(WorkflowExecutionSnapshot.capture(graph), {"collection": Path("first.json")})
        buffer = WorkflowEditBuffer(graph)
        buffer.update_node_config("edit", "prompt", "Changed settings")
        latest, applicable = project_results(buffer.document, (first,))
        self.assertEqual(latest, first.results)
        self.assertEqual(applicable, {})
        self.assertEqual(next_workflow_step(buffer.document, applicable).action, "run")
        second = RecordedWorkflowRun(WorkflowExecutionSnapshot.capture(buffer.document), {"collection": Path("second.json")})
        buffer.undo()
        latest, applicable = project_results(buffer.document, (second, first))
        self.assertEqual(latest["collection"], Path("second.json"))
        self.assertEqual(applicable["collection"], Path("first.json"))
        buffer.move_node("edit", x=50, y=10)
        buffer.update_node_title("edit", "Renamed")
        self.assertEqual(project_results(buffer.document, (first,))[1], first.results)

        buffer.select_image("choice", chosen_reference())
        completed = RecordedWorkflowRun(WorkflowExecutionSnapshot.capture(buffer.document), {"end": Path("final.json")})
        buffer.delete_node("edit")
        buffer.delete_node("source")
        applicable = project_results(buffer.document, (completed,))[1]
        self.assertEqual(next_workflow_step(buffer.document, applicable).action, "results")
        buffer.update_node_config("end", "prompt", "Another edit")
        self.assertEqual(project_results(buffer.document, (completed,))[1], {})

    def test_read_history_uses_persisted_request_including_random_seed_mode(self):
        graph = choice_graph()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot.json"
            save_workflow_document(graph, snapshot)
            run = root / "runs" / graph.workflow_id / "2026-09-08"
            atomic_write_json(run / "run.json", {"workflow_snapshot": str(snapshot), "nodes": {"collection": "result.json"}})
            records = read_workflow_runs(root, graph.workflow_id)
            self.assertEqual(records[0].snapshot.nodes["edit"].config.seed_mode, "random")
            applicable = project_results(graph, records)[1]
            self.assertEqual(next_workflow_step(graph, applicable).action, "choose")


if __name__ == "__main__":
    unittest.main()
