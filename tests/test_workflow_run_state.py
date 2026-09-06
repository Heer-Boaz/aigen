import unittest

from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_graph import (
    ImageEditConfig, ImageEditNode, ImageSourceConfig, ImageSourceNode,
    NodePortRef, WorkflowConnection, WorkflowGraph,
)
from aigen.workflow_run_state import WorkflowRunState


class WorkflowRunStateTests(unittest.TestCase):
    def setUp(self):
        nodes = (
            ImageSourceNode(id="a", title="A", config=ImageSourceConfig(path="a.png")),
            ImageSourceNode(id="b", title="B", config=ImageSourceConfig(path="b.png")),
            *(ImageEditNode(id=name, title=name, config=ImageEditConfig(backend="flux2-klein")) for name in ("left", "right", "join", "end")),
        )
        routes = (("a", "left", 0), ("b", "right", 0), ("left", "join", 0), ("right", "join", 1), ("join", "end", 0))
        self.graph = WorkflowGraph(name="Branches", nodes=nodes, connections=tuple(
            WorkflowConnection(id=f"wire{index}", source=NodePortRef(node_id=source, port="image"),
                               target=NodePortRef(node_id=target, port="references"), order=order)
            for index, (source, target, order) in enumerate(routes)
        ))
        self.state = WorkflowRunState()
        self.state.start(self.graph)
        for node in nodes:
            self.state.update(node.id, "completed")
        self.completed = {node.id: "completed" for node in nodes}

    def test_content_invalidates_only_affected_branches_and_undo_restores(self):
        buffer = WorkflowEditBuffer(self.graph)
        buffer.update_node_config("a", "path", "changed.png")
        expected = {**self.completed, **dict.fromkeys(("a", "left", "join", "end"), "outdated")}
        self.assertEqual(self.state.project(buffer.document), expected)
        buffer.undo()
        self.assertEqual(self.state.project(buffer.document), self.completed)
        buffer.redo()
        self.assertEqual(self.state.project(buffer.document), expected)
        # Unchanged graph instances reuse validity during progress updates.
        self.state.update("a", "reused")
        self.assertEqual(self.state.project(buffer.document), expected)
        buffer.undo()
        self.assertEqual(self.state.project(buffer.document)["a"], "reused")

    def test_presentation_and_connection_identity_are_not_execution_changes(self):
        buffer = WorkflowEditBuffer(self.graph)
        buffer.rename_graph("Renamed")
        buffer.update_node_title("a", "Another title")
        buffer.move_node("a", x=90, y=24)
        self.assertEqual(self.state.project(buffer.document), self.completed)
        graph = WorkflowGraph(name=self.graph.name, nodes=self.graph.nodes, connections=tuple(
            wire.model_copy(update={"id": f"changed{index}", "order": wire.order * 10})
            for index, wire in enumerate(reversed(self.graph.connections))
        ))
        self.assertEqual(self.state.project(graph), self.completed)

    def test_reorder_reconnect_and_removal_invalidate_descendants(self):
        wires = tuple(wire.model_copy(update={"order": 1 - wire.order}) if wire.target.node_id == "join" else wire
                      for wire in self.graph.connections)
        graph = WorkflowGraph(name="Reordered", nodes=self.graph.nodes, connections=wires)
        expected = {**self.completed, "join": "outdated", "end": "outdated"}
        self.assertEqual(self.state.project(graph), expected)
        wires = tuple(wire.model_copy(update={"source": NodePortRef(node_id="b", port="image")})
                      if wire.target.node_id == "left" else wire for wire in self.graph.connections)
        graph = WorkflowGraph(name="Reconnected", nodes=self.graph.nodes, connections=wires)
        self.assertEqual(self.state.project(graph), {**expected, "left": "outdated"})
        buffer = WorkflowEditBuffer(self.graph)
        buffer.delete_node("a")
        self.assertEqual(self.state.project(buffer.document), {key: value for key, value in {**expected, "left": "outdated"}.items() if key != "a"})
        buffer.undo()
        self.assertEqual(self.state.project(buffer.document), self.completed)

    def test_new_request_and_document_reset_statuses(self):
        buffer = WorkflowEditBuffer(self.graph)
        buffer.update_node_config("a", "path", "changed.png")
        self.state.start(buffer.document)
        self.assertEqual(self.state.project(buffer.document), dict.fromkeys(self.completed, "queued"))
        self.state.clear()
        self.assertEqual(self.state.project(self.graph), {})


if __name__ == "__main__":
    unittest.main()
