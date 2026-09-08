from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Collapsible, Input, Label, Select, Static, TextArea

from aigen.generation.animegen_i2v import (
    ANIMEGEN_PRECISIONS,
    ANIMEGEN_SAMPLINGS,
)
from aigen.generation.image_batch_postprocess import (
    image_batch_postprocess_model_names,
)
from aigen.generation.image_edit import (
    IMAGE_EDIT_BACKENDS,
    image_edit_backend_settings,
)
from aigen.workflow_edit_buffer import WorkflowPropertyEdit
from aigen.workflow_property_schema import primary_fields
from aigen.workflow_graph import (
    AnimeGenI2VNode,
    FramePostprocessNode,
    ImageEditNode,
    LoraSourceNode,
    CharacterEditNode,
    BindMaskNode, SamSegmentNode, CharacterRefineNode,
    ImageSelectionNode,
    ImagePostprocessNode,
    VosrPostprocessConfig,
    WorkflowGraph,
    WorkflowNode,
    Ltx23Node, HunyuanI2VNode, PositionedKeyframeNode, AudioSourceNode, AssembleVideoNode,
)
from aigen.generation.ltx23_settings import LTX23_MODEL_TYPES, LTX23_SOLVERS, LTX23_PHASES
from aigen.generation.hunyuanvideo15 import HUNYUANVIDEO15_STEPS
from aigen.workflow_property_widgets import (
    PROPERTY_ROW_MIN_WIDTH,
    PropertyInput,
    PropertyRow,
    PropertyTextArea,
)


class ConnectionOrderSelect(Select[int]):
    def __init__(self, connection_id: str, position: int, count: int) -> None:
        super().__init__(
            [(f"Position {index + 1}", index) for index in range(count)],
            value=position,
            allow_blank=False,
            compact=True,
            id="workflow-input-order",
        )
        self.connection_id = connection_id
        self.position = position


class WorkflowInspector(VerticalScroll):
    class DraftsChanged(Message):
        def __init__(self, modified: bool) -> None:
            super().__init__()
            self.modified = modified

    DEFAULT_CSS = """
    WorkflowInspector {
        width: 2fr;
        min-width: 0;
        height: 1fr;
        border: solid #5b496d;
        background: #1c1724;
        padding: 0 1;
        scrollbar-size-vertical: 1;
    }

    WorkflowInspector Collapsible { padding: 0; border: none; margin: 1 0 0 0; }
    WorkflowInspector Collapsible Contents { padding: 0 0 0 1; }

    WorkflowInspector .workflow-inspector-heading {
        height: 1;
        text-style: bold;
        color: #d8c5eb;
    }

    WorkflowInspector .workflow-inspector-kind {
        height: 1;
        color: #9e8cad;
        margin-bottom: 1;
    }

    WorkflowInspector .workflow-inspector-empty {
        color: #9e8cad;
        height: auto;
    }
    """

    def __init__(
        self,
        document: WorkflowGraph,
        selected_node_id: str | None,
        selected_connection_id: str | None,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.document = document
        self._node_id = selected_node_id
        self._connection_id = selected_connection_id
        self._draft_editors: set[PropertyInput | PropertyTextArea] = set()

    @property
    def horizontal_minimum_width(self) -> int:
        return (
            PROPERTY_ROW_MIN_WIDTH
            + self.styles.gutter.width
        )

    def compose(self) -> ComposeResult:
        if self._node_id is None:
            if self._connection_id is not None:
                connection = next(wire for wire in self.document.connections if wire.id == self._connection_id)
                source = self.document.node(connection.source.node_id)
                target = self.document.node(connection.target.node_id)
                yield Label("Connection", classes="workflow-inspector-heading")
                yield Static(f"{source.title}.{connection.source.port}\n→ {target.title}.{connection.target.port}",
                             markup=False, classes="workflow-inspector-empty")
                siblings = self.document.incoming_connections()[target.id][connection.target.port]
                if len(siblings) > 1:
                    yield Label("Input order", classes="workflow-inspector-heading")
                    yield ConnectionOrderSelect(connection.id, siblings.index(connection), len(siblings))
                    yield Static("\n".join(f"{index + 1}. {self.document.node(wire.source.node_id).title}.{wire.source.port}"
                                           for index, wire in enumerate(siblings)), markup=False,
                                 classes="workflow-inspector-empty")
            else:
                yield Label("Workflow", classes="workflow-inspector-heading")
                yield PropertyRow(node_id=None, field_name="name", label="Name", value=self.document.name, annotation=str)
            return

        node = self.document.node(self._node_id)
        yield Label(node.title, id="workflow-node-title", classes="workflow-inspector-heading", markup=False)
        visible = _visible_config_fields(node)
        fields = _config_fields(node)
        primary = primary_fields(node, fields)
        rows = {
            name: PropertyRow(node_id=node.id, field_name=name, label=_field_label(name),
                              value=getattr(node.config, name), annotation=type(node.config).model_fields[name].annotation,
                              multiline=name in {"prompt", "negative_prompt"}, visible=name in visible,
                              browse=name == "path", options=_node_property_options(node, name),
                              placeholder=_field_placeholder(node, name))
            for name in fields
        }
        yield from (rows[name] for name in primary)
        with Collapsible(title="Advanced", collapsed=True, id="workflow-advanced"):
            yield PropertyRow(node_id=node.id, field_name="title", label="Title", value=node.title, annotation=str)
            yield from (rows[name] for name in fields if name not in primary)
        text = _node_description(node)
        with Collapsible(title="Node details", collapsed=not isinstance(node, ImageSelectionNode), id="workflow-node-help") as help_group:
            help_group.display = bool(text)
            yield Static(text, id="workflow-node-description", markup=False)

    @on(Input.Changed)
    @on(TextArea.Changed)
    def property_changed(self, event: Input.Changed | TextArea.Changed) -> None:
        editor = event.control
        if isinstance(editor, (PropertyInput, PropertyTextArea)):
            editor.remove_class("-invalid")
            was_modified = bool(self._draft_editors)
            value = editor.text if isinstance(editor, PropertyTextArea) else editor.value
            if editor.is_mounted and value != editor.original_value:
                self._draft_editors.add(editor)
            else:
                self._draft_editors.discard(editor)
            if bool(self._draft_editors) != was_modified:
                self.post_message(self.DraftsChanged(bool(self._draft_editors)))

    def property_drafts(self) -> tuple[WorkflowPropertyEdit, ...]:
        drafts: list[WorkflowPropertyEdit] = []
        for row in self.query(PropertyRow):
            editor = row.editor
            if not row.display or not isinstance(editor, (PropertyInput, PropertyTextArea)):
                continue
            value = editor.text if isinstance(editor, PropertyTextArea) else editor.value
            if value != editor.original_value:
                drafts.append(WorkflowPropertyEdit(
                    node_id=row.node_id, field_name=row.field_name, raw_value=value,
                ))
        return tuple(drafts)

    def accept_properties(self) -> None:
        for row in self.query(PropertyRow):
            editor = row.editor
            if isinstance(editor, PropertyTextArea):
                editor.original_value = editor.text
                editor.history.checkpoint()
            elif isinstance(editor, PropertyInput):
                editor.original_value = editor.value
        if self._draft_editors:
            self._draft_editors.clear()
            self.post_message(self.DraftsChanged(False))

    def focus_invalid_draft(self, edit: WorkflowPropertyEdit) -> None:
        row = next(
            row for row in self.query(PropertyRow)
            if row.node_id == edit.node_id and row.field_name == edit.field_name
        )
        for ancestor in row.ancestors:
            if isinstance(ancestor, Collapsible):
                ancestor.collapsed = False
        row.editor.add_class("-invalid")
        self.call_after_refresh(row.editor.focus)

    async def show(
        self,
        document: WorkflowGraph,
        selected_node_id: str | None,
        selected_connection_id: str | None,
    ) -> None:
        previous_projection = (self.document.workflow_id, self._projection())
        previous_binding = self._binding()
        self.document = document
        self._node_id = selected_node_id
        self._connection_id = selected_connection_id
        if (document.workflow_id, self._projection()) == previous_projection:
            return
        if self._binding() != previous_binding or self._connection_id is not None:
            if self._draft_editors:
                self._draft_editors.clear()
                self.post_message(self.DraftsChanged(False))
            await self.recompose()
            return
        node = document.node(self._node_id) if self._node_id is not None else None
        visible_fields = _visible_config_fields(node) if node is not None else ()
        for row in self.query(PropertyRow):
            if row.node_id is None:
                row.show_value(document.name)
            elif row.field_name == "title":
                row.show_value(node.title)
            else:
                row.show_value(
                    getattr(node.config, row.field_name),
                    options=_node_property_options(node, row.field_name),
                )
                row.display = row.field_name in visible_fields
                if isinstance(row.editor, PropertyInput):
                    row.editor.placeholder = _field_placeholder(node, row.field_name)
        if node is not None:
            self.query_one("#workflow-node-title", Label).update(node.title)
            description = self.query_one("#workflow-node-description", Static)
            text = _node_description(node)
            description.update(text)
            self.query_one("#workflow-node-help").display = bool(text)

    def _binding(self) -> tuple[object, ...]:
        node = self.document.node(self._node_id) if self._node_id is not None else None
        return (
            self.document.workflow_id,
            self._node_id,
            self._connection_id,
            (node.kind, type(node.config)) if node is not None else None,
        )

    def _projection(self) -> tuple[object, ...]:
        if self._node_id is not None:
            node = self.document.node(self._node_id)
            return (
                self._node_id,
                node.kind,
                node.title,
                node.config,
            )
        if self._connection_id is not None:
            connection = next(
                connection
                for connection in self.document.connections
                if connection.id == self._connection_id
            )
            return (
                self._connection_id,
                connection.source,
                connection.target,
                connection.order,
                self.document.node(connection.source.node_id).title,
                self.document.node(connection.target.node_id).title,
                tuple((sibling.id, sibling.source, self.document.node(sibling.source.node_id).title)
                      for sibling in self.document.incoming_connections()[connection.target.node_id][connection.target.port]),
            )
        return (self.document.name, None)


def _field_label(field_name: str) -> str:
    return field_name.replace("_", " ").capitalize()


def _field_placeholder(node: WorkflowNode, field_name: str) -> str:
    if isinstance(node, (ImageEditNode, CharacterEditNode)):
        if field_name in {"steps", "guidance", "strength"}:
            value = getattr(image_edit_backend_settings(node.config.backend), field_name)
            return f"Default: {value}" if value is not None else "Backend default"
        if field_name in {"width", "height", "aspect_ratio"}:
            return "Auto"
    if isinstance(node, CharacterRefineNode) and field_name == "max_side":
        return "Native source size"
    return ""


def _node_property_options(
    node: WorkflowNode,
    field_name: str,
) -> tuple[tuple[str, object], ...] | None:
    values: tuple[str, ...] | None = None
    if isinstance(node, (ImageEditNode, CharacterEditNode)):
        if isinstance(node, CharacterEditNode) and field_name == "pose_mode" and node.config.backend == "flux2-klein":
            values = ("native",)
        elif field_name == "backend":
            if isinstance(node, CharacterEditNode):
                from aigen.character_edit import CHARACTER_EDIT_BACKENDS
                values = CHARACTER_EDIT_BACKENDS
            else:
                values = IMAGE_EDIT_BACKENDS
        elif field_name in {"sampler", "scheduler"}:
            settings = image_edit_backend_settings(node.config.backend)
            values = getattr(settings, f"{field_name}s")
            default = getattr(settings, field_name)
            return ((f"Backend default ({default})", None),) + tuple(
                (value, value) for value in values
            )
    elif isinstance(node, (ImagePostprocessNode, FramePostprocessNode)):
        if field_name == "model":
            values = image_batch_postprocess_model_names()
    elif isinstance(node, AnimeGenI2VNode):
        if field_name == "sampling":
            values = ANIMEGEN_SAMPLINGS
        elif field_name == "precision":
            values = ANIMEGEN_PRECISIONS
    elif isinstance(node, Ltx23Node):
        if field_name == "model":
            values = tuple(LTX23_MODEL_TYPES)
        elif field_name == "solver":
            values = tuple(sorted(LTX23_SOLVERS))
        elif field_name == "phases":
            return tuple((str(value), value) for value in sorted(LTX23_PHASES))
    elif isinstance(node, HunyuanI2VNode) and field_name == "steps":
        return tuple((str(value), value) for value in sorted(HUNYUANVIDEO15_STEPS))
    if values is None:
        return None
    return tuple((value, value) for value in values)


def _config_fields(node: WorkflowNode) -> tuple[str, ...]:
    if isinstance(node, (ImageSelectionNode, BindMaskNode)):
        return ()
    return tuple(type(node.config).model_fields)


def _visible_config_fields(node: WorkflowNode) -> tuple[str, ...]:
    fields = _config_fields(node)
    hidden: set[str] = set()
    if isinstance(node, CharacterEditNode) and node.config.backend == "flux2-klein":
        hidden.update(("structure_control", "max_sequence_length", "guidance_scale"))
    if (
        "seed_mode" in type(node.config).model_fields
        and getattr(node.config, "seed_mode") == "random"
    ):
        hidden.add("seed")
    if (
        isinstance(node, (ImagePostprocessNode, FramePostprocessNode))
        and isinstance(node.config, VosrPostprocessConfig)
    ):
        hidden.add(
            "scale"
            if node.config.sizing == "long-side"
            else "long_side"
        )
    return tuple(field for field in fields if field not in hidden)

def _node_description(node: WorkflowNode) -> str:
    paragraphs: list[str] = []
    if isinstance(node, ImageSelectionNode):
        selected = node.config.selected
        paragraphs.append(
            f"Chosen image: {selected.artifact_identity[:16]}\nResults changes this choice; Continue uses this recorded image."
            if selected else "Run the connected collection, then choose an image in Results.",
        )
    if isinstance(node, LoraSourceNode):
        paragraphs.append("Import a trained LoRA for a matching image backend. Local training currently supports FLUX.1; Klein and Qwen LoRAs can be imported. Dataset preparation creates image/caption pairs.")
    if isinstance(node, Ltx23Node):
        paragraphs.append(f"Keyframes: positions 0–{node.config.frames - 1}. Canvas rounds up to multiples of 64px before fitting. No generated audio. Pad uses white.")
    elif isinstance(node, AnimeGenI2VNode):
        paragraphs.append("Start image and optional end image. Canvas follows the start image's aspect. No generated audio. Pad uses white.")
    elif isinstance(node, HunyuanI2VNode):
        paragraphs.append("One start image; native 480p aspect buckets; 24 FPS. Frames: 4n+1. No generated audio.")
    elif isinstance(node, PositionedKeyframeNode):
        paragraphs.append("Frame position starts at 0. Connect this keyframe to LTX.")
    elif isinstance(node, AudioSourceNode):
        paragraphs.append("Container stream index; leave blank for the first audio stream. Replacement audio starts at video time zero.")
    elif isinstance(node, AssembleVideoNode):
        paragraphs.append("Retain the recorded frame timeline. Preserve, remove, or replace audio; video determines the end time. Transparent frames use the chosen background.")
    if isinstance(node, CharacterEditNode):
        paragraphs.append("Raw candidates → visual audit → bounded retry → accepted image. VOSR runs after acceptance; leave upscale long side blank for raw output. Qwen supports depth/edge/keypoint controls. Image numbers follow references, native pose, then structural controls.")
    if isinstance(node, CharacterRefineNode):
        paragraphs.append("Qwen-2511 regional edit → raw audit → accepted image. Image 1 is the source; references follow in connection order. White mask pixels are editable; original pixels and alpha outside the mask are preserved. Strength selects round(steps × strength) denoising steps. Blank max side keeps native resolution.")
    if isinstance(node, SamSegmentNode):
        paragraphs.append("Select with SAM, then connect the mask and the same source to a regional edit. Coordinates refer to the displayed source. Positive/negative points use x,y;x,y. White is editable; grey mask edges retain feathering.")
    if isinstance(node, BindMaskNode):
        paragraphs.append("Bind a supplied mask image to its source. A selection imported from a region plan retains the original source and mask checksums.")
    if isinstance(node, (ImageEditNode, CharacterEditNode)):
        capabilities = image_edit_backend_settings(node.config.backend)
        count = f"1–{capabilities.max_references}" if capabilities.max_references is not None else "1+"
        description = f"References: {count}, in connection order. Canvas alignment: {capabilities.dimension_alignment}px."
        if capabilities.image_slot_labels:
            description += "\n" + " → ".join(capabilities.image_slot_labels)
        if capabilities.lora_architecture:
            description += f"\nLoRA: {capabilities.lora_architecture}"
        if capabilities.strength_description:
            description += "\n" + capabilities.strength_description
        paragraphs.append(description)
    return "\n\n".join(paragraphs)
