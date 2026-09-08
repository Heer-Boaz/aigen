from aigen.workflow_graph import NodeKind, WorkflowNode


# Primary fields reflect each node's task. Remaining model fields stay editable
# in Advanced; backend relevance is independently owned by the inspector.
PRIMARY_FIELDS = {
    NodeKind.IMAGE_SOURCE: ('path',),
    NodeKind.VIDEO_SOURCE: ('path',),
    NodeKind.AUDIO_SOURCE: ('path', 'stream_index'),
    NodeKind.REFERENCE_PACK: ('path',),
    NodeKind.LORA_SOURCE: ('path', 'weight'),
    NodeKind.IMAGE_EDIT: ('backend', 'prompt', 'seed_mode', 'seed', 'aspect_ratio', 'width', 'height', 'strength'),
    NodeKind.CHARACTER_EDIT: ('backend', 'prompt', 'seed_mode', 'seed', 'candidates', 'aspect_ratio', 'width', 'height', 'pose_mode'),
    NodeKind.CHARACTER_REFINE: ('backend', 'prompt', 'seed_mode', 'seed', 'strength', 'candidates', 'max_side'),
    NodeKind.SAM_SEGMENT: ('engine', 'prompt_mode', 'box', 'positive_points', 'negative_points', 'mask_candidate'),
    NodeKind.BIND_MASK: (),
    NodeKind.IMAGE_SELECTION: (),
    NodeKind.IMAGE_COLLECTION: (),
    NodeKind.IMAGE_POSTPROCESS: ('model', 'sizing', 'long_side', 'scale', 'cell_size', 'mode', 'force_step'),
    NodeKind.FRAME_POSTPROCESS: ('model', 'sizing', 'long_side', 'scale', 'cell_size', 'mode', 'force_step'),
    NodeKind.ANIMEGEN_I2V: ('prompt', 'seed_mode', 'seed', 'frames', 'fps', 'keyframe_fit'),
    NodeKind.LTX23: ('prompt', 'negative_prompt', 'seed_mode', 'seed', 'frames', 'fps', 'resolution', 'keyframe_fit'),
    NodeKind.HUNYUAN_I2V: ('prompt', 'seed_mode', 'seed', 'frames'),
    NodeKind.POSITIONED_KEYFRAME: ('frame',),
    NodeKind.VIDEO_CONTACT_SHEET: (),
    NodeKind.EXTRACT_VIDEO_FRAMES: (),
    NodeKind.ASSEMBLE_VIDEO: ('audio_policy', 'background'),
}


def primary_fields(node: WorkflowNode, available: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in PRIMARY_FIELDS[node.kind] if name in available)
