from __future__ import annotations

from typing import Any

from aigen.generation.runtime_diagnostics import module_device_report


CLIP_TEXT_ENCODER_COMPONENT = "text_encoder"
T5_TEXT_ENCODER_COMPONENT = "text_encoder_2"
CLIP_TOKENIZER_COMPONENT = "tokenizer"
T5_TOKENIZER_COMPONENT = "tokenizer_2"
FLUX_TEXT_COMPONENTS_DISABLED = {
    CLIP_TEXT_ENCODER_COMPONENT: None,
    T5_TEXT_ENCODER_COMPONENT: None,
    CLIP_TOKENIZER_COMPONENT: None,
    T5_TOKENIZER_COMPONENT: None,
}


def flux_text_component_device_reports(pipeline: Any) -> dict[str, Any]:
    reports = {}
    clip_text_encoder = getattr(pipeline, CLIP_TEXT_ENCODER_COMPONENT)
    t5_text_encoder = getattr(pipeline, T5_TEXT_ENCODER_COMPONENT)
    if clip_text_encoder is not None:
        reports["clip_text_encoder"] = module_device_report(clip_text_encoder)
    if t5_text_encoder is not None:
        reports["t5_text_encoder"] = module_device_report(t5_text_encoder)
    return reports
