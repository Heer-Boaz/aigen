from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain, permutations
from pathlib import Path

from aigen.character_reference_models import CharacterReferenceError, CharacterReferencePackSpec
from aigen.vlm_qwen import QwenVlm


@dataclass(frozen=True)
class ReferenceSelection:
    selected_refs: tuple[str, ...]
    selected_ref_indices: tuple[int, ...]
    selector: str


def select_reference_subset(
    *,
    runner: QwenVlm | None,
    pack: CharacterReferencePackSpec,
    reference_paths: Mapping[str, Path],
    instruction: str,
    route_kind: str,
    output_mode: str,
    budget: int,
    path_label: str,
) -> ReferenceSelection:
    reference_ids = tuple(pack.references)
    if len(reference_ids) <= budget:
        return ReferenceSelection(
            selected_refs=reference_ids,
            selected_ref_indices=tuple(range(1, len(reference_ids) + 1)),
            selector="all_refs_within_budget",
        )
    if runner is None:
        raise CharacterReferenceError(
            f"{path_label}: {len(reference_ids)} references exceed the {budget}-reference editor budget "
            "but no VLM runner was provided for reference selection"
        )

    image_paths = [reference_paths[reference_id] for reference_id in reference_ids]
    choices = tuple(
        choice
        for choice in chain.from_iterable(
            permutations(range(1, len(reference_ids) + 1), size)
            for size in range(1, budget + 1)
        )
    )
    selected_ref_indices = runner.select_indices(
        _reference_selector_prompt(
            instruction=instruction,
            route_kind=route_kind,
            output_mode=output_mode,
            reference_count=len(reference_ids),
            budget=budget,
        ),
        image_paths,
        choices,
    )
    return ReferenceSelection(
        selected_refs=tuple(reference_ids[index - 1] for index in selected_ref_indices),
        selected_ref_indices=selected_ref_indices,
        selector="qwen_vlm_constrained_index_selection",
    )


def _reference_selector_prompt(
    *,
    instruction: str,
    route_kind: str,
    output_mode: str,
    reference_count: int,
    budget: int,
) -> str:
    return f"""Choose which reference images to feed to an image editor for one task.

The {reference_count} images are supplied in order and use 1-based indices: Image 1 through Image {reference_count}.
Task route: {route_kind}
Output mode: {output_mode}
Instruction: {instruction!r}

Choose between 1 and {budget} indices. Use only unique integers from 1 through {reference_count}.
Choose the combination whose images provide the strongest complementary visual coverage for preserving the
entire visible subject in the requested edit. Avoid redundant coverage. Do not describe the subject or any image.
Answer with the chosen 1-based indices.
"""
