from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Collection, Literal, Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from aigen.pix2pix.errors import Pix2PixError


@dataclass(frozen=True, order=True, slots=True)
class BodySlot:
    job_id: int
    gender: Literal[0, 1]


@dataclass(frozen=True, order=True, slots=True)
class BodyPoseAssignment:
    job_id: int
    gender: Literal[0, 1]
    action_base: int
    direction: int


def assign_body_pose_cells(
    body_slots: Collection[BodySlot],
    action_quotas: Mapping[int, int],
    direction_quotas: Mapping[int, int],
    forbidden_cells: Collection[BodyPoseAssignment],
    *,
    seed: int,
    domain: str,
) -> tuple[BodyPoseAssignment, ...]:
    slot_count = len(body_slots)
    if slot_count == 0:
        raise Pix2PixError("body-to-pose assignment requires at least one slot")
    if sum(action_quotas.values()) != slot_count:
        raise Pix2PixError(
            "body and action multiplicities differ: "
            f"{slot_count} body slots, {sum(action_quotas.values())} actions"
        )
    if sum(direction_quotas.values()) != slot_count:
        raise Pix2PixError(
            "body and direction multiplicities differ: "
            f"{slot_count} body slots, "
            f"{sum(direction_quotas.values())} directions"
        )
    if any(count < 0 for count in action_quotas.values()):
        raise Pix2PixError("action quotas must be non-negative")
    if any(count < 0 for count in direction_quotas.values()):
        raise Pix2PixError("direction quotas must be non-negative")

    body_counts = Counter(body_slots)
    bodies = tuple(sorted(body_counts))
    actions = tuple(
        action
        for action, count in sorted(action_quotas.items())
        if count > 0
    )
    directions = tuple(
        direction
        for direction, count in sorted(direction_quotas.items())
        if count > 0
    )
    forbidden = set(forbidden_cells)
    variable_records = []
    for body in bodies:
        for action in actions:
            for direction in directions:
                candidate = BodyPoseAssignment(
                    job_id=body.job_id,
                    gender=body.gender,
                    action_base=action,
                    direction=direction,
                )
                if candidate not in forbidden:
                    variable_records.append(candidate)
    variables = tuple(variable_records)
    if not variables:
        raise Pix2PixError("body-to-pose assignment has no allowed variables")

    by_body: dict[BodySlot, list[int]] = {body: [] for body in bodies}
    by_action: dict[int, list[int]] = {action: [] for action in actions}
    by_direction: dict[int, list[int]] = {
        direction: [] for direction in directions
    }
    by_body_action: dict[tuple[BodySlot, int], list[int]] = {
        (body, action): [] for body in bodies for action in actions
    }
    by_body_direction: dict[tuple[BodySlot, int], list[int]] = {
        (body, direction): []
        for body in bodies
        for direction in directions
    }
    by_pose: dict[tuple[int, int], list[int]] = {
        (action, direction): []
        for action in actions
        for direction in directions
    }
    for index, variable in enumerate(variables):
        body = BodySlot(job_id=variable.job_id, gender=variable.gender)
        by_body[body].append(index)
        by_action[variable.action_base].append(index)
        by_direction[variable.direction].append(index)
        by_body_action[(body, variable.action_base)].append(index)
        by_body_direction[(body, variable.direction)].append(index)
        by_pose[(variable.action_base, variable.direction)].append(index)

    rows: list[int] = []
    columns: list[int] = []
    lower_bounds: list[int] = []
    upper_bounds: list[int] = []

    def add_constraint(
        variable_indices: Collection[int],
        lower: int,
        upper: int,
    ) -> None:
        row = len(lower_bounds)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        rows.extend([row] * len(variable_indices))
        columns.extend(variable_indices)

    for body in bodies:
        add_constraint(by_body[body], body_counts[body], body_counts[body])
    for action in actions:
        count = action_quotas[action]
        add_constraint(by_action[action], count, count)
    for direction in directions:
        count = direction_quotas[direction]
        add_constraint(by_direction[direction], count, count)
    for variable_indices in by_body_action.values():
        add_constraint(variable_indices, 0, 1)
    for variable_indices in by_body_direction.values():
        add_constraint(variable_indices, 0, 1)

    joint_lower, joint_upper = joint_pose_bounds(
        action_quotas,
        direction_quotas,
    )
    for variable_indices in by_pose.values():
        add_constraint(variable_indices, joint_lower, joint_upper)

    constraint_matrix = coo_array(
        (
            np.ones(len(columns), dtype=np.float64),
            (
                np.asarray(rows, dtype=np.int32),
                np.asarray(columns, dtype=np.int32),
            ),
        ),
        shape=(len(lower_bounds), len(variables)),
        dtype=np.float64,
    ).tocsc()
    objective = np.fromiter(
        (
            int.from_bytes(
                _rank(
                    seed,
                    domain,
                    (
                        f"{variable.job_id}:{variable.gender}:"
                        f"{variable.action_base}:{variable.direction}"
                    ),
                )[:8],
                "big",
            )
            / 2**64
            for variable in variables
        ),
        dtype=np.float64,
        count=len(variables),
    )
    result = milp(
        c=objective,
        integrality=np.ones(len(variables), dtype=np.uint8),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            constraint_matrix,
            np.asarray(lower_bounds, dtype=np.float64),
            np.asarray(upper_bounds, dtype=np.float64),
        ),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise Pix2PixError(
            "cannot satisfy the body/action/direction coverage contract: "
            f"{result.message}"
        )
    assignments = tuple(
        sorted(
            variable
            for variable, selected in zip(variables, result.x, strict=True)
            if selected > 0.5
        )
    )
    if len(assignments) != slot_count:
        raise Pix2PixError(
            "coverage solver returned an invalid assignment cardinality"
        )
    return assignments


def _rank(seed: int, domain: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{domain}\0{value}".encode("utf-8")).digest()


def joint_pose_bounds(
    action_quotas: Mapping[int, int],
    direction_quotas: Mapping[int, int],
) -> tuple[int, int]:
    actions = tuple(count for count in action_quotas.values() if count > 0)
    directions = tuple(
        count for count in direction_quotas.values() if count > 0
    )
    if not actions or not directions:
        return 0, 0
    total = sum(actions)
    joint_cell_count = len(actions) * len(directions)
    lower = int(
        total >= joint_cell_count
        and min(actions) >= len(directions)
        and min(directions) >= len(actions)
    )
    upper = max(
        _ceil_div(total, joint_cell_count),
        max(_ceil_div(count, len(directions)) for count in actions),
        max(_ceil_div(count, len(actions)) for count in directions),
    )
    return lower, upper


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator
