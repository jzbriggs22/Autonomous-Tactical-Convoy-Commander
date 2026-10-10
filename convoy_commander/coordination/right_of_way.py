"""Right-of-way negotiation between convoys (Phase 12).

Simple priority-based protocol: when two convoy leaders are within
proximity_radius, the lower-priority convoy yields (speed reduced to
near-zero). Equal priority ties broken by convoy_id (higher yields).
"""

from __future__ import annotations

import math

from convoy_commander.coordination.convoy_group import ConvoyGroup


def detect_crossing(
    group_a: ConvoyGroup,
    group_b: ConvoyGroup,
    proximity_radius: float,
) -> bool:
    """Return True if the two convoys' leaders are within proximity_radius."""
    leader_a = group_a.get_leader()
    leader_b = group_b.get_leader()
    if leader_a is None or leader_b is None:
        return False
    dist = math.hypot(
        leader_a.state.x - leader_b.state.x,
        leader_a.state.y - leader_b.state.y,
    )
    return dist < proximity_radius


def resolve_right_of_way(
    group_a: ConvoyGroup,
    group_b: ConvoyGroup,
    current_time: float,
    yield_duration: float = 10.0,
) -> ConvoyGroup | None:
    """Determine which convoy should yield.

    Lower-priority convoy yields.  Equal priority → higher convoy_id yields.
    Returns the yielding group (or None if already yielding).
    """
    if group_a.yielding or group_b.yielding:
        return None  # already resolved

    # Determine which yields
    if group_a.priority < group_b.priority:
        yielder = group_a
    elif group_b.priority < group_a.priority:
        yielder = group_b
    elif group_a.convoy_id > group_b.convoy_id:
        yielder = group_a
    else:
        yielder = group_b

    yielder.yielding = True
    yielder.yield_until = current_time + yield_duration
    return yielder


def apply_yield(group: ConvoyGroup, current_time: float) -> float:
    """Return speed multiplier for this convoy.

    0.1 if yielding (near-stop), 1.0 otherwise.
    Clears yield flag when yield_until has passed.
    """
    if not group.yielding:
        return 1.0
    if current_time >= group.yield_until:
        group.yielding = False
        group.yield_until = 0.0
        return 1.0
    return 0.1
