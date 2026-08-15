"""Convoy merge and split operations (Phase 12).

Merge: all vehicles from one convoy are transferred to another.
Split: a subset of vehicles is removed from a convoy to form a new one.
"""

from __future__ import annotations

import math

from convoy_commander.coordination.convoy_group import ConvoyGroup
from convoy_commander.coordination.leader_election import LeaderElection
from convoy_commander.metrics.collector import MetricsCollector


def can_merge(
    group_a: ConvoyGroup,
    group_b: ConvoyGroup,
    merge_distance: float,
) -> bool:
    """Check if two convoys can merge (leaders within merge_distance,
    neither currently yielding)."""
    if group_a.yielding or group_b.yielding:
        return False
    leader_a = group_a.get_leader()
    leader_b = group_b.get_leader()
    if leader_a is None or leader_b is None:
        return False
    dist = math.hypot(
        leader_a.state.x - leader_b.state.x,
        leader_a.state.y - leader_b.state.y,
    )
    return dist < merge_distance


def merge_convoys(
    absorber: ConvoyGroup,
    absorbed: ConvoyGroup,
    heartbeat_timeout: float,
) -> None:
    """Move all vehicles from absorbed into absorber.

    Re-initialises elections for transferred vehicles and forces
    re-election in the absorber convoy.  The absorbed group becomes empty.
    """
    for v in list(absorbed.vehicles):
        absorbed.remove_vehicle(v.id)
        absorber.add_vehicle(v)
        absorber.elections[v.id] = LeaderElection(v, heartbeat_timeout)

    # Force re-election in absorber
    for v in absorber.vehicles:
        if v.id in absorber.elections:
            absorber.elections[v.id].reset()

    # Take max priority
    absorber.priority = max(absorber.priority, absorbed.priority)


def split_convoy(
    source: ConvoyGroup,
    vehicle_ids_for_new: list[int],
    new_convoy_id: int,
    new_destination: tuple[float, float],
    new_priority: int,
    heartbeat_timeout: float,
    min_vehicles: int = 2,
) -> ConvoyGroup | None:
    """Split vehicles out of source into a new ConvoyGroup.

    Returns the new group, or None if the split would leave either
    group below min_vehicles.
    """
    remaining = len(source.vehicles) - len(vehicle_ids_for_new)
    if remaining < min_vehicles or len(vehicle_ids_for_new) < min_vehicles:
        return None

    new_group = ConvoyGroup(
        convoy_id=new_convoy_id,
        priority=new_priority,
        vehicles=[],
        destination=new_destination,
        collector=MetricsCollector(),
    )

    for vid in vehicle_ids_for_new:
        v = source.remove_vehicle(vid)
        if v is not None:
            # Clean up old elections
            source.elections.pop(vid, None)
            source.cbba_slots.pop(vid, None)
            new_group.add_vehicle(v)
            new_group.elections[vid] = LeaderElection(v, heartbeat_timeout)

    # Force re-election in both groups
    for v in source.vehicles:
        if v.id in source.elections:
            source.elections[v.id].reset()
    for v in new_group.vehicles:
        new_group.elections[v.id].reset()

    # Designate initial leader in new group (first vehicle)
    if new_group.vehicles:
        new_group.vehicles[0].is_leader = True
        new_group.vehicles[0].leader_id = new_group.vehicles[0].id
        for v in new_group.vehicles:
            v.leader_id = new_group.vehicles[0].id
            new_group.elections[v.id].last_heartbeat_time = 0.0

    return new_group
