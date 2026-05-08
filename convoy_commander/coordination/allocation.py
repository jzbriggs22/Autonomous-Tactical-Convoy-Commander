"""Waypoint allocation using distributed greedy with tie-breaking."""

from __future__ import annotations

import math

from convoy_commander.vehicles.vehicle import Vehicle


def allocate_waypoints_greedy(
    vehicles: list[Vehicle],
    destination: tuple[float, float],
    *,
    formation_radius: float = 42.0,
    formation_spacing: float = 30.0,
) -> None:
    """Distributed greedy allocation with per-vehicle goal offsets.

    All vehicles share a high-level convoy destination, but each gets its
    own arrival point in a deterministic NxN grid centred on ``destination``.
    Spreading goals avoids the convergence pile-up at the 15m arrival
    radius when many vehicles target the same point.

    Slot assignment is by vehicle id so runs are reproducible.
    """
    op = sorted(
        [v for v in vehicles if v.is_operational],
        key=lambda v: v.id,
    )
    if not op:
        return
    cols = max(1, int(math.ceil(math.sqrt(len(op)))))
    half = (cols - 1) / 2.0
    for k, v in enumerate(op):
        ox = (k % cols - half) * formation_spacing
        oy = (k // cols - half) * formation_spacing
        mag = math.hypot(ox, oy)
        if mag > formation_radius:
            scale = formation_radius / mag
            ox *= scale
            oy *= scale
        v.assigned_destination = (destination[0] + ox, destination[1] + oy)


def allocate_multi_waypoints_greedy(
    vehicles: list[Vehicle],
    waypoints: list[tuple[float, float]],
) -> dict[int, list[tuple[float, float]]]:
    """Distributed greedy allocation for multiple waypoints.

    Each vehicle bids on waypoints by distance. Closest vehicle gets each
    waypoint with tie-breaking by vehicle ID (lower wins).
    Returns mapping of vehicle_id -> assigned waypoints.
    """
    assignments: dict[int, list[tuple[float, float]]] = {v.id: [] for v in vehicles}
    available = list(range(len(waypoints)))
    assigned_vehicles: set[int] = set()

    while available and len(assigned_vehicles) < len(vehicles):
        best_assignment: tuple[int, int, float] | None = None  # (vid, wp_idx, dist)
        for wp_idx in available:
            wp = waypoints[wp_idx]
            for v in vehicles:
                if v.id in assigned_vehicles or not v.is_operational:
                    continue
                dist = math.hypot(v.estimator.state.x - wp[0], v.estimator.state.y - wp[1])
                if best_assignment is None or dist < best_assignment[2] or (
                    dist == best_assignment[2] and v.id < best_assignment[0]
                ):
                    best_assignment = (v.id, wp_idx, dist)
        if best_assignment is None:
            break
        vid, wp_idx, _ = best_assignment
        wp = waypoints[wp_idx]
        assignments[vid].append(wp)
        available.remove(wp_idx)
        assigned_vehicles.add(vid)

    # Remaining waypoints go to nearest available vehicle
    for wp_idx in available:
        wp = waypoints[wp_idx]
        best_vid = -1
        best_dist = float("inf")
        for v in vehicles:
            if not v.is_operational:
                continue
            dist = math.hypot(v.estimator.state.x - wp[0], v.estimator.state.y - wp[1])
            if dist < best_dist:
                best_dist = dist
                best_vid = v.id
        if best_vid >= 0:
            assignments[best_vid].append(wp)

    return assignments
