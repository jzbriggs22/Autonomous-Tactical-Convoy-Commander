"""Waypoint allocation using distributed greedy with tie-breaking."""

from __future__ import annotations

import math

from convoy_commander.vehicles.vehicle import Vehicle


def allocate_waypoints_greedy(
    vehicles: list[Vehicle],
    destination: tuple[float, float],
) -> None:
    """Simple distributed greedy allocation.

    All vehicles share the same destination (convoy mission).
    Each vehicle gets the destination as its assigned goal.
    In a more complex version, intermediate waypoints would be distributed
    using CBBA-lite auction, but for convoy operations, all vehicles
    share the same destination with formation offsets handled by coordination.
    """
    for v in vehicles:
        if v.is_operational:
            v.assigned_destination = destination


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
