"""Formation control: maintain target spacing and heading alignment."""

from __future__ import annotations

import math

import numpy as np

from convoy_commander.core.physics import normalize_angle
from convoy_commander.vehicles.vehicle import Vehicle


def compute_formation_correction(
    vehicle: Vehicle,
    leader: Vehicle | None,
    neighbors: list[Vehicle],
    formation_index: int,
    spacing: float,
) -> tuple[float, float]:
    """Compute a correction vector for formation maintenance.

    Uses constant-time-headway (CTH) gap model (Phase 6):
      gap = standoff_distance + time_headway * follower_speed
    capped at ``spacing * formation_index`` so it doesn't exceed the
    configured maximum formation spacing.

    Returns (dx, dy) correction to add to the waypoint-tracking force.
    """
    correction = np.zeros(2)
    heading_correction = 0.0

    if leader is not None and leader.is_operational and leader.id != vehicle.id:
        # Desired position: behind leader in formation
        leader_pos = np.array([leader.estimator.state.x, leader.estimator.state.y])
        leader_heading = leader.estimator.state.heading

        # Constant time headway gap (Phase 6)
        coord = vehicle.config.coordination
        follower_speed = vehicle.state.speed
        desired_gap = coord.standoff_distance + coord.time_headway * follower_speed
        offset_dist = min(desired_gap * formation_index, spacing * formation_index)
        desired_x = leader_pos[0] - offset_dist * math.cos(leader_heading)
        desired_y = leader_pos[1] - offset_dist * math.sin(leader_heading)

        my_pos = np.array([vehicle.estimator.state.x, vehicle.estimator.state.y])
        formation_error = np.array([desired_x, desired_y]) - my_pos
        error_mag = float(np.linalg.norm(formation_error))

        if error_mag > 2.0:
            # Proportional correction with moderate gain — avoid overpowering
            # goal-tracking when formation error is small
            gain = min(1.0, error_mag / (spacing * 1.0))
            correction = (formation_error / error_mag) * gain

        # Heading alignment
        heading_correction = normalize_angle(leader_heading - vehicle.state.heading) * 0.3

    # Neighbor consensus: nudge toward average neighbor heading
    if neighbors:
        neighbor_headings: list[float] = []
        for n in neighbors:
            if n.id != vehicle.id and n.is_operational:
                neighbor_headings.append(n.estimator.state.heading)
        if neighbor_headings:
            # Circular mean
            sin_sum = sum(math.sin(h) for h in neighbor_headings)
            cos_sum = sum(math.cos(h) for h in neighbor_headings)
            avg_heading = math.atan2(sin_sum, cos_sum)
            heading_correction += normalize_angle(avg_heading - vehicle.state.heading) * 0.1

    return (float(correction[0]), float(correction[1]))


def get_formation_index(vehicle_id: int, leader_id: int | None, all_ids: list[int]) -> int:
    """Get the vehicle's position index in formation (0 = leader)."""
    if leader_id is not None and vehicle_id == leader_id:
        return 0
    operational_ids = sorted(i for i in all_ids if i != leader_id)
    if vehicle_id in operational_ids:
        return operational_ids.index(vehicle_id) + 1
    return len(operational_ids)
