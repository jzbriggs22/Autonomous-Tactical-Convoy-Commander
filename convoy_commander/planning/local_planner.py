"""Local planner: potential field / vector field approach with obstacle avoidance."""

from __future__ import annotations

import math

import numpy as np

from convoy_commander.core.physics import KinematicState, normalize_angle
from convoy_commander.core.world import World
from convoy_commander.vehicles.vehicle import Vehicle, VehicleCommand


def compute_command(
    vehicle: Vehicle,
    target: tuple[float, float],
    world: World,
    neighbors: list[Vehicle],
    dt: float,
) -> VehicleCommand:
    """Compute a control command to move toward target while avoiding obstacles.

    Uses a potential field approach:
    - Attractive force toward target waypoint
    - Repulsive force from obstacles and other vehicles
    - Speed control based on proximity to obstacles and target
    """
    est = vehicle.estimator.state
    pos = np.array([est.x, est.y])
    target_vec = np.array(target) - pos
    target_dist = float(np.linalg.norm(target_vec))

    if target_dist < 1.0:
        # At target, stop
        return VehicleCommand(accel=-vehicle.vcfg.max_decel * 0.5, turn_rate=0.0)

    # Attractive force
    attract = target_vec / target_dist

    # Repulsive forces from obstacles
    repulse = np.zeros(2)
    obstacle_influence_range = 40.0

    for obs in world.obstacles:
        obs_vec = pos - np.array([obs.x, obs.y])
        obs_dist = float(np.linalg.norm(obs_vec))
        clearance = obs_dist - obs.radius
        if 0 < clearance < obstacle_influence_range:
            strength = (1.0 / clearance - 1.0 / obstacle_influence_range) * 50.0
            repulse += (obs_vec / obs_dist) * strength

    # Repulsive forces from no-go zones
    for nz in world.nogo_zones:
        nz_vec = pos - np.array([nz.x, nz.y])
        nz_dist = float(np.linalg.norm(nz_vec))
        clearance = nz_dist - nz.radius
        if 0 < clearance < obstacle_influence_range * 2:
            strength = (1.0 / max(clearance, 1.0) - 1.0 / (obstacle_influence_range * 2)) * 80.0
            repulse += (nz_vec / nz_dist) * strength

    # Repulsive forces from other vehicles (collision avoidance)
    min_sep = vehicle.config.coordination.min_separation
    for other in neighbors:
        if other.id == vehicle.id or not other.is_operational:
            continue
        other_pos = np.array([other.estimator.state.x, other.estimator.state.y])
        sep_vec = pos - other_pos
        sep_dist = float(np.linalg.norm(sep_vec))
        if 0 < sep_dist < min_sep * 2:
            strength = (1.0 / max(sep_dist, 1.0) - 1.0 / (min_sep * 2)) * 100.0
            repulse += (sep_vec / sep_dist) * strength

    # Combine forces
    total_force = attract * 3.0 + repulse
    force_mag = float(np.linalg.norm(total_force))
    if force_mag > 0:
        total_force /= force_mag

    # Compute desired heading
    desired_heading = math.atan2(total_force[1], total_force[0])

    # Heading error
    heading_error = normalize_angle(desired_heading - vehicle.state.heading)

    # Turn rate proportional to heading error
    kp_turn = 2.5
    turn_rate = kp_turn * heading_error

    # Speed control
    max_speed = vehicle.effective_max_speed

    # Slow down if heading error is large
    heading_factor = max(0.1, 1.0 - abs(heading_error) / math.pi)
    # Slow down near target
    approach_factor = min(1.0, target_dist / 30.0)
    # Slow down near obstacles
    obstacle_factor = 1.0
    for obs in world.obstacles:
        obs_dist = math.hypot(est.x - obs.x, est.y - obs.y) - obs.radius
        if obs_dist < 20.0:
            obstacle_factor = min(obstacle_factor, max(0.2, obs_dist / 20.0))

    desired_speed = max_speed * heading_factor * approach_factor * obstacle_factor

    # Acceleration
    speed_error = desired_speed - vehicle.state.speed
    if speed_error > 0:
        accel = min(speed_error / dt, vehicle.vcfg.max_accel)
    else:
        accel = max(speed_error / dt, -vehicle.vcfg.max_decel)

    return VehicleCommand(accel=accel, turn_rate=turn_rate)
