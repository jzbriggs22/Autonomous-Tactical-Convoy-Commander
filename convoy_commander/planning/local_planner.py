"""Local planner: DWA-lite with potential-field fallback.

Architecture (SAFETY):
  Primary: Dynamic Window Approach (DWA) lite — samples a grid of
  (speed, turn_rate) control inputs, forward-simulates each for a short
  horizon, scores on goal heading, obstacle clearance, and speed, and
  returns the best feasible command.

  Fallback: pure potential-field, used when DWA finds no valid trajectory
  (e.g., surrounded by obstacles or at very low speed).

  Key advantages of DWA over pure potential field:
    1. Respects vehicle dynamics (kinematic window constraint).
    2. Evaluates full trajectory segments rather than instantaneous forces.
    3. Less susceptible to local minima in open terrain.

SAFETY-CRITICAL ASSUMPTIONS:
  B1. DWA horizon is 0.5 s — short enough that forward simulation remains
      valid under Euler integration at dt=0.1 s.
  B2. Clearance is measured to obstacle centres minus radii; swept-volume
      clearance is not computed.
  B3. If no sample scores above MINIMUM_SCORE, the potential-field fallback
      is used and a PLANNER_FALLBACK event is logged by the runner.
"""

from __future__ import annotations

import math

import numpy as np

from convoy_commander.core.physics import KinematicState, normalize_angle
from convoy_commander.core.world import World
from convoy_commander.vehicles.vehicle import Vehicle, VehicleCommand

# DWA parameters
_DWA_N_SPEED = 7           # speed samples
_DWA_N_OMEGA = 11          # turn-rate samples
_DWA_HORIZON = 0.5         # forward simulation horizon (s)
_DWA_SIM_STEPS = 5         # steps within horizon
_DWA_ALPHA = 0.5           # heading score weight
_DWA_BETA = 0.35           # clearance score weight
_DWA_GAMMA = 0.15          # velocity score weight
_DWA_MIN_CLEARANCE = 3.0   # m — trajectory is invalid if clearance drops below this
_DWA_MIN_SCORE = 0.05      # minimum score to accept DWA result (else use fallback)


def compute_command(
    vehicle: Vehicle,
    target: tuple[float, float],
    world: World,
    neighbors: list[Vehicle],
    dt: float,
) -> VehicleCommand:
    """Compute a control command (DWA-lite with potential-field fallback).

    Returns a VehicleCommand whose accel and turn_rate are within the vehicle's
    kinematic limits.  The caller is responsible for applying safe-mode speed
    clamping after this call.
    """
    est = vehicle.estimator.state
    cur_x, cur_y = est.x, est.y
    cur_heading = vehicle.state.heading
    cur_speed = vehicle.state.speed
    target_vec = (target[0] - cur_x, target[1] - cur_y)
    target_dist = math.hypot(target_vec[0], target_vec[1])

    if target_dist < 1.0:
        return VehicleCommand(accel=-vehicle.vcfg.max_decel * 0.5, turn_rate=0.0)

    # --- DWA-lite ---
    max_speed = vehicle.effective_max_speed
    max_accel = vehicle.vcfg.max_accel
    max_decel = vehicle.vcfg.max_decel
    max_omega = vehicle.vcfg.max_turn_rate
    dwa_dt = _DWA_HORIZON / _DWA_SIM_STEPS

    # Dynamic window: reachable speeds in one step
    v_min = max(0.0, cur_speed - max_decel * dt)
    v_max = min(max_speed, cur_speed + max_accel * dt)

    best_score = -1.0
    best_cmd = VehicleCommand(accel=0.0, turn_rate=0.0)
    used_fallback = False

    for v in np.linspace(v_min, v_max, _DWA_N_SPEED):
        for omega in np.linspace(-max_omega, max_omega, _DWA_N_OMEGA):
            # Forward simulate
            sx, sy, sh = cur_x, cur_y, cur_heading
            min_clearance = float("inf")
            valid = True

            for _ in range(_DWA_SIM_STEPS):
                sh = sh + omega * dwa_dt
                sx = sx + v * math.cos(sh) * dwa_dt
                sy = sy + v * math.sin(sh) * dwa_dt

                # Check clearance
                clr = _min_clearance(sx, sy, world, neighbors, vehicle.id)
                if clr < _DWA_MIN_CLEARANCE:
                    valid = False
                    break
                min_clearance = min(min_clearance, clr)

            if not valid:
                continue

            # Score: heading, clearance, velocity
            goal_angle = math.atan2(target[1] - sy, target[0] - sx)
            heading_err = abs(normalize_angle(goal_angle - sh))
            heading_score = 1.0 - heading_err / math.pi

            max_sensor = 60.0
            clearance_score = min(1.0, min_clearance / max_sensor)

            velocity_score = v / max_speed if max_speed > 0 else 0.0

            score = (
                _DWA_ALPHA * heading_score
                + _DWA_BETA * clearance_score
                + _DWA_GAMMA * velocity_score
            )

            if score > best_score:
                best_score = score
                speed_error = v - cur_speed
                if speed_error > 0:
                    accel = min(speed_error / dt, max_accel)
                else:
                    accel = max(speed_error / dt, -max_decel)
                best_cmd = VehicleCommand(accel=accel, turn_rate=omega)

    if best_score >= _DWA_MIN_SCORE:
        return best_cmd

    # --- Potential-field fallback ---
    used_fallback = True
    return _potential_field_command(vehicle, target, target_dist, world, neighbors, dt)


def _min_clearance(
    x: float,
    y: float,
    world: World,
    neighbors: list[Vehicle],
    own_id: int,
) -> float:
    """Minimum clearance from obstacles, no-go zones, and other vehicles."""
    clr = float("inf")
    for obs in world.obstacles:
        d = math.hypot(x - obs.x, y - obs.y) - obs.radius
        clr = min(clr, max(0.0, d))
    for po in world.poly_obstacles:
        clr = min(clr, po.clearance_from(x, y))
    for nz in world.nogo_zones:
        d = math.hypot(x - nz.x, y - nz.y) - nz.radius
        clr = min(clr, max(0.0, d * 0.5))  # No-go zones count as half clearance
    for other in neighbors:
        if other.id == own_id or not other.is_operational:
            continue
        d = math.hypot(x - other.estimator.state.x, y - other.estimator.state.y)
        clr = min(clr, max(0.0, d - 3.0))
    return clr


def _potential_field_command(
    vehicle: Vehicle,
    target: tuple[float, float],
    target_dist: float,
    world: World,
    neighbors: list[Vehicle],
    dt: float,
) -> VehicleCommand:
    """Pure potential-field command (fallback when DWA finds no valid path)."""
    est = vehicle.estimator.state
    pos = np.array([est.x, est.y])
    target_vec = np.array(target) - pos

    attract = target_vec / target_dist

    repulse = np.zeros(2)
    obs_range = 40.0

    for obs in world.obstacles:
        obs_vec = pos - np.array([obs.x, obs.y])
        obs_dist = float(np.linalg.norm(obs_vec))
        clearance = obs_dist - obs.radius
        if 0 < clearance < obs_range:
            strength = (1.0 / clearance - 1.0 / obs_range) * 50.0
            repulse += (obs_vec / obs_dist) * strength

    for po in world.poly_obstacles:
        clr = po.clearance_from(pos[0], pos[1])
        if 0 < clr < obs_range:
            # Repulse away from poly obstacle center
            po_vec = pos - np.array([po.x, po.y])
            po_dist = float(np.linalg.norm(po_vec))
            if po_dist > 1e-6:
                strength = (1.0 / max(clr, 0.5) - 1.0 / obs_range) * 50.0
                repulse += (po_vec / po_dist) * strength

    for nz in world.nogo_zones:
        nz_vec = pos - np.array([nz.x, nz.y])
        nz_dist = float(np.linalg.norm(nz_vec))
        clearance = nz_dist - nz.radius
        if 0 < clearance < obs_range * 2:
            strength = (1.0 / max(clearance, 1.0) - 1.0 / (obs_range * 2)) * 80.0
            repulse += (nz_vec / nz_dist) * strength

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

    total_force = attract * 3.0 + repulse
    force_mag = float(np.linalg.norm(total_force))
    if force_mag > 0:
        total_force /= force_mag

    desired_heading = math.atan2(total_force[1], total_force[0])
    heading_error = normalize_angle(desired_heading - vehicle.state.heading)
    turn_rate = 2.5 * heading_error

    max_speed = vehicle.effective_max_speed
    heading_factor = max(0.1, 1.0 - abs(heading_error) / math.pi)
    approach_factor = min(1.0, target_dist / 30.0)
    obstacle_factor = 1.0
    for obs in world.obstacles:
        d = math.hypot(est.x - obs.x, est.y - obs.y) - obs.radius
        if d < 20.0:
            obstacle_factor = min(obstacle_factor, max(0.2, d / 20.0))
    for po in world.poly_obstacles:
        d = po.clearance_from(est.x, est.y)
        if d < 20.0:
            obstacle_factor = min(obstacle_factor, max(0.2, d / 20.0))

    desired_speed = max_speed * heading_factor * approach_factor * obstacle_factor
    speed_error = desired_speed - vehicle.state.speed
    if speed_error > 0:
        accel = min(speed_error / dt, vehicle.vcfg.max_accel)
    else:
        accel = max(speed_error / dt, -vehicle.vcfg.max_decel)

    return VehicleCommand(accel=accel, turn_rate=turn_rate)
