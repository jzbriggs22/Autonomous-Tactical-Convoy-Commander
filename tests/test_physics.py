"""Tests for core physics module."""

import math

from convoy_commander.core.physics import FuelState, KinematicState, clamp, normalize_angle


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(15, 0, 10) == 10


def test_normalize_angle():
    assert abs(normalize_angle(0.0)) < 1e-10
    assert abs(normalize_angle(math.pi) - math.pi) < 1e-10
    assert abs(normalize_angle(3 * math.pi) - math.pi) < 1e-10
    assert abs(normalize_angle(-3 * math.pi) - (-math.pi)) < 1e-10


def test_kinematic_state_step():
    """Deterministic stepping: vehicle moves forward."""
    state = KinematicState(x=0, y=0, heading=0, speed=0)
    # Accelerate east
    state.step(accel=2.0, turn_rate=0.0, dt=1.0, max_speed=10, max_accel=5, max_decel=5, max_turn_rate=1)
    assert state.speed == 2.0
    assert state.x > 0
    assert abs(state.y) < 1e-10


def test_kinematic_state_turn():
    state = KinematicState(x=0, y=0, heading=0, speed=5)
    state.step(accel=0, turn_rate=0.5, dt=1.0, max_speed=10, max_accel=5, max_decel=5, max_turn_rate=1)
    assert state.heading > 0


def test_kinematic_distance():
    a = KinematicState(x=0, y=0)
    b = KinematicState(x=3, y=4)
    assert abs(a.distance_to(b) - 5.0) < 1e-10


def test_fuel_consumption():
    fuel = FuelState(fuel=100.0, capacity=100.0, rate_idle=0.01, rate_per_speed=0.005)
    fuel.consume(speed=10.0, dt=1.0)
    # Expected: (0.01 + 0.005 * 10) * 1 = 0.06
    assert abs(fuel.fuel - 99.94) < 1e-10
    assert not fuel.is_empty


def test_fuel_empty():
    fuel = FuelState(fuel=0.01, capacity=100.0, rate_idle=0.01, rate_per_speed=0.005)
    fuel.consume(speed=10.0, dt=1.0)
    assert fuel.is_empty
