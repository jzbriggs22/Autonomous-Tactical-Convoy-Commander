"""Tests for planning module."""

import numpy as np

from convoy_commander.core.config import WorldConfig
from convoy_commander.core.world import World
from convoy_commander.planning.global_planner import plan_route


def test_global_planner_finds_path():
    """A* should find a path between two points."""
    config = WorldConfig(
        width=500, height=500, road_graph_density=8,
        obstacle_count=3, nogo_zone_count=1, landmark_count=2,
    )
    rng = np.random.default_rng(42)
    world = World(config, rng)

    waypoints = plan_route(world, 50, 50, 400, 400)
    assert len(waypoints) >= 2
    assert waypoints[0] == (50, 50)
    assert waypoints[-1] == (400, 400)


def test_global_planner_start_equals_goal():
    """When start is near goal, should return short path."""
    config = WorldConfig(width=500, height=500, road_graph_density=8)
    rng = np.random.default_rng(42)
    world = World(config, rng)

    # Same node
    node_pos = world.get_node_pos(0)
    waypoints = plan_route(world, node_pos[0], node_pos[1], node_pos[0] + 1, node_pos[1] + 1)
    assert len(waypoints) >= 2
