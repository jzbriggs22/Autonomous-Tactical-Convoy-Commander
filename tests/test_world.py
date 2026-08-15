"""Tests for world model."""

import numpy as np

from convoy_commander.core.config import WorldConfig
from convoy_commander.core.world import World, Obstacle


def test_world_generation_deterministic():
    """Same seed should produce same world."""
    config = WorldConfig()
    w1 = World(config, np.random.default_rng(42))
    w2 = World(config, np.random.default_rng(42))

    assert len(w1.obstacles) == len(w2.obstacles)
    assert len(w1.landmarks) == len(w2.landmarks)
    assert w1.road_graph.number_of_nodes() == w2.road_graph.number_of_nodes()


def test_obstacle_contains():
    obs = Obstacle(x=100, y=100, radius=10)
    assert obs.contains(100, 100)
    assert obs.contains(105, 100)
    assert not obs.contains(200, 200)


def test_world_is_blocked():
    config = WorldConfig(obstacle_count=0, nogo_zone_count=0, landmark_count=0)
    rng = np.random.default_rng(42)
    world = World(config, rng)
    world.obstacles.append(Obstacle(x=100, y=100, radius=10))

    assert world.is_blocked(100, 100)
    assert not world.is_blocked(200, 200)


def test_world_add_obstacle():
    config = WorldConfig(obstacle_count=0, nogo_zone_count=0, landmark_count=0)
    rng = np.random.default_rng(42)
    world = World(config, rng)

    assert len(world.obstacles) == 0
    world.add_obstacle(250, 250, 15)
    assert len(world.obstacles) == 1
    assert world.is_blocked(250, 250)


def test_road_graph_connected():
    config = WorldConfig()
    rng = np.random.default_rng(42)
    world = World(config, rng)

    import networkx as nx
    assert nx.is_connected(world.road_graph)
