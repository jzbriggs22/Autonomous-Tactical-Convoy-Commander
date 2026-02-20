"""Global planner: A* on road graph."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from convoy_commander.core.world import World


def plan_route(
    world: World, start_x: float, start_y: float, goal_x: float, goal_y: float
) -> list[tuple[float, float]]:
    """Plan a route from start to goal using A* on the road graph.

    Returns list of (x, y) waypoints including start and goal.
    """
    if world.road_graph.number_of_nodes() == 0:
        return [(start_x, start_y), (goal_x, goal_y)]

    start_node = world.nearest_road_node(start_x, start_y)
    goal_node = world.nearest_road_node(goal_x, goal_y)

    if start_node == goal_node:
        return [(start_x, start_y), (goal_x, goal_y)]

    try:
        path_nodes = nx.astar_path(
            world.road_graph,
            start_node,
            goal_node,
            heuristic=lambda a, b: _heuristic(world, a, b),
            weight="weight",
        )
    except nx.NetworkXNoPath:
        # Fallback: direct path
        return [(start_x, start_y), (goal_x, goal_y)]

    # Convert node path to coordinate waypoints
    waypoints: list[tuple[float, float]] = [(start_x, start_y)]
    for node_id in path_nodes:
        pos = world.get_node_pos(node_id)
        waypoints.append(pos)
    waypoints.append((goal_x, goal_y))

    return waypoints


def _heuristic(world: World, node_a: int, node_b: int) -> float:
    """Euclidean distance heuristic for A*."""
    pos_a = world.get_node_pos(node_a)
    pos_b = world.get_node_pos(node_b)
    return math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])
