"""Global planner: multi-objective A* on road graph.

The planner supports four objectives via ``PlanningObjective``:
  - w_time: weight on travel distance (proxy for time)
  - w_fuel: weight on fuel consumption (proportional to distance)
  - w_risk: weight on per-edge risk score (proximity to obstacles / no-go zones)
  - w_slope: weight on elevation/slope cost (requires DEM data)

The composite edge cost is:
    (w_time + w_fuel) * base_weight + w_risk * edge_risk * dist + w_slope * slope_cost * dist

Setting all weights equal to the defaults (1.0, 0.3, 0.5, 0.0) biases the planner
toward shorter, less risky routes.  Enable w_slope > 0 with a DEM for
elevation-aware routing that prefers flatter terrain.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import networkx as nx

from convoy_commander.core.world import World

if TYPE_CHECKING:
    from convoy_commander.core.config import PlanningObjective


def plan_route(
    world: World,
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    objective: PlanningObjective | None = None,
) -> list[tuple[float, float]]:
    """Plan a route from start to goal using A* on the road graph.

    Args:
        world: The world model containing the road graph.
        start_x, start_y: Start position (m).
        goal_x, goal_y: Goal position (m).
        objective: Multi-objective weights.  None uses pure distance (w_time=1,
                   w_fuel=0, w_risk=0), preserving backward compatibility.

    Returns list of (x, y) waypoints including start and goal.
    """
    if world.road_graph.number_of_nodes() == 0:
        return [(start_x, start_y), (goal_x, goal_y)]

    start_node = world.nearest_road_node(start_x, start_y)
    goal_node = world.nearest_road_node(goal_x, goal_y)

    if start_node == goal_node:
        return [(start_x, start_y), (goal_x, goal_y)]

    if objective is None:
        weight_fn = "weight"
    else:
        weight_fn = _make_weight_fn(world, objective)

    try:
        path_nodes = nx.astar_path(
            world.road_graph,
            start_node,
            goal_node,
            heuristic=lambda a, b: _heuristic(world, a, b),
            weight=weight_fn,
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


def _make_weight_fn(world: World, obj: PlanningObjective):
    """Return a weight function that combines distance, fuel, and risk."""

    def weight(u: int, v: int, data: dict) -> float:
        base_w = float(data.get("weight", 1.0))
        # Approximate distance from base_weight (base_weight ≈ dist * random_factor)
        pos_u = world.get_node_pos(u)
        pos_v = world.get_node_pos(v)
        dist = math.hypot(pos_u[0] - pos_v[0], pos_u[1] - pos_v[1])
        risk = float(data.get("risk", 0.0))

        cost = (obj.w_time + obj.w_fuel) * base_w + obj.w_risk * risk * dist

        # Elevation/slope cost
        if obj.w_slope > 0:
            if world.elevation_grid is not None:
                sc = world.elevation_grid.slope_cost(
                    pos_u[0], pos_u[1], pos_v[0], pos_v[1]
                )
            else:
                # Use pre-annotated edge slope if available
                edge_slope = float(data.get("slope", 0.0))
                sc = min(abs(edge_slope) / 0.3, 1.0)  # normalise to [0,1]
            cost += obj.w_slope * sc * dist

        return cost

    return weight


def _heuristic(world: World, node_a: int, node_b: int) -> float:
    """Euclidean distance heuristic for A*."""
    pos_a = world.get_node_pos(node_a)
    pos_b = world.get_node_pos(node_b)
    return math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])
