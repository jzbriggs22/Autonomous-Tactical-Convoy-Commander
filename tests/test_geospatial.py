"""Tests for geospatial module: elevation grid, coordinate projection, and integration."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from convoy_commander.core.config import PlanningObjective, SimConfig, WorldConfig
from convoy_commander.core.world import World
from convoy_commander.geospatial.coords import Projector, create_projector
from convoy_commander.geospatial.elevation import ElevationGrid
from convoy_commander.planning.global_planner import plan_route
from convoy_commander.sim.scenarios import get_scenario


# ---------------------------------------------------------------------------
# ElevationGrid tests (pure numpy, no external deps)
# ---------------------------------------------------------------------------


class TestElevationGrid:
    def test_flat_terrain_zero_slope(self):
        data = np.full((10, 10), 100.0)
        grid = ElevationGrid(data=data, resolution_x=10.0, resolution_y=10.0)
        assert grid.elevation_at(50.0, 50.0) == pytest.approx(100.0, abs=1.0)
        assert grid.slope_between(0.0, 0.0, 90.0, 90.0) == pytest.approx(0.0)

    def test_sloped_terrain(self):
        # Linear slope: elevation increases with column (x direction)
        data = np.zeros((10, 10))
        for col in range(10):
            data[:, col] = col * 10.0
        grid = ElevationGrid(data=data, resolution_x=10.0, resolution_y=10.0)
        slope = grid.slope_between(0.0, 50.0, 90.0, 50.0)
        assert slope > 0

    def test_elevation_bilinear_interpolation(self):
        data = np.array([[0.0, 10.0], [20.0, 30.0]])
        grid = ElevationGrid(data=data, resolution_x=10.0, resolution_y=10.0)
        # Centre of 2x2 grid should be average = 15.0
        center = grid.elevation_at(5.0, 5.0)
        assert center == pytest.approx(15.0, abs=0.5)

    def test_out_of_bounds_returns_zero(self):
        data = np.full((5, 5), 50.0)
        grid = ElevationGrid(data=data, resolution_x=10.0, resolution_y=10.0)
        assert grid.elevation_at(999.0, 999.0) == 0.0

    def test_slope_cost_bounded(self):
        data = np.zeros((10, 10))
        data[:, -1] = 1000.0  # extreme cliff
        grid = ElevationGrid(data=data, resolution_x=10.0, resolution_y=10.0)
        cost = grid.slope_cost(0.0, 50.0, 90.0, 50.0)
        assert 0.0 <= cost <= 1.0

    def test_slope_cost_uphill_vs_downhill(self):
        # Ramp: elevation 0 at left, 30 at right
        data = np.zeros((5, 10))
        for col in range(10):
            data[:, col] = col * 3.0
        grid = ElevationGrid(data=data, resolution_x=10.0, resolution_y=10.0)
        uphill = grid.slope_cost(10.0, 20.0, 80.0, 20.0)
        downhill = grid.slope_cost(80.0, 20.0, 10.0, 20.0)
        # Uphill should cost more than downhill
        assert uphill > downhill

    def test_zero_distance_slope(self):
        data = np.full((5, 5), 42.0)
        grid = ElevationGrid(data=data, resolution_x=10.0, resolution_y=10.0)
        assert grid.slope_between(10.0, 10.0, 10.0, 10.0) == 0.0
        assert grid.slope_cost(10.0, 10.0, 10.0, 10.0) == 0.0


# ---------------------------------------------------------------------------
# Projector tests
# ---------------------------------------------------------------------------


class TestProjector:
    def test_roundtrip(self):
        proj = create_projector(center_lat=37.0, center_lon=-122.0)
        x, y = proj.to_local(37.01, -121.99)
        lat, lon = proj.to_latlon(x, y)
        assert lat == pytest.approx(37.01, abs=0.001)
        assert lon == pytest.approx(-121.99, abs=0.001)

    def test_origin_is_zero(self):
        proj = Projector(center_lat=40.0, center_lon=-74.0)
        x, y = proj.to_local(40.0, -74.0)
        assert x == pytest.approx(0.0, abs=0.01)
        assert y == pytest.approx(0.0, abs=0.01)

    def test_north_is_positive_y(self):
        proj = Projector(center_lat=0.0, center_lon=0.0)
        _, y = proj.to_local(1.0, 0.0)  # 1 degree north
        assert y > 0

    def test_east_is_positive_x(self):
        proj = Projector(center_lat=0.0, center_lon=0.0)
        x, _ = proj.to_local(0.0, 1.0)  # 1 degree east
        assert x > 0


# ---------------------------------------------------------------------------
# World integration tests
# ---------------------------------------------------------------------------


class TestWorldElevation:
    def test_default_world_no_elevation(self):
        config = WorldConfig()
        world = World(config, np.random.default_rng(42))
        assert world.elevation_grid is None
        assert world.elevation_at(500.0, 500.0) == 0.0
        assert world.get_slope(0.0, 0.0, 100.0, 100.0) == 0.0

    def test_backward_compat_procedural(self):
        """Default config still generates procedural world with 144 nodes."""
        config = WorldConfig()
        world = World(config, np.random.default_rng(42))
        assert world.road_graph.number_of_nodes() == 144  # 12x12 grid
        assert world.elevation_grid is None

    def test_config_geo_fields_optional(self):
        """WorldConfig validates with and without geo fields."""
        config_default = WorldConfig()
        assert config_default.elevation_tiff is None
        assert config_default.osm_source is None
        assert config_default.geo_bounds is None

        config_geo = WorldConfig(
            elevation_tiff="/some/path.tif",
            osm_source="Some Place",
            geo_bounds=(38.0, 37.0, -121.0, -122.0),
        )
        assert config_geo.elevation_tiff == "/some/path.tif"


# ---------------------------------------------------------------------------
# Planner slope weight tests
# ---------------------------------------------------------------------------


class TestPlannerSlope:
    def test_planner_slope_weight_prefers_flat(self):
        """With slope-aware A*, the planner should prefer a flat path over a hilly one."""
        # Build a small synthetic graph with 2 paths:
        # Path A: flat (nodes 0 -> 1 -> 3), longer distance
        # Path B: hilly (nodes 0 -> 2 -> 3), shorter distance but steep
        config = WorldConfig(road_graph_density=2, obstacle_count=0, nogo_zone_count=0,
                             poly_obstacle_count=0, landmark_count=0)
        world = World(config, np.random.default_rng(42))

        # Replace road graph with our synthetic one (no elevation grid — uses edge slope)
        G = nx.Graph()
        G.add_node(0, pos=(50.0, 50.0))
        G.add_node(1, pos=(500.0, 50.0))   # flat path waypoint
        G.add_node(2, pos=(500.0, 500.0))  # hilly path waypoint
        G.add_node(3, pos=(900.0, 500.0))  # goal
        G.add_edge(0, 1, weight=450.0, risk=0.0, slope=0.0)
        G.add_edge(1, 3, weight=500.0, risk=0.0, slope=0.0)
        G.add_edge(0, 2, weight=450.0, risk=0.0, slope=0.8)   # very steep
        G.add_edge(2, 3, weight=400.0, risk=0.0, slope=0.8)   # very steep
        world.road_graph = G
        world.elevation_grid = None  # use edge slope annotations

        # Plan with slope weight enabled
        obj_flat = PlanningObjective(w_time=1.0, w_fuel=0.0, w_risk=0.0, w_slope=3.0)
        route_slope = plan_route(world, 50.0, 50.0, 900.0, 500.0, objective=obj_flat)

        # With slope weight, should go through node 1 (flat)
        flat_waypoint = (500.0, 50.0)
        assert flat_waypoint in route_slope


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------


class TestTerrainScenario:
    def test_terrain_real_scenario_loads(self):
        """terrain_real scenario loads without error."""
        config = get_scenario("terrain_real")
        assert config.scenario == "terrain_real"
        assert config.planning.w_slope == pytest.approx(0.4)
        assert config.duration == 600.0

    def test_w_slope_in_planning_objective(self):
        """PlanningObjective has w_slope field defaulting to 0."""
        obj = PlanningObjective()
        assert obj.w_slope == 0.0
        obj_slope = PlanningObjective(w_slope=0.5)
        assert obj_slope.w_slope == 0.5
