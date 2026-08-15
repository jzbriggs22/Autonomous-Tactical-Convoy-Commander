"""World model: obstacles, roads (graph), no-go zones, landmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from convoy_commander.core.config import WorldConfig
from convoy_commander.ew.jammer import RFJammer


@dataclass
class Obstacle:
    """Circular obstacle."""

    x: float
    y: float
    radius: float

    def contains(self, px: float, py: float) -> bool:
        dx = px - self.x
        dy = py - self.y
        return dx * dx + dy * dy < self.radius * self.radius


@dataclass
class NoGoZone:
    """Circular no-go zone (larger, softer penalty)."""

    x: float
    y: float
    radius: float

    def contains(self, px: float, py: float) -> bool:
        dx = px - self.x
        dy = py - self.y
        return dx * dx + dy * dy < self.radius * self.radius


@dataclass
class Landmark:
    """Known landmark for position fixing."""

    x: float
    y: float
    detection_range: float = 50.0


@dataclass
class PolyObstacle:
    """Axis-aligned rectangular obstacle.

    Uses half-extents (half_w, half_h) from the centre so the obstacle
    spans [x-half_w, x+half_w] × [y-half_h, y+half_h].

    SAFETY NOTE: Clearance is the minimum Euclidean distance from the
    query point to the nearest rectangle edge, which is exact for
    axis-aligned boxes (no swept-volume approximation).
    """

    x: float
    y: float
    half_w: float  # half-width  (total width  = 2 * half_w)
    half_h: float  # half-height (total height = 2 * half_h)

    def contains(self, px: float, py: float) -> bool:
        return abs(px - self.x) < self.half_w and abs(py - self.y) < self.half_h

    def clearance_from(self, px: float, py: float) -> float:
        """Minimum distance from point to the nearest rectangle boundary."""
        dx = max(abs(px - self.x) - self.half_w, 0.0)
        dy = max(abs(py - self.y) - self.half_h, 0.0)
        return math.hypot(dx, dy)

    @property
    def bounding_radius(self) -> float:
        """Conservative bounding circle radius for quick rejection tests."""
        return math.hypot(self.half_w, self.half_h)


@dataclass
class SpoofRegion:
    """GPS spoofing zone: shifts the apparent GPS position by a fixed offset.

    Vehicles inside this region receive GPS measurements biased by
    (offset_x, offset_y) metres.  The innovation gate in the estimator
    detects and rejects anomalously large fixes.
    """

    x: float
    y: float
    radius: float
    offset_x: float  # Spoof bias: added to true_x before passing to estimator
    offset_y: float  # Spoof bias: added to true_y before passing to estimator

    def contains(self, px: float, py: float) -> bool:
        dx = px - self.x
        dy = py - self.y
        return dx * dx + dy * dy < self.radius * self.radius


class World:
    """2D continuous world with obstacles, road graph, no-go zones, and landmarks."""

    def __init__(self, config: WorldConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.width = config.width
        self.height = config.height
        self.obstacles: list[Obstacle] = []
        self.poly_obstacles: list[PolyObstacle] = []
        self.nogo_zones: list[NoGoZone] = []
        self.landmarks: list[Landmark] = []
        self.spoof_regions: list[SpoofRegion] = []
        self.jammers: list[RFJammer] = []
        self.road_graph: nx.Graph = nx.Graph()

        # Elevation grid (None when no DEM loaded)
        from convoy_commander.geospatial.elevation import ElevationGrid
        self.elevation_grid: ElevationGrid | None = None

        if config.osm_source is not None or config.elevation_tiff is not None:
            self._load_geospatial(rng)
        else:
            self._generate(rng)

    # ------------------------------------------------------------------
    # Geospatial loading
    # ------------------------------------------------------------------

    def _load_geospatial(self, rng: np.random.Generator) -> None:
        """Load world from real-world geospatial data sources."""
        cfg = self.config

        # Load elevation DEM if provided
        if cfg.elevation_tiff is not None:
            from convoy_commander.geospatial.terrain import load_elevation_grid
            self.elevation_grid = load_elevation_grid(
                cfg.elevation_tiff,
                target_width=self.width,
                target_height=self.height,
                max_slope_threshold=cfg.max_slope_threshold,
            )

        # Load road graph from OSM if provided
        if cfg.osm_source is not None:
            from convoy_commander.geospatial.osm_roads import load_osm_road_graph
            self.road_graph = load_osm_road_graph(
                cfg.osm_source,
                target_width=self.width,
                target_height=self.height,
                bounds=cfg.geo_bounds,
                network_type=cfg.osm_network_type,
                elevation_grid=self.elevation_grid,
            )
        else:
            # No OSM source but have DEM — use procedural road grid on top
            self._generate_road_grid(rng)

        # Generate procedural hazards on top of real terrain
        self._generate_hazards(rng)

        # Annotate edges with risk scores
        self._annotate_edge_risk()

        # Annotate edges with slope data if elevation is available
        if self.elevation_grid is not None:
            self._annotate_edge_slope()

    def _generate_road_grid(self, rng: np.random.Generator) -> None:
        """Generate the procedural road graph grid (extracted from _generate)."""
        n = self.config.road_graph_density
        spacing_x = self.width / (n + 1)
        spacing_y = self.height / (n + 1)

        for i in range(n):
            for j in range(n):
                node_id = i * n + j
                x = spacing_x * (i + 1)
                y = spacing_y * (j + 1)
                self.road_graph.add_node(node_id, pos=(x, y))

        for i in range(n):
            for j in range(n):
                node_id = i * n + j
                pos = self.road_graph.nodes[node_id]["pos"]
                if i + 1 < n:
                    neighbor = (i + 1) * n + j
                    npos = self.road_graph.nodes[neighbor]["pos"]
                    dist = math.hypot(npos[0] - pos[0], npos[1] - pos[1])
                    weight = dist * (1.0 + 0.3 * rng.random())
                    self.road_graph.add_edge(node_id, neighbor, weight=weight)
                if j + 1 < n:
                    neighbor = i * n + (j + 1)
                    npos = self.road_graph.nodes[neighbor]["pos"]
                    dist = math.hypot(npos[0] - pos[0], npos[1] - pos[1])
                    weight = dist * (1.0 + 0.3 * rng.random())
                    self.road_graph.add_edge(node_id, neighbor, weight=weight)
                if i + 1 < n and j + 1 < n and rng.random() < 0.3:
                    neighbor = (i + 1) * n + (j + 1)
                    npos = self.road_graph.nodes[neighbor]["pos"]
                    dist = math.hypot(npos[0] - pos[0], npos[1] - pos[1])
                    weight = dist * (1.0 + 0.3 * rng.random())
                    self.road_graph.add_edge(node_id, neighbor, weight=weight)

        edges = list(self.road_graph.edges())
        for e in edges:
            if rng.random() < 0.1:
                self.road_graph.remove_edge(*e)
                if not nx.is_connected(self.road_graph):
                    pos_a = self.road_graph.nodes[e[0]]["pos"]
                    pos_b = self.road_graph.nodes[e[1]]["pos"]
                    dist = math.hypot(pos_b[0] - pos_a[0], pos_b[1] - pos_a[1])
                    self.road_graph.add_edge(e[0], e[1], weight=dist)

    def _generate_hazards(self, rng: np.random.Generator) -> None:
        """Generate obstacles, no-go zones, landmarks, spoof regions."""
        # No-go zones
        for _ in range(self.config.nogo_zone_count):
            for _attempt in range(50):
                r = rng.uniform(*self.config.nogo_zone_radius_range)
                x = rng.uniform(r, self.width - r)
                y = rng.uniform(r, self.height - r)
                if x > 150 and x < self.width - 150:
                    self.nogo_zones.append(NoGoZone(x=x, y=y, radius=r))
                    break

        # Circular obstacles
        for _ in range(self.config.obstacle_count):
            for _attempt in range(50):
                r = rng.uniform(*self.config.obstacle_radius_range)
                x = rng.uniform(r, self.width - r)
                y = rng.uniform(r, self.height - r)
                ok = True
                for nz in self.nogo_zones:
                    if math.hypot(x - nz.x, y - nz.y) < r + nz.radius:
                        ok = False
                        break
                if x < 100 and y < 200:
                    ok = False
                if ok:
                    self.obstacles.append(Obstacle(x=x, y=y, radius=r))
                    break

        # Remove road edges through obstacles/nogo zones
        edges_to_remove: list[tuple[int, int]] = []
        for u, v in self.road_graph.edges():
            pos_u = np.array(self.road_graph.nodes[u]["pos"])
            pos_v = np.array(self.road_graph.nodes[v]["pos"])
            blocked = False
            for obs in self.obstacles:
                if self._segment_intersects_circle(pos_u, pos_v, np.array([obs.x, obs.y]), obs.radius):
                    blocked = True
                    break
            if not blocked:
                for nz in self.nogo_zones:
                    if self._segment_intersects_circle(pos_u, pos_v, np.array([nz.x, nz.y]), nz.radius):
                        blocked = True
                        break
            if blocked:
                edges_to_remove.append((u, v))
        for e in edges_to_remove:
            self.road_graph.remove_edge(*e)

        # Ensure connectivity
        if self.road_graph.number_of_nodes() > 0 and not nx.is_connected(self.road_graph):
            components = list(nx.connected_components(self.road_graph))
            largest = max(components, key=len)
            for comp in components:
                if comp is largest:
                    continue
                best_dist = float("inf")
                best_pair = (list(comp)[0], list(largest)[0])
                for a in comp:
                    pa = np.array(self.road_graph.nodes[a]["pos"])
                    for b in largest:
                        pb = np.array(self.road_graph.nodes[b]["pos"])
                        d = float(np.linalg.norm(pa - pb))
                        if d < best_dist:
                            best_dist = d
                            best_pair = (a, b)
                self.road_graph.add_edge(best_pair[0], best_pair[1], weight=best_dist * 1.5)

        # Poly obstacles
        for _ in range(self.config.poly_obstacle_count):
            for _attempt in range(50):
                half_w = rng.uniform(8.0, 30.0)
                half_h = rng.uniform(8.0, 20.0)
                x = rng.uniform(half_w + 10, self.width - half_w - 10)
                y = rng.uniform(half_h + 10, self.height - half_h - 10)
                if x < 120 and y < 220:
                    continue
                ok = True
                for nz in self.nogo_zones:
                    if math.hypot(x - nz.x, y - nz.y) < math.hypot(half_w, half_h) + nz.radius:
                        ok = False
                        break
                if ok:
                    self.poly_obstacles.append(PolyObstacle(x=x, y=y, half_w=half_w, half_h=half_h))
                    break

        # Remove edges blocked by poly obstacles
        poly_blocked: list[tuple[int, int]] = []
        for u, v in self.road_graph.edges():
            pos_u = np.array(self.road_graph.nodes[u]["pos"])
            pos_v = np.array(self.road_graph.nodes[v]["pos"])
            for po in self.poly_obstacles:
                if self._segment_intersects_rect(pos_u, pos_v, po.x, po.y, po.half_w, po.half_h):
                    poly_blocked.append((u, v))
                    break
        for e in poly_blocked:
            if self.road_graph.has_edge(*e):
                self.road_graph.remove_edge(*e)
                if self.road_graph.number_of_nodes() > 0 and not nx.is_connected(self.road_graph):
                    self.road_graph.add_edge(e[0], e[1], weight=500.0, risk=0.9)

        # Landmarks
        for _ in range(self.config.landmark_count):
            x = rng.uniform(50, self.width - 50)
            y = rng.uniform(50, self.height - 50)
            self.landmarks.append(
                Landmark(x=x, y=y, detection_range=self.config.landmark_detection_range)
            )

        # Spoof regions
        for _ in range(self.config.spoof_region_count):
            spoof_r = 80.0
            x = rng.uniform(self.width * 0.3, self.width * 0.7)
            y = rng.uniform(self.height * 0.3, self.height * 0.7)
            angle = rng.uniform(0, 2 * math.pi)
            magnitude = rng.uniform(self.config.spoof_offset_max * 0.5, self.config.spoof_offset_max)
            offset_x = magnitude * math.cos(angle)
            offset_y = magnitude * math.sin(angle)
            self.spoof_regions.append(
                SpoofRegion(x=x, y=y, radius=spoof_r, offset_x=offset_x, offset_y=offset_y)
            )

    def _annotate_edge_slope(self) -> None:
        """Annotate road graph edges with slope from elevation grid."""
        if self.elevation_grid is None:
            return
        for u, v in self.road_graph.edges():
            pos_u = self.road_graph.nodes[u]["pos"]
            pos_v = self.road_graph.nodes[v]["pos"]
            slope = self.elevation_grid.slope_between(pos_u[0], pos_u[1], pos_v[0], pos_v[1])
            self.road_graph[u][v]["slope"] = slope

    # ------------------------------------------------------------------
    # Elevation queries
    # ------------------------------------------------------------------

    def elevation_at(self, x: float, y: float) -> float:
        """Return elevation at (x, y). Returns 0.0 if no elevation data loaded."""
        if self.elevation_grid is None:
            return 0.0
        return self.elevation_grid.elevation_at(x, y)

    def get_slope(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Return slope between two points. Returns 0.0 if no elevation data."""
        if self.elevation_grid is None:
            return 0.0
        return self.elevation_grid.slope_between(x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # Procedural generation (original)
    # ------------------------------------------------------------------

    def _generate(self, rng: np.random.Generator) -> None:
        """Generate world features deterministically."""
        # Road graph: regular grid with some edges removed and weighted
        n = self.config.road_graph_density
        spacing_x = self.width / (n + 1)
        spacing_y = self.height / (n + 1)

        for i in range(n):
            for j in range(n):
                node_id = i * n + j
                x = spacing_x * (i + 1)
                y = spacing_y * (j + 1)
                self.road_graph.add_node(node_id, pos=(x, y))

        # Add edges (4-connected grid + some diagonals)
        for i in range(n):
            for j in range(n):
                node_id = i * n + j
                pos = self.road_graph.nodes[node_id]["pos"]
                # Right neighbor
                if i + 1 < n:
                    neighbor = (i + 1) * n + j
                    npos = self.road_graph.nodes[neighbor]["pos"]
                    dist = math.hypot(npos[0] - pos[0], npos[1] - pos[1])
                    weight = dist * (1.0 + 0.3 * rng.random())
                    self.road_graph.add_edge(node_id, neighbor, weight=weight)
                # Up neighbor
                if j + 1 < n:
                    neighbor = i * n + (j + 1)
                    npos = self.road_graph.nodes[neighbor]["pos"]
                    dist = math.hypot(npos[0] - pos[0], npos[1] - pos[1])
                    weight = dist * (1.0 + 0.3 * rng.random())
                    self.road_graph.add_edge(node_id, neighbor, weight=weight)
                # Diagonal (add some)
                if i + 1 < n and j + 1 < n and rng.random() < 0.3:
                    neighbor = (i + 1) * n + (j + 1)
                    npos = self.road_graph.nodes[neighbor]["pos"]
                    dist = math.hypot(npos[0] - pos[0], npos[1] - pos[1])
                    weight = dist * (1.0 + 0.3 * rng.random())
                    self.road_graph.add_edge(node_id, neighbor, weight=weight)

        # Remove some random edges to make it more interesting
        edges = list(self.road_graph.edges())
        for e in edges:
            if rng.random() < 0.1:
                self.road_graph.remove_edge(*e)
                # Re-add if graph becomes disconnected
                if not nx.is_connected(self.road_graph):
                    pos_a = self.road_graph.nodes[e[0]]["pos"]
                    pos_b = self.road_graph.nodes[e[1]]["pos"]
                    dist = math.hypot(pos_b[0] - pos_a[0], pos_b[1] - pos_a[1])
                    self.road_graph.add_edge(e[0], e[1], weight=dist)

        # Generate no-go zones (avoid center corridor)
        for _ in range(self.config.nogo_zone_count):
            for _attempt in range(50):
                r = rng.uniform(*self.config.nogo_zone_radius_range)
                x = rng.uniform(r, self.width - r)
                y = rng.uniform(r, self.height - r)
                # Keep away from start/end corridors
                if x > 150 and x < self.width - 150:
                    self.nogo_zones.append(NoGoZone(x=x, y=y, radius=r))
                    break

        # Generate obstacles (avoid no-go zones to prevent overlap)
        for _ in range(self.config.obstacle_count):
            for _attempt in range(50):
                r = rng.uniform(*self.config.obstacle_radius_range)
                x = rng.uniform(r, self.width - r)
                y = rng.uniform(r, self.height - r)
                # Don't place on top of no-go zones
                ok = True
                for nz in self.nogo_zones:
                    if math.hypot(x - nz.x, y - nz.y) < r + nz.radius:
                        ok = False
                        break
                # Keep clear of spawn area
                if x < 100 and y < 200:
                    ok = False
                if ok:
                    self.obstacles.append(Obstacle(x=x, y=y, radius=r))
                    break

        # Remove road graph edges that pass through obstacles or no-go zones
        edges_to_remove: list[tuple[int, int]] = []
        for u, v in self.road_graph.edges():
            pos_u = np.array(self.road_graph.nodes[u]["pos"])
            pos_v = np.array(self.road_graph.nodes[v]["pos"])
            blocked = False
            for obs in self.obstacles:
                if self._segment_intersects_circle(
                    pos_u, pos_v, np.array([obs.x, obs.y]), obs.radius
                ):
                    blocked = True
                    break
            if not blocked:
                for nz in self.nogo_zones:
                    if self._segment_intersects_circle(
                        pos_u, pos_v, np.array([nz.x, nz.y]), nz.radius
                    ):
                        blocked = True
                        break
            if blocked:
                edges_to_remove.append((u, v))

        for e in edges_to_remove:
            self.road_graph.remove_edge(*e)

        # Ensure connectivity after removal
        if not nx.is_connected(self.road_graph):
            components = list(nx.connected_components(self.road_graph))
            # Connect each component to the largest one
            largest = max(components, key=len)
            for comp in components:
                if comp is largest:
                    continue
                # Find closest pair of nodes between comp and largest
                best_dist = float("inf")
                best_pair = (list(comp)[0], list(largest)[0])
                for a in comp:
                    pa = np.array(self.road_graph.nodes[a]["pos"])
                    for b in largest:
                        pb = np.array(self.road_graph.nodes[b]["pos"])
                        d = float(np.linalg.norm(pa - pb))
                        if d < best_dist:
                            best_dist = d
                            best_pair = (a, b)
                self.road_graph.add_edge(best_pair[0], best_pair[1], weight=best_dist * 1.5)

        # Generate axis-aligned rectangle obstacles
        for _ in range(self.config.poly_obstacle_count):
            for _attempt in range(50):
                half_w = rng.uniform(8.0, 30.0)
                half_h = rng.uniform(8.0, 20.0)
                x = rng.uniform(half_w + 10, self.width - half_w - 10)
                y = rng.uniform(half_h + 10, self.height - half_h - 10)
                # Keep clear of spawn area and no-go zones
                if x < 120 and y < 220:
                    continue
                ok = True
                for nz in self.nogo_zones:
                    if math.hypot(x - nz.x, y - nz.y) < math.hypot(half_w, half_h) + nz.radius:
                        ok = False
                        break
                if ok:
                    self.poly_obstacles.append(PolyObstacle(x=x, y=y, half_w=half_w, half_h=half_h))
                    break

        # Remove road graph edges blocked by poly obstacles
        poly_blocked: list[tuple[int, int]] = []
        for u, v in self.road_graph.edges():
            pos_u = np.array(self.road_graph.nodes[u]["pos"])
            pos_v = np.array(self.road_graph.nodes[v]["pos"])
            for po in self.poly_obstacles:
                if self._segment_intersects_rect(pos_u, pos_v, po.x, po.y, po.half_w, po.half_h):
                    poly_blocked.append((u, v))
                    break
        for e in poly_blocked:
            if self.road_graph.has_edge(*e):
                self.road_graph.remove_edge(*e)
                if not nx.is_connected(self.road_graph):
                    self.road_graph.add_edge(e[0], e[1], weight=500.0, risk=0.9)

        # Generate landmarks
        for _ in range(self.config.landmark_count):
            x = rng.uniform(50, self.width - 50)
            y = rng.uniform(50, self.height - 50)
            self.landmarks.append(
                Landmark(x=x, y=y, detection_range=self.config.landmark_detection_range)
            )

        # Annotate edges with risk score [0,1] based on proximity to hazards
        self._annotate_edge_risk()

        # Generate GPS spoofing regions (placed mid-route to be encountered)
        for _ in range(self.config.spoof_region_count):
            spoof_r = 80.0
            x = rng.uniform(self.width * 0.3, self.width * 0.7)
            y = rng.uniform(self.height * 0.3, self.height * 0.7)
            # Random offset within the configured max
            angle = rng.uniform(0, 2 * math.pi)
            magnitude = rng.uniform(
                self.config.spoof_offset_max * 0.5, self.config.spoof_offset_max
            )
            offset_x = magnitude * math.cos(angle)
            offset_y = magnitude * math.sin(angle)
            self.spoof_regions.append(
                SpoofRegion(x=x, y=y, radius=spoof_r, offset_x=offset_x, offset_y=offset_y)
            )

    @staticmethod
    def _segment_intersects_rect(
        p1: np.ndarray, p2: np.ndarray, cx: float, cy: float, hw: float, hh: float
    ) -> bool:
        """Check if line segment p1->p2 intersects an axis-aligned rectangle.

        Uses the slab (parametric clipping) method.  Endpoints inside the
        rectangle also count as an intersection.
        """
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        t_min, t_max = 0.0, 1.0

        for axis_d, axis_p, center_c, half in [
            (dx, float(p1[0]), cx, hw),
            (dy, float(p1[1]), cy, hh),
        ]:
            if abs(axis_d) < 1e-9:
                # Segment parallel to slab; check if outside
                if abs(axis_p - center_c) > half:
                    return False
            else:
                t1 = (center_c - half - axis_p) / axis_d
                t2 = (center_c + half - axis_p) / axis_d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return False
        return True

    @staticmethod
    def _segment_intersects_circle(
        p1: np.ndarray, p2: np.ndarray, center: np.ndarray, radius: float
    ) -> bool:
        """Check if line segment p1->p2 intersects circle."""
        d = p2 - p1
        f = p1 - center
        a = float(np.dot(d, d))
        b = 2.0 * float(np.dot(f, d))
        c = float(np.dot(f, f)) - radius * radius

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return False

        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)

        return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)

    def _annotate_edge_risk(self) -> None:
        """Annotate road graph edges with a risk score in [0, 1].

        Risk is inversely proportional to clearance from obstacles and no-go
        zones.  This attribute is used by the multi-objective global planner.
        """
        max_influence = 60.0  # m — beyond this, risk contribution is 0
        for u, v in self.road_graph.edges():
            pos_u = np.array(self.road_graph.nodes[u]["pos"])
            pos_v = np.array(self.road_graph.nodes[v]["pos"])
            mid = (pos_u + pos_v) * 0.5

            risk = 0.0
            for obs in self.obstacles:
                d = math.hypot(mid[0] - obs.x, mid[1] - obs.y) - obs.radius
                if d < max_influence:
                    risk = max(risk, 1.0 - max(d, 0.0) / max_influence)
            for po in self.poly_obstacles:
                d = po.clearance_from(mid[0], mid[1])
                if d < max_influence:
                    risk = max(risk, 1.0 - d / max_influence)
            for nz in self.nogo_zones:
                d = math.hypot(mid[0] - nz.x, mid[1] - nz.y) - nz.radius
                if d < max_influence:
                    risk = max(risk, 0.5 * (1.0 - max(d, 0.0) / max_influence))

            self.road_graph[u][v]["risk"] = min(1.0, risk)

    def is_blocked(self, x: float, y: float) -> bool:
        """Check if a position is inside an obstacle (circular or rectangular)."""
        for obs in self.obstacles:
            if obs.contains(x, y):
                return True
        for po in self.poly_obstacles:
            if po.contains(x, y):
                return True
        return False

    def is_nogo(self, x: float, y: float) -> bool:
        """Check if a position is in a no-go zone."""
        for nz in self.nogo_zones:
            if nz.contains(x, y):
                return True
        return False

    def in_bounds(self, x: float, y: float) -> bool:
        return 0 <= x <= self.width and 0 <= y <= self.height

    def nearest_road_node(self, x: float, y: float) -> int:
        """Find nearest road graph node to position."""
        best_node = 0
        best_dist = float("inf")
        for node_id, data in self.road_graph.nodes(data=True):
            pos = data["pos"]
            dist = math.hypot(pos[0] - x, pos[1] - y)
            if dist < best_dist:
                best_dist = dist
                best_node = node_id
        return best_node

    def get_node_pos(self, node_id: int) -> tuple[float, float]:
        """Get position of a road graph node."""
        return self.road_graph.nodes[node_id]["pos"]

    def add_obstacle(self, x: float, y: float, radius: float) -> None:
        """Add an obstacle at runtime (for dynamic scenarios)."""
        self.obstacles.append(Obstacle(x=x, y=y, radius=radius))

    def get_spoof_offset(self, x: float, y: float) -> tuple[float, float] | None:
        """Return GPS spoof offset (ox, oy) if position is inside a spoof region.

        Returns None if the position is not spoofed.  If multiple regions
        overlap, returns the first match (deterministic by generation order).
        """
        for region in self.spoof_regions:
            if region.contains(x, y):
                return (region.offset_x, region.offset_y)
        return None

    def landmarks_in_range(self, x: float, y: float) -> list[Landmark]:
        """Return landmarks within detection range of position."""
        result: list[Landmark] = []
        for lm in self.landmarks:
            if math.hypot(lm.x - x, lm.y - y) <= lm.detection_range:
                result.append(lm)
        return result

    # ------------------------------------------------------------------
    # Electronic warfare (Phase 11)
    # ------------------------------------------------------------------

    def add_jammer(self, jammer: RFJammer) -> None:
        """Add an RF jammer to the world."""
        jammer._world_w = self.width
        jammer._world_h = self.height
        self.jammers.append(jammer)

    def get_jammer_degradation(self, px: float, py: float) -> float:
        """Return combined SNR loss multiplier from all jammers at position.

        Returns 1.0 (no degradation) if no jammers affect this position.
        """
        max_deg = 1.0
        for j in self.jammers:
            deg = j.snr_degradation(px, py)
            if deg > max_deg:
                max_deg = deg
        return max_deg

    def is_gps_jammed(self, px: float, py: float) -> bool:
        """Return True if any GPS jammer covers this position."""
        for j in self.jammers:
            if j.gps_denied_at(px, py):
                return True
        return False

    def annotate_edge_threat(self) -> None:
        """Annotate road graph edges with jammer threat score in [0, 1].

        Same inverse-distance pattern as ``_annotate_edge_risk()``.
        Called after jammers are placed or triangulated to update costs.
        """
        if not self.jammers:
            return
        for u, v in self.road_graph.edges():
            pos_u = self.road_graph.nodes[u]["pos"]
            pos_v = self.road_graph.nodes[v]["pos"]
            mid_x = (pos_u[0] + pos_v[0]) * 0.5
            mid_y = (pos_u[1] + pos_v[1]) * 0.5

            threat = 0.0
            for j in self.jammers:
                d = math.hypot(mid_x - j.x, mid_y - j.y)
                if d < j.radius:
                    threat = max(threat, 1.0 - d / j.radius)
            self.road_graph[u][v]["threat"] = min(1.0, threat)
