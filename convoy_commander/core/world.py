"""World model: obstacles, roads (graph), no-go zones, landmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from convoy_commander.core.config import WorldConfig


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
        self.road_graph: nx.Graph = nx.Graph()

        self._generate(rng)

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
            self.landmarks.append(Landmark(x=x, y=y))

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
