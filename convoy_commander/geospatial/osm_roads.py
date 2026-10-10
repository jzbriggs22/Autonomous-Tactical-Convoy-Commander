"""OpenStreetMap road network loading and projection to local metric graph."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from convoy_commander.geospatial.coords import Projector
from convoy_commander.geospatial.elevation import ElevationGrid


def load_osm_road_graph(
    osm_source: str | Path,
    target_width: float = 1000.0,
    target_height: float = 1000.0,
    bounds: tuple[float, float, float, float] | None = None,
    network_type: str = "drive",
    elevation_grid: ElevationGrid | None = None,
) -> nx.Graph:
    """Load an OSM road network and project to local metre coordinates.

    Args:
        osm_source: Either a ``.graphml`` file path **or** a place name
            string for ``osmnx.graph_from_place``.
        target_width: Scale the graph to fit this width in metres.
        target_height: Scale the graph to fit this height in metres.
        bounds: (north, south, east, west) in WGS84 for bounding-box query.
            Ignored if *osm_source* is a file.
        network_type: osmnx network type (``"drive"``, ``"walk"``, etc.).
        elevation_grid: If provided, annotate edges with slope data.

    Returns:
        ``nx.Graph`` with nodes having ``pos=(x, y)`` and edges having
        ``weight`` (distance) and ``risk`` (default 0.0).  Node IDs are
        sequential integers starting at 0.
    """
    from convoy_commander.geospatial.compat import require_osmnx

    ox = require_osmnx()

    source_path = Path(osm_source)

    # Load graph
    if source_path.exists() and source_path.suffix in (".graphml", ".osm"):
        G_raw = ox.load_graphml(source_path)
    elif bounds is not None:
        north, south, east, west = bounds
        G_raw = ox.graph_from_bbox(north, south, east, west, network_type=network_type)
    else:
        G_raw = ox.graph_from_place(str(osm_source), network_type=network_type)

    # Convert MultiDiGraph → undirected Graph
    G_undirected = nx.Graph(G_raw.to_undirected())

    # Project nodes to local metric coordinates
    # Find centre of bounding box for equirectangular projection
    lats = [d.get("y", d.get("lat", 0.0)) for _, d in G_undirected.nodes(data=True)]
    lons = [d.get("x", d.get("lon", 0.0)) for _, d in G_undirected.nodes(data=True)]

    if not lats or not lons:
        raise ValueError("OSM graph has no nodes with coordinate attributes")

    proj = Projector(
        center_lat=(min(lats) + max(lats)) / 2.0,
        center_lon=(min(lons) + max(lons)) / 2.0,
    )

    # Convert all node positions to local metres
    local_positions: dict[int, tuple[float, float]] = {}
    for node, data in G_undirected.nodes(data=True):
        lat = data.get("y", data.get("lat", 0.0))
        lon = data.get("x", data.get("lon", 0.0))
        x, y = proj.to_local(lat, lon)
        local_positions[node] = (x, y)

    # Translate so min x/y = 0 and scale to target dimensions
    all_x = [p[0] for p in local_positions.values()]
    all_y = [p[1] for p in local_positions.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span_x = max_x - min_x if max_x > min_x else 1.0
    span_y = max_y - min_y if max_y > min_y else 1.0

    # Add margin (5% on each side)
    margin = 0.05
    scale_x = target_width * (1 - 2 * margin) / span_x
    scale_y = target_height * (1 - 2 * margin) / span_y

    for node in G_undirected.nodes():
        raw_x, raw_y = local_positions[node]
        nx_val = (raw_x - min_x) * scale_x + target_width * margin
        ny_val = (raw_y - min_y) * scale_y + target_height * margin
        G_undirected.nodes[node]["pos"] = (nx_val, ny_val)

    # Compute edge weights from projected Euclidean distance
    for u, v in G_undirected.edges():
        p1 = G_undirected.nodes[u]["pos"]
        p2 = G_undirected.nodes[v]["pos"]
        dist = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
        G_undirected[u][v]["weight"] = max(dist, 0.1)
        G_undirected[u][v]["risk"] = 0.0  # baseline — overwritten by World._annotate_edge_risk

        # Add slope data if elevation available
        if elevation_grid is not None:
            slope = elevation_grid.slope_between(p1[0], p1[1], p2[0], p2[1])
            G_undirected[u][v]["slope"] = slope

    # Relabel to sequential integers (existing code assumes int node IDs)
    G_out = nx.convert_node_labels_to_integers(G_undirected, first_label=0)

    return G_out
