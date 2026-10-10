"""Coordinate projection: WGS84 lat/lon to local metric and back."""

from __future__ import annotations

import math


class Projector:
    """Convert between WGS84 lat/lon and a local metric coordinate system.

    Uses a simple equirectangular approximation centred on (center_lat,
    center_lon).  Accuracy is within ~0.1% for areas up to ~50 km across,
    which is sufficient for tactical convoy scenarios.

    For higher accuracy over larger areas, replace with a UTM projection
    via pyproj (requires the ``[geo]`` extra).
    """

    def __init__(self, center_lat: float, center_lon: float) -> None:
        self.center_lat = center_lat
        self.center_lon = center_lon
        self._m_per_deg_lat = 111_320.0
        self._m_per_deg_lon = 111_320.0 * math.cos(math.radians(center_lat))

    def to_local(self, lat: float, lon: float) -> tuple[float, float]:
        """Convert (lat, lon) to local (x, y) in metres."""
        x = (lon - self.center_lon) * self._m_per_deg_lon
        y = (lat - self.center_lat) * self._m_per_deg_lat
        return (x, y)

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        """Convert local (x, y) in metres to (lat, lon)."""
        lat = y / self._m_per_deg_lat + self.center_lat
        lon = x / self._m_per_deg_lon + self.center_lon
        return (lat, lon)


def create_projector(center_lat: float, center_lon: float) -> Projector:
    """Create a lat/lon ↔ local metric converter centred on the given point."""
    return Projector(center_lat, center_lon)
