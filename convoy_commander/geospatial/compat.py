"""Optional dependency guards for geospatial packages."""

from __future__ import annotations

from typing import Any


def require_rasterio() -> Any:
    """Return the rasterio module or raise with install instructions."""
    try:
        import rasterio
        return rasterio
    except ImportError:
        raise ImportError(
            "rasterio is required for GeoTIFF terrain loading. "
            "Install with: pip install 'convoy_commander[geo]'"
        ) from None


def require_osmnx() -> Any:
    """Return the osmnx module or raise with install instructions."""
    try:
        import osmnx
        return osmnx
    except ImportError:
        raise ImportError(
            "osmnx is required for OpenStreetMap road loading. "
            "Install with: pip install 'convoy_commander[geo]'"
        ) from None


def require_pyproj() -> Any:
    """Return the pyproj module or raise with install instructions."""
    try:
        import pyproj
        return pyproj
    except ImportError:
        raise ImportError(
            "pyproj is required for coordinate projection. "
            "Install with: pip install 'convoy_commander[geo]'"
        ) from None
