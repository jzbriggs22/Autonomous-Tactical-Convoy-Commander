"""GeoTIFF DEM loading and resampling to ElevationGrid."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from convoy_commander.geospatial.elevation import ElevationGrid


def load_elevation_grid(
    tiff_path: str | Path,
    target_width: float = 1000.0,
    target_height: float = 1000.0,
    bounds: tuple[float, float, float, float] | None = None,
    max_slope_threshold: float = 0.3,
) -> ElevationGrid:
    """Load a GeoTIFF DEM and resample to a local metric grid.

    Args:
        tiff_path: Path to a GeoTIFF elevation file (SRTM, ASTER, etc.).
        target_width: Desired world width in metres for resampling.
        target_height: Desired world height in metres for resampling.
        bounds: Optional (west, south, east, north) crop window in the
            raster's native CRS.  If *None* the full raster extent is used.
        max_slope_threshold: Slope (rise/run) that maps to cost 1.0.

    Returns:
        An ``ElevationGrid`` whose *data* array covers
        ``target_width × target_height`` metres.
    """
    from convoy_commander.geospatial.compat import require_rasterio

    rasterio = require_rasterio()
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds as window_from_bounds

    tiff_path = Path(tiff_path)
    if not tiff_path.exists():
        raise FileNotFoundError(f"DEM file not found: {tiff_path}")

    with rasterio.open(tiff_path) as src:
        if bounds is not None:
            window = window_from_bounds(*bounds, transform=src.transform)
            data = src.read(1, window=window)
        else:
            data = src.read(1)

        # Determine native resolution in metres (approximate for geographic CRS)
        pixel_w = abs(src.transform.a)
        pixel_h = abs(src.transform.e)
        native_rows, native_cols = data.shape

        # If CRS is geographic (degrees), convert pixel size to metres roughly
        if src.crs and src.crs.is_geographic:
            center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
            import math
            m_per_deg = 111_320.0 * math.cos(math.radians(center_lat))
            pixel_w *= m_per_deg
            pixel_h *= 111_320.0

        native_width = native_cols * pixel_w
        native_height = native_rows * pixel_h

    # Resample to target grid dimensions
    target_cols = max(2, int(target_width / max(pixel_w, 1.0)))
    target_rows = max(2, int(target_height / max(pixel_h, 1.0)))

    # Cap at reasonable resolution to avoid memory issues
    target_cols = min(target_cols, 2000)
    target_rows = min(target_rows, 2000)

    from scipy.ndimage import zoom  # type: ignore[import-untyped]

    zoom_y = target_rows / data.shape[0]
    zoom_x = target_cols / data.shape[1]
    resampled = zoom(data.astype(np.float64), (zoom_y, zoom_x), order=1)

    res_x = target_width / resampled.shape[1]
    res_y = target_height / resampled.shape[0]

    return ElevationGrid(
        data=resampled.astype(np.float64),
        resolution_x=res_x,
        resolution_y=res_y,
        origin_x=0.0,
        origin_y=0.0,
        max_slope_threshold=max_slope_threshold,
    )
