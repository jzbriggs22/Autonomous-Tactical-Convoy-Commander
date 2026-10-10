"""ElevationGrid: queryable elevation surface aligned to World coordinates.

Pure numpy — no external dependencies required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ElevationGrid:
    """2-D elevation grid in local metric coordinates.

    The grid is aligned to the World coordinate frame: cell (0, 0) corresponds
    to world position (origin_x, origin_y) and each cell spans
    ``resolution_x`` × ``resolution_y`` metres.
    """

    data: np.ndarray  # shape (rows, cols), elevation in metres
    resolution_x: float  # metres per column
    resolution_y: float  # metres per row
    origin_x: float = 0.0
    origin_y: float = 0.0
    max_slope_threshold: float = field(default=0.3)  # slope that maps to cost 1.0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def elevation_at(self, x: float, y: float) -> float:
        """Bilinear-interpolated elevation at world (x, y).

        Returns 0.0 if the query point is outside the grid.
        """
        col_f = (x - self.origin_x) / self.resolution_x
        row_f = (y - self.origin_y) / self.resolution_y

        rows, cols = self.data.shape
        if col_f < 0 or col_f >= cols - 1 or row_f < 0 or row_f >= rows - 1:
            # Clamp to edge or return 0 if completely outside
            if 0 <= col_f < cols and 0 <= row_f < rows:
                return float(self.data[min(int(row_f), rows - 1), min(int(col_f), cols - 1)])
            return 0.0

        c0, r0 = int(col_f), int(row_f)
        dc, dr = col_f - c0, row_f - r0

        z00 = self.data[r0, c0]
        z10 = self.data[r0, c0 + 1]
        z01 = self.data[r0 + 1, c0]
        z11 = self.data[r0 + 1, c0 + 1]

        return float(
            z00 * (1 - dc) * (1 - dr)
            + z10 * dc * (1 - dr)
            + z01 * (1 - dc) * dr
            + z11 * dc * dr
        )

    def slope_between(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Absolute slope (rise / run) between two world points.

        Returns 0.0 if horizontal distance is negligible.
        """
        run = math.hypot(x2 - x1, y2 - y1)
        if run < 1e-6:
            return 0.0
        dz = self.elevation_at(x2, y2) - self.elevation_at(x1, y1)
        return abs(dz) / run

    def slope_cost(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Normalised slope cost in [0, 1] with uphill/downhill asymmetry.

        Uphill (positive elevation change): full cost.
        Downhill (negative elevation change): 30% cost (gravity assist).
        """
        run = math.hypot(x2 - x1, y2 - y1)
        if run < 1e-6:
            return 0.0
        dz = self.elevation_at(x2, y2) - self.elevation_at(x1, y1)
        slope = dz / run

        if slope >= 0:
            return min(slope / self.max_slope_threshold, 1.0)
        else:
            return min(abs(slope) / self.max_slope_threshold, 1.0) * 0.3
