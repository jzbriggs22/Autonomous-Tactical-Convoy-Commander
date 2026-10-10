"""Spatial hashing for efficient neighbor queries.

Replaces O(N^2) pairwise distance checks with O(N) insert + O(k) query,
where k is the number of entities in nearby cells.  Rebuild per step is O(N).
"""

from __future__ import annotations

import math
from collections import defaultdict


class SpatialHash:
    """Grid-based spatial hash for 2-D point entities."""

    __slots__ = ("cell_size", "_inv_cell", "_grid")

    def __init__(self, cell_size: float) -> None:
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size}")
        self.cell_size = cell_size
        self._inv_cell = 1.0 / cell_size
        self._grid: dict[tuple[int, int], list[int]] = defaultdict(list)

    def clear(self) -> None:
        """Remove all entities."""
        self._grid.clear()

    def insert(self, entity_id: int, x: float, y: float) -> None:
        """Insert an entity at position (x, y)."""
        cx = int(math.floor(x * self._inv_cell))
        cy = int(math.floor(y * self._inv_cell))
        self._grid[(cx, cy)].append(entity_id)

    def query_radius(
        self,
        x: float,
        y: float,
        radius: float,
        positions: dict[int, tuple[float, float]],
    ) -> list[int]:
        """Return entity IDs within *radius* of (x, y).

        ``positions`` maps entity_id -> (px, py) for exact distance filtering.
        """
        r2 = radius * radius
        cx = x * self._inv_cell
        cy = y * self._inv_cell
        # Number of cells the radius spans
        span = int(math.ceil(radius * self._inv_cell)) + 1

        cx_i = int(math.floor(cx))
        cy_i = int(math.floor(cy))

        result: list[int] = []
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                cell = self._grid.get((cx_i + dx, cy_i + dy))
                if cell is None:
                    continue
                for eid in cell:
                    px, py = positions[eid]
                    ddx = px - x
                    ddy = py - y
                    if ddx * ddx + ddy * ddy <= r2:
                        result.append(eid)
        return result
