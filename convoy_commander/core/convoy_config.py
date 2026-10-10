"""Multi-convoy configuration models (Phase 12).

Defines ConvoySpec (per-convoy parameters) and MultiConvoyConfig
(top-level config for multi-convoy simulations).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from convoy_commander.core.config import SimConfig


class ConvoySpec(BaseModel):
    """Specification for a single convoy within a multi-convoy scenario."""

    convoy_id: int
    num_vehicles: int = Field(default=4, ge=1, le=32)
    start_x: float = 50.0
    start_y: float = 50.0
    dest_x: float = 920.0
    dest_y: float = 920.0
    priority: int = Field(default=0, ge=0, description="Right-of-way priority (higher wins)")
    start_heading: float = 0.3


class MultiConvoyConfig(BaseModel):
    """Top-level config for multi-convoy simulation."""

    base: SimConfig = Field(default_factory=SimConfig)
    convoys: list[ConvoySpec] = Field(min_length=1)
    right_of_way_radius: float = Field(
        default=60.0, gt=0,
        description="Radius for detecting route-crossing proximity",
    )
    merge_distance: float = Field(
        default=40.0, gt=0,
        description="Max distance between convoy leaders to trigger merge",
    )
    split_min_vehicles: int = Field(
        default=2, ge=2,
        description="Minimum vehicles per convoy after a split",
    )

    @property
    def total_vehicles(self) -> int:
        return sum(c.num_vehicles for c in self.convoys)
