"""Per-convoy state container (Phase 12).

ConvoyGroup bundles all mutable state for one convoy: vehicles, leader
election instances, CBBA slot assignments, destination, and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from convoy_commander.coordination.leader_election import LeaderElection
from convoy_commander.metrics.collector import MetricsCollector
from convoy_commander.vehicles.vehicle import Vehicle


@dataclass
class ConvoyGroup:
    """Mutable state for one convoy."""

    convoy_id: int
    priority: int
    vehicles: list[Vehicle]
    destination: tuple[float, float]
    elections: dict[int, LeaderElection] = field(default_factory=dict)
    cbba_slots: dict[int, int] = field(default_factory=dict)
    cbba_realloc_timer: float = 0.0
    collector: MetricsCollector = field(default_factory=MetricsCollector)
    broadcast_timer: float = 0.0
    # Right-of-way state
    yielding: bool = False
    yield_until: float = 0.0

    @property
    def operational_vehicles(self) -> list[Vehicle]:
        return [v for v in self.vehicles if v.is_operational]

    def get_leader(self) -> Vehicle | None:
        for v in self.vehicles:
            if v.is_leader and v.is_operational:
                return v
        return None

    def vehicle_ids(self) -> set[int]:
        return {v.id for v in self.vehicles}

    def remove_vehicle(self, vehicle_id: int) -> Vehicle | None:
        """Remove a vehicle from this convoy, return it."""
        for i, v in enumerate(self.vehicles):
            if v.id == vehicle_id:
                return self.vehicles.pop(i)
        return None

    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add a vehicle to this convoy."""
        vehicle.convoy_id = self.convoy_id
        self.vehicles.append(vehicle)
