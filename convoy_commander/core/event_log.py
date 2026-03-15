"""Structured event log for safety-critical audit trail.

Every safety-relevant decision, state transition, and anomaly is recorded
as an immutable event with a monotonic timestamp. Events are written to
a JSON-lines file for post-hoc analysis and are summarised in the
markdown report.

Design assumptions (SAFETY-CRITICAL):
  A1. The event log is append-only; no event is ever deleted or mutated.
  A2. Events are recorded synchronously on the simulation thread, so
      ordering is guaranteed by sim-time monotonicity.
  A3. This is a *simulator* log, not a vehicle-local log. In a real
      system each vehicle would maintain its own tamper-evident log.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path


class Severity(Enum):
    """Event severity for triage and filtering."""

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


class EventKind(Enum):
    """Taxonomy of loggable events."""

    # --- lifecycle ---
    SIM_START = "sim_start"
    SIM_END = "sim_end"
    CONFIG_VALIDATED = "config_validated"
    CONFIG_WARNING = "config_warning"

    # --- vehicle state ---
    VEHICLE_SPAWNED = "vehicle_spawned"
    VEHICLE_ARRIVED = "vehicle_arrived"
    VEHICLE_BREAKDOWN = "vehicle_breakdown"
    VEHICLE_FUEL_LOW = "vehicle_fuel_low"
    VEHICLE_FUEL_EMPTY = "vehicle_fuel_empty"

    # --- safety mode ---
    SAFE_MODE_ENTER = "safe_mode_enter"
    SAFE_MODE_EXIT = "safe_mode_exit"
    EMERGENCY_STOP = "emergency_stop"

    # --- estimation ---
    ESTIMATOR_FIX_APPLIED = "estimator_fix_applied"
    ESTIMATOR_FIX_REJECTED = "estimator_fix_rejected"
    ESTIMATOR_UNCERTAINTY_HIGH = "estimator_uncertainty_high"
    ESTIMATOR_UNCERTAINTY_RECOVERED = "estimator_uncertainty_recovered"
    GPS_SPOOFED = "gps_spoofed"
    DRIFT_SPIKE = "drift_spike"

    # --- coordination ---
    LEADER_ELECTED = "leader_elected"
    LEADER_LOST = "leader_lost"
    ELECTION_STARTED = "election_started"
    FORMATION_DEGRADED = "formation_degraded"

    # --- collision / separation ---
    COLLISION = "collision"
    NEAR_MISS = "near_miss"
    SEPARATION_VIOLATION = "separation_violation"

    # --- comms ---
    COMMS_LOST = "comms_lost"
    COMMS_RESTORED = "comms_restored"
    COMMS_MODE_CHANGE = "comms_mode_change"
    BLACKOUT_ENTERED = "blackout_entered"
    BLACKOUT_EXITED = "blackout_exited"

    # --- planning ---
    ROUTE_PLANNED = "route_planned"
    ROUTE_REPLAN = "route_replan"
    OBSTACLE_DETECTED = "obstacle_detected"
    PLANNER_FALLBACK = "planner_fallback"

    # --- supervisor ---
    SUPERVISOR_ACTION = "supervisor_action"

    # --- invariant ---
    INVARIANT_VIOLATION = "invariant_violation"
    BOUNDARY_VIOLATION = "boundary_violation"
    SPEED_LIMIT_EXCEEDED = "speed_limit_exceeded"

    # --- scenario ---
    SCENARIO_EVENT = "scenario_event"

    # --- weather ---
    WEATHER_UPDATED = "weather_updated"
    WEATHER_FRICTION_LOW = "weather_friction_low"
    WEATHER_VISIBILITY_LOW = "weather_visibility_low"
    WEATHER_SAFE_MODE = "weather_safe_mode"
    WEATHER_API_FALLBACK = "weather_api_fallback"


@dataclass(frozen=True)
class Event:
    """A single audit-trail event. Immutable once created."""

    time: float
    kind: str  # EventKind.value
    severity: str  # Severity.name
    vehicle_id: int | None = None
    message: str = ""
    details: dict[str, object] = field(default_factory=dict)


class EventLog:
    """Append-only structured event log."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._last_time: float = -1.0

    def log(
        self,
        time: float,
        kind: EventKind,
        severity: Severity,
        vehicle_id: int | None = None,
        message: str = "",
        **details: object,
    ) -> None:
        """Record an event. Time must be monotonically non-decreasing."""
        if time < self._last_time:
            # Monotonicity violation — record it but do not crash the sim
            self._events.append(Event(
                time=time,
                kind=EventKind.INVARIANT_VIOLATION.value,
                severity=Severity.CRITICAL.name,
                message=f"Event log monotonicity violated: {time} < {self._last_time}",
            ))
        self._last_time = max(self._last_time, time)

        self._events.append(Event(
            time=time,
            kind=kind.value,
            severity=severity.name,
            vehicle_id=vehicle_id,
            message=message,
            details=details,
        ))

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def filter(
        self,
        kind: EventKind | None = None,
        severity_min: Severity | None = None,
        vehicle_id: int | None = None,
    ) -> list[Event]:
        """Return filtered view of events."""
        severity_order = list(Severity)
        min_idx = severity_order.index(severity_min) if severity_min else 0
        result: list[Event] = []
        for e in self._events:
            if kind and e.kind != kind.value:
                continue
            if vehicle_id is not None and e.vehicle_id != vehicle_id:
                continue
            try:
                e_sev = Severity[e.severity]
            except KeyError:
                continue
            if severity_order.index(e_sev) < min_idx:
                continue
            result.append(e)
        return result

    def count_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._events:
            counts[e.kind] = counts.get(e.kind, 0) + 1
        return counts

    def count_by_severity(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._events:
            counts[e.severity] = counts.get(e.severity, 0) + 1
        return counts

    def save(self, path: Path) -> None:
        """Persist event log as JSON-lines."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for event in self._events:
                f.write(json.dumps(asdict(event), default=str) + "\n")

    def __len__(self) -> int:
        return len(self._events)
