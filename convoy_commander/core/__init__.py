"""Core world model, physics, time management, and event logging."""

from convoy_commander.core.config import SimConfig
from convoy_commander.core.event_log import EventKind, EventLog, Severity
from convoy_commander.core.world import World
from convoy_commander.core.physics import KinematicState, clamp

__all__ = ["SimConfig", "World", "KinematicState", "clamp", "EventKind", "EventLog", "Severity"]
