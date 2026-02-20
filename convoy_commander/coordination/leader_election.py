"""Leader election using a simplified Bully algorithm over unreliable comms."""

from __future__ import annotations

from convoy_commander.comms.messages import (
    Message,
    MessageType,
    make_leader_election,
    make_leader_heartbeat,
)
from convoy_commander.vehicles.vehicle import Vehicle


class LeaderElection:
    """Bully-style leader election.

    Each vehicle has a priority score (based on ID, fuel, status).
    The highest-priority operational vehicle becomes leader.
    Leader sends heartbeats; if heartbeat times out, election restarts.
    """

    def __init__(self, vehicle: Vehicle, heartbeat_timeout: float) -> None:
        self.vehicle = vehicle
        self.heartbeat_timeout = heartbeat_timeout
        self.last_heartbeat_time: float = 0.0
        self.election_in_progress: bool = False
        self.highest_seen_candidate: int = -1
        self.highest_seen_priority: float = -1.0

    def compute_priority(self) -> float:
        """Priority: higher is better. Based on fuel, ID (lower ID = higher tie-break)."""
        if not self.vehicle.is_operational:
            return -1.0
        return self.vehicle.fuel.fraction * 100.0 + (1000 - self.vehicle.id) * 0.01

    def on_heartbeat(self, msg: Message, current_time: float) -> None:
        """Process leader heartbeat."""
        self.last_heartbeat_time = current_time
        leader_id = msg.sender_id
        self.vehicle.leader_id = leader_id
        self.election_in_progress = False

    def on_election_message(self, msg: Message, current_time: float) -> None:
        """Process election message."""
        candidate_id = int(msg.payload["candidate_id"])  # type: ignore[arg-type]
        priority = float(msg.payload["priority"])  # type: ignore[arg-type]
        if priority > self.highest_seen_priority or (
            priority == self.highest_seen_priority and candidate_id < self.highest_seen_candidate
        ):
            self.highest_seen_priority = priority
            self.highest_seen_candidate = candidate_id

    def check_and_elect(self, current_time: float) -> list[Message]:
        """Check if election needed, produce messages."""
        msgs: list[Message] = []

        if not self.vehicle.is_operational:
            self.vehicle.is_leader = False
            return msgs

        # Leader sends heartbeats
        if self.vehicle.is_leader:
            msgs.append(make_leader_heartbeat(self.vehicle.id, current_time))
            return msgs

        # Check heartbeat timeout
        if current_time - self.last_heartbeat_time > self.heartbeat_timeout:
            self.election_in_progress = True

        if self.election_in_progress:
            my_priority = self.compute_priority()
            msgs.append(
                make_leader_election(self.vehicle.id, current_time, self.vehicle.id, my_priority)
            )
            # If I'm the highest candidate I've seen, declare victory
            if my_priority > self.highest_seen_priority or (
                my_priority == self.highest_seen_priority
                and self.vehicle.id <= self.highest_seen_candidate
            ):
                self.vehicle.is_leader = True
                self.vehicle.leader_id = self.vehicle.id
                self.election_in_progress = False
                self.highest_seen_candidate = self.vehicle.id
                self.highest_seen_priority = my_priority
                msgs.append(make_leader_heartbeat(self.vehicle.id, current_time))

        return msgs

    def reset(self) -> None:
        """Reset election state (e.g., when leader is known to have failed)."""
        self.election_in_progress = True
        self.highest_seen_candidate = -1
        self.highest_seen_priority = -1.0
        self.vehicle.is_leader = False
        self.vehicle.leader_id = None
        self.last_heartbeat_time = 0.0
