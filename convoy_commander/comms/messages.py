"""Message types for inter-vehicle communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class MessageType(Enum):
    STATE_BROADCAST = auto()  # periodic position/status broadcast
    INTENT = auto()  # planned waypoint / heading
    HAZARD = auto()  # obstacle / danger alert
    LEADER_ELECTION = auto()  # leader election message
    LEADER_HEARTBEAT = auto()  # heartbeat from leader
    CONSENSUS = auto()  # formation consensus
    WAYPOINT_BID = auto()  # auction bid for waypoint allocation


@dataclass
class Message:
    """A message sent between vehicles."""

    msg_type: MessageType
    sender_id: int
    timestamp: float  # sim time when sent
    payload: dict[str, object] = field(default_factory=dict)
    # Set by comms model on delivery
    delivered_at: float = 0.0


def make_state_broadcast(
    sender_id: int,
    timestamp: float,
    x: float,
    y: float,
    heading: float,
    speed: float,
    uncertainty: float,
    status: str,
    fuel: float,
) -> Message:
    return Message(
        msg_type=MessageType.STATE_BROADCAST,
        sender_id=sender_id,
        timestamp=timestamp,
        payload={
            "x": x,
            "y": y,
            "heading": heading,
            "speed": speed,
            "uncertainty": uncertainty,
            "status": status,
            "fuel": fuel,
        },
    )


def make_intent(
    sender_id: int, timestamp: float, target_x: float, target_y: float
) -> Message:
    return Message(
        msg_type=MessageType.INTENT,
        sender_id=sender_id,
        timestamp=timestamp,
        payload={"target_x": target_x, "target_y": target_y},
    )


def make_hazard(
    sender_id: int, timestamp: float, hx: float, hy: float, radius: float
) -> Message:
    return Message(
        msg_type=MessageType.HAZARD,
        sender_id=sender_id,
        timestamp=timestamp,
        payload={"x": hx, "y": hy, "radius": radius},
    )


def make_leader_election(
    sender_id: int, timestamp: float, candidate_id: int, priority: float
) -> Message:
    return Message(
        msg_type=MessageType.LEADER_ELECTION,
        sender_id=sender_id,
        timestamp=timestamp,
        payload={"candidate_id": candidate_id, "priority": priority},
    )


def make_leader_heartbeat(sender_id: int, timestamp: float) -> Message:
    return Message(
        msg_type=MessageType.LEADER_HEARTBEAT,
        sender_id=sender_id,
        timestamp=timestamp,
    )


def make_waypoint_bid(
    sender_id: int, timestamp: float, waypoint_idx: int, bid_value: float
) -> Message:
    return Message(
        msg_type=MessageType.WAYPOINT_BID,
        sender_id=sender_id,
        timestamp=timestamp,
        payload={"waypoint_idx": waypoint_idx, "bid_value": bid_value},
    )
