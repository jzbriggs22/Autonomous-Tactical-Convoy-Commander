"""Communications network model with range, latency, and packet loss."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from convoy_commander.comms.messages import Message
from convoy_commander.core.config import CommsConfig
from convoy_commander.core.spatial import SpatialHash


@dataclass
class CommsBlackoutRegion:
    """Region where comms are degraded or blocked."""

    x: float
    y: float
    radius: float
    loss_multiplier: float = 5.0  # multiplied with base loss


class CommsNetwork:
    """Simulates range-based ad-hoc network with loss and latency."""

    def __init__(self, config: CommsConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.blackout_regions: list[CommsBlackoutRegion] = []

        # In-flight messages: list of (message, delivery_time, recipient_id)
        self._in_flight: list[tuple[Message, float, int]] = []

        # Delivered message inbox per vehicle
        self.inboxes: dict[int, list[Message]] = defaultdict(list)

        # Aggregate stats
        self.total_sent: int = 0
        self.total_delivered: int = 0
        self.total_dropped: int = 0
        self.total_bytes_approx: int = 0

        # Per-message-type stats (keyed by MessageType.name)
        self.sent_by_type: dict[str, int] = {}
        self.delivered_by_type: dict[str, int] = {}
        self.dropped_by_type: dict[str, int] = {}

    def send_broadcast(
        self,
        msg: Message,
        sender_pos: tuple[float, float],
        all_vehicles: dict[int, tuple[float, float]],
        current_time: float,
        spatial_hash: SpatialHash | None = None,
    ) -> None:
        """Broadcast message to all vehicles in range.

        If ``spatial_hash`` is provided, only nearby vehicles are checked
        (O(k) instead of O(N)).
        """
        if spatial_hash is not None:
            nearby = spatial_hash.query_radius(
                sender_pos[0], sender_pos[1], self.config.max_range, all_vehicles,
            )
            for vid in nearby:
                if vid == msg.sender_id:
                    continue
                self._try_send(msg, sender_pos, vid, all_vehicles[vid], current_time)
        else:
            for vid, vpos in all_vehicles.items():
                if vid == msg.sender_id:
                    continue
                self._try_send(msg, sender_pos, vid, vpos, current_time)

    def send_to(
        self,
        msg: Message,
        sender_pos: tuple[float, float],
        recipient_id: int,
        recipient_pos: tuple[float, float],
        current_time: float,
    ) -> None:
        """Send message to specific vehicle."""
        self._try_send(msg, sender_pos, recipient_id, recipient_pos, current_time)

    def _try_send(
        self,
        msg: Message,
        sender_pos: tuple[float, float],
        recipient_id: int,
        recipient_pos: tuple[float, float],
        current_time: float,
    ) -> None:
        """Attempt to send, applying range check, loss, and latency."""
        self.total_sent += 1
        self.total_bytes_approx += 64  # approximate message size
        type_name = msg.msg_type.name
        self.sent_by_type[type_name] = self.sent_by_type.get(type_name, 0) + 1

        dist = math.hypot(sender_pos[0] - recipient_pos[0], sender_pos[1] - recipient_pos[1])

        # Range check
        if dist > self.config.max_range:
            self.total_dropped += 1
            self.dropped_by_type[type_name] = self.dropped_by_type.get(type_name, 0) + 1
            return

        # Distance-based loss increase
        range_factor = (dist / self.config.max_range) ** 2
        loss_prob = self.config.packet_loss + range_factor * 0.3

        # Blackout region multiplier
        for region in self.blackout_regions:
            d_sender = math.hypot(sender_pos[0] - region.x, sender_pos[1] - region.y)
            d_recip = math.hypot(recipient_pos[0] - region.x, recipient_pos[1] - region.y)
            if d_sender < region.radius or d_recip < region.radius:
                loss_prob = min(1.0, loss_prob * region.loss_multiplier)

        # Apply packet loss
        if self.rng.random() < loss_prob:
            self.total_dropped += 1
            self.dropped_by_type[type_name] = self.dropped_by_type.get(type_name, 0) + 1
            return

        # Compute latency
        latency_ms = max(1.0, self.rng.normal(self.config.latency_mean_ms, self.config.latency_std_ms))
        delivery_time = current_time + latency_ms / 1000.0

        delivered_msg = Message(
            msg_type=msg.msg_type,
            sender_id=msg.sender_id,
            timestamp=msg.timestamp,
            payload=msg.payload.copy(),
            delivered_at=delivery_time,
        )

        self._in_flight.append((delivered_msg, delivery_time, recipient_id))

    def tick(self, current_time: float) -> None:
        """Deliver messages whose delivery time has passed."""
        still_in_flight: list[tuple[Message, float, int]] = []
        for msg, delivery_time, recipient_id in self._in_flight:
            if current_time >= delivery_time:
                self.inboxes[recipient_id].append(msg)
                self.total_delivered += 1
                type_name = msg.msg_type.name
                self.delivered_by_type[type_name] = self.delivered_by_type.get(type_name, 0) + 1
            else:
                still_in_flight.append((msg, delivery_time, recipient_id))
        self._in_flight = still_in_flight

    def get_inbox(self, vehicle_id: int) -> list[Message]:
        """Get and clear inbox for a vehicle."""
        msgs = self.inboxes[vehicle_id]
        self.inboxes[vehicle_id] = []
        return msgs

    def get_adjacency(
        self,
        positions: dict[int, tuple[float, float]],
        spatial_hash: SpatialHash | None = None,
    ) -> dict[int, list[int]]:
        """Get comms adjacency graph based on distance.

        If ``spatial_hash`` is provided, queries use the grid (O(N*k)
        instead of O(N^2)).
        """
        adj: dict[int, list[int]] = {vid: [] for vid in positions}
        if spatial_hash is not None:
            for vid, (vx, vy) in positions.items():
                nearby = spatial_hash.query_radius(vx, vy, self.config.max_range, positions)
                for nid in nearby:
                    if nid != vid:
                        adj[vid].append(nid)
        else:
            vids = list(positions.keys())
            for i in range(len(vids)):
                for j in range(i + 1, len(vids)):
                    a, b = vids[i], vids[j]
                    dist = math.hypot(
                        positions[a][0] - positions[b][0],
                        positions[a][1] - positions[b][1],
                    )
                    if dist <= self.config.max_range:
                        adj[a].append(b)
                        adj[b].append(a)
        return adj

    def get_stats_by_type(self) -> dict[str, dict[str, int]]:
        """Return per-message-type send/deliver/drop counts.

        Returns a dict keyed by MessageType.name with nested keys
        ``sent``, ``delivered``, ``dropped``.
        """
        all_types = set(self.sent_by_type) | set(self.delivered_by_type) | set(self.dropped_by_type)
        return {
            t: {
                "sent": self.sent_by_type.get(t, 0),
                "delivered": self.delivered_by_type.get(t, 0),
                "dropped": self.dropped_by_type.get(t, 0),
            }
            for t in sorted(all_types)
        }

    def add_blackout_region(self, x: float, y: float, radius: float, loss_mult: float = 5.0) -> None:
        self.blackout_regions.append(CommsBlackoutRegion(x, y, radius, loss_mult))
