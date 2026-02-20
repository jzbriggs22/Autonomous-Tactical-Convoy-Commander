"""Tests for communications model."""

import numpy as np

from convoy_commander.comms.messages import MessageType, make_state_broadcast
from convoy_commander.comms.network import CommsNetwork
from convoy_commander.core.config import CommsConfig


def test_comms_in_range_delivery():
    """Messages should be delivered between vehicles in range."""
    config = CommsConfig(max_range=200.0, packet_loss=0.0, latency_mean_ms=10.0, latency_std_ms=1.0)
    rng = np.random.default_rng(42)
    network = CommsNetwork(config, rng)

    msg = make_state_broadcast(
        sender_id=0, timestamp=0.0, x=10, y=10, heading=0, speed=5,
        uncertainty=1.0, status="ACTIVE", fuel=90.0,
    )
    positions = {0: (10.0, 10.0), 1: (50.0, 50.0)}
    network.send_broadcast(msg, (10.0, 10.0), positions, 0.0)

    # Tick forward past latency
    network.tick(1.0)
    inbox = network.get_inbox(1)
    assert len(inbox) == 1
    assert inbox[0].msg_type == MessageType.STATE_BROADCAST


def test_comms_out_of_range():
    """Messages should not reach vehicles out of range."""
    config = CommsConfig(max_range=50.0, packet_loss=0.0, latency_mean_ms=10.0, latency_std_ms=1.0)
    rng = np.random.default_rng(42)
    network = CommsNetwork(config, rng)

    msg = make_state_broadcast(
        sender_id=0, timestamp=0.0, x=10, y=10, heading=0, speed=5,
        uncertainty=1.0, status="ACTIVE", fuel=90.0,
    )
    positions = {0: (10.0, 10.0), 1: (500.0, 500.0)}
    network.send_broadcast(msg, (10.0, 10.0), positions, 0.0)
    network.tick(1.0)
    inbox = network.get_inbox(1)
    assert len(inbox) == 0


def test_comms_packet_loss():
    """With 100% loss, no messages should be delivered."""
    config = CommsConfig(max_range=1000.0, packet_loss=1.0, latency_mean_ms=10.0, latency_std_ms=1.0)
    rng = np.random.default_rng(42)
    network = CommsNetwork(config, rng)

    positions = {0: (10.0, 10.0), 1: (50.0, 50.0)}
    for _ in range(100):
        msg = make_state_broadcast(
            sender_id=0, timestamp=0.0, x=10, y=10, heading=0, speed=5,
            uncertainty=1.0, status="ACTIVE", fuel=90.0,
        )
        network.send_broadcast(msg, (10.0, 10.0), positions, 0.0)

    network.tick(10.0)
    inbox = network.get_inbox(1)
    assert len(inbox) == 0
    assert network.total_dropped > 0


def test_comms_latency():
    """Messages should not arrive before their delivery time."""
    config = CommsConfig(max_range=200.0, packet_loss=0.0, latency_mean_ms=500.0, latency_std_ms=1.0)
    rng = np.random.default_rng(42)
    network = CommsNetwork(config, rng)

    msg = make_state_broadcast(
        sender_id=0, timestamp=0.0, x=10, y=10, heading=0, speed=5,
        uncertainty=1.0, status="ACTIVE", fuel=90.0,
    )
    positions = {0: (10.0, 10.0), 1: (50.0, 50.0)}
    network.send_broadcast(msg, (10.0, 10.0), positions, 0.0)

    # Tick at 0.1s - should not be delivered yet (latency ~500ms)
    network.tick(0.1)
    inbox = network.get_inbox(1)
    assert len(inbox) == 0

    # Tick at 1.0s - should be delivered now
    network.tick(1.0)
    inbox = network.get_inbox(1)
    assert len(inbox) == 1
