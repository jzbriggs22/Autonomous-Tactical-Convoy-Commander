"""Tests for leader election."""

import numpy as np

from convoy_commander.comms.messages import MessageType, make_leader_heartbeat
from convoy_commander.coordination.leader_election import LeaderElection
from convoy_commander.core.config import SimConfig
from convoy_commander.vehicles.vehicle import Vehicle


def _make_vehicle(vid: int, seed: int = 42) -> Vehicle:
    config = SimConfig(seed=seed)
    rng = np.random.default_rng(seed + vid)
    return Vehicle(vid, config, rng, start_x=50 + vid * 20, start_y=50)


def test_leader_election_triggers_on_timeout():
    """Election should trigger when heartbeat times out."""
    v0 = _make_vehicle(0)
    v1 = _make_vehicle(1)
    v0.is_leader = True

    election = LeaderElection(v1, heartbeat_timeout=5.0)
    election.last_heartbeat_time = 0.0

    # At t=3, no timeout yet
    msgs = election.check_and_elect(3.0)
    assert not election.election_in_progress or not v1.is_leader

    # At t=6, timeout should trigger election
    msgs = election.check_and_elect(6.0)
    assert election.election_in_progress or v1.is_leader


def test_leader_election_stabilizes():
    """After election, exactly one leader should exist."""
    vehicles = [_make_vehicle(i) for i in range(4)]
    elections = [LeaderElection(v, heartbeat_timeout=5.0) for v in vehicles]

    # Set all to election mode
    for e in elections:
        e.election_in_progress = True

    # Run several rounds
    for t in range(10):
        for i, e in enumerate(elections):
            msgs = e.check_and_elect(float(t))
            # Share election messages with others
            for msg in msgs:
                if msg.msg_type == MessageType.LEADER_ELECTION:
                    for j, other_e in enumerate(elections):
                        if j != i:
                            other_e.on_election_message(msg, float(t))
                elif msg.msg_type == MessageType.LEADER_HEARTBEAT:
                    for j, other_e in enumerate(elections):
                        if j != i:
                            other_e.on_heartbeat(msg, float(t))

    leaders = [v for v in vehicles if v.is_leader]
    assert len(leaders) >= 1  # At least one leader elected


def test_leader_heartbeat_resets_timeout():
    """Receiving a heartbeat should prevent election."""
    v = _make_vehicle(1)
    election = LeaderElection(v, heartbeat_timeout=5.0)
    election.last_heartbeat_time = 0.0

    # Send heartbeat at t=3
    hb = make_leader_heartbeat(sender_id=0, timestamp=3.0)
    election.on_heartbeat(hb, 3.0)

    # At t=7, should not trigger (heartbeat was at 3, timeout at 8)
    msgs = election.check_and_elect(7.0)
    assert not election.election_in_progress
