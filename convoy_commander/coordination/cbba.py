"""Consensus-Based Bundle Algorithm (CBBA-lite) for task/position allocation.

CBBA is a distributed auction-based algorithm where each agent maintains:
  - A *bundle*: ordered list of tasks it has won.
  - A *winning bid list*: best known bid for each task across the fleet.
  - A *winner list*: which agent holds each task.

Agents iterate between two phases:
  1. **Bundling**: each agent greedily adds tasks to its bundle if it can
     outbid the current winner.
  2. **Consensus**: agents exchange bid tables and resolve conflicts using
     the highest-bid-wins rule with vehicle-ID tie-breaking.

This implementation is a *lite* variant optimised for convoy formation
position assignment:
  - Tasks = formation slot indices (0 = lead, 1 = second, …).
  - Bid = negative distance from the vehicle's current estimated position
    to the ideal formation slot position, plus a small fuel-fraction bonus.
  - Lower cost (higher bid) wins.

SAFETY NOTE: CBBA-lite is used for *formation position* allocation only.
It does NOT override safe-mode or breakdown status.  Vehicles that are
non-operational are excluded from bidding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from convoy_commander.vehicles.vehicle import Vehicle


@dataclass
class CBBAState:
    """Per-agent CBBA state."""

    agent_id: int
    bundle: list[int] = field(default_factory=list)         # task indices this agent has won
    winning_bids: dict[int, float] = field(default_factory=dict)  # task -> best bid
    winners: dict[int, int] = field(default_factory=dict)         # task -> winning agent


def _compute_bid(
    vehicle: Vehicle,
    slot_position: tuple[float, float],
) -> float:
    """Compute the bid for a formation slot.

    Higher is better.  We use *negative distance* (closer is better) plus
    a small fuel bonus so that vehicles with more fuel prefer leading
    positions.
    """
    est = vehicle.estimator.state
    dist = math.hypot(est.x - slot_position[0], est.y - slot_position[1])
    fuel_bonus = vehicle.fuel.fraction * 5.0  # max 5.0 bonus at full fuel
    return -dist + fuel_bonus


def compute_formation_slots(
    leader_x: float,
    leader_y: float,
    leader_heading: float,
    num_slots: int,
    spacing: float,
) -> list[tuple[float, float]]:
    """Compute ideal formation slot positions behind the leader.

    Slot 0 is the leader position itself.  Slots 1..N-1 are spaced
    behind the leader along the *reverse* heading direction.
    """
    slots: list[tuple[float, float]] = [(leader_x, leader_y)]
    for i in range(1, num_slots):
        offset = spacing * i
        sx = leader_x - offset * math.cos(leader_heading)
        sy = leader_y - offset * math.sin(leader_heading)
        slots.append((sx, sy))
    return slots


def cbba_allocate(
    vehicles: list[Vehicle],
    slot_positions: list[tuple[float, float]],
    max_iterations: int = 20,
) -> dict[int, int]:
    """Run CBBA-lite to allocate formation slots to vehicles.

    Parameters
    ----------
    vehicles : list[Vehicle]
        Operational vehicles participating in the auction.
    slot_positions : list[tuple[float, float]]
        One (x, y) per formation slot.  ``len(slot_positions) >= len(vehicles)``.
    max_iterations : int
        Maximum consensus rounds.

    Returns
    -------
    dict[int, int]
        Mapping of ``vehicle_id -> slot_index``.  A vehicle may be unassigned
        if there are more vehicles than slots (unlikely in practice).
    """
    n_tasks = len(slot_positions)
    agents: dict[int, CBBAState] = {}
    for v in vehicles:
        st = CBBAState(agent_id=v.id)
        # Initialise winning bids to -inf (no winner yet)
        for t in range(n_tasks):
            st.winning_bids[t] = float("-inf")
            st.winners[t] = -1
        agents[v.id] = st

    # Build a bid table: agent -> task -> bid value (computed once)
    bid_table: dict[int, dict[int, float]] = {}
    for v in vehicles:
        bid_table[v.id] = {}
        for t in range(n_tasks):
            bid_table[v.id][t] = _compute_bid(v, slot_positions[t])

    for _iteration in range(max_iterations):
        changed = False

        # Phase 1: Bundling — each agent greedily adds the best available task
        for v in vehicles:
            st = agents[v.id]
            if len(st.bundle) >= 1:
                continue  # each vehicle gets at most one slot
            best_task = -1
            best_bid = float("-inf")
            for t in range(n_tasks):
                my_bid = bid_table[v.id][t]
                if my_bid > st.winning_bids[t]:
                    if my_bid > best_bid:
                        best_bid = my_bid
                        best_task = t
                    elif my_bid == best_bid and t < best_task:
                        best_task = t  # tie-break by task index
            if best_task >= 0:
                st.bundle.append(best_task)
                st.winning_bids[best_task] = best_bid
                st.winners[best_task] = v.id
                changed = True

        # Phase 2: Consensus — all agents share their winning bids
        # In a real distributed system this happens over comms.
        # Here we simulate the global consensus step directly.
        global_bids: dict[int, float] = {}
        global_winners: dict[int, int] = {}
        for t in range(n_tasks):
            best_bid = float("-inf")
            best_winner = -1
            for vid, st in agents.items():
                if st.winning_bids[t] > best_bid or (
                    st.winning_bids[t] == best_bid and vid < best_winner
                ):
                    if st.winning_bids[t] > best_bid:
                        best_bid = st.winning_bids[t]
                        best_winner = vid
            global_bids[t] = best_bid
            global_winners[t] = best_winner

        # Update all agents with global consensus
        for vid, st in agents.items():
            for t in range(n_tasks):
                if global_winners[t] != st.winners[t]:
                    # Outbid — drop task from bundle if we held it
                    if t in st.bundle and global_winners[t] != vid:
                        st.bundle.remove(t)
                        changed = True
                st.winning_bids[t] = global_bids[t]
                st.winners[t] = global_winners[t]

        if not changed:
            break

    # Build result
    result: dict[int, int] = {}
    for vid, st in agents.items():
        if st.bundle:
            result[vid] = st.bundle[0]

    # Assign unassigned vehicles to the nearest free slot
    assigned_slots = set(result.values())
    unassigned = [v for v in vehicles if v.id not in result]
    free_slots = [t for t in range(n_tasks) if t not in assigned_slots]
    for v in unassigned:
        if not free_slots:
            break
        best_slot = min(free_slots, key=lambda t: -bid_table[v.id][t])
        result[v.id] = best_slot
        free_slots.remove(best_slot)

    return result
