"""Aggregate multi-convoy metrics (Phase 12)."""

from __future__ import annotations

from dataclasses import dataclass, field

from convoy_commander.metrics.collector import SimMetrics


@dataclass
class MultiConvoyMetrics:
    """Aggregate metrics across multiple convoys."""

    per_convoy: dict[int, SimMetrics] = field(default_factory=dict)
    total_vehicles: int = 0
    total_arrived: int = 0
    total_collisions: int = 0
    inter_convoy_collisions: int = 0
    total_near_misses: int = 0
    overall_mission_success: bool = False
    merge_count: int = 0
    split_count: int = 0
    right_of_way_yields: int = 0
    avg_cohesion_per_convoy: dict[int, float] = field(default_factory=dict)


def compute_multi_convoy_metrics(
    per_convoy_metrics: dict[int, SimMetrics],
    inter_convoy_collisions: int = 0,
    merge_count: int = 0,
    split_count: int = 0,
    right_of_way_yields: int = 0,
) -> MultiConvoyMetrics:
    """Compute aggregate metrics from per-convoy SimMetrics."""
    m = MultiConvoyMetrics()
    m.per_convoy = per_convoy_metrics
    m.inter_convoy_collisions = inter_convoy_collisions
    m.merge_count = merge_count
    m.split_count = split_count
    m.right_of_way_yields = right_of_way_yields

    for cid, sm in per_convoy_metrics.items():
        m.total_vehicles += sm.vehicles_total
        m.total_arrived += sm.vehicles_arrived
        m.total_collisions += sm.collision_count
        m.total_near_misses += sm.near_miss_count
        m.avg_cohesion_per_convoy[cid] = sm.convoy_cohesion_score

    m.total_collisions += inter_convoy_collisions
    m.overall_mission_success = m.total_arrived >= max(1, m.total_vehicles // 2)

    return m
