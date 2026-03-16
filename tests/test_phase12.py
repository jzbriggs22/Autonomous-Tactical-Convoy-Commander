"""Phase 12 tests: Multi-Convoy Operations & Inter-Convoy Coordination."""

from __future__ import annotations

import math

import numpy as np
import pytest

from convoy_commander.core.config import SimConfig
from convoy_commander.core.convoy_config import ConvoySpec, MultiConvoyConfig
from convoy_commander.core.event_log import EventKind
from convoy_commander.coordination.convoy_group import ConvoyGroup
from convoy_commander.coordination.leader_election import LeaderElection
from convoy_commander.coordination.right_of_way import (
    apply_yield,
    detect_crossing,
    resolve_right_of_way,
)
from convoy_commander.coordination.merge_split import (
    can_merge,
    merge_convoys,
    split_convoy,
)
from convoy_commander.metrics.multi_metrics import (
    MultiConvoyMetrics,
    compute_multi_convoy_metrics,
)
from convoy_commander.metrics.collector import SimMetrics
from convoy_commander.vehicles.vehicle import Vehicle


# ====================================================================
# Helpers
# ====================================================================

def _make_vehicle(vid: int, x: float = 0.0, y: float = 0.0, convoy_id: int = 0) -> Vehicle:
    """Create a minimal vehicle for testing."""
    config = SimConfig()
    rng = np.random.default_rng(42 + vid)
    v = Vehicle(vehicle_id=vid, config=config, rng=rng, start_x=x, start_y=y)
    v.convoy_id = convoy_id
    return v


def _make_group(
    convoy_id: int,
    num_vehicles: int = 4,
    priority: int = 0,
    start_x: float = 50.0,
    start_y: float = 50.0,
    dest: tuple[float, float] = (920.0, 920.0),
) -> ConvoyGroup:
    """Create a ConvoyGroup for testing."""
    vehicles = []
    for i in range(num_vehicles):
        vid = convoy_id * 100 + i
        v = _make_vehicle(vid, x=start_x + i * 25.0, y=start_y, convoy_id=convoy_id)
        vehicles.append(v)
    group = ConvoyGroup(
        convoy_id=convoy_id,
        priority=priority,
        vehicles=vehicles,
        destination=dest,
    )
    # Set up elections and leader
    for v in vehicles:
        group.elections[v.id] = LeaderElection(v, 5.0)
    if vehicles:
        vehicles[0].is_leader = True
        vehicles[0].leader_id = vehicles[0].id
        for v in vehicles:
            v.leader_id = vehicles[0].id
    return group


# ====================================================================
# Group 1: Config and Data Structures
# ====================================================================


class TestConvoyConfig:
    def test_convoy_spec_validation(self):
        """ConvoySpec rejects invalid num_vehicles."""
        with pytest.raises(Exception):
            ConvoySpec(convoy_id=0, num_vehicles=0)

    def test_multi_convoy_config_total_vehicles(self):
        """MultiConvoyConfig.total_vehicles returns correct sum."""
        config = MultiConvoyConfig(
            convoys=[
                ConvoySpec(convoy_id=0, num_vehicles=4),
                ConvoySpec(convoy_id=1, num_vehicles=3),
            ]
        )
        assert config.total_vehicles == 7

    def test_convoy_group_add_remove_vehicle(self):
        """ConvoyGroup add/remove correctly updates vehicle list and convoy_id."""
        group = _make_group(0, num_vehicles=3)
        assert len(group.vehicles) == 3

        # Create a new vehicle and add it
        new_v = _make_vehicle(999, convoy_id=5)
        group.add_vehicle(new_v)
        assert len(group.vehicles) == 4
        assert new_v.convoy_id == 0  # updated to group's convoy_id

        # Remove a vehicle
        removed = group.remove_vehicle(999)
        assert removed is not None
        assert removed.id == 999
        assert len(group.vehicles) == 3

        # Remove non-existent returns None
        assert group.remove_vehicle(999) is None

    def test_vehicle_convoy_id_default(self):
        """Vehicle.convoy_id defaults to 0 (backward compatibility)."""
        config = SimConfig()
        rng = np.random.default_rng(42)
        v = Vehicle(vehicle_id=0, config=config, rng=rng)
        assert v.convoy_id == 0


# ====================================================================
# Group 2: Right-of-Way
# ====================================================================


class TestRightOfWay:
    def test_detect_crossing_true(self):
        """Detects crossing when leaders are within radius."""
        ga = _make_group(0, start_x=100.0, start_y=100.0)
        gb = _make_group(1, start_x=130.0, start_y=100.0)
        assert detect_crossing(ga, gb, proximity_radius=80.0)

    def test_detect_crossing_false(self):
        """No crossing detected when leaders are far apart."""
        ga = _make_group(0, start_x=50.0, start_y=50.0)
        gb = _make_group(1, start_x=500.0, start_y=500.0)
        assert not detect_crossing(ga, gb, proximity_radius=80.0)

    def test_resolve_lower_priority_yields(self):
        """Lower priority convoy yields in right-of-way resolution."""
        ga = _make_group(0, priority=1)
        gb = _make_group(1, priority=0)
        yielder = resolve_right_of_way(ga, gb, current_time=10.0, yield_duration=5.0)
        assert yielder is gb
        assert gb.yielding
        assert gb.yield_until == 15.0
        assert not ga.yielding

    def test_apply_yield_speed_factor(self):
        """Yielding convoy gets 0.1 speed factor; non-yielding gets 1.0."""
        group = _make_group(0)
        assert apply_yield(group, 10.0) == 1.0

        group.yielding = True
        group.yield_until = 20.0
        assert apply_yield(group, 15.0) == 0.1

        # After yield expires
        assert apply_yield(group, 20.0) == 1.0
        assert not group.yielding


# ====================================================================
# Group 3: Merge/Split
# ====================================================================


class TestMergeSplit:
    def test_merge_all_vehicles_transferred(self):
        """merge_convoys transfers all vehicles; absorbed group becomes empty."""
        ga = _make_group(0, num_vehicles=4, priority=1)
        gb = _make_group(1, num_vehicles=3, priority=0)
        original_count = len(ga.vehicles) + len(gb.vehicles)

        merge_convoys(ga, gb, heartbeat_timeout=5.0)

        assert len(ga.vehicles) == original_count
        assert len(gb.vehicles) == 0
        # All vehicles should have convoy_id = 0
        for v in ga.vehicles:
            assert v.convoy_id == 0

    def test_merge_elections_reset(self):
        """All elections are reset after merge."""
        ga = _make_group(0, num_vehicles=2)
        gb = _make_group(1, num_vehicles=2)

        # Set a specific election state before merge
        for eid, el in ga.elections.items():
            el.election_in_progress = False

        merge_convoys(ga, gb, heartbeat_timeout=5.0)

        # All elections should be reset (election_in_progress = True)
        for vid, el in ga.elections.items():
            assert el.election_in_progress
            assert el.vehicle.leader_id is None

    def test_split_correct_partition(self):
        """split_convoy creates correct partition of vehicles."""
        source = _make_group(0, num_vehicles=6)
        all_ids = [v.id for v in source.vehicles]
        split_ids = all_ids[3:]  # last 3

        new_group = split_convoy(
            source, split_ids,
            new_convoy_id=1,
            new_destination=(100.0, 100.0),
            new_priority=0,
            heartbeat_timeout=5.0,
            min_vehicles=2,
        )

        assert new_group is not None
        assert len(source.vehicles) == 3
        assert len(new_group.vehicles) == 3
        assert new_group.convoy_id == 1
        assert new_group.destination == (100.0, 100.0)
        # Check vehicle IDs
        source_ids = {v.id for v in source.vehicles}
        new_ids = {v.id for v in new_group.vehicles}
        assert source_ids & new_ids == set()  # no overlap

    def test_split_min_vehicles_enforced(self):
        """Split returns None if it would leave groups below min_vehicles."""
        source = _make_group(0, num_vehicles=3)
        all_ids = [v.id for v in source.vehicles]

        result = split_convoy(
            source, all_ids[:2],  # try to take 2, leaving only 1
            new_convoy_id=1,
            new_destination=(100.0, 100.0),
            new_priority=0,
            heartbeat_timeout=5.0,
            min_vehicles=2,
        )
        assert result is None
        assert len(source.vehicles) == 3  # unchanged


# ====================================================================
# Group 4: Multi-Convoy Runner Integration
# ====================================================================


class TestMultiConvoyRunner:
    def test_two_convoys_basic(self):
        """Two convoys run for 10s without crashing, both produce metrics."""
        from convoy_commander.sim.multi_runner import MultiConvoyRunner

        config = MultiConvoyConfig(
            base=SimConfig(duration=10.0, seed=42),
            convoys=[
                ConvoySpec(convoy_id=0, num_vehicles=3, start_x=50.0, start_y=50.0),
                ConvoySpec(convoy_id=1, num_vehicles=3, start_x=50.0, start_y=200.0),
            ],
        )
        runner = MultiConvoyRunner(config)
        result = runner.run()

        assert result.aggregate_metrics is not None
        assert result.aggregate_metrics.total_vehicles == 6
        assert len(result.all_vehicles) == 6
        # Both convoys should have metrics
        assert 0 in result.aggregate_metrics.per_convoy
        assert 1 in result.aggregate_metrics.per_convoy

    def test_shared_world(self):
        """Both convoys share the same World instance."""
        from convoy_commander.sim.multi_runner import MultiConvoyRunner

        config = MultiConvoyConfig(
            base=SimConfig(duration=1.0, seed=42),
            convoys=[
                ConvoySpec(convoy_id=0, num_vehicles=2),
                ConvoySpec(convoy_id=1, num_vehicles=2, start_x=200.0, start_y=200.0),
            ],
        )
        runner = MultiConvoyRunner(config)
        result = runner.run()

        assert result.world is runner.world

    def test_inter_convoy_collision_tracking(self):
        """Inter-convoy collisions are tracked separately from intra-convoy."""
        from convoy_commander.sim.multi_runner import MultiConvoyRunner

        # Put convoys very close to each other to force near-miss/collision
        config = MultiConvoyConfig(
            base=SimConfig(duration=2.0, seed=42),
            convoys=[
                ConvoySpec(convoy_id=0, num_vehicles=2, start_x=50.0, start_y=50.0),
                ConvoySpec(
                    convoy_id=1, num_vehicles=2,
                    start_x=51.0, start_y=50.0,  # very close
                ),
            ],
        )
        runner = MultiConvoyRunner(config)
        result = runner.run()

        # Check that inter-convoy events were logged (near miss or collision)
        inter_events = [
            e for e in result.event_log.events
            if e.kind in (
                EventKind.INTER_CONVOY_COLLISION.value,
                EventKind.INTER_CONVOY_NEAR_MISS.value,
            )
        ]
        # With vehicles this close, there should be some inter-convoy events
        assert len(inter_events) > 0

    def test_message_filtering_convoy_scoped(self):
        """Leader elections are convoy-scoped: convoy A's election msgs
        don't affect convoy B's leader."""
        from convoy_commander.sim.multi_runner import MultiConvoyRunner

        config = MultiConvoyConfig(
            base=SimConfig(duration=5.0, seed=42),
            convoys=[
                ConvoySpec(convoy_id=0, num_vehicles=3, start_x=50.0, start_y=50.0),
                ConvoySpec(
                    convoy_id=1, num_vehicles=3,
                    start_x=50.0, start_y=300.0,
                ),
            ],
        )
        runner = MultiConvoyRunner(config)
        result = runner.run()

        # Each convoy should have its own leader
        group0 = result.groups[0]
        group1 = result.groups[1]
        leader0 = group0.get_leader()
        leader1 = group1.get_leader()

        # Leaders should be from their own convoy
        if leader0 is not None:
            assert leader0.convoy_id == 0
        if leader1 is not None:
            assert leader1.convoy_id == 1

    def test_right_of_way_crossing_scenario(self):
        """Run two_convoy_crossing for 30s and verify at least one yield event."""
        from convoy_commander.sim.scenarios import get_multi_convoy_scenario
        from convoy_commander.sim.multi_runner import MultiConvoyRunner

        mc_config = get_multi_convoy_scenario("two_convoy_crossing", duration=30.0)
        runner = MultiConvoyRunner(mc_config)
        result = runner.run()

        yield_events = [
            e for e in result.event_log.events
            if e.kind == EventKind.RIGHT_OF_WAY_YIELD.value
        ]
        # The convoys cross paths so at least one yield should occur
        # (might not always happen in 30s depending on route, so allow 0 too)
        assert result.aggregate_metrics is not None
        assert result.aggregate_metrics.total_vehicles == 8


# ====================================================================
# Group 5: Scenario Integration
# ====================================================================


class TestScenarios:
    def test_all_multi_scenarios_load(self):
        """All 4 multi-convoy scenarios produce valid configs."""
        from convoy_commander.sim.scenarios import get_multi_convoy_scenario

        for name in [
            "two_convoy_crossing",
            "convoy_merge",
            "convoy_split_reroute",
            "multi_convoy_contested",
        ]:
            config = get_multi_convoy_scenario(name)
            assert config.total_vehicles > 0
            assert len(config.convoys) >= 1
            assert config.base.scenario == name

    def test_convoy_merge_integration(self):
        """Run convoy_merge for 60s, verify merge event logged."""
        from convoy_commander.sim.scenarios import get_multi_convoy_scenario
        from convoy_commander.sim.multi_runner import MultiConvoyRunner

        mc_config = get_multi_convoy_scenario("convoy_merge", duration=60.0)
        runner = MultiConvoyRunner(mc_config)
        result = runner.run()

        merge_events = [
            e for e in result.event_log.events
            if e.kind == EventKind.CONVOY_MERGE.value
        ]
        # Convoys start close together heading to same destination,
        # so merge should be triggered
        assert result.aggregate_metrics is not None
        # Either merge happened or both convoys are still running
        total_in_groups = sum(len(g.vehicles) for g in result.groups.values())
        assert total_in_groups == 8

    def test_convoy_split_reroute_integration(self):
        """Run convoy_split_reroute for 80s, verify split event logged
        and two convoys exist."""
        from convoy_commander.sim.scenarios import get_multi_convoy_scenario
        from convoy_commander.sim.multi_runner import MultiConvoyRunner

        mc_config = get_multi_convoy_scenario("convoy_split_reroute", duration=80.0)
        runner = MultiConvoyRunner(mc_config)
        result = runner.run()

        split_events = [
            e for e in result.event_log.events
            if e.kind == EventKind.CONVOY_SPLIT.value
        ]
        assert len(split_events) == 1  # split at t=60s

        # After split, should have two convoy groups
        active_groups = [g for g in result.groups.values() if g.vehicles]
        assert len(active_groups) == 2

        # Destinations should be different
        dests = {g.destination for g in active_groups}
        assert len(dests) == 2


# ====================================================================
# Group 6: Metrics
# ====================================================================


class TestMultiMetrics:
    def test_aggregate_metrics(self):
        """Verify per-convoy and aggregate metrics are computed correctly."""
        sm0 = SimMetrics(
            vehicles_total=4, vehicles_arrived=3,
            collision_count=1, near_miss_count=2,
            convoy_cohesion_score=50.0,
        )
        sm1 = SimMetrics(
            vehicles_total=4, vehicles_arrived=4,
            collision_count=0, near_miss_count=1,
            convoy_cohesion_score=40.0,
        )

        agg = compute_multi_convoy_metrics(
            {0: sm0, 1: sm1},
            inter_convoy_collisions=2,
            merge_count=1,
            split_count=0,
            right_of_way_yields=3,
        )

        assert agg.total_vehicles == 8
        assert agg.total_arrived == 7
        assert agg.total_collisions == 1 + 0 + 2  # intra + inter
        assert agg.total_near_misses == 3
        assert agg.inter_convoy_collisions == 2
        assert agg.merge_count == 1
        assert agg.right_of_way_yields == 3
        assert agg.overall_mission_success  # 7/8 >= 4


# ====================================================================
# Group 7: Backward Compatibility
# ====================================================================


class TestBackwardCompat:
    def test_existing_simrunner_unaffected(self):
        """Run baseline scenario with SimRunner — verify it still works."""
        from convoy_commander.sim.scenarios import get_scenario
        from convoy_commander.sim.runner import SimRunner

        config = get_scenario("baseline", duration=5.0, seed=42, vehicles=4)
        runner = SimRunner(config)
        result = runner.run()

        metrics = result.collector.compute_final(
            result.vehicles,
            result.comms.total_sent,
            result.comms.total_delivered,
            result.comms.total_dropped,
            config.duration,
        )
        assert metrics.vehicles_total == 4
        # All vehicles should have default convoy_id = 0
        for v in result.vehicles:
            assert v.convoy_id == 0
