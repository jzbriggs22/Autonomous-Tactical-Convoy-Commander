"""Command-line interface for convoy_commander."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="convoy_commander",
        description="Autonomous Tactical Convoy Commander",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run a simulation scenario")
    run_parser.add_argument("--scenario", default="baseline", help="Scenario name")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--vehicles", type=int, default=8, help="Number of vehicles")
    run_parser.add_argument("--loss", type=float, default=None, help="Packet loss rate")
    run_parser.add_argument("--latency", type=float, default=None, help="Mean latency ms")
    run_parser.add_argument("--duration", type=float, default=None, help="Sim duration seconds")
    run_parser.add_argument("--output", type=str, default=None, help="Output directory")

    # report command
    report_parser = subparsers.add_parser("report", help="Regenerate report from last run")
    report_parser.add_argument("--last", action="store_true", help="Use last run")
    report_parser.add_argument("--dir", type=str, default=None, help="Run directory")

    # test command
    subparsers.add_parser("test", help="Run tests")

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "report":
        _cmd_report(args)
    elif args.command == "test":
        _cmd_test()
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_run(args: argparse.Namespace) -> None:
    """Run a simulation."""
    from convoy_commander.sim.scenarios import get_scenario
    from convoy_commander.sim.runner import SimRunner
    from convoy_commander.viz.report import generate_report

    overrides: dict[str, object] = {"seed": args.seed, "vehicles": args.vehicles}
    if args.loss is not None:
        overrides["loss"] = args.loss
    if args.latency is not None:
        overrides["latency"] = args.latency
    if args.duration is not None:
        overrides["duration"] = args.duration

    config = get_scenario(args.scenario, **overrides)

    print(f"=== Convoy Commander ===")
    print(f"Scenario: {config.scenario}")
    print(f"Seed: {config.seed}")
    print(f"Vehicles: {config.num_vehicles}")
    print(f"Duration: {config.duration}s")
    print(f"GPS: {'available' if config.gps_available else 'denied'}")
    print(f"Packet loss: {config.comms.packet_loss:.0%}")
    print(f"Latency: {config.comms.latency_mean_ms:.0f}ms")

    # Print safety warnings
    warnings = config.safety_warnings()
    if warnings:
        print(f"\n  SAFETY WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")
    print()

    runner = SimRunner(config)

    start_time = time.time()

    def progress(step: int, total: int) -> None:
        pct = step / total * 100
        print(f"\r  Simulating... {pct:5.1f}%", end="", flush=True)

    result = runner.run(progress_callback=progress)
    elapsed = time.time() - start_time
    print(f"\r  Simulation complete in {elapsed:.1f}s        ")

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs") / f"{config.scenario}_{timestamp}"

    report_path = generate_report(result, output_dir)
    print(f"\n  Report:    {report_path}")
    print(f"  Plots:     {output_dir / 'plots'}/")
    print(f"  Data:      {output_dir / 'metrics.json'}")
    print(f"  Audit log: {output_dir / 'event_log.jsonl'}")

    # Print summary
    metrics = result.collector.compute_final(
        result.vehicles,
        result.comms.total_sent,
        result.comms.total_delivered,
        result.comms.total_dropped,
        config.duration,
    )
    print(f"\n=== Results ===")
    print(f"  Mission: {'SUCCESS' if metrics.mission_success else 'FAILURE'}")
    print(f"  Arrived: {metrics.vehicles_arrived}/{metrics.vehicles_total}")
    print(f"  Avg time to dest: {metrics.avg_time_to_destination:.1f}s")
    print(f"  Total fuel: {metrics.total_fuel_used:.1f}")
    print(f"  Cohesion: {metrics.convoy_cohesion_score:.1f}m")
    print(f"  Near misses: {metrics.near_miss_count}")
    print(f"  Collisions: {metrics.collision_count}")
    print(f"  Comms delivery: {metrics.comms_delivery_ratio:.1%}")
    print(f"  Avg pos error: {metrics.avg_position_error:.2f}m")

    # Print safety event summary
    by_sev = result.event_log.count_by_severity()
    crit = by_sev.get("CRITICAL", 0)
    warn = by_sev.get("WARNING", 0)
    print(f"\n=== Safety Audit ===")
    print(f"  Total events: {len(result.event_log)}")
    print(f"  CRITICAL: {crit}")
    print(f"  WARNING:  {warn}")
    if crit > 0:
        print(f"  ** {crit} CRITICAL event(s) — see {output_dir / 'event_log.jsonl'}")


def _cmd_report(args: argparse.Namespace) -> None:
    """Regenerate report from saved artifacts."""
    print("Report regeneration from saved artifacts is not yet implemented.")
    print("Run a new simulation with: python -m convoy_commander run --scenario baseline")
    sys.exit(0)


def _cmd_test() -> None:
    """Run tests via pytest."""
    import subprocess
    sys.exit(subprocess.call(["python", "-m", "pytest", "-q"]))


if __name__ == "__main__":
    main()
