"""Command-line interface for convoy_commander."""

from __future__ import annotations

import argparse
import json
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
    run_parser.add_argument(
        "--scenario", default="baseline",
        help="Scenario name: baseline|gps_denied|comms_degraded|leader_failure|"
             "obstacle_pop|gps_spoofed|silent_running|comms_blackout|sensor_drift_spike|"
             "platooning|mesh_relay|terrain_real|heavy_rain|winter_storm|weather_api",
    )
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--vehicles", type=int, default=8, help="Number of vehicles")
    run_parser.add_argument("--loss", type=float, default=None, help="Packet loss rate")
    run_parser.add_argument("--latency", type=float, default=None, help="Mean latency ms")
    run_parser.add_argument("--duration", type=float, default=None, help="Sim duration seconds")
    run_parser.add_argument("--output", type=str, default=None, help="Output directory")
    # Geospatial flags (Phase 9)
    run_parser.add_argument("--elevation", type=str, default=None, help="Path to DEM GeoTIFF file")
    run_parser.add_argument("--osm-source", type=str, default=None, help="OSM file path or place name")
    run_parser.add_argument("--geo-bounds", type=str, default=None,
                            help="Bounding box: north,south,east,west (WGS84 degrees)")
    run_parser.add_argument("--slope-weight", type=float, default=None,
                            help="Weight for slope cost in A* pathfinding (0=disabled)")
    # Weather flags (Phase 10)
    run_parser.add_argument("--weather", action="store_true", help="Enable weather effects")
    run_parser.add_argument("--weather-source", type=str, default=None,
                            help="Weather source: 'api' or 'static'")
    run_parser.add_argument("--weather-lat", type=float, default=None,
                            help="Latitude for weather API (WGS84)")
    run_parser.add_argument("--weather-lon", type=float, default=None,
                            help="Longitude for weather API (WGS84)")
    run_parser.add_argument("--precipitation", type=float, default=None,
                            help="Static precipitation mm/h")
    run_parser.add_argument("--wind-speed", type=float, default=None,
                            help="Static wind speed m/s")
    run_parser.add_argument("--visibility", type=float, default=None,
                            help="Static visibility m")
    run_parser.add_argument("--temperature", type=float, default=None,
                            help="Static temperature Celsius")

    # report command
    report_parser = subparsers.add_parser("report", help="Display metrics from a previous run")
    report_parser.add_argument("--last", action="store_true", help="Use most recent run")
    report_parser.add_argument("--dir", type=str, default=None, help="Run directory")

    # evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate", help="Run evaluation sweep: multiple scenarios x seeds"
    )
    eval_parser.add_argument(
        "--scenarios", type=str, default=None,
        help="Comma-separated scenario names (default: baseline,gps_denied,"
             "comms_degraded,leader_failure,comms_blackout,platooning,mesh_relay)",
    )
    eval_parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds (default: 42,123,7)",
    )
    eval_parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Duration per run in seconds (default: 60)",
    )
    eval_parser.add_argument("--vehicles", type=int, default=8, help="Number of vehicles")
    eval_parser.add_argument("--output", type=str, default=None, help="Output base directory")

    # test command
    subparsers.add_parser("test", help="Run tests")

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "report":
        _cmd_report(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)
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
    # Geospatial overrides (Phase 9)
    if args.elevation is not None:
        overrides["elevation"] = args.elevation
    if args.osm_source is not None:
        overrides["osm_source"] = args.osm_source
    if args.geo_bounds is not None:
        parts = [float(x) for x in args.geo_bounds.split(",")]
        overrides["geo_bounds"] = tuple(parts)
    if args.slope_weight is not None:
        overrides["slope_weight"] = args.slope_weight
    # Weather overrides (Phase 10)
    if args.weather:
        overrides["weather_enabled"] = True
    if args.weather_source is not None:
        overrides["weather_source"] = args.weather_source
    if args.weather_lat is not None:
        overrides["weather_lat"] = args.weather_lat
    if args.weather_lon is not None:
        overrides["weather_lon"] = args.weather_lon
    if args.precipitation is not None:
        overrides["precipitation"] = args.precipitation
    if args.wind_speed is not None:
        overrides["wind_speed"] = args.wind_speed
    if args.visibility is not None:
        overrides["visibility"] = args.visibility
    if args.temperature is not None:
        overrides["temperature"] = args.temperature

    config = get_scenario(args.scenario, **overrides)

    print(f"=== Convoy Commander ===")
    print(f"Scenario: {config.scenario}")
    print(f"Seed: {config.seed}")
    print(f"Vehicles: {config.num_vehicles}")
    print(f"Duration: {config.duration}s")
    print(f"GPS: {'available' if config.gps_available else 'denied'}")
    print(f"Packet loss: {config.comms.packet_loss:.0%}")
    print(f"Latency: {config.comms.latency_mean_ms:.0f}ms")
    if config.weather.enabled:
        w = config.weather
        print(f"Weather: {w.weather_source} | "
              f"temp={w.static_temperature_c}°C, "
              f"precip={w.static_precipitation_mm_h}mm/h, "
              f"wind={w.static_wind_speed_ms}m/s, "
              f"vis={w.static_visibility_m}m")

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
    """Display metrics from a previous run."""
    if args.dir:
        run_dir = Path(args.dir)
    elif args.last:
        run_dir = _find_last_run()
    else:
        print("Error: specify --last or --dir <path>")
        sys.exit(1)

    if not run_dir.exists():
        print(f"Error: directory not found: {run_dir}")
        sys.exit(1)

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"Error: no metrics.json in {run_dir}")
        sys.exit(1)

    with open(metrics_path) as f:
        m = json.load(f)

    print(f"=== Report: {run_dir} ===")
    print(f"  Mission:     {'SUCCESS' if m.get('mission_success') else 'FAILURE'}")
    print(f"  Arrived:     {m.get('vehicles_arrived')}/{m.get('vehicles_total')}")
    print(f"  Avg ETA:     {m.get('avg_time_to_destination', 0):.1f}s")
    print(f"  Total fuel:  {m.get('total_fuel_used', 0):.1f}")
    print(f"  Cohesion:    {m.get('convoy_cohesion_score', 0):.1f}m")
    print(f"  Collisions:  {m.get('collision_count', 0)}")
    print(f"  Near misses: {m.get('near_miss_count', 0)}")
    print(f"  Comms:       {m.get('comms_delivery_ratio', 0):.1%}")
    print(f"  Avg pos err: {m.get('avg_position_error', 0):.2f}m")

    report_path = run_dir / "report.md"
    if report_path.exists():
        print(f"\n  Full report: {report_path}")

    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            stamp = json.load(f)
        print(f"\n  Git:      {stamp.get('git_commit', 'N/A')}")
        print(f"  Python:   {stamp.get('python_version', 'N/A')}")
        print(f"  Platform: {stamp.get('platform_info', 'N/A')}")


def _find_last_run() -> Path:
    """Find the most recently modified run directory."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("Error: no runs/ directory found")
        sys.exit(1)

    candidates = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "metrics.json").exists()
    ]
    if not candidates:
        print("Error: no run directories with metrics.json found in runs/")
        sys.exit(1)

    candidates.sort(key=lambda d: (d / "metrics.json").stat().st_mtime, reverse=True)
    return candidates[0]


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation sweep."""
    from convoy_commander.evaluate import run_evaluation, DEFAULT_SCENARIOS, DEFAULT_SEEDS

    scenarios = args.scenarios.split(",") if args.scenarios else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    output_base = Path(args.output) if args.output else None

    effective_scenarios = scenarios or DEFAULT_SCENARIOS
    effective_seeds = seeds or DEFAULT_SEEDS
    total = len(effective_scenarios) * len(effective_seeds)

    print(f"=== Convoy Commander Evaluation ===")
    print(f"Scenarios: {effective_scenarios}")
    print(f"Seeds:     {effective_seeds}")
    print(f"Total runs: {total}")
    print(f"Duration:  {args.duration}s per run")
    print(f"Vehicles:  {args.vehicles}")
    print()

    def progress(done: int, total: int, scenario: str, seed: int) -> None:
        print(f"  [{done}/{total}] {scenario} seed={seed} complete")

    summary, summary_dir = run_evaluation(
        scenarios=scenarios,
        seeds=seeds,
        duration=args.duration,
        num_vehicles=args.vehicles,
        output_base=output_base,
        progress_callback=progress,
    )

    print(f"\n=== Evaluation Complete ===")
    print(f"  Success rate: {summary.overall_success_rate:.0f}%")
    print(f"  Collisions:   {summary.total_collisions}")
    print(f"  Near misses:  {summary.total_near_misses}")
    print(f"  Summary:  {summary_dir / 'summary.md'}")
    print(f"  Data:     {summary_dir / 'summary.json'}")


def _cmd_test() -> None:
    """Run tests via pytest."""
    import subprocess
    sys.exit(subprocess.call(["python", "-m", "pytest", "-q"]))


if __name__ == "__main__":
    main()
