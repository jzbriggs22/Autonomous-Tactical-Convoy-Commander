"""Visualization: trajectory plots, comms graph, error plots."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from convoy_commander.core.world import World
from convoy_commander.metrics.collector import MetricsCollector, SimMetrics
from convoy_commander.vehicles.vehicle import Vehicle


COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
]


def plot_world(world: World, ax: plt.Axes) -> None:
    """Draw obstacles, no-go zones, poly obstacles, spoof regions, roads, landmarks."""
    # No-go zones
    for nz in world.nogo_zones:
        circle = mpatches.Circle(
            (nz.x, nz.y), nz.radius, alpha=0.15, color="red", label="No-go zone"
        )
        ax.add_patch(circle)

    # Circular obstacles
    for obs in world.obstacles:
        circle = mpatches.Circle(
            (obs.x, obs.y), obs.radius, alpha=0.5, color="gray", label="Obstacle"
        )
        ax.add_patch(circle)

    # Poly (rectangular) obstacles
    for po in world.poly_obstacles:
        rect = mpatches.Rectangle(
            (po.x - po.half_w, po.y - po.half_h),
            po.half_w * 2, po.half_h * 2,
            alpha=0.5, color="dimgray", label="Rect obstacle",
        )
        ax.add_patch(rect)

    # GPS spoof regions (translucent magenta circles)
    for sr in world.spoof_regions:
        spoof_circle = mpatches.Circle(
            (sr.x, sr.y), sr.radius, alpha=0.12, color="magenta",
            linestyle="--", linewidth=1.0, fill=True, label="Spoof zone",
        )
        ax.add_patch(spoof_circle)

    # Road graph
    for u, v in world.road_graph.edges():
        pos_u = world.get_node_pos(u)
        pos_v = world.get_node_pos(v)
        ax.plot(
            [pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]],
            "k-", linewidth=0.3, alpha=0.3,
        )

    # Landmarks
    for lm in world.landmarks:
        ax.plot(lm.x, lm.y, "^", color="green", markersize=6, alpha=0.7)

    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")


def plot_trajectories(
    world: World,
    vehicles: list[Vehicle],
    output_path: Path,
) -> None:
    """Plot true and estimated trajectories for all vehicles."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # True trajectories
    ax1.set_title("True Trajectories")
    plot_world(world, ax1)
    for i, v in enumerate(vehicles):
        color = COLORS[i % len(COLORS)]
        xs = [p[0] for p in v.true_history]
        ys = [p[1] for p in v.true_history]
        ax1.plot(xs, ys, "-", color=color, linewidth=0.8, alpha=0.8, label=f"V{v.id}")
        ax1.plot(xs[0], ys[0], "o", color=color, markersize=5)
        ax1.plot(xs[-1], ys[-1], "s", color=color, markersize=5)
    ax1.legend(fontsize=7, loc="upper right")

    # Estimated trajectories
    ax2.set_title("Estimated Trajectories")
    plot_world(world, ax2)
    for i, v in enumerate(vehicles):
        color = COLORS[i % len(COLORS)]
        xs = [p[0] for p in v.est_history]
        ys = [p[1] for p in v.est_history]
        ax2.plot(xs, ys, "--", color=color, linewidth=0.8, alpha=0.8, label=f"V{v.id}")
    ax2.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_position_errors(
    vehicles: list[Vehicle],
    dt: float,
    output_path: Path,
) -> None:
    """Plot position estimation error over time for each vehicle."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Position Estimation Error Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")

    for i, v in enumerate(vehicles):
        color = COLORS[i % len(COLORS)]
        errors = []
        for (tx, ty), (ex, ey) in zip(v.true_history, v.est_history):
            errors.append(math.hypot(tx - ex, ty - ey))
        times = [j * dt for j in range(len(errors))]
        ax.plot(times, errors, "-", color=color, linewidth=0.7, alpha=0.8, label=f"V{v.id}")

    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metrics_summary(
    collector: MetricsCollector,
    vehicles: list[Vehicle],
    dt: float,
    output_path: Path,
) -> None:
    """Plot speed, fuel, uncertainty, and headway over time."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

    # Organize time series by vehicle
    vid_data: dict[int, dict[str, list[float]]] = {}
    for entry in collector.time_series:
        if entry.vehicle_id not in vid_data:
            vid_data[entry.vehicle_id] = {
                "time": [], "speed": [], "fuel": [], "uncertainty": [],
            }
        vid_data[entry.vehicle_id]["time"].append(entry.time)
        vid_data[entry.vehicle_id]["speed"].append(entry.speed)
        vid_data[entry.vehicle_id]["fuel"].append(entry.fuel)
        vid_data[entry.vehicle_id]["uncertainty"].append(entry.uncertainty)

    for i, (vid, data) in enumerate(sorted(vid_data.items())):
        color = COLORS[i % len(COLORS)]
        axes[0].plot(data["time"], data["speed"], color=color, linewidth=0.6, alpha=0.8, label=f"V{vid}")
        axes[1].plot(data["time"], data["fuel"], color=color, linewidth=0.6, alpha=0.8, label=f"V{vid}")
        axes[2].plot(data["time"], data["uncertainty"], color=color, linewidth=0.6, alpha=0.8, label=f"V{vid}")

    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Vehicle Speed Over Time")
    axes[0].legend(fontsize=6, ncol=4)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Fuel")
    axes[1].set_title("Fuel Level Over Time")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Uncertainty (m)")
    axes[2].set_title("Position Uncertainty Over Time")
    axes[2].grid(True, alpha=0.3)

    # Headway gap (4th subplot)
    if collector.headway_samples:
        by_vid_hw: dict[int, dict[str, list[float]]] = {}
        for s in collector.headway_samples:
            vid = s["vehicle_id"]
            if vid not in by_vid_hw:
                by_vid_hw[vid] = {"time": [], "actual": []}
            by_vid_hw[vid]["time"].append(s["time"])
            by_vid_hw[vid]["actual"].append(s["actual_gap"])
        for i, (vid, data) in enumerate(sorted(by_vid_hw.items())):
            color = COLORS[i % len(COLORS)]
            axes[3].plot(data["time"], data["actual"], color=color, linewidth=0.6,
                         alpha=0.8, label=f"V{vid}")
        axes[3].legend(fontsize=6, ncol=4)
    else:
        axes[3].text(0.5, 0.5, "No headway data", transform=axes[3].transAxes,
                     ha="center", va="center", fontsize=12, color="gray")
    axes[3].set_ylabel("Headway Gap (m)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Inter-Vehicle Headway Gap Over Time")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_headway_gaps(
    collector: MetricsCollector,
    config: object,
    dt: float,
    output_path: Path,
) -> None:
    """Plot actual vs desired inter-vehicle headway gaps over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Inter-Vehicle Headway Gap Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gap (m)")

    if not collector.headway_samples:
        ax.text(0.5, 0.5, "No headway data recorded", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")
    else:
        # Group by vehicle_id
        by_vid: dict[int, dict[str, list[float]]] = {}
        for s in collector.headway_samples:
            vid = s["vehicle_id"]
            if vid not in by_vid:
                by_vid[vid] = {"time": [], "actual": [], "desired": []}
            by_vid[vid]["time"].append(s["time"])
            by_vid[vid]["actual"].append(s["actual_gap"])
            by_vid[vid]["desired"].append(s["desired_gap"])

        for i, (vid, data) in enumerate(sorted(by_vid.items())):
            color = COLORS[i % len(COLORS)]
            ax.plot(data["time"], data["actual"], "-", color=color, linewidth=0.7,
                    alpha=0.8, label=f"V{vid} actual")
            ax.plot(data["time"], data["desired"], "--", color=color, linewidth=0.5,
                    alpha=0.5, label=f"V{vid} desired")

        # Reference line for standoff distance
        if hasattr(config, "coordination"):
            ax.axhline(y=config.coordination.standoff_distance, color="red",
                       linestyle=":", linewidth=0.8, alpha=0.5, label="Standoff min")

        ax.legend(fontsize=6, ncol=4)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_string_stability(
    collector: MetricsCollector,
    output_path: Path,
) -> None:
    """Plot spacing errors and RMS string stability per vehicle."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    if not collector.spacing_error_samples:
        ax1.text(0.5, 0.5, "No spacing error data recorded\n(stability window not reached)",
                 transform=ax1.transAxes, ha="center", va="center", fontsize=14, color="gray")
        ax2.text(0.5, 0.5, "No data", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=14, color="gray")
    else:
        # Group by vehicle_id
        by_vid: dict[int, dict[str, list[float]]] = {}
        for s in collector.spacing_error_samples:
            vid = s["vehicle_id"]
            if vid not in by_vid:
                by_vid[vid] = {"time": [], "error": []}
            by_vid[vid]["time"].append(s["time"])
            by_vid[vid]["error"].append(s["error"])

        # Top: spacing error over time
        ax1.set_title("Spacing Error Over Time")
        ax1.set_ylabel("Error (m)")
        for i, (vid, data) in enumerate(sorted(by_vid.items())):
            color = COLORS[i % len(COLORS)]
            ax1.plot(data["time"], data["error"], "-", color=color, linewidth=0.7,
                     alpha=0.8, label=f"V{vid}")
        ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
        ax1.legend(fontsize=6, ncol=4)
        ax1.grid(True, alpha=0.3)

        # Bottom: RMS bar chart
        ax2.set_title("RMS Spacing Error by Vehicle (String Stability)")
        ax2.set_xlabel("Vehicle ID")
        ax2.set_ylabel("RMS Error (m)")
        sorted_vids = sorted(by_vid.keys())
        rms_vals = []
        for vid in sorted_vids:
            errors = by_vid[vid]["error"]
            rms = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else 0.0
            rms_vals.append(rms)
        bar_colors = [COLORS[i % len(COLORS)] for i in range(len(sorted_vids))]
        ax2.bar([f"V{v}" for v in sorted_vids], rms_vals, color=bar_colors, alpha=0.8)
        ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_corridor_adherence(
    world: World,
    vehicles: list[Vehicle],
    collector: MetricsCollector,
    output_path: Path,
) -> None:
    """Plot trajectories and corridor distance over time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Left: trajectories on world map
    ax1.set_title("Vehicle Trajectories")
    plot_world(world, ax1)
    for i, v in enumerate(vehicles):
        color = COLORS[i % len(COLORS)]
        xs = [p[0] for p in v.true_history]
        ys = [p[1] for p in v.true_history]
        ax1.plot(xs, ys, "-", color=color, linewidth=0.8, alpha=0.8, label=f"V{v.id}")
    ax1.legend(fontsize=7, loc="upper right")

    # Right: corridor distance time series
    ax2.set_title("Corridor Distance Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Corridor Distance (m)")

    if not collector.corridor_samples:
        ax2.text(0.5, 0.5, "No corridor data recorded", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=14, color="gray")
    else:
        by_vid: dict[int, dict[str, list[float]]] = {}
        for s in collector.corridor_samples:
            vid = s["vehicle_id"]
            if vid not in by_vid:
                by_vid[vid] = {"time": [], "dist": []}
            by_vid[vid]["time"].append(s["time"])
            by_vid[vid]["dist"].append(s["corridor_dist"])

        for i, (vid, data) in enumerate(sorted(by_vid.items())):
            color = COLORS[i % len(COLORS)]
            ax2.plot(data["time"], data["dist"], "-", color=color, linewidth=0.7,
                     alpha=0.8, label=f"V{vid}")
        ax2.legend(fontsize=6, ncol=4)

    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_comms_graph(
    vehicles: list[Vehicle],
    adjacency: dict[int, list[int]],
    world: World,
    output_path: Path,
) -> None:
    """Plot current comms graph overlaid on world."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Communications Graph")
    plot_world(world, ax)

    # Draw comms links
    drawn: set[tuple[int, int]] = set()
    for vid, neighbors in adjacency.items():
        v = next((veh for veh in vehicles if veh.id == vid), None)
        if v is None:
            continue
        for nid in neighbors:
            edge = (min(vid, nid), max(vid, nid))
            if edge in drawn:
                continue
            drawn.add(edge)
            n = next((veh for veh in vehicles if veh.id == nid), None)
            if n is None:
                continue
            ax.plot(
                [v.state.x, n.state.x], [v.state.y, n.state.y],
                "b-", linewidth=0.5, alpha=0.4,
            )

    # Draw vehicles
    for i, v in enumerate(vehicles):
        color = COLORS[i % len(COLORS)]
        marker = "D" if v.is_leader else "o"
        ax.plot(v.state.x, v.state.y, marker, color=color, markersize=8, label=f"V{v.id}")

    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
