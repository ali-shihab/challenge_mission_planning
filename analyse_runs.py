#\!/usr/bin/env python3
"""
analyse_runs.py  –  Aggregate run metrics and produce all report figures.

Usage (from repo root, after sourcing setup.bash):
    python3 analyse_runs.py [--runs_root runs] [--out_dir report_figs]

Produces:
    fig_metrics.png      – bar chart: mission time, path distance, effective speed
    fig_paths_sc*.png    – top-down flight paths per scenario
    fig_boxplots.png     – box-and-whisker across 5 runs per condition
    summary_table.csv    – full table for LaTeX
"""
import argparse
import json
import math
import os
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────
SCENARIO_LABELS = {
    "scenario1": "Sc1\n(10 vp, 1 obs)",
    "scenario2": "Sc2\n(10 vp, 5 obs)",
    "scenario3": "Sc3\n(20 vp, 1 obs)",
    "scenario4": "Sc4\n(5 vp, 10 obs)",
}

PLANNER_LABELS = {
    "baseline": "Baseline\n(straight, input)",
    "astar":    "A* + 2-opt",
    "rrts":     "RRT* + 2-opt",
}

PLANNER_COLORS = {
    "baseline": "#4878CF",
    "astar":    "#6ACC65",
    "rrts":     "#D65F5F",
}

def parse_run_dir_name(name: str):
    """
    20260410_082003__scenario1__astar__drone0
    Returns (datetime_str, scenario, planner, namespace)
    """
    parts = name.split("__")
    if len(parts) < 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def load_all_runs(runs_root: Path):
    """Return list of dicts, one per completed run."""
    runs = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_run_dir_name(d.name)
        if parsed is None:
            continue
        dt_str, scenario, planner, ns = parsed
        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        # Load trajectory for path plotting
        traj = []
        traj_path = d / "trajectory.csv"
        if traj_path.exists():
            import csv as _csv
            with open(traj_path) as f2:
                reader = _csv.DictReader(f2)
                for row in reader:
                    try:
                        traj.append({k: float(v) for k, v in row.items()})
                    except Exception:
                        pass
        # Load events for ordering info
        ordering = "input"
        events_path = d / "events.jsonl"
        if events_path.exists():
            with open(events_path) as f3:
                for line in f3:
                    try:
                        ev = json.loads(line)
                        if ev.get("event") == "ordering_computed":
                            ordering = ev["payload"].get("strategy", "input")
                            break
                    except Exception:
                        pass
        # Count aruco verifications
        aruco_verified = 0
        if events_path.exists():
            with open(events_path) as f3:
                for line in f3:
                    try:
                        ev = json.loads(line)
                        if ev.get("event") == "viewpoint_verification":
                            if ev["payload"].get("aruco_verified"):
                                aruco_verified += 1
                    except Exception:
                        pass

        runs.append({
            "dir": d.name,
            "scenario": scenario,
            "planner": planner,
            "ordering": ordering,
            "dt": dt_str,
            "success": m.get("success", False),
            "duration_s": m.get("duration_s", 0.0),
            "path_length_m": m.get("path_length_m", 0.0),
            # Effective speed = distance / time (holistic, comparable across planners)
            "effective_speed_mps": (
                m.get("path_length_m", 0.0) / m.get("duration_s", 1.0)
                if m.get("duration_s", 0.0) > 1.0 else 0.0
            ),
            "aruco_verified": aruco_verified,
            "trajectory": traj,
            "git_commit": m.get("git_commit", ""),
        })
    return runs


def group_runs(runs, scenarios, planners):
    """Group by (scenario, planner) -> list of successful runs."""
    groups = defaultdict(list)
    for r in runs:
        if r["scenario"] not in scenarios:
            continue
        if r["planner"] not in planners:
            continue
        if r["success"]:
            groups[(r["scenario"], r["planner"])].append(r)
    return groups


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


# ── plotting ─────────────────────────────────────────────────────────────────
def plot_bar_metrics(groups, scenarios, planners, out_path):
    """Three-panel bar chart: mission time, path distance, effective speed."""
    metrics_keys = ["duration_s", "path_length_m", "effective_speed_mps"]
    metrics_labels = ["Mission Time (s)", "Path Distance (m)", "Effective Speed (m/s)"]
    metrics_fmt = [".0f", ".1f", ".2f"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Performance Metrics: Baseline vs A*+2-opt vs RRT*+2-opt\n"
        "(successful runs only; error bars = ±1 std dev)",
        fontsize=12, fontweight="bold"
    )

    n_sc = len(scenarios)
    n_pl = len(planners)
    width = 0.22
    x = np.arange(n_sc)

    for ax, key, ylabel, fmt in zip(axes, metrics_keys, metrics_labels, metrics_fmt):
        for pi, planner in enumerate(planners):
            means, stds = [], []
            for sc in scenarios:
                vals = [r[key] for r in groups.get((sc, planner), [])]
                m, s = mean_std(vals)
                means.append(m)
                stds.append(s)
            offset = (pi - (n_pl - 1) / 2) * width
            bars = ax.bar(
                x + offset, means, width,
                yerr=stds, capsize=3,
                label=PLANNER_LABELS.get(planner, planner),
                color=PLANNER_COLORS.get(planner, "#888"),
                alpha=0.85, error_kw={"elinewidth": 1.2}
            )
            # Value labels on bars
            for bar, val, std_v in zip(bars, means, stds):
                if not math.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (std_v if not math.isnan(std_v) else 0) + 0.5,
                        format(val, fmt), ha="center", va="bottom", fontsize=7.5
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS.get(sc, sc) for sc in scenarios], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_paths(groups, scenarios, planners, scenario_data, out_path_template):
    """Top-down path plot per scenario."""
    for sc in scenarios:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(f"Top-Down Paths: {SCENARIO_LABELS.get(sc, sc).replace(chr(10),' ')}", fontsize=11)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

        # Draw obstacles and viewpoints from scenario_data if available
        if sc in scenario_data:
            sd = scenario_data[sc]
            for obs in sd.get("obstacles", {}).values():
                ox, oy = float(obs["x"]), float(obs["y"])
                w, d = float(obs["w"]), float(obs["d"])
                rect = mpatches.FancyBboxPatch(
                    (ox - w/2, oy - d/2), w, d,
                    boxstyle="square,pad=0", linewidth=0.5,
                    edgecolor="gray", facecolor="lightgray", alpha=0.7
                )
                ax.add_patch(rect)
            vps = sd.get("viewpoint_poses", {})
            if vps:
                xs = [float(v["x"]) for v in vps.values()]
                ys = [float(v["y"]) for v in vps.values()]
                ax.scatter(xs, ys, marker="*", s=150, color="red", zorder=5, label="Viewpoints")
            start = sd.get("drone_start_pose", {})
            if start:
                ax.scatter([float(start.get("x", 0))], [float(start.get("y", 0))],
                           marker="D", s=100, color="green", zorder=6, label="Start")

        plotted = set()
        for planner in planners:
            runs_for_sc = groups.get((sc, planner), [])
            if not runs_for_sc:
                continue
            # Plot the best (shortest time) run's trajectory
            best = min(runs_for_sc, key=lambda r: r["duration_s"])
            traj = best["trajectory"]
            if traj:
                xs = [p["x"] for p in traj]
                ys = [p["y"] for p in traj]
                label = PLANNER_LABELS.get(planner, planner).replace("\n", " ")
                ax.plot(xs, ys, color=PLANNER_COLORS.get(planner, "#888"),
                        linewidth=1.5, alpha=0.85, label=label)
                plotted.add(planner)

        ax.legend(loc="best", fontsize=8)
        out = out_path_template.replace("SCENARIO", sc)
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_boxplots(groups, scenarios, planners, out_path):
    """Box-and-whisker: distribution of mission times across 5 runs."""
    metrics = [("duration_s", "Mission Time (s)"), ("path_length_m", "Path Distance (m)")]
    fig, axes = plt.subplots(len(metrics), len(scenarios), figsize=(14, 7), squeeze=False)
    fig.suptitle("Run-to-Run Variability (5 runs per condition)", fontsize=12, fontweight="bold")

    for row, (key, ylabel) in enumerate(metrics):
        for col, sc in enumerate(scenarios):
            ax = axes[row][col]
            data, labels, colors = [], [], []
            for pl in planners:
                vals = [r[key] for r in groups.get((sc, pl), [])]
                if vals:
                    data.append(vals)
                    labels.append(PLANNER_LABELS.get(pl, pl).replace("\n", " "))
                    colors.append(PLANNER_COLORS.get(pl, "#888"))
            if data:
                bp = ax.boxplot(data, patch_artist=True, tick_tick_labels=labels, widths=0.5)
                for patch, col in zip(bp["boxes"], colors):
                    patch.set_facecolor(col)
                    patch.set_alpha(0.75)
            ax.set_title(SCENARIO_LABELS.get(sc, sc).replace("\n", " "), fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.tick_params(axis="x", labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def write_csv(groups, scenarios, planners, out_path):
    rows = []
    for sc in scenarios:
        for pl in planners:
            runs_here = groups.get((sc, pl), [])
            times = [r["duration_s"] for r in runs_here]
            dists = [r["path_length_m"] for r in runs_here]
            speeds = [r["effective_speed_mps"] for r in runs_here]
            n_ok = len(runs_here)
            rows.append({
                "Scenario": sc,
                "Planner": pl,
                "N_success": n_ok,
                "Time_mean": f"{mean_std(times)[0]:.1f}",
                "Time_std": f"{mean_std(times)[1]:.1f}",
                "Dist_mean": f"{mean_std(dists)[0]:.1f}",
                "Dist_std": f"{mean_std(dists)[1]:.1f}",
                "Speed_mean": f"{mean_std(speeds)[0]:.3f}",
                "Speed_std": f"{mean_std(speeds)[1]:.3f}",
            })
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--out_dir", default="report_figs")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print("Loading runs...")
    all_runs = load_all_runs(runs_root)
    print(f"  Found {len(all_runs)} run directories")

    # Filter: only the most recent run planner naming convention
    # (astar, rrts, baseline)
    SCENARIOS = ["scenario1", "scenario2", "scenario3", "scenario4"]
    PLANNERS  = ["baseline", "astar", "rrts"]

    groups = group_runs(all_runs, SCENARIOS, PLANNERS)

    # Print summary
    print("\n=== Run Summary ===")
    for sc in SCENARIOS:
        for pl in PLANNERS:
            runs_here = groups.get((sc, pl), [])
            times = [r["duration_s"] for r in runs_here]
            m, s = mean_std(times)
            print(f"  {sc}/{pl}: n={len(runs_here)}, time={m:.1f}±{s:.1f}s")

    # Load scenario YAML for obstacle/viewpoint positions
    scenario_data = {}
    for sc in SCENARIOS:
        yaml_path = Path("scenarios") / f"{sc}.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                scenario_data[sc] = yaml.safe_load(f)

    print("\nGenerating figures...")
    plot_bar_metrics(groups, SCENARIOS, PLANNERS, out_dir / "fig_metrics.png")
    plot_paths(groups, SCENARIOS, PLANNERS, scenario_data,
               str(out_dir / "fig_paths_SCENARIO.png"))
    plot_boxplots(groups, SCENARIOS, PLANNERS, out_dir / "fig_boxplots.png")
    write_csv(groups, SCENARIOS, PLANNERS, out_dir / "summary_table.csv")

    print(f"\n✓ All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
