#!/usr/bin/env python3
"""
build_report_assets.py — report figures for the 2x3 factorial dataset.

Produces, from collection_ledger.csv + runs/:
  fig_paths_scenarioN.png  trajectories, split file-order | NN+2-opt so the
                           ordering effect is visible side by side
  fig_altitude.png         altitude vs tour progress (evidences 3D planning)
  fig_reliability.png      success rate and ArUco confirmation by condition

Each path/altitude trace is the MEDIAN-tour run of its cell, so the figure
shows a representative run rather than a cherry-picked best.
"""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent
RUNS = REPO / "runs"
SCEN = REPO / "scenarios"
OUT = REPO / "report_figs"
OUT.mkdir(exist_ok=True)

ORDERINGS = ["input", "nn_2opt"]
PLANNERS = ["straight", "astar", "rrts"]
ORD_LABEL = {"input": "File order", "nn_2opt": "NN+2-opt"}
PL_LABEL = {"straight": "Straight", "astar": "A*", "rrts": "RRT*"}
# Okabe-Ito, colourblind safe; distinct dash patterns as a second channel
STYLE = {
    "straight": dict(color="#0072B2", ls="-",  lw=1.5),
    "astar":    dict(color="#009E73", ls="-",  lw=2.2),
    "rrts":     dict(color="#D55E00", ls="--", lw=1.6),
}
NVP = {"1": 10, "2": 10, "3": 20, "4": 5}
SCENARIOS = ["1", "2", "3", "4"]


def load_cells():
    """(scenario, ordering, planner) -> list of run records, flown runs only."""
    cells = defaultdict(list)
    for row in csv.DictReader(open(REPO / "collection_ledger.csv")):
        if row["status"] not in ("ok", "ok_incomplete"):
            continue
        d = RUNS / row["run_dir"]
        if not d.is_dir():
            continue
        tour = None
        vp = aruco = 0
        xs, ys, zs = [], [], []
        ep = d / "events.jsonl"
        if ep.exists():
            for line in open(ep):
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                ev = e.get("event")
                if ev == "mission_duration":
                    tour = e["payload"].get("duration_s")
                elif ev == "viewpoint_verification":
                    vp += 1
                elif ev == "aruco_wait_done" and e.get("payload", {}).get("verified"):
                    aruco += 1
        if tour is None:
            continue
        tp = d / "trajectory.csv"
        if tp.exists():
            with open(tp) as f:
                for r in csv.DictReader(f):
                    try:
                        xs.append(float(r["x"])); ys.append(float(r["y"])); zs.append(float(r["z"]))
                    except Exception:
                        pass
        n = NVP[row["scenario"]]
        cells[(row["scenario"], row["ordering"], row["planner"])].append(
            dict(tour=tour, vp=vp, aruco=aruco, n=n, complete=vp >= n,
                 xs=xs, ys=ys, zs=zs, status=row["status"])
        )
    return cells


def median_run(runs):
    """The run whose tour time is the median of its cell."""
    if not runs:
        return None
    return sorted(runs, key=lambda r: r["tour"])[len(runs) // 2]


def scenario(sc):
    return yaml.safe_load(open(SCEN / f"scenario{sc}.yaml"))


def draw_world(ax, sd):
    for _, ob in sd.get("obstacles", {}).items():
        ax.add_patch(mpatches.Rectangle(
            (ob["x"] - ob.get("w", 1) / 2, ob["y"] - ob.get("d", 1) / 2),
            ob.get("w", 1), ob.get("d", 1),
            facecolor="#b0b0b0", edgecolor="#404040", lw=1.0, alpha=0.85, zorder=1))
    for _, vp in sd.get("viewpoint_poses", {}).items():
        ax.plot(vp["x"], vp["y"], marker="*", color="#CC79A7", ms=11,
                mec="#5a2d4a", mew=0.5, ls="none", zorder=6)
    st = sd.get("drone_start_pose", {"x": 0, "y": 0})
    ax.plot(st["x"], st["y"], marker="D", color="#111111", ms=8, ls="none", zorder=7)


def main():
    cells = load_cells()

    # ---------- path figures: ordering side by side ----------
    for sc in SCENARIOS:
        sd = scenario(sc)
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
        for ax, o in zip(axes, ORDERINGS):
            draw_world(ax, sd)
            for p in PLANNERS:
                r = median_run(cells.get((sc, o, p), []))
                if not r or not r["xs"]:
                    continue
                lab = PL_LABEL[p]
                if not r["complete"]:
                    lab += f" ({r['vp']}/{r['n']} vp)"
                step = max(1, len(r["xs"]) // 700)
                ax.plot(r["xs"][::step], r["ys"][::step], alpha=0.9,
                        zorder=3 if p == "astar" else 4, label=lab, **STYLE[p])
            ax.set_title(f"{ORD_LABEL[o]} ordering", fontsize=11)
            ax.set_xlabel("x (m)")
            # 'box' rather than 'datalim': the two panels share axes so the
            # ordering comparison is at identical scale, and datalim is
            # disallowed on shared axes.
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3, zorder=0)
            ax.legend(fontsize=8, loc="best", framealpha=0.92)
        axes[0].set_ylabel("y (m)")
        n_o = len(sd.get("obstacles", {}))
        fig.suptitle(f"Scenario {sc} — {NVP[sc]} viewpoints, {n_o} obstacle"
                     f"{'' if n_o == 1 else 's'}   (median run of each cell)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(OUT / f"fig_paths_scenario{sc}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("wrote fig_paths_scenario%s.png" % sc)

    # ---------- altitude ----------
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.4), sharey="row")
    for col, sc in enumerate(SCENARIOS):
        sd = scenario(sc)
        hmax = max([ob.get("h", 1) for ob in sd.get("obstacles", {}).values()] + [0])
        for row, o in enumerate(ORDERINGS):
            ax = axes[row][col]
            if hmax > 0:
                ax.axhspan(0, hmax, color="#b0b0b0", alpha=0.4, zorder=0)
            for p in PLANNERS:
                r = median_run(cells.get((sc, o, p), []))
                if not r or not r["zs"]:
                    continue
                z = np.array(r["zs"])
                prog = np.linspace(0, 100, len(z))
                step = max(1, len(z) // 700)
                ax.plot(prog[::step], z[::step], alpha=0.9, label=PL_LABEL[p], **STYLE[p])
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            if row == 0:
                ax.set_title(f"Scenario {sc}", fontsize=10)
            if row == 1:
                ax.set_xlabel("tour progress (%)", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{ORD_LABEL[o]}\naltitude z (m)", fontsize=9)
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Altitude profiles — grey band is the obstacle height envelope "
                 "(both planners route over obstacles as well as around them)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig_altitude.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote fig_altitude.png")

    # ---------- reliability: the headline result ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.arange(len(SCENARIOS) * 2)
    xt = [f"Sc{sc}\n{ORD_LABEL[o]}" for sc in SCENARIOS for o in ORDERINGS]
    w = 0.26
    for ax, (key, ylab) in zip(axes, [("success", "Mission success rate (%)"),
                                      ("aruco", "ArUco confirmation rate (%)")]):
        for i, p in enumerate(PLANNERS):
            vals = []
            for sc in SCENARIOS:
                for o in ORDERINGS:
                    runs = cells.get((sc, o, p), [])
                    if not runs:
                        vals.append(0); continue
                    if key == "success":
                        vals.append(100 * sum(r["complete"] for r in runs) / len(runs))
                    else:
                        vals.append(100 * sum(r["aruco"] / r["n"] for r in runs) / len(runs))
            ax.bar(x + i * w, vals, w, label=PL_LABEL[p],
                   color=STYLE[p]["color"], alpha=0.9, edgecolor="white", lw=0.5)
        ax.set_xticks(x + w)
        ax.set_xticklabels(xt, fontsize=8)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_ylim(0, 108)
        ax.axhline(100, color="#666", ls=":", lw=1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Reliability by condition (n=3 per cell)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig_reliability.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote fig_reliability.png")


if __name__ == "__main__":
    main()
