#!/usr/bin/env python3
"""
build_report_assets.py — report figures for the 2x3 factorial dataset.

Figures are generated at the size they are PRINTED at (a \textwidth figure on
A4 with 2 cm margins is 17 cm = 6.7 in). Generating wide and scaling down in
LaTeX is what made the earlier versions unreadable: an 8 pt label on a 17 in
canvas becomes ~3 pt on the page.

Outputs:
  fig_paths_scenarioN.png  trajectories, file-order | NN+2-opt side by side
  fig_altitude.png         altitude vs tour progress
  fig_reliability.png      success and ArUco confirmation by condition
  fig_effects.png          decomposed ordering and planner main effects
"""
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Printed width of a \textwidth figure: A4 (21 cm) - 2*2 cm margins = 17 cm.
TEXTWIDTH_IN = 6.7

matplotlib.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "lines.linewidth": 1.3,
    "savefig.dpi": 200,
})

REPO = Path(__file__).resolve().parent
RUNS = REPO / "runs"
SCEN = REPO / "scenarios"
OUT = REPO / "report_figs"
OUT.mkdir(exist_ok=True)

ORDERINGS = ["input", "nn_2opt"]
PLANNERS = ["straight", "astar", "rrts"]
ORD_LABEL = {"input": "File order", "nn_2opt": "NN+2-opt"}
PL_LABEL = {"straight": "Straight", "astar": "A*", "rrts": "RRT*"}
STYLE = {
    "straight": dict(color="#0072B2", ls="-",  lw=1.2),
    "astar":    dict(color="#009E73", ls="-",  lw=1.9),
    "rrts":     dict(color="#D55E00", ls="--", lw=1.4),
}
NVP = {"1": 10, "2": 10, "3": 20, "4": 5}
SCENARIOS = ["1", "2", "3", "4"]


def load_cells():
    cells = defaultdict(list)
    for row in csv.DictReader(open(REPO / "collection_ledger.csv")):
        if row["status"] not in ("ok", "ok_incomplete"):
            continue
        d = RUNS / row["run_dir"]
        if not d.is_dir():
            continue
        tour = None
        vp = aruco = 0
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
        path = None
        mp = d / "metrics.json"
        if mp.exists():
            path = json.load(open(mp)).get("path_length_m")
        xs, ys, zs = [], [], []
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
            dict(tour=tour, path=path, vp=vp, aruco=aruco, n=n,
                 complete=vp >= n, xs=xs, ys=ys, zs=zs))
    return cells


def median_run(runs):
    return sorted(runs, key=lambda r: r["tour"])[len(runs) // 2] if runs else None


def scenario(sc):
    return yaml.safe_load(open(SCEN / f"scenario{sc}.yaml"))


def draw_world(ax, sd):
    for _, ob in sd.get("obstacles", {}).items():
        ax.add_patch(mpatches.Rectangle(
            (ob["x"] - ob.get("w", 1) / 2, ob["y"] - ob.get("d", 1) / 2),
            ob.get("w", 1), ob.get("d", 1),
            facecolor="#b8b8b8", edgecolor="#404040", lw=0.8, alpha=0.9, zorder=1))
    for _, vp in sd.get("viewpoint_poses", {}).items():
        ax.plot(vp["x"], vp["y"], marker="*", color="#CC79A7", ms=9,
                mec="#5a2d4a", mew=0.4, ls="none", zorder=6)
    st_ = sd.get("drone_start_pose", {"x": 0, "y": 0})
    ax.plot(st_["x"], st_["y"], marker="D", color="#111111", ms=6, ls="none", zorder=7)


def main():
    cells = load_cells()

    # ---------- paths ----------
    for sc in SCENARIOS:
        sd = scenario(sc)
        fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 3.6), sharex=True, sharey=True)
        for ax, o in zip(axes, ORDERINGS):
            draw_world(ax, sd)
            for p in PLANNERS:
                r = median_run(cells.get((sc, o, p), []))
                if not r or not r["xs"]:
                    continue
                lab = PL_LABEL[p] + ("" if r["complete"] else f" ({r['vp']}/{r['n']})")
                step = max(1, len(r["xs"]) // 600)
                ax.plot(r["xs"][::step], r["ys"][::step], alpha=0.9,
                        zorder=3 if p == "astar" else 4, label=lab, **STYLE[p])
            ax.set_title(f"{ORD_LABEL[o]} ordering")
            ax.set_xlabel("x (m)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3, zorder=0, lw=0.5)
            ax.legend(fontsize=7, loc="upper left", framealpha=0.9, handlelength=1.6)
        axes[0].set_ylabel("y (m)")
        n_o = len(sd.get("obstacles", {}))
        fig.suptitle(f"Scenario {sc} — {NVP[sc]} viewpoints, {n_o} obstacle"
                     f"{'' if n_o == 1 else 's'}", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUT / f"fig_paths_scenario{sc}.png", bbox_inches="tight")
        plt.close()
        print("paths", sc)

    # ---------- altitude: 4 scenarios x 2 orderings, taller so labels breathe ----------
    fig, axes = plt.subplots(2, 4, figsize=(TEXTWIDTH_IN, 4.0), sharey="row", sharex=True)
    for col, sc in enumerate(SCENARIOS):
        sd = scenario(sc)
        hmax = max([ob.get("h", 1) for ob in sd.get("obstacles", {}).values()] + [0])
        for row, o in enumerate(ORDERINGS):
            ax = axes[row][col]
            if hmax > 0:
                ax.axhspan(0, hmax, color="#c8c8c8", alpha=0.55, zorder=0)
            for p in PLANNERS:
                r = median_run(cells.get((sc, o, p), []))
                if not r or not r["zs"]:
                    continue
                z = np.array(r["zs"])
                prog = np.linspace(0, 100, len(z))
                step = max(1, len(z) // 400)
                ax.plot(prog[::step], z[::step], alpha=0.9, label=PL_LABEL[p], **STYLE[p])
            ax.grid(True, alpha=0.3, lw=0.5)
            ax.set_xticks([0, 50, 100])
            if row == 0:
                ax.set_title(f"Scenario {sc}", pad=4)
            if row == 1:
                ax.set_xlabel("tour progress (%)")
            if col == 0:
                ax.set_ylabel(f"{ORD_LABEL[o]}\naltitude (m)")
    axes[0][0].legend(fontsize=7, loc="upper left", handlelength=1.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_altitude.png", bbox_inches="tight")
    plt.close()
    print("altitude")

    # ---------- reliability ----------
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 3.1), sharey=True)
    x = np.arange(len(SCENARIOS) * 2)
    xt = [f"Sc{sc}\n{'File' if o=='input' else '2-opt'}" for sc in SCENARIOS for o in ORDERINGS]
    w = 0.27
    for ax, (key, ylab) in zip(axes, [("success", "Mission success rate (%)"),
                                      ("aruco", "ArUco confirmation (%)")]):
        for i, p in enumerate(PLANNERS):
            vals = []
            for sc in SCENARIOS:
                for o in ORDERINGS:
                    runs = cells.get((sc, o, p), [])
                    if not runs:
                        vals.append(0); continue
                    vals.append(100 * sum(r["complete"] for r in runs) / len(runs) if key == "success"
                                else 100 * sum(r["aruco"] / r["n"] for r in runs) / len(runs))
            ax.bar(x + i * w, vals, w, label=PL_LABEL[p],
                   color=STYLE[p]["color"], alpha=0.92, edgecolor="white", lw=0.4)
        ax.set_xticks(x + w)
        ax.set_xticklabels(xt)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 112)
        ax.axhline(100, color="#666", ls=":", lw=0.8)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=7.5, loc="lower left", ncol=3, columnspacing=0.8,
                   handlelength=1.2, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "fig_reliability.png", bbox_inches="tight")
    plt.close()
    print("reliability")

    # ---------- main effects ----------
    def umean(runs, k):
        v = [r[k] for r in runs if r.get(k) is not None]
        return st.mean(v) if v else None

    def ok(sc, o, p):
        r = cells.get((sc, o, p), [])
        return bool(r) and all(x["complete"] for x in r)

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 3.0))

    # ordering effect on path %, planner as series, scenario on x
    xs_ = np.arange(len(SCENARIOS))
    for i, p in enumerate(PLANNERS):
        vals = []
        for sc in SCENARIOS:
            if ok(sc, "input", p) and ok(sc, "nn_2opt", p):
                a = umean(cells[(sc, "input", p)], "path")
                b = umean(cells[(sc, "nn_2opt", p)], "path")
                vals.append(100 * (b - a) / a)
            else:
                vals.append(np.nan)
        axes[0].bar(xs_ + i * w, vals, w, label=PL_LABEL[p],
                    color=STYLE[p]["color"], alpha=0.92, edgecolor="white", lw=0.4)
    axes[0].set_xticks(xs_ + w)
    axes[0].set_xticklabels([f"Sc{s}" for s in SCENARIOS])
    axes[0].set_ylabel("path change (%)")
    axes[0].set_title("Ordering effect\n(NN+2-opt vs file order, planner fixed)", pad=4)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].grid(axis="y", alpha=0.3, lw=0.5)
    axes[0].set_axisbelow(True)
    axes[0].legend(fontsize=7.5, loc="lower left", handlelength=1.2)
    # Bars go downward here, so annotate at the top where nothing is drawn.
    axes[0].text(0.5, -0.30, "missing bar = comparison invalid (baseline never completed)",
                 transform=axes[0].transAxes, ha="center", fontsize=6.5, color="#555")

    # planner effect on success rate, ordering as series
    for i, p in enumerate(["astar", "rrts"]):
        for j, o in enumerate(ORDERINGS):
            vals = []
            for sc in SCENARIOS:
                base = cells.get((sc, o, "straight"), [])
                cur = cells.get((sc, o, p), [])
                if not base or not cur:
                    vals.append(np.nan); continue
                bs = 100 * sum(r["complete"] for r in base) / len(base)
                cs = 100 * sum(r["complete"] for r in cur) / len(cur)
                vals.append(cs - bs)
            off = (i * 2 + j) * 0.21
            axes[1].bar(xs_ + off, vals, 0.20,
                        label=f"{PL_LABEL[p]}, {'File' if o=='input' else '2-opt'}",
                        color=STYLE[p]["color"], alpha=0.95 if o == "input" else 0.55,
                        edgecolor="white", lw=0.4)
    axes[1].set_xticks(xs_ + 0.315)
    axes[1].set_xticklabels([f"Sc{s}" for s in SCENARIOS])
    axes[1].set_ylabel("success rate change (pp)")
    axes[1].set_title("Planner effect vs straight-line\n(ordering fixed)", pad=4)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].grid(axis="y", alpha=0.3, lw=0.5)
    axes[1].set_axisbelow(True)
    axes[1].legend(fontsize=6.8, loc="upper left", ncol=2, columnspacing=0.7, handlelength=1.1)

    fig.tight_layout()
    fig.savefig(OUT / "fig_effects.png", bbox_inches="tight")
    plt.close()
    print("effects")


if __name__ == "__main__":
    main()
