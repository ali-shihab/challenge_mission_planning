#!/usr/bin/env python3
"""
analyse_factorial.py — analysis for the 2x3 factorial collection.

Reads the ledger written by collect_factorial.bash, pulls each run's measured
outcome from its own logs, and produces:

  1. summary_factorial.csv  — mean +/- std for every (scenario, ordering, planner)
  2. effects_factorial.csv  — the decomposed main effects
  3. fig_factorial.png      — grouped bars with error bars
  4. fig_effects.png        — ordering effect vs planner effect, side by side

The point of the factorial is attribution. The previous dataset varied ordering
and local planner together, so neither effect could be isolated. Here:

    ordering effect = (planner fixed) nn_2opt - input
    planner  effect = (ordering fixed) astar - straight, rrts - straight

Usage:
    python3 analyse_factorial.py                 # uses ./collection_ledger.csv
    python3 analyse_factorial.py --out report_figs
"""

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDERINGS = ["input", "nn_2opt"]
PLANNERS = ["straight", "astar", "rrts"]
ORD_LABEL = {"input": "file order", "nn_2opt": "NN+2-opt"}
PL_LABEL = {"straight": "Straight", "astar": "A*", "rrts": "RRT*"}
PL_COLOUR = {"straight": "#0072B2", "astar": "#009E73", "rrts": "#D55E00"}
NVP = {"1": 10, "2": 10, "3": 20, "4": 5}


def load_run(run_dir: Path):
    """Extract measured outcome from a single run directory."""
    if not run_dir.is_dir():
        return None
    out = {"tour": None, "total": None, "path": None, "vp": 0, "aruco": 0}

    metrics = run_dir / "metrics.json"
    if metrics.exists():
        m = json.load(open(metrics))
        out["total"] = m.get("duration_s")
        out["path"] = m.get("path_length_m")

    events = run_dir / "events.jsonl"
    if events.exists():
        for line in open(events):
            try:
                e = json.loads(line)
            except Exception:
                continue
            ev = e.get("event")
            if ev == "mission_duration":
                out["tour"] = e["payload"].get("duration_s")
            elif ev == "viewpoint_verification":
                out["vp"] += 1
            elif ev == "aruco_wait_done" and e.get("payload", {}).get("verified"):
                out["aruco"] += 1
    return out if out["tour"] is not None else None


def load_ledger(ledger: Path, runs_root: Path):
    """
    cells[(scenario, ordering, planner)] -> list of per-run dicts.

    Only runs that completed the harness are analysed. A timeout or harness
    error is an INFRASTRUCTURE artefact (host load, a wedged simulator), not a
    property of the planner under test, so counting it as a mission failure
    would understate the planner's success rate. The collector retries such
    cells in place, so a discarded attempt is replaced by a real one rather
    than leaving a gap. Discards are reported so they are never silent.
    """
    cells = defaultdict(list)
    discarded = defaultdict(list)
    if not ledger.exists():
        raise SystemExit(f"No ledger at {ledger}. Run collect_factorial.bash first.")

    for row in csv.DictReader(open(ledger)):
        sc = row.get("scenario")
        if not sc:
            continue
        key = (sc, row["ordering"], row["planner"])
        status = row.get("status", "")

        # 'ok_incomplete' is a flown mission that did not reach every viewpoint.
        # That is a result — it is precisely how the baseline fails in cluttered
        # scenarios — so it is analysed, not discarded. Only runs that never
        # flew (timeout, wedged startup) are dropped as infrastructure noise.
        if status not in ("ok", "ok_incomplete") or not row.get("run_dir"):
            discarded[key].append(status or "no_data")
            continue

        data = load_run(runs_root / row["run_dir"])
        if data is None:
            discarded[key].append("unreadable")
            continue

        data["usable"] = True
        data["n_vp"] = NVP.get(sc, 0)
        data["complete"] = (data["vp"] >= data["n_vp"])
        cells[key].append(data)

    if discarded:
        total = sum(len(v) for v in discarded.values())
        print(f"\nNOTE: {total} infrastructure-failed attempt(s) discarded "
              f"(not counted as mission failures):")
        for k in sorted(discarded):
            print(f"  Sc{k[0]} {k[1]}/{k[2]}: {', '.join(discarded[k])}")
    return cells


def agg(vals):
    """mean, sample std (0 when n<2), n."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def umean(runs, key):
    """Mean of the given key over analysable runs only; None if there are none."""
    vals = [r[key] for r in runs if r.get("usable") and r.get(key) is not None]
    return st.mean(vals) if vals else None


def srate(runs):
    """Success rate over ALL attempts, failures included."""
    return st.mean([1.0 if r["complete"] else 0.0 for r in runs]) * 100 if runs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="collection_ledger.csv")
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--out", default="report_figs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    cells = load_ledger(Path(args.ledger), Path(args.runs_root))
    if not cells:
        raise SystemExit("Ledger contains no successful runs yet.")

    scenarios = sorted({k[0] for k in cells})

    # ---------------- summary table ----------------
    rows = []
    print(f"\n{'Sc':<4}{'Ordering':<12}{'Planner':<10}{'n':<4}"
          f"{'Tour s':>16}{'Path m':>16}{'Cover':>8}{'ArUco':>8}{'Succ':>7}")
    print("-" * 92)
    for sc in scenarios:
        for o in ORDERINGS:
            for p in PLANNERS:
                attempts = cells.get((sc, o, p), [])
                if not attempts:
                    continue
                usable = attempts
                # timing/distance
                tm, tsd, n = agg([r["tour"] for r in usable])
                pm, psd, _ = agg([r["path"] for r in usable])
                totm, totsd, _ = agg([r["total"] for r in usable])
                nvp = attempts[0]["n_vp"]
                # coverage / aruco / success over the valid runs
                cov = st.mean([r["vp"] / nvp for r in attempts]) * 100
                aru = st.mean([r["aruco"] / nvp for r in attempts]) * 100
                succ = st.mean([1.0 if r["complete"] else 0.0 for r in attempts]) * 100
                rows.append(dict(scenario=sc, ordering=o, planner=p,
                                 n=n,
                                 tour_mean=None if tm is None else round(tm, 1),
                                 tour_std=None if tsd is None else round(tsd, 1),
                                 total_mean=None if totm is None else round(totm, 1),
                                 total_std=None if totsd is None else round(totsd, 1),
                                 path_mean=None if pm is None else round(pm, 1),
                                 path_std=None if psd is None else round(psd, 1),
                                 coverage_pct=round(cov, 1), aruco_pct=round(aru, 1),
                                 success_rate_pct=round(succ, 1)))
                ts = "        n/a     " if tm is None else f"{tm:>10.1f}+/-{tsd:<4.1f}"
                ps = "        n/a     " if pm is None else f"{pm:>10.1f}+/-{psd:<4.1f}"
                print(f"{sc:<4}{ORD_LABEL[o]:<12}{PL_LABEL[p]:<10}{n:<4}"
                      f"{ts}{ps}{cov:>7.0f}%{aru:>7.0f}%{succ:>6.0f}%")

    with open(out / "summary_factorial.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------- decomposed effects ----------------
    # This is the whole reason for running the factorial.
    eff = []
    print(f"\n{'='*90}\nMAIN EFFECTS (negative = improvement)\n{'='*90}")

    print("\nOrdering effect: NN+2-opt minus file order, local planner held constant")
    for sc in scenarios:
        for p in PLANNERS:
            a = cells.get((sc, "input", p), [])
            b = cells.get((sc, "nn_2opt", p), [])
            if not a or not b:
                continue
            at_, bt_ = umean(a, "tour"), umean(b, "tour")
            ap_, bp_ = umean(a, "path"), umean(b, "path")
            if None in (at_, bt_, ap_, bp_):
                continue
            d_t = bt_ - at_
            d_p = bp_ - ap_
            pct = 100 * d_p / ap_ if ap_ else 0
            eff.append(dict(effect="ordering", scenario=sc, held_fixed=f"planner={p}",
                            d_tour_s=round(d_t, 1), d_path_m=round(d_p, 1),
                            d_path_pct=round(pct, 1)))
            print(f"  Sc{sc} {PL_LABEL[p]:<9} tour {d_t:+7.1f}s   path {d_p:+7.1f}m ({pct:+.1f}%)")

    print("\nPlanner effect: A*/RRT* minus straight-line, ordering held constant")
    for sc in scenarios:
        for o in ORDERINGS:
            base = cells.get((sc, o, "straight"), [])
            if not base:
                continue
            bt = umean(base, "tour"); bp = umean(base, "path"); bs = srate(base)
            if None in (bt, bp, bs):
                continue
            for p in ["astar", "rrts"]:
                c = cells.get((sc, o, p), [])
                if not c:
                    continue
                ct_, cp_ = umean(c, "tour"), umean(c, "path")
                if None in (ct_, cp_):
                    continue
                d_t = ct_ - bt
                d_p = cp_ - bp
                d_s = srate(c) - bs
                eff.append(dict(effect="planner", scenario=sc,
                                held_fixed=f"ordering={o}", comparison=f"{p}-straight",
                                d_tour_s=round(d_t, 1), d_path_m=round(d_p, 1),
                                d_success_pct=round(d_s, 1)))
                print(f"  Sc{sc} {ORD_LABEL[o]:<11} {PL_LABEL[p]:<6} "
                      f"tour {d_t:+7.1f}s   path {d_p:+7.1f}m   success {d_s:+5.0f}pp")

    with open(out / "effects_factorial.csv", "w", newline="") as f:
        keys = sorted({k for e in eff for k in e})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(eff)

    # ---------------- figures ----------------
    metrics = [("tour", "Tour time (s)"), ("path", "Path distance (m)")]
    fig, axes = plt.subplots(len(metrics), len(scenarios),
                             figsize=(4.0 * len(scenarios), 3.6 * len(metrics)),
                             squeeze=False)
    x = np.arange(len(ORDERINGS))
    w = 0.26
    for r, (key, ylab) in enumerate(metrics):
        for c, sc in enumerate(scenarios):
            ax = axes[r][c]
            for i, p in enumerate(PLANNERS):
                means, errs = [], []
                for o in ORDERINGS:
                    m, s, _ = agg([q[key] for q in cells.get((sc, o, p), [])
                                   if q.get("usable")])
                    means.append(m or 0)
                    errs.append(s or 0)
                ax.bar(x + i * w, means, w, yerr=errs, capsize=3,
                       label=PL_LABEL[p], color=PL_COLOUR[p], alpha=0.9,
                       edgecolor="white", lw=0.6)
            ax.set_xticks(x + w)
            ax.set_xticklabels([ORD_LABEL[o] for o in ORDERINGS], fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=10)
            if r == 0:
                ax.set_title(f"Scenario {sc}", fontsize=11)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Ordering x local planner, mean +/- s.d. over repeats", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "fig_factorial.png", dpi=150, bbox_inches="tight")
    plt.close()

    # effects figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ord_eff = [e for e in eff if e["effect"] == "ordering"]
    if ord_eff:
        labs = [f"Sc{e['scenario']}\n{e['held_fixed'].split('=')[1]}" for e in ord_eff]
        vals = [e["d_path_pct"] for e in ord_eff]
        axes[0].bar(range(len(vals)), vals,
                    color=["#009E73" if v < 0 else "#D55E00" for v in vals], alpha=0.9)
        axes[0].set_xticks(range(len(labs)))
        axes[0].set_xticklabels(labs, fontsize=8)
        axes[0].set_ylabel("path change (%)", fontsize=10)
        axes[0].set_title("Ordering effect (NN+2-opt vs file order)\nplanner held constant", fontsize=10)
        axes[0].axhline(0, color="k", lw=0.8)
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].set_axisbelow(True)

    pl_eff = [e for e in eff if e["effect"] == "planner"]
    if pl_eff:
        labs = [f"Sc{e['scenario']}\n{e['comparison'].split('-')[0]}\n{e['held_fixed'].split('=')[1]}"
                for e in pl_eff]
        vals = [e["d_success_pct"] for e in pl_eff]
        axes[1].bar(range(len(vals)), vals,
                    color=["#009E73" if v >= 0 else "#D55E00" for v in vals], alpha=0.9)
        axes[1].set_xticks(range(len(labs)))
        axes[1].set_xticklabels(labs, fontsize=7)
        axes[1].set_ylabel("success rate change (pp)", fontsize=10)
        axes[1].set_title("Planner effect vs straight-line\nordering held constant", fontsize=10)
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out / "fig_effects.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---------------- coverage warnings ----------------
    thin = [(k, sum(1 for x in v if x.get("usable"))) for k, v in cells.items()
            if sum(1 for x in v if x.get("usable")) < 2]
    print(f"\n{'='*90}")
    print(f"Wrote {out/'summary_factorial.csv'}, {out/'effects_factorial.csv'},")
    print(f"      {out/'fig_factorial.png'}, {out/'fig_effects.png'}")
    if thin:
        print(f"\nWARNING: {len(thin)} cell(s) have n<2, so their std is 0 and "
              f"differences there are not interpretable:")
        for (sc, o, p), n in sorted(thin):
            print(f"  Sc{sc} {o}/{p}: n={n}")
    missing = [(sc, o, p) for sc in scenarios for o in ORDERINGS for p in PLANNERS
               if (sc, o, p) not in cells]
    if missing:
        print(f"\nWARNING: {len(missing)} cell(s) have no successful runs at all:")
        for sc, o, p in missing:
            print(f"  Sc{sc} {o}/{p}")


if __name__ == "__main__":
    main()
