#!/bin/bash
# =============================================================================
# collect_factorial.bash — unattended factorial data collection
#
# DESIGN
#   2 x 3 factorial, repeated N times, on each of 4 scenarios:
#       ordering ∈ {input, nn_2opt}     (file order vs NN+2-opt)
#       planner  ∈ {straight, astar, rrts}
#
#   This resolves the confound in the previous dataset. Previously only three
#   cells were run (straight+input, astar+nn_2opt, rrts+nn_2opt), so ordering
#   and local planner always changed together and neither effect could be
#   isolated. The full grid gives:
#       ordering effect  = straight+input      vs straight+nn_2opt   (planner fixed)
#       planner effect   = straight+input      vs astar+input        (ordering fixed)
#       interaction      = whether the planner gain depends on ordering
#
# CONTROL
#   The simulator is torn down and relaunched for EVERY run, so every run
#   starts from an identical world state at the same origin. Slower than
#   reusing a session, but it is the only way to keep the comparison clean.
#
# ROBUSTNESS
#   - Resumable: completed cells are recorded in a ledger and skipped on rerun.
#   - Readiness gate: refuses to start a mission until the sim clock is ticking
#     AND the drone is publishing pose (catches silent drone-spawn failures).
#   - Hard timeout per run; a hung run is killed and recorded, batch continues.
#   - Never leaves a simulator running between runs.
#
# USAGE
#   ./collect_factorial.bash                    # all 4 scenarios, N=3
#   ./collect_factorial.bash -n 5               # N=5 repeats
#   ./collect_factorial.bash -s "1 4"           # only scenarios 1 and 4
#   ./collect_factorial.bash -i 8000            # raise RRT* iteration budget
#   ./collect_factorial.bash --dry-run          # print the plan, run nothing
#
#   Resume after interruption: just run the same command again.
# =============================================================================

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# ── configuration ────────────────────────────────────────────────────────────
N_REPEATS=3
SCENARIOS="1 2 3 4"
RRT_ITER=3000
DRY_RUN=0
HEADLESS=0
NS="drone0"

# Headless is OFF by default and is a temporary, self-reverting change.
# The GUI setting lives in tmuxinator/aerostack2.yaml; --headless edits it for
# the duration of the batch and restores it on ANY exit (normal, error, Ctrl-C).
# If a previous run was killed hard, the stale backup is detected and restored
# at startup, so the repo is never left silently headless.
TMUXCONF="$REPO/tmuxinator/aerostack2.yaml"
TMUXBAK="$REPO/tmuxinator/.aerostack2.yaml.prebatch"

ORDERINGS=("input" "nn_2opt")
PLANNERS=("straight" "astar" "rrts")

# per-run mission parameters (identical across all cells)
SPEED=3.0
INFLATION=0.4
GRID_RES=0.5
GOTO_TIMEOUT=60.0
STUCK_TIMEOUT=20.0
DWELL=0.2
FINAL_TOL=0.75
YAW_TOL=30.0
ARUCO_TOPIC="/${NS}/sensor_measurements/hd_camera/image_raw"

# timing guards
SIM_READY_TIMEOUT=180     # max seconds to wait for a usable simulator
TEARDOWN_SETTLE=8         # seconds to let processes die after stop.bash

LEDGER="$REPO/collection_ledger.csv"
LOGDIR="$REPO/collection_logs"

# ── argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--repeats)   N_REPEATS="$2"; shift 2 ;;
    -s|--scenarios) SCENARIOS="$2";  shift 2 ;;
    -i|--rrt-iter)  RRT_ITER="$2";   shift 2 ;;
    --headless)     HEADLESS=1;      shift ;;
    --dry-run)      DRY_RUN=1;       shift ;;
    -h|--help)      sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOGDIR"
[[ -f "$LEDGER" ]] || echo "timestamp,scenario,ordering,planner,rep,status,run_dir,tour_s,coverage,aruco" > "$LEDGER"

# ── helpers ──────────────────────────────────────────────────────────────────
log() { echo "[$(date +%H:%M:%S)] $*"; }

restore_gui() {
  if [[ -f "$TMUXBAK" ]]; then
    mv -f "$TMUXBAK" "$TMUXCONF"
    log "GUI setting restored in tmuxinator/aerostack2.yaml"
  fi
}

enable_headless() {
  cp "$TMUXCONF" "$TMUXBAK"
  if grep -q 'headless:=' "$TMUXCONF"; then
    sed -i 's/headless:=false/headless:=true/' "$TMUXCONF"
  else
    # insert the argument into the launch_simulation.py invocation
    sed -i 's|\(ros2 launch as2_gazebo_assets launch_simulation.py\)|\1\n            headless:=true|' "$TMUXCONF"
  fi
  grep -q 'headless:=true' "$TMUXCONF" \
    && log "HEADLESS enabled for this batch (will auto-restore on exit)" \
    || { log "ERROR: could not enable headless; restoring and aborting"; restore_gui; exit 1; }
}

# Recover from a previously killed batch that never got to restore.
if [[ -f "$TMUXBAK" ]]; then
  echo "NOTE: found a leftover pre-batch backup of tmuxinator/aerostack2.yaml"
  echo "      (a previous batch was killed before it could restore). Restoring now."
  mv -f "$TMUXBAK" "$TMUXCONF"
fi

# Per-run mission timeout: scenario 3 has 20 viewpoints so needs a bigger budget.
timeout_for_scenario() {
  case "$1" in
    3) echo 900 ;;
    *) echo 600 ;;
  esac
}

already_done() {
  # $1=sc $2=ord $3=pl $4=rep  — a cell counts as done only if it SUCCEEDED
  grep -q ",$1,$2,$3,$4,ok," "$LEDGER" 2>/dev/null
}

teardown() {
  ./stop.bash >/dev/null 2>&1
  # stop.bash is best-effort; make sure nothing survives to poison the next run
  pkill -9 -f 'ign gazebo'      >/dev/null 2>&1
  pkill -9 -f 'gz sim'          >/dev/null 2>&1
  pkill -9 -f 'ros_gz_bridge'   >/dev/null 2>&1
  pkill -9 -f 'as2_'            >/dev/null 2>&1
  pkill -9 -f 'mission_scenario'>/dev/null 2>&1
  tmux kill-server              >/dev/null 2>&1
  sleep "$TEARDOWN_SETTLE"
}

# Wait until the simulator is genuinely usable, not merely launched.
# Two gates: the sim clock must advance, and the drone must publish a pose.
# The second gate is what catches a world that came up without a drone in it.
wait_for_sim() {
  local deadline=$((SECONDS + SIM_READY_TIMEOUT))
  local clock_ok=0 pose_ok=0

  while (( SECONDS < deadline )); do
    if (( ! clock_ok )); then
      timeout 12 ros2 topic echo /clock --once >/dev/null 2>&1 && { clock_ok=1; log "    sim clock is ticking"; }
    fi
    if (( clock_ok && ! pose_ok )); then
      timeout 12 ros2 topic echo "/${NS}/self_localization/pose" --once >/dev/null 2>&1 \
        && { pose_ok=1; log "    drone is publishing pose"; }
    fi
    (( clock_ok && pose_ok )) && return 0
    sleep 5
  done

  log "    READINESS FAILED (clock=$clock_ok pose=$pose_ok)"
  return 1
}

# Pull the outcome of the most recent run directory back out of its own logs,
# so the ledger is a real summary rather than just an exit code.
summarise_latest_run() {
  python3 - <<'PY' 2>/dev/null || echo "unknown,,,"
import json, glob, os
ds = sorted(glob.glob('runs/*/'), key=os.path.getmtime)
if not ds:
    print("unknown,,,"); raise SystemExit
d = ds[-1]
tour = ''; vp = 0; ar = 0
try:
    for line in open(os.path.join(d, 'events.jsonl')):
        e = json.loads(line); ev = e['event']
        if ev == 'mission_duration':
            tour = round(e['payload']['duration_s'], 1)
        elif ev == 'viewpoint_verification':
            vp += 1
        elif ev == 'aruco_wait_done' and e['payload'].get('verified'):
            ar += 1
except Exception:
    pass
print(f"{os.path.basename(d.rstrip('/'))},{tour},{vp},{ar}")
PY
}

# ── plan ─────────────────────────────────────────────────────────────────────
TOTAL=0; TODO=0
for sc in $SCENARIOS; do
  for ord in "${ORDERINGS[@]}"; do
    for pl in "${PLANNERS[@]}"; do
      for r in $(seq 1 "$N_REPEATS"); do
        TOTAL=$((TOTAL+1))
        already_done "$sc" "$ord" "$pl" "$r" || TODO=$((TODO+1))
      done
    done
  done
done

echo "======================================================================"
echo " Factorial collection"
echo "   scenarios : $SCENARIOS"
echo "   ordering  : ${ORDERINGS[*]}"
echo "   planner   : ${PLANNERS[*]}"
echo "   repeats   : $N_REPEATS"
echo "   RRT* iter : $RRT_ITER"
echo "   headless  : $( ((HEADLESS)) && echo 'yes (temporary, auto-restored)' || echo 'no (GUI)' )"
echo "   cells     : $TOTAL total, $TODO remaining"
echo "   estimate  : ~$(( TODO * 6 / 60 ))h$(( (TODO * 6) % 60 ))m  (~6 min/run incl. sim restart)"
echo "   ledger    : $LEDGER"
echo "======================================================================"

if (( DRY_RUN )); then
  echo "(dry run — nothing executed)"; exit 0
fi
if (( TODO == 0 )); then
  echo "Nothing to do; all cells already recorded as successful."; exit 0
fi

# The ROS/colcon setup scripts reference unbound variables (COLCON_TRACE and
# friends), which is fatal under `set -u`. Disable it just for the source.
set +u
source "$REPO/setup.bash" >/dev/null 2>&1
set -u

# EXIT covers every path out of the script, so the GUI setting is always put
# back: normal completion, error, or Ctrl-C.
trap 'restore_gui' EXIT
trap 'echo; log "interrupted — tearing down"; teardown; restore_gui; exit 130' INT TERM

(( HEADLESS )) && enable_headless

# ── execute ──────────────────────────────────────────────────────────────────
DONE=0
for sc in $SCENARIOS; do
  SCEN_FILE="scenarios/scenario${sc}.yaml"
  RUN_TIMEOUT=$(timeout_for_scenario "$sc")

  for ord in "${ORDERINGS[@]}"; do
    for pl in "${PLANNERS[@]}"; do
      for r in $(seq 1 "$N_REPEATS"); do

        if already_done "$sc" "$ord" "$pl" "$r"; then
          log "SKIP  sc${sc} ${ord}/${pl} rep${r} (already complete)"
          continue
        fi

        DONE=$((DONE+1))
        TAG="sc${sc}_${ord}_${pl}_r${r}"
        log "───────────────────────────────────────────────────────────"
        log "RUN ${DONE}/${TODO}  scenario=${sc} ordering=${ord} planner=${pl} rep=${r}"

        teardown

        log "  launching simulator"
        ./launch_as2.bash -s "$SCEN_FILE" > "$LOGDIR/${TAG}_sim.log" 2>&1 &
        SIM_PID=$!

        if ! wait_for_sim; then
          log "  ABORT: simulator never became ready"
          echo "$(date -Is),$sc,$ord,$pl,$r,sim_not_ready,,,," >> "$LEDGER"
          teardown
          continue
        fi

        log "  running mission (timeout ${RUN_TIMEOUT}s)"
        timeout "$RUN_TIMEOUT" python3 mission_scenario.py \
            "$SCEN_FILE" \
            --namespace "$NS" \
            --use_sim_time \
            --ordering "$ord" \
            --local_planner "$pl" \
            --rrt_max_iter "$RRT_ITER" \
            --speed "$SPEED" \
            --obstacle_inflation_m "$INFLATION" \
            --grid_resolution "$GRID_RES" \
            --goto_timeout_s "$GOTO_TIMEOUT" \
            --stuck_timeout_s "$STUCK_TIMEOUT" \
            --dwell_s "$DWELL" \
            --final_pos_tolerance_m "$FINAL_TOL" \
            --yaw_tolerance_deg "$YAW_TOL" \
            --return_to_start \
            --aruco_topic "$ARUCO_TOPIC" \
            > "$LOGDIR/${TAG}_mission.log" 2>&1
        RC=$?

        case $RC in
          0)   STATUS="ok" ;;
          124) STATUS="timeout" ;;
          *)   STATUS="error_rc${RC}" ;;
        esac

        SUMMARY=$(summarise_latest_run)
        echo "$(date -Is),$sc,$ord,$pl,$r,$STATUS,$SUMMARY" >> "$LEDGER"
        log "  -> $STATUS  ($SUMMARY)"

        teardown
      done
    done
  done
done

log "═══════════════════════════════════════════════════════════"
log "BATCH COMPLETE"
awk -F, 'NR>1{c[$6]++} END{for(s in c) printf "  %-14s %d\n", s, c[s]}' "$LEDGER"
log "Ledger: $LEDGER"
log "Next:   python3 analyse_factorial.py"
