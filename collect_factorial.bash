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
# Trajectory sampling rate. Must be pinned: path distance is integrated from
# these samples, so a rate that drifts between runs biases the comparison.
# 50 Hz gives ~6 cm resolution at the 3 m/s cruise speed.
SAMPLE_HZ=50
ARUCO_TOPIC="/${NS}/sensor_measurements/hd_camera/image_raw"

# timing guards
SIM_READY_TIMEOUT=180     # max seconds to wait for a usable simulator
TEARDOWN_SETTLE=8         # seconds to let processes die after stop.bash
SIM_SETTLE=12             # extra settle after pose appears, before arming
# Refuse to start a cell below this much free disk. Sized against the actual
# workload: with sampling pinned at SAMPLE_HZ and the logger's row cap, a run
# writes ~1-2 MB and the whole 72-run batch under 150 MB. 5 GB therefore still
# catches genuine exhaustion (or a regression in the sampler) with a wide
# margin, without false-tripping the way a 15 GB threshold did.
MIN_FREE_GB=5
MAX_ATTEMPTS=3            # in-place retries per cell before giving up on it

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

# Refuse to run two batches at once: they would fight over the simulator, the
# ledger and the tmuxinator config.
exec 9>"$REPO/.collect_factorial.lock"
if ! flock -n 9; then
  echo "ERROR: another collect_factorial.bash appears to be running."
  echo "       If you are sure it is not, remove $REPO/.collect_factorial.lock"
  exit 1
fi

# Recover from a previously killed batch. A SIGKILL cannot run the EXIT trap,
# so the config may still be headless and sim processes may still be alive.
if [[ -f "$TMUXBAK" ]]; then
  echo "NOTE: found a leftover pre-batch backup of tmuxinator/aerostack2.yaml"
  echo "      (a previous batch was killed before it could restore it). Restoring,"
  echo "      and clearing any simulator processes it left behind."
  mv -f "$TMUXBAK" "$TMUXCONF"
  ./stop.bash >/dev/null 2>&1
  pkill -9 -f 'ign gazebo' >/dev/null 2>&1
  pkill -9 -f 'ros_gz_bridge' >/dev/null 2>&1
  sleep 3
fi

# Per-run mission timeout.
#
# Sized against the WORST case, not the typical one. If every leg stalls, the
# watchdog still walks the whole tour: n_viewpoints * goto_timeout_s, plus the
# return-to-start leg and landing. For 20 viewpoints at 60 s that is already
# ~1290 s. A budget below that kills legitimately-degraded runs and records
# them as infrastructure timeouts, which both loses real data (a slow, partly
# failed run IS a result) and triggers pointless retries.
timeout_for_scenario() {
  case "$1" in
    3) echo 1500 ;;   # 20 viewpoints
    4) echo  600 ;;   #  5 viewpoints
    *) echo  900 ;;   # 10 viewpoints
  esac
}

already_done() {
  # $1=sc $2=ord $3=pl $4=rep
  # A cell is done once the mission actually FLEW. 'ok_incomplete' counts: the
  # tour ran but did not reach every viewpoint, which is a result to keep, not
  # a run to repeat. Only infrastructure failures leave a cell outstanding.
  grep -qE ",$1,$2,$3,$4,(ok|ok_incomplete)," "$LEDGER" 2>/dev/null
}

teardown() {
  ./stop.bash >/dev/null 2>&1
  # stop.bash is best-effort; make sure nothing survives to poison the next run.
  # NOTE: deliberately NOT `tmux kill-server` — that would kill every tmux
  # session on the machine including the one this script may be running in.
  # stop.bash already kills the drone/ground_station sessions by name.
  tmux kill-session -t "$NS"             >/dev/null 2>&1
  tmux kill-session -t ground_station    >/dev/null 2>&1
  pkill -9 -f 'ign gazebo'               >/dev/null 2>&1
  pkill -9 -f 'gz sim'                   >/dev/null 2>&1
  pkill -9 -f 'ros_gz_bridge'            >/dev/null 2>&1
  pkill -9 -f 'as2_'                     >/dev/null 2>&1
  pkill -9 -f 'mission_scenario'         >/dev/null 2>&1
  sleep "$TEARDOWN_SETTLE"
}

# Refuse to start another cell if the disk is nearly full. A prior bug in this
# project produced multi-GB trajectory files; unattended, that fills the VM and
# every later cell fails for reasons that look unrelated.
check_disk() {
  local free_gb=$(( $(df -Pk "$REPO" | awk 'NR==2{print $4}') / 1024 / 1024 ))
  if (( free_gb < MIN_FREE_GB )); then
    log "ABORT: only ${free_gb} GB free (need ${MIN_FREE_GB}) — stopping before data is corrupted"
    teardown; exit 2
  fi
}

# Wait until the simulator is genuinely usable, not merely launched.
# Two gates: the sim clock must advance, and the drone must publish a pose.
# The second gate is what catches a world that came up without a drone in it.
wait_for_sim() {
  local deadline=$((SECONDS + SIM_READY_TIMEOUT))
  local clock_ok=0 pose_ok=0

  while (( SECONDS < deadline )); do
    if (( ! clock_ok )); then
      # -k is mandatory on every `timeout` wrapping a ROS CLI call: rclpy
      # ignores SIGTERM, so a bare `timeout` waits on the child forever. One
      # such probe blocked this batch for 71 minutes.
      timeout -k 5 12 ros2 topic echo /clock --once >/dev/null 2>&1 && { clock_ok=1; log "    sim clock is ticking"; }
    fi
    if (( clock_ok && ! pose_ok )); then
      timeout -k 5 12 ros2 topic echo "/${NS}/self_localization/pose" --once >/dev/null 2>&1 \
        && { pose_ok=1; log "    drone is publishing pose"; }
    fi
    if (( clock_ok && pose_ok )); then
      # Pose publishing does not mean the platform will accept arm/offboard.
      # A failed arm here costs a whole run, so pay a few seconds to let the
      # platform and behaviour servers finish coming up.
      log "    settling ${SIM_SETTLE}s before starting the mission"
      sleep "$SIM_SETTLE"
      return 0
    fi
    sleep 5
  done

  log "    READINESS FAILED (clock=$clock_ok pose=$pose_ok)"
  return 1
}

# Summarise ONE named run directory. The directory is passed in explicitly,
# identified by diffing runs/ before and after the mission. Picking "newest by
# mtime" instead would silently attribute the PREVIOUS cell's results to this
# one whenever a mission exits 0 without creating a directory, which quietly
# duplicates a run into an unrelated factorial cell.
summarise_run() {
  RUN_DIR="$1" python3 - <<'PY' 2>/dev/null || echo "unknown,,,"
import json, os
d = os.environ['RUN_DIR']
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

        check_disk

        DONE=$((DONE+1))
        log "───────────────────────────────────────────────────────────"
        log "RUN ${DONE}/${TODO}  scenario=${sc} ordering=${ord} planner=${pl} rep=${r}"

        # A timeout or a simulator that never came up is an infrastructure
        # artefact (host load, wedged Gazebo), not a result for this planner.
        # Retry the cell in place so the design keeps its full N, instead of
        # moving on and silently leaving a thin cell behind.
        ATTEMPT=1
        STATUS=""   # referenced in the retry banner; must exist under `set -u`
        while (( ATTEMPT <= MAX_ATTEMPTS )); do
        TAG="sc${sc}_${ord}_${pl}_r${r}_a${ATTEMPT}"
        (( ATTEMPT > 1 )) && log "  RETRY ${ATTEMPT}/${MAX_ATTEMPTS} (previous attempt: $STATUS)"

        teardown

        log "  launching simulator"
        ./launch_as2.bash -s "$SCEN_FILE" > "$LOGDIR/${TAG}_sim.log" 2>&1 &
        SIM_PID=$!

        if ! wait_for_sim; then
          log "  simulator never became ready"
          STATUS="sim_not_ready"
          echo "$(date -Is),$sc,$ord,$pl,$r,$STATUS,,,," >> "$LEDGER"
          teardown
          ATTEMPT=$((ATTEMPT+1))
          continue
        fi

        log "  running mission (timeout ${RUN_TIMEOUT}s)"
        BEFORE_DIRS=$(ls -1d runs/*/ 2>/dev/null | sort)
        # -k is essential: plain `timeout` only sends SIGTERM, which rclpy
        # swallows, so a wedged mission survives its own timeout and blocks the
        # batch indefinitely. SIGKILL 60 s later guarantees the cell ends.
        timeout -k 60 "$RUN_TIMEOUT" python3 mission_scenario.py \
            "$SCEN_FILE" \
            --namespace "$NS" \
            --use_sim_time \
            --ordering "$ord" \
            --local_planner "$pl" \
            --rrt_max_iter "$RRT_ITER" \
            --sample_hz "$SAMPLE_HZ" \
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

        AFTER_DIRS=$(ls -1d runs/*/ 2>/dev/null | sort)
        NEW_DIR=$(comm -13 <(echo "$BEFORE_DIRS") <(echo "$AFTER_DIRS") | head -n1)

        # Distinguish MISSION outcome from INFRASTRUCTURE outcome.
        #
        # mission_scenario.py exits non-zero when the tour was incomplete, which
        # is a legitimate RESULT (the baseline genuinely cannot reach every
        # viewpoint in a dense obstacle field) and must be kept, not retried.
        # The thing that warrants a retry is a run that never flew at all.
        #
        # The discriminator is therefore the presence of a mission_duration
        # event, not the exit status: if the tour was timed, the mission ran.
        case $RC in
          124) STATUS="timeout" ;;
          0)   STATUS="ok" ;;
          *)   STATUS="ok_incomplete" ;;   # flew, but did not reach every viewpoint
        esac

        # A zero exit code is not proof the mission ran. If no new run directory
        # appeared, downgrade the status so the cell is retried on resume rather
        # than silently recorded against another run's data.
        if [[ -z "$NEW_DIR" ]]; then
          [[ "$STATUS" == "ok" ]] && STATUS="ok_no_rundir"
          SUMMARY="none,,,"
          log "  WARNING: mission produced no run directory"
        else
          SUMMARY=$(summarise_run "$NEW_DIR")
          # No mission_duration means the tour never started (a failed
          # arm/offboard/takeoff falls straight through to disarm). That is an
          # infrastructure failure and must be retried, whatever the exit code.
          if [[ -z "$(cut -d, -f2 <<<"$SUMMARY")" ]]; then
            [[ "$STATUS" != "timeout" ]] && STATUS="no_mission"
            log "  WARNING: no mission_duration — the mission never flew"
          fi
        fi

        echo "$(date -Is),$sc,$ord,$pl,$r,$STATUS,$SUMMARY" >> "$LEDGER"
        log "  -> $STATUS  ($SUMMARY)"

        teardown

        # A flown mission ends the cell, complete or not. Anything else is an
        # infrastructure failure and gets another attempt.
        [[ "$STATUS" == "ok" || "$STATUS" == "ok_incomplete" ]] && break
        ATTEMPT=$((ATTEMPT+1))
        done

        if [[ "$STATUS" != "ok" && "$STATUS" != "ok_incomplete" ]]; then
          log "  GAVE UP on this cell after ${MAX_ATTEMPTS} attempts (last: $STATUS)"
          log "  -> rerun the script later to retry it; the cell is not marked done"
        fi
      done
    done
  done
done

log "═══════════════════════════════════════════════════════════"
log "BATCH COMPLETE"
awk -F, 'NR>1{c[$6]++} END{for(s in c) printf "  %-14s %d\n", s, c[s]}' "$LEDGER"
log "Ledger: $LEDGER"
log "Next:   python3 analyse_factorial.py"
