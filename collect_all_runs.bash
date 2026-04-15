#\!/bin/bash
# collect_all_runs.bash
#
# Run each scenario × planner 5 times with the tuned parameters.
# Launch the AS2 simulation FIRST in another terminal:
#   ./launch_as2.bash -s scenarios/scenarioN.yaml
# Then run this script in a second terminal.
#
# Order: scenario1 → scenario2 → scenario3 → scenario4
# Planners per scenario: baseline (straight/input), astar (nn_2opt), rrts (nn_2opt)
# 5 runs each = 60 total mission executions across all 4 scenarios

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/setup.bash"

PYTHON="python3"
MISSION="$REPO/mission_scenario.py"
ARUCO_TOPIC="/drone0/sensor_measurements/hd_camera/image_raw"
N=1
NS="drone0"

run_scenario() {
    local SC=$1
    local PLANNER=$2
    local ORDERING=$3
    local N_RUNS=$4

    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "  Scenario: $SC  |  Planner: $PLANNER  |  Order: $ORDERING"
    echo "  Runs: $N_RUNS"
    echo "╚══════════════════════════════════════════════════════╝"

    for i in $(seq 1 $N_RUNS); do
        echo ""
        echo "  ── Run $i/$N_RUNS ──"
        $PYTHON "$MISSION" \
            "scenarios/${SC}.yaml" \
            --namespace "$NS" \
            --use_sim_time \
            --local_planner  "$PLANNER" \
            --ordering       "$ORDERING" \
            --speed          3.0 \
            --obstacle_inflation_m 0.4 \
            --grid_resolution      0.5 \
            --goto_timeout_s  60.0 \
            --stuck_timeout_s 20.0 \
            --dwell_s         0.2 \
            --final_pos_tolerance_m 0.75 \
            --yaw_tolerance_deg     30.0 \
            --return_to_start \
            --aruco_topic "$ARUCO_TOPIC"
        echo "  ── Run $i done ──"
        sleep 5   # let sim settle before next run
    done
}

# ── SCENARIO 1 ────────────────────────────────────────────────────────────────
echo "=== SCENARIO 1 (10 vp, 1 obs) ==="
echo "NOTE: Make sure ./launch_as2.bash -s scenarios/scenario1.yaml is running\!"
echo "Press Enter to start..."
read -r

run_scenario scenario1 straight  input   $N   # baseline
run_scenario scenario1 astar     nn_2opt $N
run_scenario scenario1 rrts      nn_2opt $N

# ── SCENARIO 2 ────────────────────────────────────────────────────────────────
echo ""
echo "=== SCENARIO 2 (10 vp, 5 obs) ==="
echo "Restart sim: ./launch_as2.bash -s scenarios/scenario2.yaml then press Enter..."
read -r

run_scenario scenario2 straight  input   $N
run_scenario scenario2 astar     nn_2opt $N
run_scenario scenario2 rrts      nn_2opt $N

# ── SCENARIO 3 ────────────────────────────────────────────────────────────────
echo ""
echo "=== SCENARIO 3 (20 vp, 1 obs) ==="
echo "Restart sim: ./launch_as2.bash -s scenarios/scenario3.yaml then press Enter..."
read -r

run_scenario scenario3 straight  input   $N
run_scenario scenario3 astar     nn_2opt $N
run_scenario scenario3 rrts      nn_2opt $N

# ── SCENARIO 4 ────────────────────────────────────────────────────────────────
echo ""
echo "=== SCENARIO 4 (5 vp, 10 obs) ==="
echo "Restart sim: ./launch_as2.bash -s scenarios/scenario4.yaml then press Enter..."
read -r

run_scenario scenario4 straight  input   $N
run_scenario scenario4 astar     nn_2opt $N
run_scenario scenario4 rrts      nn_2opt $N

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "  ALL RUNS COMPLETE — now generate figures:            "
echo "  python3 analyse_runs.py --out_dir report_figs        "
echo "╚══════════════════════════════════════════════════════╝"
