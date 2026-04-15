#\!/bin/bash
# run_batch.bash - Run a single scenario with given planner/ordering N times
# Usage: ./run_batch.bash <scenario> <planner> <ordering> <n_runs>
# Example: ./run_batch.bash scenarios/scenario1.yaml astar nn_2opt 5

set -e
source "$(dirname "$0")/setup.bash"

SCENARIO=${1:-scenarios/scenario1.yaml}
PLANNER=${2:-astar}
ORDERING=${3:-nn_2opt}
N=${4:-5}
NAMESPACE=${5:-drone0}

echo "=== Batch: scenario=$SCENARIO planner=$PLANNER ordering=$ORDERING runs=$N ==="

for i in $(seq 1 $N); do
    echo ""
    echo "--- Run $i/$N ---"
    python3 "$(dirname "$0")/mission_scenario.py" \
        "$SCENARIO" \
        --namespace "$NAMESPACE" \
        --use_sim_time \
        --local_planner "$PLANNER" \
        --ordering "$ORDERING" \
        --speed 3.0 \
        --obstacle_inflation_m 0.4 \
        --goto_timeout_s 60.0 \
        --stuck_timeout_s 20.0 \
        --dwell_s 0.2 \
        --return_to_start \
        --aruco_topic "/drone0/sensor_measurements/hd_camera/image_raw"
    echo "--- Run $i/$N done ---"
    sleep 3
done

echo "=== Batch complete ==="
