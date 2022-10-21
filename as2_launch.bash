#!/bin/bash

if [ "$#" -le 0 ]; then
	echo "usage: $0 [drone_namespace] "
	exit 1
fi

# Arguments
drone_namespace=$1

source ./launch_tools.bash

new_session $drone_namespace

new_window 'controller_manager' "ros2 launch controller_manager controller_manager_launch.py \
    drone_id:=$drone_namespace \
    use_bypass:=true \
    config:=config/controller.yaml \
    base_frame_id:=base_link \
    use_sim_time:=true"

new_window 'state_estimator' "ros2 launch basic_state_estimator basic_state_estimator_launch.py \
    namespace:=$drone_namespace \
    odom_only:=false \
    ground_truth:=true \
    base_frame:="\/$drone_namespace" \
    use_sim_time:=true"

new_window 'ignition_interface' "ros2 launch ignition_platform ignition_platform_launch.py \
    drone_id:=$drone_namespace \
    config_file:=sim_config/test.json \
    use_sim_time:=true"

new_window 'traj_generator' "ros2 launch trajectory_generator trajectory_generator_launch.py  \
    drone_id:=$drone_namespace \
    use_sim_time:=true"

new_window 'basic_behaviours' "ros2 launch as2_basic_behaviours all_basic_behaviours_launch.py \
    drone_id:=$drone_namespace \
    config_follow_path:=config/follow_path_behaviour.yaml \
    config_takeoff:=config/takeoff_behaviour.yaml \
    config_land:=config/land_behaviour.yaml \
    config_goto:=config/goto_behaviour.yaml \
    use_sim_time:=true"

new_window 'gps_translator' "ros2 launch gps_utils gps_translator_launch.py \
    namespace:=$drone_namespace"

