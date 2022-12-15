#!/bin/bash

if [ "$#" -le 0 ]; then
	echo "usage: $0 [drone_namespace] "
	exit 1
fi

# Arguments
drone_namespace=$1
use_sim_time=true
behavior_type="position" # position or trajectory

source ./utils/launch_tools.bash

new_session $drone_namespace

new_window 'as2_ignition_platform' "ros2 launch as2_ignition_platform ignition_platform_launch.py \
    drone_id:=$drone_namespace \
    config_file:=simulation_config/default.json \
    use_sim_time:=$use_sim_time"

new_window 'as2_controller_manager' "ros2 launch as2_controller_manager controller_manager_launch.py \
    namespace:=$drone_namespace \
    use_bypass:=true \
    config:=config/$controller/controller.yaml \
    use_sim_time:=$use_sim_time"

new_window 'as2_state_estimator' "ros2 launch as2_state_estimator state_estimator_launch.py \
    namespace:=$drone_namespace \
    config:=config/state_estimator.yaml \
    base_frame:="\/$drone_namespace" \
    use_sim_time:=$use_sim_time" 

new_window 'as2_platform_behaviors' "ros2 launch as2_platform_behaviors as2_platform_behaviors_launch.py \
    namespace:=$drone_namespace \
    config_takeoff:=config/$behavior_type/takeoff_behaviour.yaml \
    config_goto:=config/$behavior_type/goto_behaviour.yaml \
    config_follow_path:=config/$behavior_type/follow_path_behaviour.yaml \
    config_land:=config/land_behaviour.yaml \
    use_sim_time:=$use_sim_time"

if [[ "$behavior_type" == "trajectory" ]]
then
    new_window 'traj_generator' "ros2 launch trajectory_generator trajectory_generator_launch.py  \
        namespace:=$drone_namespace \
        use_sim_time:=$use_sim_time"
fi