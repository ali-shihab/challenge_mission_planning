#!/bin/bash

config_path="$1"
config_path=${config_path:="sim_config/test.json"}

PROJECT_PATH="$(pwd)"
AS2_MODELS="${PROJECT_PATH}/assets/ignition/models"
AS2_WORLDS="${PROJECT_PATH}/assets/ignition/worlds"

export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:$AS2_MODELS:$AS2_WORLDS

export RUN_ON_START=1

SCRIPT_PATH="${AEROSTACK2_PATH}/simulation/ignition_assets/scripts"
# $SCRIPT_PATH/run_ign.sh ${config_path}
ros2 launch ignition_assets launch_simulation.py config_file:=${config_path}