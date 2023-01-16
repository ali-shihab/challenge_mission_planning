#!/bin/bash

config_path="$1"
config_path=${config_path:="simulation_config/default.json"}

PROJECT_PATH="$(pwd)"
AS2_MODELS="${PROJECT_PATH}/as2_simulation_assets/as2_ign_gazebo_assets/models"
AS2_WORLDS="${PROJECT_PATH}/as2_simulation_assets/as2_ign_gazebo_assets/worlds"

export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:$AS2_MODELS:$AS2_WORLDS

export RUN_ON_START=1

SCRIPT_PATH="${AEROSTACK2_PATH}/as2_simulation_assets/as2_ign_gazebo_assets/scripts"
ros2 launch as2_ign_gazebo_assets launch_simulation.py config_file:=${config_path}