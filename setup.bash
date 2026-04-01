#!/bin/bash

# Source WORKSPACE (mission_planning_ws) - this sets GZ_SIM_RESOURCE_PATH correctly
# including ground_plane and other built-in models from aerostack2/mission_planning_ws
export WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
source $WORKSPACE/install/setup.bash
echo "Sourced WORKSPACE at $WORKSPACE"

# APPEND scenario-specific model paths (obstacle_1, aruco markers) to existing path
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GZ_SIM_RESOURCE_PATH="${_REPO}/config_sim/gazebo/models:${_REPO}/config_sim/world/models:${GZ_SIM_RESOURCE_PATH}"
export IGN_GAZEBO_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH}"
export AS2_EXTRA_DRONE_MODELS=crazyflie_led_ring

# Display / rendering for headless Xvfb
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3
