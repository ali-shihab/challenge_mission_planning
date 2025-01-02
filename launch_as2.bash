#!/bin/bash

usage() {
    echo "  options:"
    echo "      -m: multi agent (only needed to be set for simulation). Default not set"
    echo "      -n: select drones namespace to launch, values are comma separated. By default, it will get all drones from world description file"
    echo "      -c: if set, the real crazyflie interface will be launched instead of the simulation. Defaults to false"
    echo "      -g: launch using gnome-terminal instead of tmux. Default not set"
}

# Initialize variables with default values
swarm="false"
drones_namespace_comma=""
launch_simulation="true"
use_gnome="false"

# Arg parser
while getopts "mn:cg" opt; do
  case ${opt} in
    m )
      swarm="true"
      ;;
    n )
      drones_namespace_comma="${OPTARG}"
      ;;
    c )
      launch_simulation="false"
      ;;
    g )
      use_gnome="true"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    : )
      if [[ ! $OPTARG =~ ^[wrt]$ ]]; then
        echo "Option -$OPTARG requires an argument" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CONFIG_SIM="${SCRIPT_DIR}/config_sim"
CONFIG_REAL="${SCRIPT_DIR}/config_real"

config_folder=""
simulation_config_file=""
if [[ ${launch_simulation} == "true" ]]; then
  config_folder="${CONFIG_SIM}"
  # Ensure this folders gazebo packages are on the path for both aerostack2 and gazebo to read...
  export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:"${config_folder}/gazebo/models"
  export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:"${config_folder}/gazebo/models"
  export AS2_EXTRA_DRONE_MODELS=crazyflie_led_ring
else
  config_folder="${CONFIG_REAL}"
fi

drone_config="${config_folder}/config/config.yaml"

# Set simulation world description config file
if [[ ${swarm} == "true" ]]; then
  simulation_config_file="${CONFIG_SIM}/config/world_swarm.yaml"
else
  simulation_config_file="${CONFIG_SIM}/config/world.yaml"
fi


# If no drone namespaces are provided, get them from the world description config file 
if [ -z "$drones_namespace_comma" ]; then

  if [[ ${launch_simulation} == "true" ]]; then
    dnamespace_lookup_file="${simulation_config_file}"
  else
    dnamespace_lookup_file="${drone_config}"
  fi

  drones_namespace_comma=$(python3 utils/get_drones.py -p ${dnamespace_lookup_file} --sep ',')
fi
IFS=',' read -r -a drone_namespaces <<< "$drones_namespace_comma"

# Select between tmux and gnome-terminal
tmuxinator_mode="start"
tmuxinator_end="wait"
tmp_file="/tmp/as2_project_launch_${drone_namespaces[@]}.txt"
if [[ ${use_gnome} == "true" ]]; then
  tmuxinator_mode="debug"
  tmuxinator_end="> ${tmp_file} && python3 utils/tmuxinator_to_genome.py -p ${tmp_file} && wait"
fi

# Launch aerostack2 for each drone namespace
for namespace in ${drone_namespaces[@]}; do
  base_launch="false"
  if [[ ${namespace} == ${drone_namespaces[0]} ]]; then
    base_launch="true"
  fi
  eval "tmuxinator ${tmuxinator_mode} -n ${namespace} -p ${SCRIPT_DIR}/tmuxinator/aerostack2.yaml \
    drone_namespace=${namespace} \
    script_folder=${SCRIPT_DIR} \
    simulation=${launch_simulation} \
    config_dir=${config_folder}/config \
    config_file=${drone_config} \
    simulation_config_file=${simulation_config_file} \
    base_launch=${base_launch} \
    ${tmuxinator_end}"

  sleep 0.1 # Wait for tmuxinator to finish
done

# Attach to tmux session
if [[ ${use_gnome} == "false" ]]; then
  tmux attach-session -t ${drone_namespaces[0]}
# If tmp_file exists, remove it
elif [[ -f ${tmp_file} ]]; then
  rm "${tmp_file}"
fi
