#!/bin/bash

drone_namespaces=('drone_sim_0' 'drone_sim_1' 'drone_sim_2')
simulation_config="simulation_config/swarm.json"
launch_keyboard_teleop="false"

# For each drone namespace
for drone_namespace in "${drone_namespaces[@]}"
do
    ./as2_launch.bash "${drone_namespace}" $simulation_config $launch_keyboard_teleop
done

# Attach to the first session
session=${drone_namespaces[0]}
# if inside a tmux session detach before attaching to the session
if [ -n "$TMUX" ]; then
    tmux switch-client -t $session
else
    tmux attach -t $session:0
fi

source ./utils/launch_tools.bash
new_session ${drone_namespaces[0]}_keyboard_teleop
new_window 'keyboard_teleop' "ros2 launch as2_keyboard_teleoperation as2_keyboard_teleoperation_launch.py  \
        namespace:=drone_sim_0,drone_sim_1,drone_sim_2 \
        use_sim_time:=true"
