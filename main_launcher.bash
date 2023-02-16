#!/bin/bash

drone_namespaces=('drone_sim_0')
simulation_config="simulation_config/default.json"
launch_keyboard_teleop="true"

# Run nodes
./as2_launch.bash "${drone_namespaces[0]}" $simulation_config $launch_keyboard_teleop

session=${drone_namespace}$(($n - 1))

# if inside a tmux session detach before attaching to the session
if [ -n "$TMUX" ]; then
    tmux switch-client -t $session
else
    tmux attach -t $session:0
fi