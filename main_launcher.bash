#!/bin/bash

num_drones=$1
num_drones=${num_drones:=1}

DIR_SCRIPT="${0%/*}"

n=0
drone_namespace="drone_sim_"
while [ $n -lt $num_drones ]; do
    ${DIR_SCRIPT}/as2_launch.bash $drone_namespace$n
    n=$(($n + 1))
done

session=${drone_namespace}$(($n - 1))

# if inside a tmux session detach before attaching to the session
if [ -n "$TMUX" ]; then
    tmux switch-client -t $session
else
    tmux attach -t $session:0
fi