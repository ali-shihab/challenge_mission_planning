#!/bin/bash
drone_namespace=${AEROSTACK2_SIMULATION_DRONE_ID::-1}

tmux ls | grep -Po "${drone_namespace}\d+" | xargs -I % sh -c 'tmux kill-session -t %'

tmux kill-session -t "${drone_namespace}UI"