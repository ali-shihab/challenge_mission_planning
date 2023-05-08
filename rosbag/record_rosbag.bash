#!/bin/bash

drone_namespace=$1

if [[ "$drone_namespace" == "" ]] 
then
    echo "Please specify the drone namespace as the first argument"
    exit 1
fi

mkdir rosbag/rosbags 2>/dev/null
cd rosbag/rosbags &&\
ros2 bag record \
"/$drone_namespace/platform/info" \
"/$drone_namespace/self_localization/pose" \
"/$drone_namespace/self_localization/twist" \
"/$drone_namespace/actuator_command/twist" \
"/tf" \
"/tf_static" \
--include-hidden-topics