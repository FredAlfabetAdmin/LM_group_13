#!/usr/bin/env bash

source /root/catkin_ws/ip.sh
rm /root/catkin_ws/ip.sh

cd /root/catkin_ws

source /opt/ros/noetic/setup.bash
catkin_make

chmod -R u+x /root/catkin_ws/

source /root/catkin_ws/devel/setup.bash
source /root/setup.bash

echo "Starting ROS"

rosrun learning_machines learning_robobo_controller.py "$@"
