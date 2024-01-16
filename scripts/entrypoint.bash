#!/usr/bin/env bash

source /root/catkin_ws/bootinstructs.sh
rm /root/catkin_ws/bootinstructs.sh

cd /root/catkin_ws
find /root/catkin_ws -type f -wholename '**/*.py' -exec sed -i 's/\r$//' {} \;

source /opt/ros/noetic/setup.bash
catkin_make
chmod -R u+x /root/catkin_ws/

source /root/setup.bash
source /root/catkin_ws/devel/setup.bash

echo $DISPLAY

echo "Starting Coppeliasim"
# sudo /root/coppeliasim/coppeliaSim -gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE ./scenes/Robobo_Scene.ttt
echo "Starting ROS"

rosrun learning_machines learning_robobo_controller.py "$@"
