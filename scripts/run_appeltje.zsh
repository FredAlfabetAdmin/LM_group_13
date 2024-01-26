#!/usr/bin/env zsh
ip_address=$(ifconfig en0 | awk '/inet / {print $2}')
ip_hw="10.15.2.155"

echo $ip_address
echo $ip_hw
set -xe
docker build --platform linux/amd64 --tag lm .
echo "#!/bin/bash\nexport GLOBAL_IP_ADRESS="$ip_address"\nHW_IP="$ip_hw"" > ./catkin_ws/ip.sh

# Mounting to a directory that does not exist creates it.
# Mounting to relative paths works since docker engine 23
docker run -t --rm --platform linux/amd64 -p 45100:45100 -p 45101:45101 -v $(pwd)/results:/root/results -v $(pwd)/catkin_ws:/root/catkin_ws lm "$@"
# Because docker runs as root, this means the files will be owned by the root user.
# Change this with:
# sudo chown "$USER":"$USER" ./results -R