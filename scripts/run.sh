#!/bin/bash
ip_address=$(hostname -I | awk '{print $1}')

set -xe
docker build --tag lm . --build-arg IP_ADRESS=$ip_address

# Mounting to a directory that does not exist creates it.
# Mounting to relative paths works since docker engine 23

echo "#!/bin/bash\nexport GLOBAL_IP_ADRESS="$ip_address"" > ./catkin_ws/ip.sh

docker run -t --rm -p 45100:45100 -p 45101:45101 -v $(pwd)/results:/root/results -v $(pwd)/catkin_ws:/root/catkin_ws lm "$@"

# Because docker runs as root, this means the files will be owned by the root user.
# Change this with:
# sudo chown "$USER":"$USER" ./results -R
