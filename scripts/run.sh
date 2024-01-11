#!/bin/bash
ip_address=$(hostname -I | awk '{print $1}')

set -xe

docker build --tag lm . --build-arg IP_ADRESS=$ip_address
# Mounting to a directory that does not exist creates it.
# Mounting to relative paths works since docker engine 23
docker run -t --rm -p 45100:45100 -p 45101:45101 -v /home/bas/projects/LM2023/results:/root/results lm "$@"

# Because docker runs as root, this means the files will be owned by the root user.
# Change this with:
# sudo chown "$USER":"$USER" ./results -R
