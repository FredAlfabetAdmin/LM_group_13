#!/bin/bash

ip_address=$(hostname -I | awk '{print $1}')
ip_hw="10.15.2.155"

echo $ip_address
echo $ip_hw

set -xe
# docker build --tag lm .

echo "#!/bin/bash\nexport GLOBAL_IP_ADRESS="$ip_address"\nHW_IP="$ip_hw"" > ./catkin_ws/ip.sh

docker run -t --rm -p 45100:45100 -p 45101:45101 -v $(pwd)/results:/root/results -v $(pwd)/catkin_ws:/root/catkin_ws lm "$@"