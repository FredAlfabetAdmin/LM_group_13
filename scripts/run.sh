#!/bin/bash

ip_address=$(hostname -I | awk '{print $1}')

echo $ip_address

set -xe
docker build --tag lm .

echo "#!/bin/bash\nexport GLOBAL_IP_ADRESS="$ip_address"" > ./catkin_ws/ip.sh

docker run -t --rm -p 45100:45100 -p 45101:45101 -v $(pwd)/results:/root/results -v $(pwd)/catkin_ws:/root/catkin_ws lm "$@"