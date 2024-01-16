#!/bin/bash
ip_address=$(hostname -I | awk '{print $1}')
xcookie="bas-xps/unix:  MIT-MAGIC-COOKIE-1  ba898f21129869fdb1ba6bcf288e4826 #ffff#6261732d787073#:  MIT-MAGIC-COOKIE-1  ba898f21129869fdb1ba6bcf288e4826"

set -xe
docker build --tag lm .

echo "#!/bin/bash\nexport GLOBAL_IP_ADRESS="$ip_address"\nexport DISPLAY=${DISPLAY}\nexport xcookie=$xcookie" > ./catkin_ws/bootinstructs.sh

docker run -t --rm -p 45100:45100 -p 45101:45101 -v $(pwd)/results:/root/results -v $(pwd)/catkin_ws:/root/catkin_ws lm "$@"