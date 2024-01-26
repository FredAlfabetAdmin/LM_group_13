#!/usr/bin/env bash

# replace localhost with the port you see on the smartphone
export ROS_MASTER_URI="http://$HW_IP:11311"

# You want your local IP, usually starting with 192.168, following RFC1918
# Windows powershell:
#    (Get-NetIPAddress | Where-Object { $_.AddressState -eq "Preferred" -and $_.ValidLifetime -lt "24:00:00" }).IPAddress
# Unix:
#    hostname -I | awk '{print $1}'
export COPPELIA_SIM_IP="$GLOBAL_IP_ADRESS"



