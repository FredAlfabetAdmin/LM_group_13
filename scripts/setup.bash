ip_address=$(hostname -I | awk '{print $1}')

# # Check if the IP address is not empty
# if [ -n "$ip_address" ]; then
#     echo "Detected IP address: $ip_address"
#     # Now you can use $ip_address in your script
# else
#     echo "Unable to detect IP address."
# fi

# replace localhost with the port you see on the smartphone
export ROS_MASTER_URI="http://localhost:11311"

# You want your local IP, usually starting with 192.168, following RFC1918
# Windows powershell:
#    (Get-NetIPAddress | Where-Object { $_.AddressState -eq "Preferred" -and $_.ValidLifetime -lt "24:00:00" }).IPAddress
# Unix:
#    hostname -I | awk '{print $1}'
export COPPELIA_SIM_IP="$GLOBAL_IP_ADRESS"