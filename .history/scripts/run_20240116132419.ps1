# # if PowerShell scripts don't work, make sure to:
# # `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned`
# # in a powershell running in administrator mode.
# #
# # Sadly, there is no good equivilant to "$@" in ps1
# param($p1)

# $ip_address = (Test-Connection -ComputerName $env:COMPUTERNAME -Count 1).IPAddressToString)

# docker build --tag learning_machines . --build-arg IP_ADRESS=$ip_address
# # Mounting to a directory that does not exist creates it.
# # Mounting to relative paths works since docker engine 23
# docker run -t --rm -p 45100:45100 -p 45101:45101 -v ./results:/root/results learning_machines $PSBoundParameters["p1"]

powershell -noexit "& ""coppeliaSim.exe \start_coppelia_sim.ps1 .\scenes\Robobo_Scene.ttt"""


param([string]$mode)

# Get IP address
$ipAddress = (Get-NetIPAddress | Where-Object { $_.AddressState -eq "Preferred" -and $_.ValidLifetime -lt "24:00:00" }).IPAddress

# Build Docker image
#docker build --tag lm --build-arg IP_ADRESS=$ipAddress .
Write-Host $ipAddress
# Create IP script
Set-Content -Path "./catkin_ws/ip.sh" -Value "#!/bin/bash`nexport GLOBAL_IP_ADRESS=`"$ipAddress`""

# Run Docker container
docker run -t --rm -p 45100:45100 -p 45101:45101 -v $pwd\results:/root/results -v $pwd\catkin_ws:/root/catkin_ws lm $mode

# Change ownership of results directory
# This assumes the equivalent of "sudo chown "$USER":"$USER" ./results -R" in PowerShell
# Get-ChildItem -Path "./results" -Recurse | ForEach-Object {
#     $_ | Get-Acl | ForEach-Object {
#         $_.SetOwner([System.Security.Principal.NTAccount]::new($env:USERNAME))
#         $_ | Set-Acl $_.Path
#     }
# }
