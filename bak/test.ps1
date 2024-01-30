$ipAddress = (Get-NetIPAddress | Where-Object { $_.AddressState -eq "Preferred" -and $_.ValidLifetime -lt "24:00:00" }).IPAddress
echo $ipAddress

Set-Content -Path "./ip.ps1" -Value "export GLOBAL_IP_ADRESS=`"$ipAddress`""