@echo off
title Explainable AI Cloud UBA - Shutdown
cls
echo ====================================================================
echo        Stopping Cloud UBA Dashboard and Stream Daemon
echo ====================================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*realtime_stream.py*' -or $_.CommandLine -like '*dashboard/app.py*' -or $_.CommandLine -like '*dashboard\app.py*' } | ForEach-Object { Write-Host 'Terminating Process ID:' $_.ProcessId; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

echo.
echo All Cloud UBA services have been stopped cleanly.
timeout /t 3 /nobreak >nul
