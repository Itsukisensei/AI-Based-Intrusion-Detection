@echo off
title Explainable AI Cloud UBA - Startup
cls
echo ====================================================================
echo        Explainable AI Cloud UBA (User Behavior Analytics)
echo ====================================================================
echo.

cd /d "%~dp0"

echo [1/3] Starting Real-Time Event Streaming Daemon...
start /b "" python detection\realtime_stream.py

echo [2/3] Starting Cloud SOC Dashboard Server...
start /b "" python -m streamlit run dashboard\app.py --server.headless=true

echo [3/3] Waiting for server initialization...
timeout /t 3 /nobreak >nul

echo Launching Opera GX at http://localhost:8501 ...
if exist "C:\Users\lenovo\AppData\Local\Programs\Opera GX\opera.exe" (
    start "" "C:\Users\lenovo\AppData\Local\Programs\Opera GX\opera.exe" "http://localhost:8501"
) else (
    start http://localhost:8501
)

echo.
echo ====================================================================
echo   SUCCESS! Dashboard is now running and opened in Opera GX.
echo   URL: http://localhost:8501
echo ====================================================================
echo Keep this window open while using the platform.
echo To stop everything, simply close this window or run stop_dashboard.bat.
echo.
pause
