@echo off
title Cloud UBA - Public Internet Link
cls
echo ====================================================================
echo        Generating Public Internet Link for Cloud UBA Dashboard
echo ====================================================================
echo.
echo Anyone on ANY PC or phone in the world can open this link!
echo No Wi-Fi or file copying required.
echo.
echo Connecting to public tunnel...
echo (Look below for the "https://..." link)
echo.
ssh -o StrictHostKeyChecking=no -R 80:localhost:8501 nokey@localhost.run
pause
