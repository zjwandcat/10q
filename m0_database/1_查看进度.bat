@echo off
chcp 65001 >nul 2>&1
title M0 Pipeline Progress
cd /d "%~dp0.."
set PYTHONPATH=%cd%
echo ============================================================
echo M0 Pipeline Progress Monitor
echo ============================================================
echo.
python _monitor_progress.py
echo.
echo ============================================================
echo Press any key to exit...
pause >nul
