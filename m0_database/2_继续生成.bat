@echo off
chcp 65001 >nul 2>&1
title M0 Pipeline Generator
cd /d "%~dp0.."
set PYTHONPATH=%cd%
echo ============================================================
echo M0 Pipeline Generator (Resume Mode)
echo ============================================================
echo.
echo Starting pipeline...
echo Press Ctrl+C to safely interrupt
echo ============================================================
echo.
python m0_database\pipeline.py
echo.
echo ============================================================
echo Pipeline finished
echo Press any key to exit...
pause >nul
