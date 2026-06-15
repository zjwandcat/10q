@echo off
chcp 65001 >nul 2>&1
title M0 Regenerator B1/B2/F/G

REM cd to project root (bat is in m0_database/, go up one level)
cd /d "%~dp0"
cd ..
set "PYTHONPATH=%cd%"

echo ============================================================
echo M0 Regenerator - two_phase mode (memory safe)
echo   Phase 1: 1 worker x 228 months (cache)
echo   Phase 2: 2 workers x 912 tasks (neutralize)
echo ============================================================
echo.
echo Starting...
echo Press Ctrl+C to interrupt (finished files kept)
echo ============================================================
echo.

python m0_database\regenerator.py --n_jobs 2 --phase two_phase
set "RC=%errorlevel%"

echo.
echo ============================================================
if %RC% equ 0 (
    echo ALL OK
) else (
    echo Finished with failures. Check:
    echo   logs\m0\regenerator.log
    echo   output\m0\regenerator_bad_months.json
)
echo ============================================================
pause
exit /b %RC%
