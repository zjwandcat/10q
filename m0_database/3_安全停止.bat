@echo off
chcp 65001 >nul 2>&1
title M0 Pipeline Stop
echo ============================================================
echo M0 Pipeline Safe Stop
echo ============================================================
echo.
echo Stopping all Python processes...
taskkill /f /im python.exe 2>nul
if %errorlevel%==0 (
    echo.
    echo [OK] Python processes stopped safely
    echo.
    echo Tip: Run "2_继续生成.bat" to resume next time
) else (
    echo.
    echo [Info] No running Python processes found
)
echo.
echo ============================================================
echo Press any key to exit...
pause >nul
