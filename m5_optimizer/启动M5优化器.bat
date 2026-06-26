@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title TTHH M5 Optimizer
cls

set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

where py >nul 2>&1 && (set PY=py) || (set PY=python)

cd /d %~dp0
cd ..
set "PYTHONPATH=%cd%"

echo.
echo  ========================================================
echo        TTHH M5 Bayesian Hyperparameter Optimizer
echo  ========================================================
echo.
echo  Python:      %PY%
echo  Working dir: %cd%
echo.

for %%D in ("output\m5" "output\m5\sessions" "logs\m5") do (
    if not exist "%%~D" mkdir "%%~D"
)

set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set GRADIO_ANALYTICS_ENABLED=false
set GRADIO_OFFLINE=true

%PY% -m m5_optimizer._launcher

if errorlevel 1 (
    echo.
    echo  ========================================================
    echo              Startup Failed!
    echo  ========================================================
    echo.
    if exist logs\m5\crash.log (
        echo  Crash log:
        type logs\m5\crash.log
        echo.
    )
    pause
)
