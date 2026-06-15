@echo off
chcp 65001 >nul
title TTHH M5 Optimizer
cls

REM Switch to project root directory
cd /d %~dp0
cd ..
set PYTHONPATH=%cd%

echo.
echo  ========================================================
echo        TTHH M5 Bayesian Hyperparameter Optimizer
echo  ========================================================
echo.
echo  Working directory: %cd%
echo.

REM Create necessary directories
if not exist "output\m5" mkdir output\m5
if not exist "output\m5\sessions" mkdir output\m5\sessions
if not exist "logs\m5" mkdir logs\m5

REM Set environment variables
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set GRADIO_ANALYTICS_ENABLED=false

echo  Updating dependencies from Tsinghua mirror...
echo.
py -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
py -m pip install optuna gradio pyyaml psutil numpy pandas lightgbm xgboost scikit-learn scipy -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
echo.
echo  Dependencies updated successfully!
echo.
echo ========================================================
echo.

echo  Running diagnostics...
py _diagnose.py
echo.
if exist logs\m5\diag.log (
    echo  Diagnostic results:
    type logs\m5\diag.log
    echo.
)
echo  Starting Gradio server...
echo ========================================================
echo.

py -m m5_optimizer.app

if errorlevel 1 (
    echo.
    echo  ========================================================
    echo              Startup Failed!
    echo  ========================================================
    echo.
    if exist logs\m5\diag.log (
        echo  Latest diagnostic:
        type logs\m5\diag.log
        echo.
    )
    pause
)