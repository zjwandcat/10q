@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title TTHH M5 Optimizer
cls

REM ── Win11 / Python 3.14 兼容 ──
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

REM Python 启动器回退（优先 py，否则用 python）
where py >nul 2>&1 && (set PY=py) || (set PY=python)

REM Switch to project root directory
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

REM ── 创建必要目录 ──
for %%D in ("output\m5" "output\m5\sessions" "logs\m5") do (
    if not exist "%%~D" mkdir "%%~D"
)

REM ── 环境变量 ──
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set GRADIO_ANALYTICS_ENABLED=false

REM ── 重复启动检测（纯 cmd，不依赖 PowerShell） ──
tasklist /FI "IMAGENAME eq python.exe" /V /NH 2>nul | findstr /I "m5_optimizer" >nul 2>&1
if not errorlevel 1 (
    echo  检测到 M5 优化器已在运行中，请勿重复启动
    echo  如需强制启动，请关闭已有的 M5 窗口后重试
    echo.
    pause
    exit /b 0
)

REM ── 网络检测 (2秒超时) ──
echo  Checking network connectivity...
ping -n 1 -w 2000 223.5.5.5 >nul 2>&1
if errorlevel 1 (
    echo  无网络连接，跳过依赖升级
    echo.
    goto :skip_pip
)

REM ── 智能依赖升级 ──
set "PIP_CHECK=logs\m5\.pip_check"

if exist "%PIP_CHECK%" (
    echo  依赖校验标记存在，跳过升级
    echo  如需强制升级，请删除 logs\m5\.pip_check
    echo.
    goto :skip_pip
)

echo.
echo  Updating dependencies from Tsinghua mirror...
echo.
%PY% -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>&1
%PY% -m pip install optuna gradio pyyaml psutil numpy pandas lightgbm xgboost scikit-learn scipy -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --quiet
if errorlevel 1 (
    echo  镜像源升级失败，尝试直接升级...
    %PY% -m pip install optuna gradio pyyaml psutil numpy pandas lightgbm xgboost scikit-learn scipy --upgrade --quiet
)
if errorlevel 1 (
    echo  ⚠ 依赖升级失败，将继续尝试启动（可能缺少某些依赖）
) else (
    echo  依赖升级完成！
)
echo checked >"%PIP_CHECK%"
echo.

:skip_pip
echo ========================================================
echo.

REM ── 诊断脚本（可选） ──
if exist _diagnose.py (
    echo  Running diagnostics...
    %PY% _diagnose.py
    echo.
    if exist logs\m5\diag.log (
        echo  Diagnostic results:
        type logs\m5\diag.log
        echo.
    )
) else (
    echo  Diagnostic script not found, skipping.
    echo.
)

echo  Starting Gradio server...
echo ========================================================
echo.

%PY% -m m5_optimizer.app

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
