@echo off
chcp 65001 >nul 2>&1
title M0 Incremental Rebuild (skip existing)

cd /d "%~dp0"
cd ..
set "PYTHONPATH=%cd%"

echo ============================================================
echo M0 增量续跑 - force_rebuild=False (自动跳过已存在文件)
echo   新增因子: momentum_return_240d, jq_price_position_60d
echo ============================================================
echo.

echo ============================================================
echo [1/8] scheme_a
echo ============================================================
python -c "import sys; sys.path.insert(0,'.'); from m0_database.pipeline import run_pipeline; run_pipeline(schemes=['scheme_a'], force_rebuild=False)"
echo.

echo ============================================================
echo [2/8] scheme_b
echo ============================================================
python -c "import sys; sys.path.insert(0,'.'); from m0_database.pipeline import run_pipeline; run_pipeline(schemes=['scheme_b'], force_rebuild=False)"
echo.

echo ============================================================
echo [3/8] scheme_b1
echo ============================================================
python -c "import sys; sys.path.insert(0,'.'); from m0_database.pipeline import run_pipeline; run_pipeline(schemes=['scheme_b1'], force_rebuild=False)"
echo.

echo ============================================================
echo [4/8] scheme_b2
echo ============================================================
python -c "import sys; sys.path.insert(0,'.'); from m0_database.pipeline import run_pipeline; run_pipeline(schemes=['scheme_b2'], force_rebuild=False)"
echo.

echo ============================================================
echo [5/8] scheme_d
echo ============================================================
python -c "import sys; sys.path.insert(0,'.'); from m0_database.pipeline import run_pipeline; run_pipeline(schemes=['scheme_d'], force_rebuild=False)"
echo.

echo ============================================================
echo [6/8] scheme_e
echo ============================================================
python -c "import sys; sys.path.insert(0,'.'); from m0_database.pipeline import run_pipeline; run_pipeline(schemes=['scheme_e'], force_rebuild=False)"
echo.

echo ============================================================
echo [7/8] scheme_f
echo ============================================================
python -c "import sys; sys.path.insert(0,'.'); from m0_database.pipeline import run_pipeline; run_pipeline(schemes=['scheme_f'], force_rebuild=False)"
echo.

echo ============================================================
echo [8/8] scheme_g
echo ============================================================
python -c "import sys; sys.path.insert(0,'.'); from m0_database.pipeline import run_pipeline; run_pipeline(schemes=['scheme_g'], force_rebuild=False)"
echo.

echo ============================================================
echo 验证新因子是否存在
echo ============================================================
python -c "import pandas as pd; df=pd.read_parquet('data/pool_v2_scheme_b/202501.parquet'); cols=list(df.columns); print('momentum_return_240d:', 'momentum_return_240d' in cols); print('jq_price_position_60d:', 'jq_price_position_60d' in cols)"
echo.

echo ============================================================
echo ALL DONE
echo ============================================================
pause
