"""Pre-flight check script for M0 pipeline parquet generation"""
import sys
from pathlib import Path

BASE = Path(r"e:\10q\10q-202604gpu")
sys.path.insert(0, str(BASE))
import os
os.chdir(str(BASE))

import yaml
import numpy as np
import pandas as pd

issues = []

# ── 1. config.yaml ──
print("=" * 60)
print("1. config.yaml")
print("=" * 60)
cfg = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))
# 脱敏打印：只显示 token 前 4 位 + 后 4 位
_tok = cfg.get("tushare", {}).get("token", "") or ""
_mask = (_tok[:4] + "***" + _tok[-4:]) if len(_tok) >= 8 else "(empty)"
print(f"  tushare token: {_mask} (len={len(_tok)})")
print(f"  active_scheme: {cfg['data']['neutralization']['active_scheme']}")
print(f"  available_schemes: {list(cfg['data']['neutralization']['available_schemes'].keys())}")
print(f"  pool_dirs: {list(cfg['data']['pool_dirs'].keys())}")
print(f"  raw_cache_dir: {cfg['data']['raw_cache_dir']}")

# Check stock_filter config alignment
sf = cfg["data"]["stock_filter"]
print(f"  stock_filter: {sf}")

# ── 2. Output directories ──
print()
print("=" * 60)
print("2. Output Directories")
print("=" * 60)
for scheme, dir_path in cfg["data"]["pool_dirs"].items():
    p = BASE / dir_path
    exists = p.exists()
    count = len(list(p.glob("*.parquet"))) if exists else 0
    print(f"  [{scheme}] {dir_path} | exists={exists} | parquet_count={count}")

cache_dir = BASE / cfg["data"]["raw_cache_dir"]
cache_count = len(list(cache_dir.glob("*.pkl"))) if cache_dir.exists() else 0
print(f"  raw_cache | exists={cache_dir.exists()} | pkl_count={cache_count}")

# ── 3. Data files ──
print()
print("=" * 60)
print("3. Data Files")
print("=" * 60)
excel_path = BASE / "data/000906perf.xlsx"
print(f"  benchmark excel: exists={excel_path.exists()}")
if not excel_path.exists():
    issues.append("CRITICAL: data/000906perf.xlsx not found - benchmark return will fail")

# ── 4. Python dependencies ──
print()
print("=" * 60)
print("4. Python Dependencies")
print("=" * 60)
for dep in ["tushare", "pandas", "numpy", "scipy", "pyarrow"]:
    try:
        m = __import__(dep)
        v = getattr(m, "__version__", "unknown")
        print(f"  {dep}: OK ({v})")
    except ImportError:
        print(f"  {dep}: MISSING!")
        issues.append(f"CRITICAL: {dep} not installed")

try:
    import akshare
    print(f"  akshare: OK ({akshare.__version__})")
except ImportError:
    print("  akshare: NOT INSTALLED (optional fallback)")

# ── 5. Module imports ──
print()
print("=" * 60)
print("5. Module Imports")
print("=" * 60)
modules = [
    ("m0_database.data_fetcher", "TushareFetcher"),
    ("m0_database.factor_calculator", "FactorCalculator"),
    ("m0_database.stock_filter", "filter_stock_pool"),
    ("m0_database.neutralization", "apply_neutralization"),
    ("m0_database.benchmark_loader", "get_benchmark_return_for_month"),
    ("m0_database.pipeline", "run_pipeline"),
]
for mod_name, attr_name in modules:
    try:
        mod = __import__(mod_name, fromlist=[attr_name])
        obj = getattr(mod, attr_name)
        print(f"  {mod_name}.{attr_name}: OK")
    except Exception as e:
        print(f"  {mod_name}.{attr_name}: FAIL ({e})")
        issues.append(f"CRITICAL: Cannot import {mod_name}.{attr_name}: {e}")

# ── 6. Tushare API connectivity ──
print()
print("=" * 60)
print("6. Tushare API Connectivity")
print("=" * 60)
try:
    from m0_database.data_fetcher import TushareFetcher
    fetcher = TushareFetcher()
    
    # Test trade calendar
    cal = fetcher.get_trade_calendar("20210301", "20210331")
    print(f"  trade_calendar(202103): {len(cal)} trading days")
    
    # Test stock basic
    stock_basic = fetcher.get_stock_basic()
    print(f"  stock_basic: {len(stock_basic)} stocks")
    
    # Test monthly basic
    daily_basic = fetcher.get_monthly_basic("20210331")
    print(f"  monthly_basic(20210331): {len(daily_basic)} stocks")
    
    # Test daily data
    daily_df = fetcher.get_daily_data("000001.SZ", "20210331", n_days=100)
    print(f"  daily_data(000001.SZ): {len(daily_df)} rows, cols={daily_df.columns.tolist()}")
    
    # Test financial data
    fin = fetcher.get_financial_data("000001.SZ", "20210331")
    print(f"  financial_data(000001.SZ): {len(fin)} fields")
    if len(fin) < 5:
        issues.append("WARNING: financial_data returns very few fields - check fina_indicator API")
    
    # Test valuation data
    val = fetcher.get_daily_valuation("20210331")
    print(f"  daily_valuation(20210331): {len(val)} stocks")
    
    # Test balance sheet
    bs = fetcher.get_balance_sheet("000001.SZ", "20210331")
    print(f"  balance_sheet(000001.SZ): {len(bs)} fields")
    
    # Test macro data
    macro = fetcher.get_macro_data("20210331")
    print(f"  macro_data(20210331): {len(macro)} variables")
    
except Exception as e:
    print(f"  Tushare API test FAILED: {e}")
    issues.append(f"CRITICAL: Tushare API test failed: {e}")

# ── 7. Factor calculator test ──
print()
print("=" * 60)
print("7. Factor Calculator Test")
print("=" * 60)
try:
    from m0_database.factor_calculator import FactorCalculator
    calc = FactorCalculator()
    
    # Use real daily data
    if not daily_df.empty and len(daily_df) >= 60:
        financial_data = fin if fin else {}
        # Supplement with valuation
        if not val.empty:
            val_row = val[val["ts_code"] == "000001.SZ"]
            if not val_row.empty:
                for col in ["pe", "pb", "ps", "pcf"]:
                    v = val_row.iloc[0].get(col, np.nan)
                    if not pd.isna(v):
                        financial_data[col] = float(v)
        
        basic_data = {"total_mv": 500000.0, "close": 22.01}
        macro_data = macro if macro else {f"var_{i}": 0.0 for i in range(68)}
        
        factors = calc.calculate_all(daily_df, financial_data, macro_data, "20210331", basic_data)
        print(f"  Total factors: {len(factors)}")
        
        # Check for placeholder factors
        placeholders = [k for k in factors if "placeholder" in k]
        print(f"  Placeholder factors: {len(placeholders)}")
        if placeholders:
            issues.append(f"WARNING: {len(placeholders)} placeholder factors remain: {placeholders[:5]}")
        
        # Check NaN ratio
        nan_count = sum(1 for v in factors.values() if isinstance(v, float) and np.isnan(v))
        print(f"  NaN factors: {nan_count}/{len(factors)} ({nan_count/len(factors)*100:.1f}%)")
        
        # Check key new factors
        key_factors = [
            "value_pe_ratio", "value_pb_ratio", "growth_revenue_yoy",
            "quality_roe_ex_nonrecurring", "dividend_payout_ratio",
            "size_log_mcap", "reversal_1m", "reversal_3m",
        ]
        for kf in key_factors:
            v = factors.get(kf, "NOT_FOUND")
            is_ok = isinstance(v, (int, float)) and not np.isnan(v)
            print(f"  {kf}: {'OK' if is_ok else 'NaN/MISSING'} ({v})")
    else:
        print("  Skipped: daily data not available")
        
except Exception as e:
    print(f"  Factor calculator test FAILED: {e}")
    issues.append(f"CRITICAL: Factor calculator test failed: {e}")

# ── 8. Stock filter test ──
print()
print("=" * 60)
print("8. Stock Filter Test")
print("=" * 60)
try:
    from m0_database.stock_filter import filter_stock_pool
    if not daily_basic.empty:
        df_test = daily_basic.merge(
            stock_basic[["ts_code", "name", "industry", "list_date"]],
            on="ts_code", how="left"
        )
        df_test = df_test.rename(columns={
            "ts_code": "stock_code", "name": "stock_name",
            "close": "close_price", "total_mv": "market_cap",
            "turnover_rate": "avg_turnover_rate",
        })
        df_test["list_date"] = pd.to_datetime(df_test["list_date"])
        df_test["days_listed"] = 500
        df_test["market_cap"] = df_test["market_cap"] / 10000
        df_test["avg_turnover_rate"] = df_test["avg_turnover_rate"] / 100
        df_test["suspend_days"] = 0
        
        df_filtered = filter_stock_pool(df_test, verbose=False)
        print(f"  Before filter: {len(df_test)} stocks")
        print(f"  After filter: {len(df_filtered)} stocks")
        if len(df_filtered) < 15:
            issues.append("WARNING: Stock pool too small after filtering")
    else:
        print("  Skipped: daily_basic not available")
except Exception as e:
    print(f"  Stock filter test FAILED: {e}")
    issues.append(f"CRITICAL: Stock filter test failed: {e}")

# ── 9. Neutralization test ──
print()
print("=" * 60)
print("9. Neutralization Test")
print("=" * 60)
try:
    from m0_database.neutralization import apply_neutralization
    # Quick test with dummy data
    test_df = pd.DataFrame({
        "industry": ["银行"] * 10 + ["地产"] * 10,
        "market_cap": [100 + i * 10 for i in range(20)],
        "factor1": np.random.randn(20),
        "factor2": np.random.randn(20),
    })
    result = apply_neutralization(test_df, ["factor1", "factor2"], "scheme_d")
    print(f"  scheme_d neutralization: OK ({len(result)} rows, {len(result.columns)} cols)")
except Exception as e:
    print(f"  Neutralization test FAILED: {e}")
    issues.append(f"CRITICAL: Neutralization test failed: {e}")

# ── 10. Benchmark loader test ──
print()
print("=" * 60)
print("10. Benchmark Loader Test")
print("=" * 60)
try:
    from m0_database.benchmark_loader import get_benchmark_return_for_month
    bm = get_benchmark_return_for_month("202103")
    print(f"  benchmark_return(202103): {bm}")
except Exception as e:
    print(f"  Benchmark loader test FAILED: {e}")
    issues.append(f"WARNING: Benchmark loader test failed: {e}")

# ── Summary ──
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
if issues:
    print(f"  Found {len(issues)} issues:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("  All checks passed! Ready for parquet generation.")

print()
print("  Pipeline command examples:")
print("    # Single scheme, single month test:")
print("    run_pipeline(schemes=['scheme_d'], months=['202103'], force_rebuild=True)")
print("    # All schemes, single month:")
print("    run_pipeline(schemes=['scheme_a','scheme_b','scheme_d','scheme_e'], months=['202103'], force_rebuild=True)")
print("    # Full generation (all schemes, all months):")
print("    run_pipeline(schemes=['scheme_a','scheme_b','scheme_d','scheme_e'], force_rebuild=True)")
