"""
M1数据加载模块
并行读取M0输出的parquet文件，合并为单一DataFrame

关键设计：
  - ThreadPoolExecutor并行读取，自适应并发数
  - 全量数据转float32（节省约50%内存）
  - df.copy()消除内存碎片化
  - 支持多中性化方案切换（通过config.yaml）
  - 支持指定数据目录（供M5方案对比使用）
"""
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# ── 模块级常量 ──────────────────────────────────────
META_COLS = frozenset([
    "trade_date", "stock_code", "stock_name",
    "industry", "list_date",
])

_CODE_COLS = frozenset([
    "ts_code", "stock_code", "symbol", "code",
])

# 自适应并发配置
config_path = (Path(__file__).parent.parent /
               "config" / "concurrency_config.py")
if config_path.exists():
    sys.path.insert(0, str(config_path.parent))
    from concurrency_config import DATA_LOADER_MAX_WORKERS
    sys.path.pop(0)
else:
    import os
    DATA_LOADER_MAX_WORKERS = min(os.cpu_count() or 8, 8)

logger = logging.getLogger("m1.loader")


def _load_config() -> dict:
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_single_parquet(path: Path) -> pd.DataFrame:
    """读取单个parquet文件"""
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        return df
    except Exception as e:
        logger.warning(f"读取失败: {path.name} - {e}")
        return None


class DataLoader:
    """
    M0数据加载器

    用法：
        loader = DataLoader()                    # 使用config中的active_scheme
        loader = DataLoader(scheme="scheme_a")   # 指定方案
        loader = DataLoader(pool_dir="data/pool_v2_scheme_b/")  # 直接指定目录
        factor_df = loader.load()
    """

    def __init__(
        self,
        scheme: str = None,       # 中性化方案名，None时从config读取
        pool_dir: str = None,     # 直接指定目录，优先于scheme
        max_workers: int = None,  # 并发数，None时自适应
    ):
        cfg = _load_config()

        if pool_dir is not None:
            self.pool_dir = Path(pool_dir)
        elif scheme is not None:
            self.pool_dir = Path(cfg["data"]["pool_dirs"][scheme])
        else:
            # 从config读取active_scheme
            active = cfg["data"]["neutralization"]["active_scheme"]
            self.pool_dir = Path(cfg["data"]["pool_dirs"][active])

        self.max_workers = max_workers or DATA_LOADER_MAX_WORKERS
        logger.info(
            f"DataLoader初始化: {self.pool_dir} "
            f"并发={self.max_workers}"
        )

    def load(self) -> pd.DataFrame:
        """
        并行读取所有parquet文件，合并为完整DataFrame

        返回：
            factor_df: 完整因子数据，按trade_date升序排列
                      数值列已转为float32
        """
        # 扫描文件（按日期排序）
        files = sorted(self.pool_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"目录为空: {self.pool_dir}\n"
                f"请先运行M0生成数据"
            )

        logger.info(
            f"发现{len(files)}个parquet文件，"
            f"开始并行读取（{self.max_workers}线程）..."
        )

        # ThreadPoolExecutor并行读取
        dfs = [None] * len(files)
        with ThreadPoolExecutor(
                max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(_read_single_parquet, f): i
                for i, f in enumerate(files)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()
                if result is not None:
                    dfs[idx] = result
                completed += 1
                if completed % 20 == 0:
                    logger.info(f"  已读取 {completed}/{len(files)}")

        # 过滤失败的文件
        dfs = [d for d in dfs if d is not None]
        if not dfs:
            raise RuntimeError("所有parquet文件读取失败")

        logger.info(f"合并{len(dfs)}个DataFrame...")

        # 合并
        factor_df = pd.concat(dfs, ignore_index=True)

        # 确保trade_date为datetime类型
        if not pd.api.types.is_datetime64_any_dtype(
                factor_df["trade_date"]):
            factor_df["trade_date"] = pd.to_datetime(
                factor_df["trade_date"])

        # 按日期升序排列
        factor_df = factor_df.sort_values(
            "trade_date").reset_index(drop=True)

        # 数值列转float32（节省50%内存）
        numeric_cols = [
            c for c in factor_df.select_dtypes(
                include=["float64", "float32", "int64", "int32"]
            ).columns
            if c not in META_COLS
        ]
        if numeric_cols:
            factor_df[numeric_cols] = (
                factor_df[numeric_cols].astype(np.float32)
            )

        # copy消除内存碎片
        factor_df = factor_df.copy()

        # 股票代码列转 Categorical（节省 40-60% 该列内存，加速 groupby）
        for _col in _CODE_COLS:
            if _col in factor_df.columns and factor_df[_col].dtype == object:
                factor_df[_col] = factor_df[_col].astype("category")
                logger.info(f"股票代码列 {_col!r} 已转为 Categorical")
                break

        logger.info(
            f"加载完成: {len(factor_df):,}行 × "
            f"{len(factor_df.columns)}列\n"
            f"  日期范围: "
            f"{factor_df['trade_date'].min().strftime('%Y-%m')} ~ "
            f"{factor_df['trade_date'].max().strftime('%Y-%m')}\n"
            f"  股票池大小: "
            f"{factor_df['stock_code'].nunique()}只（均值）"
        )

        return factor_df
