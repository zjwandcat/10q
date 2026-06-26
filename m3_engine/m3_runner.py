from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger("m3.runner")


def run_m3(
    m2_path: str = "output/all_portfolios.parquet",
    m0_dir: Optional[str] = None,
    output_path: str = "output/all_portfolios_m3.parquet",
    config_path: str = "config/config_m3.yaml",
    scheme: str = "scheme_d",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Optional[pd.DataFrame]:
    try:
        from .tet_engine import M3Config, TETEngine
    except ImportError:
        logger.warning("M3 引擎导入失败，已回退至纯M2模式")
        if progress_callback:
            progress_callback("⚠️ M3引擎导入失败，已回退至纯M2模式")
        return None

    try:
        config = M3Config.from_yaml(Path(config_path))
        config.scheme = scheme
    except FileNotFoundError:
        logger.warning("⚠️ M3配置文件缺失，已回退至纯M2模式")
        if progress_callback:
            progress_callback("⚠️ M3配置文件缺失，已回退至纯M2模式")
        return None
    except AssertionError as e:
        logger.warning(f"⚠️ M3配置非法: {e}，已回退至纯M2模式")
        if progress_callback:
            progress_callback(f"⚠️ M3配置非法: {e}，已回退至纯M2模式")
        return None

    try:
        engine = TETEngine(config)

        _m0_dir = Path(m0_dir) if m0_dir else Path(f"data/pool_v2_{scheme}")

        import tqdm as _tqdm

        _original_tqdm_init = _tqdm.tqdm.__init__

        def _silent_tqdm_init(self, *args, **kwargs):
            kwargs["disable"] = True
            _original_tqdm_init(self, *args, **kwargs)

        _tqdm.tqdm.__init__ = _silent_tqdm_init
        try:
            result_path = engine.run(
                m2_path=Path(m2_path),
                m0_dir=_m0_dir,
                output_path=Path(output_path),
            )
        finally:
            _tqdm.tqdm.__init__ = _original_tqdm_init

        m3_df = pd.read_parquet(result_path)
        if m3_df.empty:
            logger.warning("⚠️ M3输出为空，仅展示纯M2轨道")
            if progress_callback:
                progress_callback("⚠️ M3输出为空，仅展示纯M2轨道")
            return None

        logger.info(f"M3风控处理完成: {len(m3_df)} 行, 输出={output_path}")
        return m3_df

    except Exception as e:
        logger.warning(f"❌ M3风控失败: {e}，已回退至纯M2模式")
        if progress_callback:
            progress_callback(f"❌ M3风控失败: {e}，已回退至纯M2模式")
        return None