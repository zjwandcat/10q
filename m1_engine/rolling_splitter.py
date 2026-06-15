"""
滚动窗口生成模块

窗口参数（从config.yaml读取）：
  train_months: 36  （训练集）
  valid_months: 12  （验证集）
  test_months:  1   （预测集，即下月）
  step_months:  1   （每次滑动1个月）

关键设计：
  - 生成器模式（yield），不一次性加载所有窗口到内存
  - 每个窗口切片后执行.copy()，防止内存泄漏
  - 严格按时间顺序，train < val < pred，禁止数据泄露
  - pred_df是下一个月的截面数据（用于预测当月持仓）
"""
import pandas as pd
import yaml
import logging
from typing import Dict, Generator

logger = logging.getLogger("m1.splitter")


def _load_config() -> dict:
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ts_fmt(ts_obj) -> str:
    """将 Timestamp 格式化为 YYYYMM 字符串"""
    return ts_obj.strftime("%Y%m")


class RollingSplitter:
    """
    滚动窗口切分器

    用法：
        splitter = RollingSplitter()
        for window in splitter.split(factor_df):
            train_df = window["train_df"]
            val_df   = window["val_df"]
            pred_df  = window["pred_df"]
    """

    def __init__(self, train_months: int = None):
        cfg = _load_config()["rolling"]
        self.train_months = train_months or cfg["train_months"]  # 36
        self.valid_months = cfg["valid_months"]                  # 12
        self.test_months = cfg["test_months"]                    # 1
        self.step_months = cfg["step_months"]                    # 1
        self.window_size = (
            self.train_months +
            self.valid_months +
            self.test_months                                     # 49
        )

    def split(
        self,
        factor_df: pd.DataFrame,
        copy: bool = True,
    ) -> Generator[Dict[str, pd.DataFrame], None, None]:
        """
        生成滚动窗口（生成器模式）

        参数：
            factor_df: 完整因子DataFrame，必须包含trade_date列
                      已按trade_date升序排列

        生成：
            每次yield一个dict：
            {
                "train_df": DataFrame,  # 36个月的训练数据
                "val_df":   DataFrame,  # 12个月的验证数据
                "pred_df":  DataFrame,  # 1个月的预测数据
                "window_idx": int,      # 窗口编号（从0开始）
                "pred_month": str,      # 预测月份YYYYMM
            }

        时间边界（严格无泄露）：
            train: months[i]   ~ months[i+35]
            val:   months[i+36] ~ months[i+47]
            pred:  months[i+48]
        """
        # 获取所有唯一月份（升序）
        all_months = sorted(factor_df["trade_date"].unique())
        n_months = len(all_months)

        if n_months < self.window_size:
            raise ValueError(
                f"数据月份数{n_months}不足{self.window_size}，"
                f"无法生成任何窗口"
            )

        n_windows = (
            (n_months - self.window_size) //
            self.step_months + 1
        )
        logger.info(
            f"数据共{n_months}个月，"
            f"可生成{n_windows}个滚动窗口"
        )

        for i in range(0, n_months - self.window_size + 1,
                       self.step_months):
            # 计算各集的月份边界
            train_start = i
            train_end = i + self.train_months             # 不含
            val_start = train_end
            val_end = val_start + self.valid_months       # 不含
            pred_start = val_end
            pred_end = pred_start + self.test_months      # 不含

            train_months_list = all_months[
                train_start:train_end
            ]
            val_months_list = all_months[
                val_start:val_end
            ]
            pred_months_list = all_months[
                pred_start:pred_end
            ]

            # 切片（copy=True时深拷贝防止内存泄漏）
            train_df = factor_df[
                factor_df["trade_date"].isin(train_months_list)
            ]
            val_df = factor_df[
                factor_df["trade_date"].isin(val_months_list)
            ]
            pred_df = factor_df[
                factor_df["trade_date"].isin(pred_months_list)
            ]
            if copy:
                train_df = train_df.copy()
                val_df = val_df.copy()
                pred_df = pred_df.copy()

            # 预测月份字符串（用于日志和持仓文件命名）
            pred_month = pd.Timestamp(
                pred_months_list[0]
            ).strftime("%Y%m")

            window_idx = i // self.step_months

            if window_idx % 20 == 0:
                logger.info(
                    f"  窗口{window_idx:03d}: "
                    f"train={_ts_fmt(train_months_list[0])}"
                    f"~{_ts_fmt(train_months_list[-1])} "
                    f"val={_ts_fmt(val_months_list[0])}"
                    f"~{_ts_fmt(val_months_list[-1])} "
                    f"pred={pred_month}"
                )

            yield {
                "train_df": train_df,
                "val_df": val_df,
                "pred_df": pred_df,
                "window_idx": window_idx,
                "pred_month": pred_month,
            }

    def get_n_windows(self, factor_df: pd.DataFrame) -> int:
        """预估窗口数量（不实际切片）"""
        n_months = factor_df["trade_date"].nunique()
        if n_months < self.window_size:
            return 0
        return (
            (n_months - self.window_size) //
            self.step_months + 1
        )
