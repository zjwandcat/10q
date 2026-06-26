# Copyright 2026 zjwandcat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
持仓构建模块
按score降序选Top20，标记is_holding前10只
仓位分配：永远满仓 - Top5(13%×5=65%) + Next5(7%×5=35%)
is_penalized只作为置信度标记，不影响仓位
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging

logger = logging.getLogger("m2.portfolio")


def calculate_turnover_cost(
    prev_holdings: Dict[str, float],   # {stock: weight}
    curr_holdings: Dict[str, float],   # {stock: weight}
    stamp_duty: float = 0.001,
    commission: float = 0.0003,
    slippage: float = 0.001,
) -> float:
    """
    计算本月换仓的总交易成本
    返回值：占组合净值的比例（正数，从收益中扣除）
    """
    # 卖出的股票
    sold = {s: w for s, w in prev_holdings.items()
            if s not in curr_holdings}
    # 买入的股票
    bought = {s: w for s, w in curr_holdings.items()
              if s not in prev_holdings}
    # 调仓的股票（权重变化）
    adjusted_sell = sum(
        max(prev_holdings.get(s, 0) - curr_holdings.get(s, 0), 0)
        for s in set(prev_holdings) & set(curr_holdings)
    )
    adjusted_buy = sum(
        max(curr_holdings.get(s, 0) - prev_holdings.get(s, 0), 0)
        for s in set(prev_holdings) & set(curr_holdings)
    )

    total_sell = sum(sold.values()) + adjusted_sell
    total_buy = sum(bought.values()) + adjusted_buy

    sell_cost = total_sell * (stamp_duty + commission + slippage)
    buy_cost = total_buy * (commission + slippage)

    return sell_cost + buy_cost


class PortfolioBuilder:
    """月度持仓构建器"""

    def build(
        self,
        pred_df_with_scores: pd.DataFrame,
        val_ic:   float,
        ic_gap:   float,
        is_penalized: bool,
        lgbm_model,
        feature_cols: List[str],
        pred_month: str,
        compute_shap: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        构建当月持仓

        输出列：
            stock_code, stock_name, industry, score, score_cv,
            tier(High/Low/Reserve), weight, is_holding,
            val_ic, ic_gap, is_penalized, pred_month, trade_date
            （compute_shap=True 时附加 shap_top{1..3}_factor/value，
              完整 SHAP 矩阵存到 top20.attrs['shap_values'] + attrs['shap_features']）
        """
        df = pred_df_with_scores.copy()

        if len(df) < 15:
            logger.warning(f"{pred_month}: 股票池仅{len(df)}只，低于最低门槛15，跳过该窗口")
            return None

        df = df.sort_values("score", ascending=False)

        # 选Top20（前20只）
        top20 = df.head(20).copy()
        top20["pred_month"] = pred_month
        top20["val_ic"]     = val_ic
        top20["ic_gap"]     = ic_gap
        top20["is_penalized"] = is_penalized

        # 分层
        top20["tier"] = "Reserve"
        top20.iloc[:5,  top20.columns.get_loc("tier")] = "High"
        top20.iloc[5:10,top20.columns.get_loc("tier")] = "Low"

        # 仓位：永远满仓，不受is_penalized影响
        # Top5: 13% × 5 = 65%
        # Next5: 7% × 5 = 35%
        # 总仓位 = 100%
        weight_map = {
            "High": 0.13,
            "Low":  0.07,
            "Reserve": 0.0,
        }
        top20["weight"] = top20["tier"].map(weight_map)
        top20["is_holding"] = top20["tier"].isin(
            ["High", "Low"])

        # 置信度标记（不影响仓位，只用于M4报告统计）
        top20["confidence_flag"] = (
            "HIGH" if not is_penalized else "LOW"
        )

        # SHAP归因（M5调用时跳过）
        if compute_shap and lgbm_model is not None:
            try:
                import shap
                X_holding = pred_df_with_scores.loc[
                    top20.index, feature_cols]
                explainer = shap.TreeExplainer(
                    lgbm_model.model_)
                shap_vals = explainer.shap_values(X_holding)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0]
                # ★ 改造: Top3 因子列保留（向后兼容），同时把完整 SHAP 矩阵
                #   存到 top20.attrs，M4 报告模块统一展示。
                for k in range(min(3, shap_vals.shape[1])):
                    top20[f"shap_top{k+1}_factor"] = (
                        feature_cols[k])
                    top20[f"shap_top{k+1}_value"] = (
                        shap_vals[:, k])
                top20.attrs["shap_values"] = (
                    shap_vals.astype(np.float32)
                )
                top20.attrs["shap_features"] = list(feature_cols)
                top20.attrs["shap_source"] = "shap.TreeExplainer"
            except ImportError:
                # ★ 新增: shap 未装时回退到 LightGBM 内置特征重要性
                # 首次警告，后续不再 spam
                if not getattr(PortfolioBuilder,
                               "_shap_warned", False):
                    logger.warning(
                        "未安装 shap，回退到 LightGBM "
                        "feature_importance_。M4 报告仍可展示"
                        "因子贡献（精度低于真 SHAP）。"
                        "安装: pip install shap")
                    PortfolioBuilder._shap_warned = True
                try:
                    fi = (lgbm_model.model_.feature_importances_
                          if hasattr(lgbm_model, "model_")
                          and hasattr(lgbm_model.model_,
                                       "feature_importances_")
                          else None)
                    if fi is not None and len(fi) == len(feature_cols):
                        order = np.argsort(-fi)[:3]
                        for k, idx in enumerate(order):
                            top20[f"shap_top{k+1}_factor"] = (
                                feature_cols[idx])
                            top20[f"shap_top{k+1}_value"] = (
                                float(fi[idx]))
                        # 构造"伪 SHAP 矩阵"：每行 = 因子重要性 / N
                        n_holding = len(top20)
                        pseudo = np.tile(
                            fi / max(fi.sum(), 1e-9),
                            (n_holding, 1)
                        ).astype(np.float32) * float(n_holding)
                        top20.attrs["shap_values"] = pseudo
                        top20.attrs["shap_features"] = list(
                            feature_cols)
                        top20.attrs["shap_source"] = (
                            "lgbm.feature_importances_"
                            " (shap未安装,回退)")
                except Exception as e2:
                    logger.warning(f"SHAP回退也失败: {e2}")
            except Exception as e:
                logger.warning(f"SHAP计算失败: {e}")

        return top20
