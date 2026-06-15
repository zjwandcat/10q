"""
持仓构建模块
按score降序选Top20，标记is_holding前10只
仓位分配：永远满仓 - Top5(13%×5=65%) + Next5(7%×5=35%)
is_penalized只作为置信度标记，不影响仓位
"""
import pandas as pd
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
                for k in range(min(3, shap_vals.shape[1])):
                    top20[f"shap_top{k+1}_factor"] = (
                        feature_cols[k])
                    top20[f"shap_top{k+1}_value"] = (
                        shap_vals[:, k])
            except Exception as e:
                logger.warning(f"SHAP计算失败: {e}")

        return top20
