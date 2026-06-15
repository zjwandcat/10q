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
标签生成模块
对每个trade_date截面内，将Target_Return_1M转化为截面排名分位数

关键设计：
  - label_rank = 截面内升序排名 / 截面股票数（范围0~1）
  - 值越高代表下月收益越好
  - NaN值保留为NaN（M2训练时dropna处理）
  - label_rank转float32
  - 不跨期共享统计量（严禁数据泄露）
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("m1.label")


class LabelMaker:
    """
    标签生成器

    用法：
        factor_df = LabelMaker().make_labels(factor_df)
    """

    def make_labels(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        生成label_rank列

        参数：
            df: 包含trade_date和Target_Return_1M列的DataFrame

        返回：
            新增label_rank列的DataFrame
            label_rank范围[0,1]，NaN表示Target_Return_1M缺失
        """
        if "Target_Return_1M" not in df.columns:
            raise ValueError(
                "缺少Target_Return_1M列，请检查M0数据")

        logger.info("开始生成label_rank...")

        # 截面内升序排名（pct=True直接得到0~1的分位数）
        # na_option="keep"：NaN值保持NaN不参与排名
        df["label_rank"] = (
            df.groupby("trade_date")["Target_Return_1M"]
            .transform(
                lambda x: x.rank(
                    method="average",
                    ascending=True,
                    pct=True,
                    na_option="keep",
                )
            )
            .astype(np.float32)
        )

        # 统计标签覆盖率
        total = len(df)
        valid = df["label_rank"].notna().sum()
        coverage = valid / total
        logger.info(
            f"label_rank生成完成: "
            f"有效{valid:,}/{total:,} "
            f"（覆盖率{coverage:.1%}）"
        )

        if coverage < 0.90:
            logger.warning(
                f"⚠️ 标签覆盖率{coverage:.1%}低于90%，"
                f"请检查Target_Return_1M数据质量")

        return df
