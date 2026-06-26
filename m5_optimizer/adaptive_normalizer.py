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
自适应分位数归一化器（Adaptive Normalizer）

基于经验分位数的自适应归一化系统，支持跨项目/跨策略横向对比。

背景：
    中性化策略 B 盲跑 100 个 Trial 后发现 val_ic 全在 0.083~0.093 之间，
    远超原静态 bounds [-0.05, 0.05]，原硬编码归一化已彻底失效。
    本模块用经验分位数 [P5, P50, P95] 替代一切硬编码边界。

数学映射原则：
    1. Linear 线性机制：
       - 传统 bounds [min_b, max_b] 全面替换为该策略分布的 [P5, P95]
       - 输入值 x 首先被 clip 到 [P5, P95] 区间
       - 映射公式：score = (x - P5) / (P95 - P5) - 0.5  (值域 [-0.5, 0.5])

    2. Tanh 渐进机制：
       - 废除硬编码的静态 scale
       - 动态计算自适应中心点 center = P50 (中位数)
       - 自适应缩放因子 adaptive_scale = (P95 - P5) / 2
       - 映射公式：score = tanh((x - P50) / max(adaptive_scale, 1e-6))
         (值域 [-1.0, 1.0])

    3. 跨策略可比性逻辑（【方案A：全局单一锚定】）：
       默认 use_global_anchor=True。
       无论当前 Trial 实际跑的是什么中性化策略（如 strategy_c / scheme_d），
       归一化计算统一读取 _GLOBAL_ANCHOR_KEY 指定的 json（默认 strategy_b）。
       由此实现【绝对水位可比】：策略 C 若整体被压缩，映射分自然偏低，
       完美呈现不同中性化策略之间的绝对实力差距。

       ⚠️ 元数据隔离（极其重要）：
       归一化计算使用全局锚定标尺，但 trial 元数据
       （meta_neutralization_type）必须记录【实际运行的策略】，
       严禁把锚定策略误写为元数据，否则会丢失策略分组能力。

    4. identity / 未被 NORM_CONFIG 覆盖的指标：
       全面自动纳入自适应 Tanh 桶，使用各自对应的分位数数据激活归一化，
       严禁原值直通打破权重平衡。

三级安全 Fallback 链（防崩溃）：
    Level 1: 尝试加载 _GLOBAL_ANCHOR_KEY 指定的全局锚定 json
    Level 2: 降级为加载当前策略自身 json（带 scheme→strategy 别名）
    Level 3: 强制 fallback 到 search_space.py 中的静态 NORM_CONFIG
    任意一级失败时仅记录 Warning/Error 日志，绝不抛出异常。

污染隔离机制：
    若某指标在 json 中的 missing_rate > 0.9 或 zero_rate > 0.9
    （例如 val_rolling6m_excess），则标记为 is_corrupted = True，
    评分时直接赋予 score = 0.0，不参与任何加减分，也【不计入】
    有效指标数量，彻底终结缺失指标拖垮全局总分的 Bug。
"""
import json
import logging
import os
from typing import Dict, Optional, Any, Set, Tuple

import numpy as np

from m5_optimizer.search_space import NORM_CONFIG

logger = logging.getLogger("m5.adaptive_normalizer")

# 数值稳定常数
_EPS = 1e-6
# 污染判定阈值：missing_rate 或 zero_rate 超过此值则标记为污染
_CORRUPTION_THRESHOLD = 0.9
# 中性化 scheme 命名 → 分位数文件命名的别名映射
# config.yaml 中 active_scheme 历史上使用 "scheme_xxx" 命名，
# 而早期生成的分位数配置文件采用 "strategy_xxx" 命名。
# 此映射保证两套命名体系自动对齐，无需重命名 json 文件。
_SCHEME_TO_STRATEGY_ALIAS: Dict[str, str] = {
    "scheme_a": "strategy_a",
    "scheme_b": "strategy_b",
    "scheme_d": "strategy_d",
    "scheme_e": "strategy_e",
    # 兼容 None / 空字符串 / 显式 default
    "": "strategy_b",
    "none": "strategy_b",
    "null": "strategy_b",
}
# ★★★ 方案A 全局单一锚定策略（Absolute Anchor）★★★
# 无论当前 Trial 跑的是 scheme_b / scheme_c / scheme_d，
# 归一化分位数一律读取此 key 对应的 json。
# 默认 strategy_b 是历史最完整、trial 数最多的基准。
_GLOBAL_ANCHOR_KEY: str = "strategy_b"


class AdaptiveNormalizer:
    """自适应分位数归一化器

    根据中性化策略类型动态加载对应的经验分位数配置，
    实现跨策略可比的归一化映射。

    默认启用【方案A：全局单一锚定】（use_global_anchor=True），
    所有策略的绝对值统一在 _GLOBAL_ANCHOR_KEY 指定的策略分布尺子上度量，
    由此保证跨中性化策略的【绝对水位可比性】。

    三级 Fallback 链确保系统永不崩溃：
      Level 1: 全局锚定 json (默认 strategy_b_quantiles.json)
      Level 2: 当前策略自身 json
      Level 3: 静态 NORM_CONFIG

    用法:
        normalizer = AdaptiveNormalizer(
            neutralization_type="strategy_c",
            use_global_anchor=True,
        )
        if normalizer.is_corrupted("val_rolling6m_excess"):
            # 跳过该指标
            score = 0.0
        else:
            score = normalizer.normalize("val_ic", 0.089, method="linear")
        # 注意：归一化用的是 strategy_b 的分位数，但 trial 的
        # meta_neutralization_type 必须仍写 "strategy_c"（由调用方负责）
    """

    def __init__(
        self,
        neutralization_type: Optional[str] = None,
        use_global_anchor: bool = True,
    ):
        """初始化自适应归一化器。

        参数:
            neutralization_type: 当前 Trial 实际运行的中性化策略类型
                （如 "strategy_c"、"scheme_d"、None）。仅用于：
                (a) Level 2 降级路径找自身 json
                (b) 业务侧元数据 trial.set_user_attr("meta_neutralization_type", ...)
                【不会】影响归一化计算所使用的分位数基准。
                None 时默认为 "strategy_b"。
            use_global_anchor: 是否启用【方案A 全局单一锚定】。
                True  (默认): 归一化一律读 _GLOBAL_ANCHOR_KEY 对应 json
                False        : 归一化读当前 neutralization_type 自身 json
                              （无锚定，跨策略不绝对可比，但单策略内最贴切）
        """
        self.neutralization_type: str = neutralization_type or "strategy_b"
        self.use_global_anchor: bool = use_global_anchor
        # 指标名 -> 分位数统计字典
        self._quantiles: Dict[str, Dict[str, Any]] = {}
        # 污染指标集合（missing_rate 或 zero_rate > 0.9）
        self._corrupted: Set[str] = set()
        # 是否处于静态 fallback 模式（JSON 全部失败时）
        self._fallback_to_static: bool = False
        # 是否成功加载分位数数据
        self._loaded: bool = False
        # 实际生效的分位数策略（用于诊断/元数据）
        self._active_anchor_key: str = ""
        # 加载层级（1=全局锚定 / 2=自身 / 3=静态兜底）
        self._load_level: int = 0
        self._load_quantiles()

    def _resolve_file_key(self, raw: Optional[str]) -> str:
        """将中性化类型标识解析为文件名 key（scheme_xxx → strategy_xxx）。"""
        if raw is None:
            raw = ""
        return _SCHEME_TO_STRATEGY_ALIAS.get(raw, raw)

    def _try_load_from_path(self, config_path: str) -> Optional[Dict[str, Any]]:
        """尝试从指定路径加载并解析分位数配置。

        成功返回 metrics 字典；失败返回 None（不抛异常）。
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            metrics = data.get("metrics", {})
            if not metrics:
                logger.warning(
                    "分位数配置 %s 的 metrics 为空，跳过此文件",
                    config_path,
                )
                return None
            logger.info(
                "分位数配置加载成功: %s (指标数=%d)",
                config_path, len(metrics),
            )
            return metrics
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as e:
            logger.warning(
                "分位数配置 JSON 解析失败: %s, %s",
                config_path, e,
            )
            return None
        except Exception as e:
            logger.warning(
                "加载分位数配置失败: %s, %s",
                config_path, e,
            )
            return None

    def _load_quantiles(self) -> None:
        """加载分位数配置（三级安全 Fallback 链）。

        Level 1: 尝试读取 _GLOBAL_ANCHOR_KEY 对应的全局锚定 json
                 （use_global_anchor=True 时启用）
        Level 2: 降级为读取当前策略自身 json
        Level 3: 强制 fallback 到静态 NORM_CONFIG（不抛异常）
        """
        config_dir = os.path.join(os.path.dirname(__file__), "configs")

        # ---------- Level 1: 全局锚定 ----------
        if self.use_global_anchor:
            anchor_key = self._GLOBAL_ANCHOR_KEY if hasattr(
                self, "_GLOBAL_ANCHOR_KEY"
            ) else _GLOBAL_ANCHOR_KEY
            anchor_path = os.path.join(
                config_dir, f"{anchor_key}_quantiles.json"
            )
            metrics = self._try_load_from_path(anchor_path)
            if metrics is not None:
                self._apply_metrics(metrics, anchor_key, level=1)
                return
            logger.warning(
                "[Level 1 失败] 全局锚定文件不可用: %s，"
                "降级为读取当前策略自身 json",
                anchor_path,
            )

        # ---------- Level 2: 当前策略自身 ----------
        self_key = self._resolve_file_key(self.neutralization_type)
        self_path = os.path.join(config_dir, f"{self_key}_quantiles.json")
        # 若 self_key 与 anchor_key 相同但 Level 1 失败过，不要再次尝试
        if self_path == os.path.join(
            config_dir, f"{_GLOBAL_ANCHOR_KEY}_quantiles.json"
        ) and self.use_global_anchor:
            # 已在 Level 1 试过且失败，跳过
            logger.warning(
                "[Level 2 跳过] 当前策略键与全局锚定键相同且 Level 1 已失败，"
                "直接进入 Level 3",
            )
        else:
            metrics = self._try_load_from_path(self_path)
            if metrics is not None:
                self._apply_metrics(metrics, self_key, level=2)
                return
            logger.warning(
                "[Level 2 失败] 当前策略 json 不可用: %s，"
                "降级到 Level 3 静态 NORM_CONFIG",
                self_path,
            )

        # ---------- Level 3: 静态 NORM_CONFIG 兜底 ----------
        # 关键：绝不能崩溃！必须强制 fallback 到 search_space.NORM_CONFIG
        logger.error(
            "[Level 3 兜底] 所有分位数 json 加载失败，"
            "强制 fallback 到静态 NORM_CONFIG（向下兼容）"
        )
        self._fallback_to_static = True
        self._load_level = 3
        self._active_anchor_key = ""  # 静态模式无 anchor 概念

    def _apply_metrics(
        self,
        metrics: Dict[str, Any],
        anchor_key: str,
        level: int,
    ) -> None:
        """把加载到的 metrics 应用到内部状态（含污染检测）。"""
        for name, stats in metrics.items():
            self._quantiles[name] = stats
            # 污染检测：missing_rate 或 zero_rate > 0.9
            missing_rate = float(stats.get("missing_rate", 0) or 0)
            zero_rate = float(stats.get("zero_rate", 0) or 0)
            if (missing_rate > _CORRUPTION_THRESHOLD
                    or zero_rate > _CORRUPTION_THRESHOLD):
                self._corrupted.add(name)
                logger.info(
                    "指标 %s 被标记为污染 "
                    "(missing_rate=%.2f, zero_rate=%.2f)，评分时跳过",
                    name, missing_rate, zero_rate,
                )
        self._loaded = True
        self._active_anchor_key = anchor_key
        self._load_level = level
        logger.info(
            "自适应归一化器加载成功 [Level %d] anchor=%s, "
            "指标数=%d, 污染数=%d",
            level, anchor_key,
            len(self._quantiles), len(self._corrupted),
        )

    def get_metric_stats(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """获取指标的统计分位数数据。

        参数:
            metric_name: 指标名称

        返回:
            该指标的分位数统计字典（含 p5/p50/p95/missing_rate 等），
            若指标不在配置中则返回 None。
        """
        return self._quantiles.get(metric_name)

    def is_corrupted(self, metric_name: str) -> bool:
        """检查指标是否被标记为污染。

        污染判定标准：missing_rate > 0.9 或 zero_rate > 0.9。
        污染指标不参与任何加减分，直接赋予 score = 0.0，
        彻底终结缺失指标拖垮全局总分的 Bug。

        参数:
            metric_name: 指标名称

        返回:
            True 表示该指标被污染，应跳过评分
        """
        return metric_name in self._corrupted

    def is_fallback(self) -> bool:
        """是否处于静态 fallback 模式。

        JSON 配置文件缺失或读取失败时返回 True，
        此时使用静态 NORM_CONFIG 进行归一化。
        """
        return self._fallback_to_static

    def is_loaded(self) -> bool:
        """是否成功加载了分位数数据。"""
        return self._loaded

    def get_anchor_strategy(self) -> str:
        """返回归一化计算实际使用的分位数策略 key。

        - Level 1 (全局锚定): 返回 _GLOBAL_ANCHOR_KEY（如 "strategy_b"）
        - Level 2 (自身)   : 返回当前 neutralization_type 解析后的 key
        - Level 3 (静态)   : 返回空串 ""

        ⚠️ 注意：这是【归一化标尺】，不是当前 Trial 真实策略。
        真实策略请用 self.neutralization_type。
        """
        return self._active_anchor_key

    def get_load_level(self) -> int:
        """返回分位数配置实际加载到的层级（1/2/3，0 表示未加载）。"""
        return self._load_level

    def get_corrupted_metrics(self) -> Set[str]:
        """返回被标记为污染的指标集合（missing_rate/zero_rate > 0.9）。"""
        return set(self._corrupted)

    def normalize(
        self,
        metric_name: str,
        value: float,
        method: str,
        static_config: Optional[Dict] = None,
    ) -> float:
        """对单个指标应用自适应归一化映射。

        参数:
            metric_name: 指标名称
            value: 原始值（已通过 _safe_float 处理）
            method: 原始归一化方法 (linear/tanh/signed_log/identity)
                - linear: 使用 [P5, P95] 作为动态 bounds
                - tanh/signed_log/identity: 全面纳入自适应 Tanh 桶
            static_config: 静态 NORM_CONFIG 字典（fallback 模式使用），
                None 时使用模块级 NORM_CONFIG

        返回:
            归一化后的值。
            - Linear 机制: 值域 [-0.5, 0.5]
            - Tanh 机制: 值域 [-1.0, 1.0]
            - 污染指标: 0.0
            - fallback 模式: 使用静态 NORM_CONFIG
        """
        # NaN / Inf 防御
        if not np.isfinite(value):
            return 0.0

        # 第一步：污染剔除
        if self.is_corrupted(metric_name):
            return 0.0

        # fallback 模式：使用静态 NORM_CONFIG
        if self._fallback_to_static:
            return _apply_static_normalization(metric_name, value, static_config)

        stats = self._quantiles.get(metric_name)
        if stats is None:
            # 指标不在分位数配置中，fallback 到静态配置
            return _apply_static_normalization(metric_name, value, static_config)

        p5 = stats.get("p5")
        p50 = stats.get("p50")
        p95 = stats.get("p95")

        # 分位数缺失（null）则视为污染，返回 0.0
        if p5 is None or p50 is None or p95 is None:
            return 0.0

        p5 = float(p5)
        p50 = float(p50)
        p95 = float(p95)

        # 第二步：动态替换参数
        if method == "linear":
            # Linear 线性机制：使用 [P5, P95] 作为动态 bounds
            return self._norm_linear_adaptive(value, p5, p95)
        else:
            # Tanh 渐进机制 + identity/signed_log 全面纳入自适应 Tanh 桶
            # 废除硬编码 scale，使用 P50 作为中心，(P95-P5)/2 作为缩放
            return self._norm_tanh_adaptive(value, p50, p5, p95)

    @staticmethod
    def _norm_linear_adaptive(value: float, p5: float, p95: float) -> float:
        """自适应线性映射

        公式: score = (clip(x, P5, P95) - P5) / max(P95 - P5, 1e-6) - 0.5
        值域: [-0.5, 0.5]

        三重防御：
          1) 极端值双向 clip：先把 x 截断到 [P5, P95]，避免越界污染
          2) 分母零防御：denominator = max(P95 - P5, 1e-6)，避免 ZeroDivisionError
          3) 结果 NaN 防御：最后一道 np.isfinite 兜底，返回 0.0
        """
        if not np.isfinite(value):
            return 0.0
        # 分母零防御（特化盲跑数据中某指标 P95==P5 的极端情况）
        denominator = max(p95 - p5, _EPS)
        if denominator < _EPS:
            return 0.0
        # 极端值双向 clip 防御
        clipped = float(np.clip(value, p5, p95))
        score = (clipped - p5) / denominator - 0.5
        # 严格值域防御（双保险）：确保结果在 [-0.5, 0.5] 之间
        if not np.isfinite(score):
            return 0.0
        return float(np.clip(score, -0.5, 0.5))

    @staticmethod
    def _norm_tanh_adaptive(
        value: float, p50: float, p5: float, p95: float
    ) -> float:
        """自适应 Tanh 映射

        公式: score = tanh((x - P50) / max(adaptive_scale, 1e-6))
        其中 adaptive_scale = (P95 - P5) / 2
        值域: [-1.0, 1.0]

        三重防御：
          1) 自适应缩放因子分母零防御
          2) np.tanh 已天然值域受限 [-1, 1]，但对极端输入仍防 inf
          3) 结果 NaN 防御
        """
        if not np.isfinite(value):
            return 0.0
        adaptive_scale = (p95 - p5) / 2.0
        s = max(adaptive_scale, _EPS)
        if s < _EPS:
            return 0.0
        shifted = (value - p50) / s
        # 防止极端 shifted 触发 overflow（np.tanh 自身有 inf 防护，但更稳）
        shifted = float(np.clip(shifted, -50.0, 50.0))
        score = float(np.tanh(shifted))
        if not np.isfinite(score):
            return 0.0
        return float(np.clip(score, -1.0, 1.0))


def _apply_static_normalization(
    metric_name: str,
    value: float,
    static_config: Optional[Dict] = None,
) -> float:
    """静态 NORM_CONFIG 归一化（fallback 模式使用）。

    当 AdaptiveNormalizer 处于 fallback 模式，或指标不在分位数配置中时，
    使用此函数回退到静态 NORM_CONFIG 进行归一化。

    与 objective._apply_normalization 保持一致的语义，
    但本函数独立实现以避免循环依赖。

    参数:
        metric_name: 指标名称
        value: 原始值
        static_config: 静态归一化配置字典，None 时使用模块级 NORM_CONFIG

    返回:
        归一化后的值
    """
    cfg = (static_config or NORM_CONFIG).get(metric_name)
    if cfg is None:
        return value

    if not np.isfinite(value):
        return 0.0

    method = cfg.get("method", "identity")
    if method == "tanh":
        s = cfg.get("scale", 1.0)
        s = s if s > 0 else 1.0
        return float(np.tanh(value / s))
    elif method == "signed_log":
        sign = 1.0 if value >= 0 else -1.0
        return sign * float(np.log(1.0 + abs(value) + _EPS))
    elif method == "linear":
        bounds = cfg.get("bounds", [0.0, 1.0])
        lo, hi = float(bounds[0]), float(bounds[1])
        if hi - lo < _EPS:
            return 0.0
        clipped = float(np.clip(value, lo, hi))
        return (clipped - lo) / (hi - lo + _EPS) - 0.5
    else:
        return value
