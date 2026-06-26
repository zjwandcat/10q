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
M3 TET 风控外挂模块 CLI 入口

用法：
    python run_m3.py --scheme scheme_d --config config/config_m3.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

from . import M3Config, TETEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M3 TET 风控外挂模块 - 基于 TET 理论对 M2 持仓做体检"
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="scheme_d",
        help="M0 pool 目录后缀（默认 scheme_d）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_m3.yaml",
        help="YAML 配置文件路径（默认 config/config_m3.yaml）",
    )
    parser.add_argument(
        "--m2-path",
        type=str,
        default="output/all_portfolios.parquet",
        help="M2 持仓文件路径（默认 output/all_portfolios.parquet）",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output/all_portfolios_m3.parquet",
        help="M3 输出文件路径（默认 output/all_portfolios_m3.parquet）",
    )
    parser.add_argument(
        "--m0-dir",
        type=str,
        default=None,
        help="M0 因子目录（默认 data/pool_v2_{scheme}/）",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认 INFO）",
    )
    parser.add_argument(
        "--no-debug-csv",
        action="store_true",
        help="禁用 debug CSV 落盘（覆盖 YAML 配置）",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    m0_dir = Path(args.m0_dir) if args.m0_dir else Path(f"data/pool_v2_{args.scheme}")
    config_path = Path(args.config)
    m2_path = Path(args.m2_path)
    output_path = Path(args.output_path)

    try:
        config = M3Config.from_yaml(config_path)
        if args.no_debug_csv:
            config.debug_csv = False
        # INCONSIST-02: --scheme CLI 覆盖 YAML
        config.scheme = args.scheme
        logger = logging.getLogger(__name__)
        logger.info(
            f"已加载配置: scheme={config.scheme}, "
            f"sell_threshold={config.sell_threshold}, "
            f"hys_band={config.hys_band}"
        )
    # INCONSIST-01: 第一个 try 块不涉及 pred_month 校验（from_yaml 不会触发），简化 except
    except AssertionError as e:
        print(f"[M3_CONFIG_ERROR] 配置非法: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"[M3_FILE_NOT_FOUND] 文件缺失: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"[M3_UNKNOWN_ERROR] 未知错误: {e}", file=sys.stderr)
        return 5

    try:
        if not m2_path.exists():
            print(f"[M3_FILE_NOT_FOUND] M2 文件缺失: {m2_path}", file=sys.stderr)
            return 3

        engine = TETEngine(config)
        result_path = engine.run(m2_path, m0_dir, output_path)
        print(f"M3 完成，输出文件: {result_path}")
        return 0
    # INCONSIST-04: match-case 重构第二个 try 块 except
    except AssertionError as e:
        match "pred_month" in str(e):
            case True:
                print(f"[M3_FORMAT_ERROR] pred_month 格式非法: {e}", file=sys.stderr)
                return 4
            case _:
                print(f"[M3_CONFIG_ERROR] 配置非法: {e}", file=sys.stderr)
                return 1
    except ValueError as e:
        print(f"[M3_TIMELINE_ERROR] 时序校验失败: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"[M3_FILE_NOT_FOUND] 文件缺失: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[M3_UNKNOWN_ERROR] 未知异常: {e}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    sys.exit(main())
