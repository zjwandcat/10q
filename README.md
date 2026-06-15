# TTHH A股量化选股系统 (m01245)

> 私有仓库 · 由 zjwandcat 维护

基于 Tushare 数据 + 多模块流水线 + GPU 加速 XGBoost/LightGBM 的 A 股量化选股回测与超参优化系统。

## 模块结构

| 模块 | 路径 | 职责 |
|---|---|---|
| **M0** 数据 | `m0_database/` | Tushare 拉取、股票筛选、因子计算、中性化、Parquet 落盘 |
| **M1** 滚动切分 | `m1_engine/` | 时序滚动训练/验证/测试集切分、Label 生成 |
| **M2** 模型训练 | `m2_engine/`, `m2_engine_gpu/` | 特征工程、LGBM/XGBoost 训练、集成、组合构建 |
| **M3** 评估 | （合并入 M2/M4） | — |
| **M4** 报告 | `m4_report/` | 回测指标计算、报告生成 |
| **M5** 优化器 | `m5_optimizer/` | Optuna TPE 贝叶斯超参搜索（Phase1 全局 + Phase2 局部）、Gradio Web UI |

## 快速开始

### 1. 环境
- Python 3.11
- Windows 10/11
- 16GB+ RAM，推荐 NVIDIA GPU（CUDA 12.x）

### 2. 安装依赖
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. 配置 Tushare Token
**必须**通过环境变量设置（**不要**写进 yaml）：
```powershell
$env:TUSHARE_TOKEN = "你的tushare_pro_token"
```
去 https://tushare.pro 注册并在个人中心获取。

### 4. 跑通流程
```powershell
# M0 拉数据 + 计算因子（首次需要数小时，缓存到 data/raw_cache/）
python run_m0_full.py

# M1 滚动切分
python m1_engine\run_m1.py

# M2 训练 + 组合构建
python m2_engine\run_m2.py

# M4 报告
python m4_report\report_generator.py

# M5 启动优化器 Web UI
.\m5_optimizer\启动M5优化器.bat
# 浏览器打开 http://127.0.0.1:7860
```

## 目录约定

```
10q-202604gpu/
├── config/             # 配置（yaml + 线程常量）
├── m0_database/        # M0 数据模块
├── m1_engine/          # M1 切分模块
├── m2_engine/          # M2 CPU 训练
├── m2_engine_gpu/      # M2 GPU 训练
├── m4_report/          # M4 报告
├── m5_optimizer/       # M5 贝叶斯优化
├── run_*.py            # 顶层入口
├── bench_*.py          # 一次性 benchmark 脚本（不入仓）
└── *.md                # 项目文档
```

> ⚠️ **不进入版本控制**：`data/`、`logs/`、`output/`、所有 `*.pkl/*.parquet/*.db` 缓存。
> 详见 [.gitignore](.gitignore)。

## 性能约束

- 滚动训练窗口：36 个月训练 / 12 个月验证 / 1 个月测试
- GPU 模式（M2 + M5）：通过 `m2_engine_gpu` + XGBoost CUDA
- 内存峰值约 12-14 GB（CPU 模式）

## 安全提示

- **Tushare token 严禁提交**！仓库已配置为从环境变量 `TUSHARE_TOKEN` 读取
- 历史 commit 已剔除明文 token
- 如发现 token 泄露，请立即在 https://tushare.pro 重置

## 许可证

仅供学习研究使用。
