# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

QuantOptima 是一个 A 股技术分析指标优化工具。核心思路是用遗传算法（GA）在 30 年 A 股历史数据上优化技术指标参数，针对不同市场体制（牛市/熊市/震荡/暴跌/反弹）分别产出多套指标体系。

优化目标：6-18 个月持有期，正收益，>80% 概率超 10% 收益。

## 常用命令

### 安装

```bash
pip install -r requirements.txt
```

依赖：pandas, numpy, akshare, pyarrow。可选：torch（用于 Apple Silicon GPU 加速回测）。

### 阶段 1：数据获取（需联网）

```bash
python fetch_data.py                       # 默认 500 只
python fetch_data.py --n 1000              # 1000 只
python fetch_data.py --full                # 全量 A 股 (~5000 只，约 8-12 小时)
python fetch_data.py --symbols 000001 600519  # 指定股票
python fetch_data.py --workers 4 --delay 0.3  # 调整并发与请求间隔
```

输出到 `stock_data/historical/{symbol}.parquet` 和 `stock_data/regime_labels.csv`。

### 阶段 2：参数优化（离线，CPU/GPU）

```bash
python run_optimize.py                              # standard preset
python run_optimize.py --preset quick               # ~30min
python run_optimize.py --preset thorough            # ~8-12h
python run_optimize.py --mode full                  # 完整流程 (global→regime→robust→iterative)
python run_optimize.py --n-stocks 80 --pop 100 --gen 50  # 自定义
python run_optimize.py --gpu                        # 启用 Apple Silicon MPS 加速
```

输出：`stock_data/optimization_result.json` + `stock_data/reports/{txt,md,html}`。

### 日常信号生成（消费优化结果）

```bash
python skills/daily_signal_generator.py --universe 50 --top 10
python skills/daily_signal_generator.py --symbols 000001 600519 --output report.txt
```

读取 `optimization_result.json` → 判断当前大盘体制 → 选用对应指标体系参数 → 输出买卖信号。

### 打包（迁移到其他机器）

```bash
python pack.py    # 打包 skills/ 到 optimization_package/skills/（输出目录保留旧名）
```


### 入口脚本

- `fetch_data.py` — 阶段 1 入口
- `run_optimize.py` — 阶段 2 入口
- `run_fetch.bat` / `run_optimize.bat` — Windows 包装脚本

## 架构

两阶段流水线：**Data → Optimize → Deploy (signals)**。

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  Data fetchers  │───▶│ HistoricalDataManager│───▶│ MarketRegimeClassif │
│  akshare/eastm  │    │   (Parquet 存储)      │    │ (regime_labels.csv)│
│  sina/tencent   │    │                       │    │                     │
└─────────────────┘    └──────────────────────┘    └──────────┬──────────┘
                                                              │
                                                              ▼
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  IndicatorSys   │◀───│  OptimizationEngine  │◀───│   ParameterSpace    │
│  Report (txt/   │    │   (GA 多体制优化)     │    │   (25 维参数)        │
│  md/html)       │    │                       │    │                     │
└─────────────────┘    └──────────┬───────────┘    └─────────────────────┘
                                  │                ┌─────────────────────┐
                                  │                │ VectorizedBacktest  │
                                  │                │  (CPU, 单股 ~10ms)   │
                                  │                │  GPUBacktest (MPS)  │
                                  │                └─────────────────────┘
                                  ▼
                       stock_data/optimization_result.json
                                  │
                                  ▼
                  ┌──────────────────────────────┐
                  │   daily_signal_generator.py  │
                  │  (读 result.json → 体制 →   │
                  │   选体系 → 算指标 → 出信号)  │
                  └──────────────────────────────┘
```

### 核心模块（`skills/`）

**数据层**
- `historical_data_manager.py` — Parquet 存储、增量更新、多线程批量获取、`batch_fetch()`、`get_universe()`
- `market_regime_classifier.py` — 基于沪深 300 指数的 `Regime` 枚举 (BULL/BEAR/SIDEWAYS/CRASH/RECOVERY)，输出 `regime_labels.csv`
- `eastmoney_fetcher.py` / `sina_fetcher.py` / `tencent_fetcher.py` — akshare 的备选源（requests 直接调用）
- `stock_data_fetcher.py` — 高层数据拉取封装

**优化层**
- `parameter_space.py` — `ParamDef` + `ParameterSpace`（4 组共 25 维：指标 13 + 评分权重 4 + 信号阈值 4 + 仓位 4）
- `vectorized_backtest.py` — `VectorizedBacktester` / `BacktestMetrics`（无逐行循环的向量化回测）
- `gpu_backtest.py` — `GPUBacktester`（PyTorch MPS 批量回测，多股×多参数并行）
- `multi_stock_backtester.py` — 多股票组合回测
- `optimization_engine.py` — `OptimizationEngine` + `IndicatorSystem` 数据结构；提供 `optimize_global` / `optimize_by_regime` / `optimize_robust` / `iterative_optimize`
- `optimization_config.py` — `OptimizationConfig` / 四个 preset（quick/standard/thorough/production），可序列化为 JSON

**信号与报告**
- `daily_signal_generator.py` — 消费 `optimization_result.json` 出每日 BUY/HOLD/SELL
- `indicator_system_report.py` — 优化结果的可读报告生成（`PARAM_NAMES` 把参数 key 翻译成中文）
- `signal_generator.py` / `technical_analyzer.py` / `advanced_indicators.py` / `trend_indicators.py` / `pattern_recognition.py` / `risk_management.py` — 指标计算与信号逻辑
- `backtest_framework.py` — 旧版/更详细的回测框架（与 `vectorized_backtest` 并存；优化用前者，详细分析用后者）

### 25 维参数空间（关键约束）

```
指标 (13): ma_fast(3-30), ma_mid(10-60), ma_slow(15-120),
          macd_fast(6-20), macd_slow(18-40), macd_signal(5-15),
          rsi_period(5-30), rsi_oversold(15-40), rsi_overbought(60-85),
          kdj_n(5-21), bb_period(10-40), bb_std(1.0-3.5), atr_period(7-28)
评分 (4):  w_trend / w_momentum / w_risk / w_performance (权重，合计 100)
阈值 (4):  buy/sell signal strength, score buy/sell threshold
仓位 (4):  stop_loss(3-15%), take_profit(10-50%), position(30-100%), trail_stop(3-10%)
```

### 适应度函数

```
fitness = 0.40 * 达标率(>10% 收益)
        + 0.25 * 中位收益率
        + 0.20 * 中位夏普比率
        + 0.15 * 回撤惩罚(越小越好)
```

### 优化模式

| 模式 | 含义 |
|------|------|
| `global` | 全局通用基线参数 (~1h) |
| `regime` | 按体制分别优化 (~2h) |
| `robust` | 跨体制稳健参数 (~1h) |
| `iterative` | 多轮迭代精炼 (~2h) |
| `full` | 上述四步串行 (~6h) |

## 数据目录布局

```
stock_data/
├── historical/                  # Parquet, 每只股票一个文件
│   └── {symbol}.parquet
├── metadata.json                # HistoricalDataManager 元数据
├── regime_labels.csv            # 每日大盘体制标签
├── regime_report.txt            # 体制统计摘要
├── optimization_result.json     # ★ 核心产物：多套指标体系
├── optimization_result_*.json   # 不同 preset 的结果
├── reports/
│   └── indicator_systems.{txt,md,html}
└── 持仓.csv                     # 手工维护的持仓表（消费信号时用）
```

`stock_data/` 已被 `.gitignore` 忽略。

## GPU 加速

- `run_optimize.py --gpu` 启用 MPS 路径。`OptimizationEngine` 在 import 时 `try/except ImportError` 导入 `gpu_backtest`，没有 torch 也能跑（自动 fallback 到 CPU）。
- 加速目标：单代 GA 评估 ~7500s → ~300s（25x）。
- 仅 Apple Silicon (MPS)。CUDA 路径不存在（`get_device()` 只检查 `mps.is_available()`）。

## 约定与注意

- 所有 Python 模块在 `skills/` 根目录，`fetch_data.py` / `run_optimize.py` 通过 `sys.path.insert(0, 'skills')` 加载它们。
- 指标参数的中文名在 `indicator_system_report.py:PARAM_NAMES`，加新参数要同步这里。
- 体制枚举在 `market_regime_classifier.py:Regime`。GA 输出的 `applicable_regimes` 字段是这套枚举的字符串。
- 旧版 `backtest_framework.py`（1076 行）保留作为详细分析；优化场景统一用 `vectorized_backtest`。
- 旧版 `backtest_framework.py`（1076 行）保留作为详细分析；优化场景统一用 `vectorized_backtest`。
- `.claude/` 已在 `.gitignore` 忽略。
