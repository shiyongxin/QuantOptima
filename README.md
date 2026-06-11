# 技术分析指标优化工具 - 部署包

## 概述

本工具通过遗传算法(GA)对A股30年历史数据进行多轮次优化，为不同市场体制(牛市/熊市/震荡/暴跌/反弹)分别找到最优技术分析参数，产出多套指标体系。

**优化目标**: 6-18个月持有期，正收益率，>80%概率超10%收益。

## 目录结构

```
optimization_package/
├── README.md               # 本文档
├── requirements.txt        # Python依赖
├── pack.py                 # 打包脚本(在原项目目录运行)
├── fetch_data.py           # 阶段1: 数据获取(需联网)
├── run_optimize.py         # 阶段2: 参数优化(离线)
├── skills/                 # 核心模块(由pack.py生成)
│   ├── market_regime_classifier.py   # 大盘态势分类
│   ├── historical_data_manager.py    # Parquet数据管理
│   ├── parameter_space.py            # 25维参数空间
│   ├── vectorized_backtest.py        # 向量化回测引擎
│   ├── optimization_engine.py        # GA优化引擎
│   ├── multi_stock_backtester.py     # 多股票回测
│   ├── optimization_config.py        # 配置管理
│   ├── daily_signal_generator.py     # 每日信号生成
│   ├── indicator_system_report.py    # 报告生成
│   └── ...                           # 其他依赖模块
└── stock_data/             # 数据目录(运行后生成)
    ├── historical/         # Parquet历史数据
    ├── regime_labels.csv   # 大盘体制标签
    ├── optimization_result.json  # 优化结果
    └── reports/            # 优化报告
```

## 快速开始

### 1. 环境准备

```bash
# 安装Python依赖
pip install -r requirements.txt

# 验证安装
python -c "import pandas, numpy, akshare, pyarrow; print('OK')"
```

### 2. 打包(在原项目目录运行)

```bash
cd Stocks
python optimization_package/pack.py
```

这会将所有需要的模块复制到 `optimization_package/skills/` 目录。
之后将整个 `optimization_package/` 目录复制到目标电脑。

### 3. 数据获取(需联网, 约2-4小时)

```bash
# 获取500只股票 + 大盘指数 + 体制标签
python fetch_data.py

# 或获取更多
python fetch_data.py --n 1000

# 或全量A股(~5000只, 约8-12小时)
python fetch_data.py --full
```

### 4. 参数优化(离线, 约4-8小时)

```bash
# 标准优化(推荐)
python run_optimize.py

# 快速测试(约30分钟)
python run_optimize.py --preset quick

# 深度优化(约8-12小时)
python run_optimize.py --preset thorough

# 自定义参数
python run_optimize.py --n-stocks 80 --pop 100 --gen 50 --mode full
```

### 5. 查看结果

```bash
# 优化结果
cat stock_data/optimization_result.json

# 文本报告
cat stock_data/reports/indicator_systems.txt

# HTML报告(浏览器打开)
start stock_data/reports/indicator_systems.html
```

### 6. 复制结果回原项目

```bash
# 将以下文件复制回原项目的 stock_data/ 目录
cp stock_data/optimization_result.json /原项目路径/stock_data/
cp stock_data/reports/* /原项目路径/stock_data/reports/
```

## 详细说明

### 优化模式

| 模式 | 说明 | 耗时 |
|------|------|------|
| `global` | 全局优化，找通用基线参数 | ~1h |
| `regime` | 按体制分别优化，找体制专属参数 | ~2h |
| `robust` | 稳健优化，找跨体制都不差的参数 | ~1h |
| `iterative` | 多轮迭代精炼 | ~2h |
| `full` | 完整流程: 全局→体制→稳健→迭代 | ~6h |

### 预设配置

| 预设 | 种群 | 代数 | 样本 | 耗时 | 适用场景 |
|------|------|------|------|------|----------|
| `quick` | 30 | 10 | 20 | ~30min | 验证流程 |
| `standard` | 100 | 50 | 50 | ~4h | 日常优化 |
| `production` | 100 | 50 | 80 | ~6h | 生产环境 |
| `thorough` | 150 | 80 | 100 | ~10h | 最终优化 |

### 参数空间(25维)

**指标参数 (13个)**:
- MA: fast(3-30), mid(10-60), slow(15-120)
- MACD: fast(6-20), slow(18-40), signal(5-15)
- RSI: period(5-30), oversold(15-40), overbought(60-85)
- KDJ: n(5-21)
- BB: period(10-40), std(1.0-3.5)
- ATR: period(7-28)

**评分权重 (4个)**: 趋势/动量/风险/近期表现

**信号阈值 (4个)**: 买入/卖出信号强度, 综合评分阈值

**仓位参数 (4个)**: 止损(3-15%)/止盈(10-50%)/仓位(30-100%)/追踪止损(3-10%)

### 适应度函数

```
fitness = 0.40 * 达标率(>10%收益)
        + 0.25 * 中位收益率
        + 0.20 * 中位夏普比率
        + 0.15 * 回撤惩罚(越小越好)
```

### Walk-Forward验证

- 训练窗口: 504天(约2年)
- 测试窗口: 180天(约6个月)
- 步进: 63天(约3个月)
- 30年数据约产生110个样本外窗口

## 预期输出

优化完成后产出类似以下的多套指标体系:

```
==============================================================================
  指标体系 #1: "牛市追涨型"
  适用场景: 大盘处于上升趋势，MA60>MA120，市场情绪乐观
==============================================================================
  历史表现:
    中位收益率: +18.3%
    超10%收益概率: 87%
    中位夏普比: 1.42
    中位最大回撤: -12.1%

  指标参数:
    MA: fast=8, slow=30
    MACD: fast=10, slow=24, signal=7
    RSI: period=10, oversold=35, overbought=75
    评分权重: 趋势40% + 动量30% + 风险10% + 近期表现20%
    买入阈值: score>=55  止损/止盈: 8%/25%

==============================================================================
  指标体系 #2: "熊市防御型"
  适用场景: 大盘处于下降趋势，MA60<MA120，市场恐慌
==============================================================================
  ...

==============================================================================
  指标体系 #3: "震荡市高抛低吸型"
  适用场景: 大盘横盘整理，无明确方向
==============================================================================
  ...

==============================================================================
  指标体系 #4: "通用稳健型"
  适用场景: 不确定市场方向时的默认策略
==============================================================================
  ...
```

## 故障排除

**Q: akshare获取数据失败**
- 检查网络连接
- 尝试增加 `--delay 0.5`
- 部分股票可能已退市，属正常跳过

**Q: 优化太慢**
- 使用 `--preset quick` 先验证流程
- 减少 `--n-stocks` 或 `--pop` / `--gen`
- 每代约5分钟(15个体), 50代约4小时

**Q: 内存不足**
- 减少 `--n-stocks` (每只股票约占1MB内存)
- 使用64位Python

**Q: 优化结果不理想**
- 增加样本量 `--n-stocks 100`
- 增加代数 `--gen 80`
- 检查体制标签是否合理

## 文件说明

| 文件 | 用途 |
|------|------|
| `optimization_result.json` | 优化产出的多套指标体系(核心产物) |
| `regime_labels.csv` | 大盘每日体制标签 |
| `regime_report.txt` | 体制分类统计报告 |
| `reports/indicator_systems.*` | 可读的优化报告(text/md/html) |
| `historical/*.parquet` | 股票历史数据缓存 |
