# QuantOptima 致命级 Bug 修复计划

> 制定日期：2026-06-27
> 制定方式：grilling 会话逐题盘问
> 执行节奏：1 天集中完成（~15-22 小时）

## 背景

`skills/optimization_engine.py` 1358 行 + `skills/vectorized_backtest.py` 548 行审计发现 6 项致命级 bug，导致回测结果系统性偏差。本计划修复这些 bug 并建立严格的数据切分和验证流程。

## 修复总览

| 阶段 | 优先级 | 工作量 | 关键改动文件 |
|---|---|---|---|
| Phase 0 数据切分 | P0 | 1-2h | 新增 `data_splitter.py` |
| Phase 1 仓位/Sharpe | P0 | 4-6h | `vectorized_backtest.py` |
| Phase 2 T+1 信号化 | P0 | 4-6h | `vectorized_backtest.py` |
| Phase 3 统一 evaluate | P0 | 4-6h | `optimization_engine.py` |
| Phase 4 regime 文档 | P1 | 1h | `market_regime_classifier.py` |
| Phase 5 survivorship | P1 | 2-4h | `historical_data_manager.py` |
| 测试 + 报告 | - | 2-3h | 新增测试 + 报告 |

## 详细决策

### Phase 0 — 数据切分（P0）

| 决策项 | 决定 | 理由 |
|---|---|---|
| 切分依据 | 大盘日历 | 单股日历导致不同股票"train"覆盖不同周期 |
| train / val / test | 1996-2014 / 2015-2019 / 2020-2026 | 覆盖完整牛熊周期 |
| 体制标签同时间窗生成 | 是 | 防 regime 标签跨越 train/test 边界 |
| 阈值 BULL/BEAR | ≥ 300 天/bucket | 覆盖完整牛/熊段（30 年 A 股常见 2-3 年长熊） |
| 阈值 SIDEWAYS | ≥ 500 天/bucket | 震荡周期更长（2010-2014 长震 4 年） |
| 阈值 CRASH/RECOVERY | ≥ 60 天/bucket | 短周期 |
| 原始数据保留 | 是（全集不动） | 用于全集对照实验 |
| 数据布局 | `historical/*.parquet`（全集）+ `splits/{train,val,test}_symbols.txt` + `regime_labels_{train,val,test}.csv` | 清晰切分 |

### Phase 1 — GA 设计（P0）

| 决策项 | 决定 | 理由 |
|---|---|---|
| GA 训练数据 | **只在 train 上跑** | 防 data-snooping |
| 验证策略 | **selection-on-val** | GA 跑出 TOP-K，val 上评估选 best |
| 测试策略 | **test-once** | 选完之后 test 上只跑 1 次 |
| 排序逻辑 | 先 train_fitness 排序 → val 上重排 → 选 val-best | 标准做法（参考 López de Prado 第 11 章） |
| cash_drag | **取消**（保守） | 用户决策：实盘可加现金管理，backtest 不算 |
| 仓位公式 | 入场 E, stock=E×p×(1-fee), cash=E-stock; 出场 stock_value=stock×(1+pnl)×(1-fee_sell); total_equity=cash+stock_value | cash 不增长（保守） |
| 手续费 | fee_buy=0.0003, fee_sell=0.0013（佣金+印花税） | A 股标准费率 |
| Sharpe 公式 | equity curve 日对数收益 + sqrt(252) 年化 | 标准做法 |
| 无交易日 | log return = 0 | 不惩罚"没交易的日子" |
| 爆仓处理 | 单日损失 clip 到 -50% | 数据异常保护 |
| min_history_days | 180 天（一个 train 120 + 一个最小 test 60） | 降低门槛让更多短历史股票进入 |

### Phase 2 — T+1 信号化（P0）

| 决策项 | 决定 | 理由 |
|---|---|---|
| 指标输入 | `df['收盘'].shift(1)` | 防当日 close 引入 look-ahead |
| KDJ low/high | 也 shift(1) | 跟 close 保持一致 |
| 信号 shift | buy 和 sell 都 shift(1) | 严格 T+1 |
| 买入价 | 次日 open × buy_cost | 实盘 T+1 交易制度 |
| 止损 | 当日 low ≤ stop_price → 当日 stop_price 成交 | 保留（止损单当日有效） |
| 止盈 | 当日 high ≥ take_profit → 当日 take_profit 成交 | 保留（限价单当日有效） |
| 信号 sell | T 日 close 触发 → T+1 open 成交 | 防 look-ahead |
| 同日触发优先级 | **止损 > 止盈 > 信号 sell** | 风险递减 |
| warmup | `max(10, len(df)//3, max_indicator_period+1)` | 保证 warmup 覆盖最慢指标 |
| warmup 期间 | equity 不变（return=0） | 跟 Q8 一致 |

### Phase 3 — 统一 evaluate（P0）

| 决策项 | 决定 | 理由 |
|---|---|---|
| train_window | 120 天（全空间 ma_slow.high 固定） | 保证所有 params 看到同样起点 |
| 体制 walk-forward 子窗口 | 252 天（一年完整） | 覆盖季节性 |
| 分桶 vs 共享 GA | 分桶（每个 regime 单独 GA） | 实现简单 |
| target_rate 聚合 | median | 离散统计 |
| return 聚合 | mean | 让牛市大赚能反映 |
| sharpe 聚合 | median | 鲁棒 |
| drawdown 聚合 | median | worst-case 鲁棒 |
| val fitness | 同 train 公式 | 便于 gap 计算 |
| overfit_gap 阈值 | 0.2 | 中等 |
| 淘汰策略 | 放宽阈值重试（0.2 → 0.3 → 0.4） | 自动化 |
| 跨 regime 边界窗口 | 丢弃 | 防 regime 混合 |

### Phase 4 — Regime 防泄漏 ✓ 完成

| 决策项 | 决定 | 理由 |
|---|---|---|
| classify() 改造 | **不改**（已用历史窗口） | 原代码 pct_change 是历史的，无 look-ahead |
| 标签滞后 | 接受，report 写明"regime 标签有 0-20 天滞后" | 滞后 = 用历史窗口算标签的固有特性 |
| 滞后检测测试 | 跳过，docstring 写明 | time-reversal 测试不适用 |
| 辅助指标 | 保持简单（不加波动率/量能/宽度） | 加指标 → 维度增加 → regime 数据量更少 |
| 跨 regime 边界窗口 | 丢弃 | 跟 E2/R5 一致 |

### Phase 5 — Survivorship（P1）✓ 完成

| 决策项 | 决定 | 理由 |
|---|---|---|
| 数据源 | akshare `stock_zh_a_stop_em` + `survivorship_report.py` | 免费 + 可维护 |
| 残缺数据 | 截断到最后有效日期 | 实现简单 |
| 权重 | 跟现存股票**等权** | 默认无偏 |
| 报告 | 每次 run 自动生成 `survivorship_report.md` | 透明度 |
| 估算退市股 | **2235 只**（已知覆盖：1942 现存 + 293 停牌） | A 股 30 年估算 |

**实际结果（2026-06-27）**：
- Universe：1942 只（2002-2026）
- 停牌：293 只（akshare `stock_zh_a_stop_em`）
- 全A覆盖：2235/5867（38.1%）
- 估算 bias：2%/年，30 年累计 **≈80%**（复利）
- 报告：`stock_data/survivorship_report.md`

### Meta 决策

| 决策项 | 决定 |
|---|---|
| M1 旧路径 | (b) 保留 `legacy_evaluate()` 作为对照 |
| M2 文件命名 | (c) 旧版自动备份 `optimization_result_legacy.json` |
| M3 审计段 | 加"修复审计"段（修复前 vs 修复后） |
| M4 节奏 | 1 天集中完成 |

## 执行顺序

```
P0-1: 数据切分 (Phase 0)              [阻塞后续所有]
  ↓
P0-2: T+1 信号化 (Phase 2)            [vectorized_backtest.py]
P0-3: 仓位公式重写 (Phase 1 Q7)        [同上]
P0-4: Sharpe 重写 (Phase 1 Q8)         [同上]
  (P0-2/3/4 一起改)
  ↓
P0-5: 统一 evaluate 路径 (Phase 3)     [依赖 P0-2/3/4]
  ↓
P0-6: GA-on-train + selection-on-val   [依赖 P0-5]
  ↓
P1-1: regime 标签生成 (Phase 0)        [跟 P0-1 一起]
P1-2: regime classifier docstring     [文档]
  ↓
P1-3: 退市股数据 (Phase 5)            [独立]
  ↓
P1-4: survivorship_report 生成        [依赖 P1-3]
```

## 修复审计模板

最终报告需包含：

```markdown
## 修复审计

| 决策 | 修复前行为 | 修复后行为 | 影响 |
|---|---|---|---|
| 数据切分 | 全部数据用于 GA | train/val/test 切分 | 消除过拟合 |
| T+1 信号 | 当日 close 入场 | 次日 open 入场 | 消除 look-ahead |
| 仓位公式 | cumprod 简化 | 含手续费 + cash 不增长 | 反映真实资金曲线 |
| Sharpe 公式 | 错误年化 | equity curve 日对数 + sqrt(252) | 标准化 |
| train_window | 504 天但未真正训练 | 120 天用于 warmup | 统一 warmup |
| 体制 walk-forward | 不一致路径 | 统一 252 天子窗口 | 可比性 |
| cash_drag | 默认 0 | 0（用户决策） | 保守 |
| survivorship | 仅现存股票 | + 退市股（~500） | 消除 ~5-15% 收益高估 |
| overfit 检测 | 无 | gap 阈值 0.2 + 放宽重试 | 自动检测过拟合 |
| 修复审计 | 无 | 本段 | 透明度 |
```

## 关键代码位置

- `skills/optimization_engine.py` — 主改动（GA、evaluate、fitness）
- `skills/vectorized_backtest.py` — 主改动（T+1、仓位、Sharpe）
- `skills/market_regime_classifier.py` — 仅文档改动
- `skills/historical_data_manager.py` — survivorship 改动
- 新增 `skills/data_splitter.py` — 数据切分工具
- 新增 `skills/legacy_evaluate.py` — 旧路径保留

## 不在本次修复范围

- S 级别（严重级）问题：留待下个 sprint
- M/L 级别（中/轻级）问题：留待 code review
- 加辅助 regime 指标：用户决定不加
- 跨 regime 复杂处理：当前丢弃窗口
