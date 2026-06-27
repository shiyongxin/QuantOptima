# -*- coding: utf-8 -*-
"""
向量化回测引擎 - Vectorized Backtest Engine (P0-1 修复版本)

修复日志（2026-06-27，见 FIX_PLAN.md）：
- Phase 1 仓位公式：cash 不增长；手续费算入；stock_fund = E × p × (1 - fee_buy)
- Phase 1 Sharpe：equity curve 日对数收益 + sqrt(252) 年化；爆仓 clip -50%
- Phase 2 T+1 信号：所有指标 shift(1)；buy/sell 都 shift(1)；信号出场用次日 open
- 同日触发优先级：stop_loss > trailing_stop > take_profit > signal sell
- 旧路径保留为 _legacy_* / legacy_backtest()，用于 M1 决策的修复前/后对照

新路径：backtest()  →  _compute_indicators_v2 / _generate_signals_v2 / _simulate_trades_v2
旧路径：legacy_backtest()  →  _compute_indicators_legacy / _generate_signals_legacy / _simulate_trades_legacy
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List


# ==================== 数据结构 ====================

@dataclass
class BacktestMetrics:
    """回测结果指标"""
    total_return: float = 0.0       # 总收益率(%)
    annualized_return: float = 0.0  # 年化收益率(%)
    max_drawdown: float = 0.0       # 最大回撤(%)
    sharpe_ratio: float = 0.0       # 夏普比率
    win_rate: float = 0.0           # 胜率(%)
    profit_factor: float = 0.0      # 盈亏比
    num_trades: int = 0             # 交易次数
    avg_holding_days: float = 0.0   # 平均持有天数
    avg_return_per_trade: float = 0.0  # 平均每笔收益(%)
    max_consecutive_losses: int = 0 # 最大连续亏损次数
    # 新增：equity curve（v2 路径用）
    equity_curve: List[float] = field(default_factory=list)


@dataclass
class TradeDetail:
    """单笔交易详情"""
    entry_idx: int          # 入场日在原始数据中的索引（v2: 实际入场日，v1: 信号日）
    exit_idx: int           # 出场日在原始数据中的索引
    entry_price: float
    exit_price: float
    pnl_pct: float          # 收益率(%)
    holding_days: int
    exit_reason: str        # "stop_loss" / "take_profit" / "trailing_stop" / "signal" / "end"


# ==================== 主类 ====================

class VectorizedBacktester:
    """
    向量化回测引擎

    默认 backtest() 走 v2（T+1 + 新仓位公式 + 新 Sharpe）；
    legacy_backtest() 保留修复前行为用于对照。
    """

    def __init__(self, commission=0.0003, slippage=0.001,
                 fee_buy=0.0003, fee_sell=0.0013,
                 risk_free_rate=0.03,
                 crash_clip_pct=0.5):
        """
        Parameters
        ----------
        commission : float
            默认佣金率（双向，legacy 用）
        slippage : float
            滑点率
        fee_buy : float
            买入费率（佣金 + 杂费）
        fee_sell : float
            卖出席费率（佣金 + 印花税 + 杂费）
        risk_free_rate : float
            年化无风险利率（Sharpe 计算用，0.03 = 3%）
        crash_clip_pct : float
            单日最大损失 clip（0.5 = 50%）
        """
        self.commission = commission
        self.slippage = slippage
        self.fee_buy = fee_buy
        self.fee_sell = fee_sell
        self.risk_free_rate = risk_free_rate
        self.crash_clip_pct = crash_clip_pct

    # ========== 公共入口 ==========

    def backtest(self, data: pd.DataFrame, params: dict) -> BacktestMetrics:
        """
        对单只股票运行回测（v2 路径：T+1 + 新仓位 + 新 Sharpe）

        Parameters
        ----------
        data : pd.DataFrame
            历史数据，必须包含 日期, 开盘, 最高, 最低, 收盘, 成交量
        params : dict
            参数字典，来自 ParameterSpace
        """
        if len(data) < 15:
            return BacktestMetrics()
        df = self._compute_indicators_v2(data, params)
        signals = self._generate_signals_v2(df, params)
        return self._simulate_trades_v2(df, signals, params)

    def legacy_backtest(self, data: pd.DataFrame, params: dict) -> BacktestMetrics:
        """修复前的旧路径（保留用于 M1 决策：修复前/后对照）"""
        if len(data) < 15:
            return BacktestMetrics()
        df = self._compute_indicators_legacy(data, params)
        signals = self._generate_signals_legacy(df, params)
        return self._simulate_trades_legacy(df, signals, params)

    def backtest_with_details(self, data: pd.DataFrame, params: dict,
                              legacy: bool = False) -> tuple:
        """
        回测并返回交易详情

        Returns
        -------
        (BacktestMetrics, list[TradeDetail])
        """
        if len(data) < 15:
            return BacktestMetrics(), []
        if legacy:
            df = self._compute_indicators_legacy(data, params)
            signals = self._generate_signals_legacy(df, params)
            return self._simulate_trades_legacy(df, signals, params, return_details=True)
        df = self._compute_indicators_v2(data, params)
        signals = self._generate_signals_v2(df, params)
        return self._simulate_trades_v2(df, signals, params, return_details=True)

    def backtest_batch(self, stock_data_dict: dict, params: dict,
                       legacy: bool = False) -> pd.DataFrame:
        """
        同一参数回测多只股票

        Returns
        -------
        pd.DataFrame : 每只股票一行，包含各项指标
        """
        results = []
        for symbol, data in stock_data_dict.items():
            try:
                metrics = self.legacy_backtest(data, params) if legacy else self.backtest(data, params)
                results.append({
                    'symbol': symbol,
                    'total_return': metrics.total_return,
                    'annualized_return': metrics.annualized_return,
                    'max_drawdown': metrics.max_drawdown,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'num_trades': metrics.num_trades,
                    'avg_holding_days': metrics.avg_holding_days,
                    'avg_return_per_trade': metrics.avg_return_per_trade,
                })
            except Exception:
                pass
        return pd.DataFrame(results)

    def walk_forward(self, data: pd.DataFrame, params: dict,
                     train_days=504, test_days=180, step_days=63) -> list:
        """
        Walk-forward 验证（修复前/后都用同一个简单窗口实现）

        注：Phase 3 会替换为统一 evaluate 路径。当前保留供 optimization_engine.py 兼容。
        """
        if len(data) < train_days + test_days:
            return []
        results = []
        idx = train_days
        while idx + test_days <= len(data):
            test_data = data.iloc[idx:idx + test_days]
            metrics = self.backtest(test_data, params)
            results.append(metrics)
            idx += step_days
        return results

    # ======================================================================
    #   v2 路径（T+1 + 新仓位公式 + 新 Sharpe）
    # ======================================================================

    def _compute_indicators_v2(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        v2: 所有指标用昨日 close 计算（防 look-ahead）

        D1 决策：所有指标 shift(1)，KDJ 的 low/high 也 shift
        """
        df = data.copy()
        # 关键：所有价格数据先 shift(1)，让指标基于"昨日及之前"的数据
        close_prev = df['收盘'].shift(1).astype(float)
        high_prev = df['最高'].shift(1).astype(float)
        low_prev = df['最低'].shift(1).astype(float)
        # volume 不影响 look-ahead（公开数据），可不 shift；但保守起见也 shift
        vol_prev = df['成交量'].shift(1).astype(float)

        # ---- 移动平均线 ----
        ma_fast_period = int(params.get('ma_fast', 5))
        ma_slow_period = int(params.get('ma_slow', 20))
        ma_mid_period = int(params.get('ma_mid', 10))
        df['ma_fast'] = close_prev.rolling(ma_fast_period).mean()
        df['ma_slow'] = close_prev.rolling(ma_slow_period).mean()
        df['ma_mid'] = close_prev.rolling(ma_mid_period).mean()

        # ---- MACD ----
        macd_fast = int(params.get('macd_fast', 12))
        macd_slow = int(params.get('macd_slow', 26))
        macd_signal_period = int(params.get('macd_signal', 9))
        ema_fast = close_prev.ewm(span=macd_fast, adjust=False).mean()
        ema_slow = close_prev.ewm(span=macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=macd_signal_period, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ---- RSI ----
        rsi_period = int(params.get('rsi_period', 14))
        delta = close_prev.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # ---- KDJ (用昨日 high/low/close) ----
        kdj_n = int(params.get('kdj_n', 9))
        low_n = low_prev.rolling(kdj_n).min()
        high_n = high_prev.rolling(kdj_n).max()
        rsv = (close_prev - low_n) / (high_n - low_n).replace(0, np.nan) * 100
        df['k'] = rsv.ewm(span=3, adjust=False).mean()
        df['d'] = df['k'].ewm(span=3, adjust=False).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']

        # ---- 布林带 ----
        bb_period = int(params.get('bb_period', 20))
        bb_std_mult = float(params.get('bb_std', 2.0))
        df['bb_mid'] = close_prev.rolling(bb_period).mean()
        bb_std = close_prev.rolling(bb_period).std()
        df['bb_upper'] = df['bb_mid'] + bb_std_mult * bb_std
        df['bb_lower'] = df['bb_mid'] - bb_std_mult * bb_std

        # ---- ATR ----
        atr_period = int(params.get('atr_period', 14))
        tr1 = high_prev - low_prev
        tr2 = (high_prev - close_prev.shift(1)).abs()
        tr3 = (low_prev - close_prev.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(atr_period).mean()

        # ---- 成交量均线 ----
        df['vol_ma5'] = vol_prev.rolling(5).mean()
        df['vol_ma20'] = vol_prev.rolling(20).mean()

        # ---- 涨跌幅（不参与信号生成，仅供报告）----
        df['ret_1d'] = df['收盘'].pct_change() * 100
        df['ret_5d'] = df['收盘'].pct_change(5) * 100
        df['ret_20d'] = df['收盘'].pct_change(20) * 100

        return df

    def _generate_signals_v2(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        v2: 信号在 T 日 close 时基于 T-1 指标生成；buy/sell 都 shift(1)

        D2 决策：buy 和 sell 都 shift(1)
        返回的 signals[t]=1 表示 T 日 close 决定 → T+1 日 open 入场
        """
        buy_threshold = float(params.get('buy_threshold', 2.0))
        sell_threshold = float(params.get('sell_threshold', 2.0))
        rsi_oversold = float(params.get('rsi_oversold', 30))
        rsi_overbought = float(params.get('rsi_overbought', 70))

        # 买入得分（基于昨日指标的信号）
        buy_score = pd.Series(0.0, index=df.index)
        ma_cross_up = (df['ma_fast'] > df['ma_slow']) & (df['ma_fast'].shift(1) <= df['ma_slow'].shift(1))
        buy_score += ma_cross_up.astype(float) * 1.0
        price_above_slow = df['收盘'] > df['ma_slow']  # 收盘 vs 昨日 MA 慢线
        buy_score += price_above_slow.astype(float) * 0.3
        macd_cross_up = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        buy_score += macd_cross_up.astype(float) * 1.0
        macd_hist_turn_pos = (df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0)
        buy_score += macd_hist_turn_pos.astype(float) * 0.5
        rsi_oversold_signal = df['rsi'] < rsi_oversold
        buy_score += rsi_oversold_signal.astype(float) * 0.5
        kdj_cross_up = (df['k'] > df['d']) & (df['k'].shift(1) <= df['d'].shift(1))
        buy_score += kdj_cross_up.astype(float) * 0.8
        bb_lower_touch = (df['收盘'] <= df['bb_lower'] * 1.02) & (df['收盘'] > df['bb_lower'] * 0.98)
        buy_score += bb_lower_touch.astype(float) * 0.5
        volume_surge = df['成交量'] > df['vol_ma5'] * 1.5
        buy_score += volume_surge.astype(float) * 0.3

        # 卖出得分
        sell_score = pd.Series(0.0, index=df.index)
        ma_cross_down = (df['ma_fast'] < df['ma_slow']) & (df['ma_fast'].shift(1) >= df['ma_slow'].shift(1))
        sell_score += ma_cross_down.astype(float) * 1.0
        price_below_slow = df['收盘'] < df['ma_slow']
        sell_score += price_below_slow.astype(float) * 0.3
        macd_cross_down = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        sell_score += macd_cross_down.astype(float) * 1.0
        rsi_overbought_signal = df['rsi'] > rsi_overbought
        sell_score += rsi_overbought_signal.astype(float) * 0.5
        kdj_cross_down = (df['k'] < df['d']) & (df['k'].shift(1) >= df['d'].shift(1))
        sell_score += kdj_cross_down.astype(float) * 0.8
        bb_upper_touch = (df['收盘'] >= df['bb_upper'] * 0.98) & (df['收盘'] < df['bb_upper'] * 1.02)
        sell_score += bb_upper_touch.astype(float) * 0.5

        # 生成最终信号
        signals = pd.Series(0, index=df.index)
        signals[buy_score >= buy_threshold] = 1
        signals[sell_score >= sell_threshold] = -1

        # 同日冲突：卖出优先
        conflict = (buy_score >= buy_threshold) & (sell_score >= sell_threshold)
        signals[conflict] = -1

        # D2 关键：buy 和 sell 都 shift(1)
        # shift 后 signals[t]=1 表示"基于 t-1 日 close 决定的信号，t 日 open 执行"
        signals = signals.shift(1).fillna(0).astype(int)

        # Warmup：覆盖最慢指标周期 +1
        max_ind = max(
            int(params.get('ma_slow', 20)),
            int(params.get('macd_slow', 26)),
            int(params.get('bb_period', 20)),
            int(params.get('atr_period', 14)),
        )
        warmup = max(10, min(60, len(df) // 3), max_ind + 1)
        signals.iloc[:warmup] = 0

        return signals

    def _simulate_trades_v2(self, df: pd.DataFrame, signals: pd.Series,
                            params: dict, return_details: bool = False):
        """
        v2: T+1 入场/出场 + 新仓位公式 + equity curve 追踪

        仓位公式（Q7 决策）：
          入场：stock_fund = E × p × (1 - fee_buy); cash = E × (1 - p)
          出场：stock_value = stock_fund × (1 + pnl/100) × (1 - fee_sell)
          total_equity = cash + stock_value
          cash 不增长（保守）

        Sharpe（Q8 决策）：
          equity curve 日对数收益 + sqrt(252) 年化
          无交易日 = 0；爆仓 clip -50%
        """
        if len(df) < 15:
            return (BacktestMetrics(), []) if return_details else BacktestMetrics()

        # 提取参数
        stop_loss_pct = float(params.get('stop_loss_pct', 0.08))
        take_profit_pct = float(params.get('take_profit_pct', 0.20))
        position_size_pct = float(params.get('position_size_pct', 0.8))
        trailing_stop_pct = float(params.get('trailing_stop_pct', 0.05))
        min_holding = int(params.get('min_holding_days', 20))

        close = df['收盘'].values.astype(np.float64)
        high = df['最高'].values.astype(np.float64)
        low = df['最低'].values.astype(np.float64)
        open_ = df['开盘'].values.astype(np.float64)
        signals_arr = signals.values.astype(np.int8)

        n = len(close)
        if n < 2:
            return (BacktestMetrics(), []) if return_details else BacktestMetrics()

        # 在 v2 中，signals[t]=1 表示"t 日 open 入场"
        # （即"t-1 日 close 看到的买入信号"）
        buy_action_days = np.where(signals_arr == 1)[0]
        # 卖出信号同样：signals[t]=-1 表示"t 日 open 出场"
        sell_action_days = set(np.where(signals_arr == -1)[0].tolist())

        # Warmup：与 _generate_signals_v2 一致
        max_ind = max(
            int(params.get('ma_slow', 20)),
            int(params.get('macd_slow', 26)),
            int(params.get('bb_period', 20)),
            int(params.get('atr_period', 14)),
        )
        warmup = max(10, min(60, n // 3), max_ind + 1)
        # 过滤掉 warmup 内的 entry
        buy_action_days = buy_action_days[buy_action_days >= warmup]

        # 状态变量
        current_equity = 1.0
        equity_curve = np.full(n, 1.0)  # 默认 1.0（无交易的日子）
        trades = []
        trade_details = [] if return_details else None

        fee_buy = self.fee_buy
        fee_sell = self.fee_sell
        slip = self.slippage

        for i, entry_day in enumerate(buy_action_days):
            if entry_day >= n:
                break

            # ---- 入场（v2: open[t] 含滑点）----
            E = current_equity
            cash = E * (1 - position_size_pct)         # cash 不增长
            stock_fund = E * position_size_pct * (1 - fee_buy)
            entry_price = open_[entry_day] * (1 + slip)

            if entry_price <= 0:
                continue  # 异常数据，跳过

            # 持仓期上界：下一笔 entry day（如果有）
            next_entry = int(buy_action_days[i + 1]) if i + 1 < len(buy_action_days) else n
            # 持仓期：[entry_day, next_entry)
            # 但 entry_day 当天已经发生入场；扫描从 entry_day + 1 开始

            # 准备追踪止损状态
            cummax = entry_price
            min_hold_end = entry_day + min_holding
            stop_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)

            exit_day = None
            exit_price = None
            exit_reason = None

            # ---- 扫描持仓期内的出场 ----
            for day in range(entry_day + 1, next_entry):
                if day >= n:
                    break

                # 优先级 1: 硬止损（任何时候可触发）
                if low[day] <= stop_price:
                    exit_day = day
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                    break

                # 最小持有期内不触发止盈/追踪
                if day < min_hold_end:
                    continue

                # 更新 cummax
                if high[day] > cummax:
                    cummax = high[day]

                trail_stop = cummax * (1 - trailing_stop_pct)

                # 优先级 2: 追踪止损
                if low[day] <= trail_stop:
                    exit_day = day
                    exit_price = trail_stop
                    exit_reason = "trailing_stop"
                    break

                # 优先级 3: 止盈
                if high[day] >= take_profit_price:
                    exit_day = day
                    exit_price = take_profit_price
                    exit_reason = "take_profit"
                    break

                # 优先级 4: 信号 sell（signals[day]=-1 表示 day open 出场）
                if day in sell_action_days:
                    exit_day = day
                    exit_price = open_[day] * (1 - slip - fee_sell)
                    exit_reason = "signal"
                    break

            # 如果没找到出场点 → 持有到持仓期结束
            if exit_day is None:
                exit_day = min(next_entry - 1, n - 1)
                if exit_day <= entry_day:
                    exit_day = entry_day
                exit_price = close[exit_day]
                exit_reason = "end"

            # ---- 计算 pnl 和新 equity ----
            holding_days = exit_day - entry_day
            pnl_pct = (exit_price / entry_price - 1) * 100

            # v2 仓位公式
            stock_value = stock_fund * (1 + pnl_pct / 100) * (1 - fee_sell)
            new_equity = cash + stock_value

            # ---- 记录交易 ----
            trades.append({
                'entry_idx': int(entry_day),
                'exit_idx': int(exit_day),
                'entry_price': float(entry_price),
                'exit_price': float(exit_price),
                'pnl_pct': float(pnl_pct),
                'holding_days': int(holding_days),
                'exit_reason': exit_reason,
            })
            if return_details:
                trade_details.append(TradeDetail(
                    entry_idx=int(entry_day),
                    exit_idx=int(exit_day),
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    pnl_pct=float(pnl_pct),
                    holding_days=int(holding_days),
                    exit_reason=exit_reason,
                ))

            # ---- 填充 equity curve ----
            # 从 entry_day 到 exit_day：equity = cash + position_value
            # entry_day 收盘：position_value = stock_fund * (close[entry_day] / entry_price)
            if entry_day < n:
                pos_val_entry = stock_fund * (close[entry_day] / entry_price)
                equity_curve[entry_day] = cash + pos_val_entry
            for day in range(entry_day + 1, exit_day):
                if day < n:
                    pos_val = stock_fund * (close[day] / entry_price)
                    equity_curve[day] = cash + pos_val
            # exit_day：出场后 equity = new_equity
            if exit_day < n:
                equity_curve[exit_day] = new_equity
            # exit_day+1 到 next_entry-1：保持 new_equity（cash 不增长）
            for day in range(exit_day + 1, next_entry):
                if day < n:
                    equity_curve[day] = new_equity

            current_equity = new_equity

        # ---- 计算 BacktestMetrics ----
        if not trades:
            return (BacktestMetrics(equity_curve=equity_curve.tolist()), []) if return_details \
                   else BacktestMetrics(equity_curve=equity_curve.tolist())

        # 基本交易统计
        pnl_arr = np.array([t['pnl_pct'] for t in trades], dtype=np.float64)
        holding_arr = np.array([t['holding_days'] for t in trades], dtype=np.float64)
        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr <= 0]
        win_rate = (len(wins) / len(pnl_arr)) * 100
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

        # 最大连续亏损
        is_loss = pnl_arr <= 0
        if np.any(is_loss):
            loss_streaks = np.diff(np.where(np.concatenate([[False], is_loss, [False]]))[0])
            max_consec = int(loss_streaks.max()) if len(loss_streaks) > 0 else 0
        else:
            max_consec = 0

        # 总收益 / 年化（基于 equity curve）
        total_return = (equity_curve[-1] - 1) * 100
        years = n / 252
        if years > 0 and equity_curve[-1] > 0:
            annualized_return = (equity_curve[-1] ** (1 / years) - 1) * 100
        else:
            annualized_return = 0

        # 最大回撤（基于 equity curve）
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak * 100
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Sharpe（基于 equity curve 日对数收益 + sqrt(252) + 爆仓 clip）
        sharpe = self._compute_sharpe_from_equity(equity_curve, n)

        metrics = BacktestMetrics(
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            max_drawdown=float(max_drawdown),
            sharpe_ratio=float(sharpe),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            num_trades=len(trades),
            avg_holding_days=float(np.mean(holding_arr)),
            avg_return_per_trade=float(np.mean(pnl_arr)),
            max_consecutive_losses=int(max_consec),
            equity_curve=equity_curve.tolist(),
        )

        if return_details:
            return metrics, trade_details
        return metrics

    def _compute_sharpe_from_equity(self, equity_curve: np.ndarray, n: int) -> float:
        """
        从 equity curve 计算 Sharpe（Q8 决策）

        - 日对数收益：r[t] = log(equity[t] / equity[t-1])
        - 无变化的日子：r[t] = 0（不惩罚"没交易的日子"）
        - 单日损失 clip -50%（防异常值）
        - annual_sharpe = (mean(r) * 252 - rf) / (std(r) * sqrt(252))
        """
        if n < 2:
            return 0.0
        # 日对数收益
        prev = equity_curve[:-1]
        curr = equity_curve[1:]
        # 避免 log(0) / log(负数)
        prev = np.maximum(prev, 1e-9)
        curr = np.maximum(curr, 1e-9)
        daily_returns = np.log(curr / prev)
        # 单日损失 clip 到 -50%
        daily_returns = np.maximum(daily_returns, -self.crash_clip_pct)
        if len(daily_returns) < 2:
            return 0.0
        mean_r = np.mean(daily_returns)
        std_r = np.std(daily_returns)
        if std_r <= 1e-9:
            return 0.0
        # 年化 Sharpe（无风险利率 0.03）
        annual_sharpe = (mean_r * 252 - self.risk_free_rate) / (std_r * np.sqrt(252))
        return float(annual_sharpe)

    # ======================================================================
    #   Legacy 路径（保留修复前行为，M1 决策要求修复前/后对照）
    # ======================================================================

    def _compute_indicators_legacy(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        """原始指标计算（用当日 close，无 shift）"""
        df = data.copy()
        close = df['收盘'].astype(float)
        high = df['最高'].astype(float)
        low = df['最低'].astype(float)
        volume = df['成交量'].astype(float)

        ma_fast_period = int(params.get('ma_fast', 5))
        ma_slow_period = int(params.get('ma_slow', 20))
        ma_mid_period = int(params.get('ma_mid', 10))
        df['ma_fast'] = close.rolling(ma_fast_period).mean()
        df['ma_slow'] = close.rolling(ma_slow_period).mean()
        df['ma_mid'] = close.rolling(ma_mid_period).mean()

        macd_fast = int(params.get('macd_fast', 12))
        macd_slow = int(params.get('macd_slow', 26))
        macd_signal_period = int(params.get('macd_signal', 9))
        ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=macd_signal_period, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        rsi_period = int(params.get('rsi_period', 14))
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        kdj_n = int(params.get('kdj_n', 9))
        low_n = low.rolling(kdj_n).min()
        high_n = high.rolling(kdj_n).max()
        rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100
        df['k'] = rsv.ewm(span=3, adjust=False).mean()
        df['d'] = df['k'].ewm(span=3, adjust=False).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']

        bb_period = int(params.get('bb_period', 20))
        bb_std_mult = float(params.get('bb_std', 2.0))
        df['bb_mid'] = close.rolling(bb_period).mean()
        bb_std = close.rolling(bb_period).std()
        df['bb_upper'] = df['bb_mid'] + bb_std_mult * bb_std
        df['bb_lower'] = df['bb_mid'] - bb_std_mult * bb_std

        atr_period = int(params.get('atr_period', 14))
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(atr_period).mean()

        df['vol_ma5'] = volume.rolling(5).mean()
        df['vol_ma20'] = volume.rolling(20).mean()

        df['ret_1d'] = close.pct_change() * 100
        df['ret_5d'] = close.pct_change(5) * 100
        df['ret_20d'] = close.pct_change(20) * 100
        return df

    def _generate_signals_legacy(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """原始信号生成（无 shift，买入用 close 当日）"""
        buy_threshold = float(params.get('buy_threshold', 2.0))
        sell_threshold = float(params.get('sell_threshold', 2.0))
        rsi_oversold = float(params.get('rsi_oversold', 30))
        rsi_overbought = float(params.get('rsi_overbought', 70))

        buy_score = pd.Series(0.0, index=df.index)
        ma_cross_up = (df['ma_fast'] > df['ma_slow']) & (df['ma_fast'].shift(1) <= df['ma_slow'].shift(1))
        buy_score += ma_cross_up.astype(float) * 1.0
        price_above_slow = df['收盘'] > df['ma_slow']
        buy_score += price_above_slow.astype(float) * 0.3
        macd_cross_up = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        buy_score += macd_cross_up.astype(float) * 1.0
        macd_hist_turn_pos = (df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0)
        buy_score += macd_hist_turn_pos.astype(float) * 0.5
        rsi_oversold_signal = df['rsi'] < rsi_oversold
        buy_score += rsi_oversold_signal.astype(float) * 0.5
        kdj_cross_up = (df['k'] > df['d']) & (df['k'].shift(1) <= df['d'].shift(1))
        buy_score += kdj_cross_up.astype(float) * 0.8
        bb_lower_touch = (df['收盘'] <= df['bb_lower'] * 1.02) & (df['收盘'] > df['bb_lower'] * 0.98)
        buy_score += bb_lower_touch.astype(float) * 0.5
        volume_surge = df['成交量'] > df['vol_ma5'] * 1.5
        buy_score += volume_surge.astype(float) * 0.3

        sell_score = pd.Series(0.0, index=df.index)
        ma_cross_down = (df['ma_fast'] < df['ma_slow']) & (df['ma_fast'].shift(1) >= df['ma_slow'].shift(1))
        sell_score += ma_cross_down.astype(float) * 1.0
        price_below_slow = df['收盘'] < df['ma_slow']
        sell_score += price_below_slow.astype(float) * 0.3
        macd_cross_down = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        sell_score += macd_cross_down.astype(float) * 1.0
        rsi_overbought_signal = df['rsi'] > rsi_overbought
        sell_score += rsi_overbought_signal.astype(float) * 0.5
        kdj_cross_down = (df['k'] < df['d']) & (df['k'].shift(1) >= df['d'].shift(1))
        sell_score += kdj_cross_down.astype(float) * 0.8
        bb_upper_touch = (df['收盘'] >= df['bb_upper'] * 0.98) & (df['收盘'] < df['bb_upper'] * 1.02)
        sell_score += bb_upper_touch.astype(float) * 0.5

        signals = pd.Series(0, index=df.index)
        signals[buy_score >= buy_threshold] = 1
        signals[sell_score >= sell_threshold] = -1
        conflict = (buy_score >= buy_threshold) & (sell_score >= sell_threshold)
        signals[conflict] = -1

        warmup = min(60, max(10, len(df) // 3))
        signals.iloc[:warmup] = 0
        return signals

    def _simulate_trades_legacy(self, df: pd.DataFrame, signals: pd.Series,
                                 params: dict, return_details: bool = False):
        """原始交易模拟（当日 close 入场，错误年化 Sharpe）"""
        if len(df) < 15:
            return (BacktestMetrics(), []) if return_details else BacktestMetrics()

        stop_loss_pct = float(params.get('stop_loss_pct', 0.08))
        take_profit_pct = float(params.get('take_profit_pct', 0.20))
        position_size_pct = float(params.get('position_size_pct', 0.8))
        trailing_stop_pct = float(params.get('trailing_stop_pct', 0.05))
        min_holding = int(params.get('min_holding_days', 20))

        close = df['收盘'].values.astype(np.float64)
        high_arr = df['最高'].values.astype(np.float64)
        low_arr = df['最低'].values.astype(np.float64)
        signals_arr = signals.values.astype(np.int8)

        n = len(close)
        if n < 2:
            return (BacktestMetrics(), []) if return_details else BacktestMetrics()

        buy_cost = 1 + self.slippage + self.commission
        sell_cost = 1 - self.slippage - self.commission
        buy_indices = np.where(signals_arr == 1)[0]
        if len(buy_indices) == 0:
            return (BacktestMetrics(), []) if return_details else BacktestMetrics()

        trades_pnl = []
        trades_holding = []
        trade_details = [] if return_details else None

        for idx in range(len(buy_indices)):
            entry_idx = buy_indices[idx]
            entry_price = close[entry_idx] * buy_cost
            take_profit_price = entry_price * (1 + take_profit_pct)
            stop_price = entry_price * (1 - stop_loss_pct)

            if idx + 1 < len(buy_indices):
                end_idx = buy_indices[idx + 1]
            else:
                end_idx = n

            period_close = close[entry_idx:end_idx]
            period_high = high_arr[entry_idx:end_idx]
            period_low = low_arr[entry_idx:end_idx]
            period_sell_signals = signals_arr[entry_idx:end_idx]

            cummax = np.maximum.accumulate(period_high)
            trailing_stops = cummax * (1 - trailing_stop_pct)
            effective_stops = np.maximum(stop_price, trailing_stops)

            check_start = min(1, len(period_close) - 1)
            period_len = len(period_close)
            min_hold_end = min(min_holding, period_len)

            hit_stop_hard = period_low[check_start:] <= stop_price
            stop_hard_triggers = np.where(hit_stop_hard)[0] + check_start

            if min_hold_end < period_len:
                hit_stop_trail = period_low[min_hold_end:] <= trailing_stops[min_hold_end:]
                stop_trail_triggers = np.where(hit_stop_trail)[0] + min_hold_end
            else:
                stop_trail_triggers = np.array([], dtype=int)

            if min_hold_end < period_len:
                hit_tp = period_high[min_hold_end:] >= take_profit_price
                tp_triggers = np.where(hit_tp)[0] + min_hold_end
            else:
                tp_triggers = np.array([], dtype=int)

            if min_hold_end < period_len:
                hit_sell = period_sell_signals[min_hold_end:] == -1
                sell_triggers = np.where(hit_sell)[0] + min_hold_end
            else:
                sell_triggers = np.array([], dtype=int)

            exit_reason = None
            exit_offset = len(period_close)
            if len(stop_hard_triggers) > 0:
                exit_reason = "stop_loss"
                exit_offset = stop_hard_triggers[0]
            elif len(stop_trail_triggers) > 0:
                exit_reason = "trailing_stop"
                exit_offset = stop_trail_triggers[0]
            elif len(tp_triggers) > 0:
                exit_reason = "take_profit"
                exit_offset = tp_triggers[0]
            elif len(sell_triggers) > 0:
                exit_reason = "signal"
                exit_offset = sell_triggers[0]

            if exit_offset >= len(period_close):
                exit_offset = len(period_close) - 1

            if exit_reason == "stop_loss":
                exit_price = stop_price
            elif exit_reason == "trailing_stop":
                exit_price = trailing_stops[exit_offset]
            elif exit_reason == "take_profit":
                exit_price = take_profit_price
            else:
                exit_price = period_close[exit_offset] * sell_cost

            pnl_pct = (exit_price / entry_price - 1) * 100
            holding_days = exit_offset

            trades_pnl.append(pnl_pct)
            trades_holding.append(holding_days)
            if return_details:
                trade_details.append(TradeDetail(
                    entry_idx=int(entry_idx),
                    exit_idx=int(entry_idx + exit_offset),
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    pnl_pct=float(pnl_pct),
                    holding_days=int(holding_days),
                    exit_reason=str(exit_reason) if exit_reason else "end",
                ))

        if len(trades_pnl) == 0:
            return (BacktestMetrics(), []) if return_details else BacktestMetrics()

        pnl_arr = np.array(trades_pnl, dtype=np.float64)
        holding_arr = np.array(trades_holding, dtype=np.float64)
        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr <= 0]
        win_rate = (len(wins) / len(pnl_arr)) * 100 if len(pnl_arr) > 0 else 0
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

        cumulative = np.cumprod(1 + pnl_arr / 100 * position_size_pct)
        total_return = (cumulative[-1] - 1) * 100 if len(cumulative) > 0 else 0
        years = n / 252
        annualized_return = 0
        if years > 0 and cumulative[-1] > 0:
            annualized_return = (cumulative[-1] ** (1 / years) - 1) * 100

        equity = np.concatenate([[1.0], cumulative])
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        sharpe = 0
        if len(pnl_arr) > 1:
            daily_returns = pnl_arr / 100
            avg_ret = np.mean(daily_returns) * 252 / np.mean(holding_arr) if np.mean(holding_arr) > 0 else 0
            std_ret = np.std(daily_returns) * np.sqrt(252 / np.mean(holding_arr)) if np.mean(holding_arr) > 0 else 1
            sharpe = (avg_ret - 0.03) / std_ret if std_ret > 0 else 0

        is_loss = pnl_arr <= 0
        max_consec = 0
        if np.any(is_loss):
            loss_streaks = np.diff(np.where(np.concatenate([[False], is_loss, [False]]))[0])
            max_consec = int(loss_streaks.max()) if len(loss_streaks) > 0 else 0

        metrics = BacktestMetrics(
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            max_drawdown=float(max_drawdown),
            sharpe_ratio=float(sharpe),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            num_trades=len(trades_pnl),
            avg_holding_days=float(np.mean(holding_arr)),
            avg_return_per_trade=float(np.mean(pnl_arr)),
            max_consecutive_losses=int(max_consec),
        )
        if return_details:
            return metrics, trade_details
        return metrics
