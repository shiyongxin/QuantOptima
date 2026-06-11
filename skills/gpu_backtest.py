# -*- coding: utf-8 -*-
"""
GPU 批量回测引擎 - GPU Batch Backtest Engine

利用 PyTorch MPS (Metal Performance Shaders) 在 Apple Silicon GPU 上
并行计算技术指标、生成信号、模拟交易。

核心优化:
- 多只股票同时计算 (batch stocks)
- 多组参数同时评估 (batch params)
- 向量化交易模拟 (逐日扫描，无逐笔循环)

目标: 将单代 GA 评估从 ~7500s 降至 ~300s (25x 加速)
"""

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from vectorized_backtest import BacktestMetrics


def get_device():
    """获取最优计算设备"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def prepare_stock_tensor(stock_data_dict: dict, max_len: int = None):
    """
    将多只股票数据打包为统一的 GPU tensor

    Parameters:
    -----------
    stock_data_dict : dict
        {symbol: DataFrame}，DataFrame 必须包含 开盘/最高/最低/收盘/成交量
    max_len : int or None
        统一长度，None 则使用最长股票的长度

    Returns:
    --------
    prices : tensor [n_stocks, max_len, 5]  (open, high, low, close, volume)
    mask : tensor [n_stocks, max_len]  (True=有效数据)
    symbols : list[str]  股票代码列表
    lengths : list[int]  每只股票的实际长度
    """
    symbols = list(stock_data_dict.keys())
    n_stocks = len(symbols)

    if n_stocks == None:
        return None, None, [], []

    # 收集每只股票的数据
    all_data = []
    lengths = []
    for sym in symbols:
        df = stock_data_dict[sym]
        cols = ['开盘', '最高', '最低', '收盘', '成交量']
        data = df[cols].values.astype(np.float32)
        all_data.append(data)
        lengths.append(len(data))

    if max_len is None:
        max_len = max(lengths)

    # Pad 到统一长度 (前面填充 0)
    prices = np.zeros((n_stocks, max_len, 5), dtype=np.float32)
    mask = np.zeros((n_stocks, max_len), dtype=bool)

    for i, data in enumerate(all_data):
        L = len(data)
        if L >= max_len:
            prices[i] = data[-max_len:]
            mask[i] = True
        else:
            prices[i, max_len - L:] = data
            mask[i, max_len - L:] = True

    return (
        torch.from_numpy(prices),
        torch.from_numpy(mask),
        symbols,
        lengths,
    )


class GPUBacktester:
    """
    GPU 批量回测引擎

    在 GPU 上并行完成: 指标计算 → 信号生成 → 交易模拟 → 指标统计
    """

    def __init__(self, commission=0.0003, slippage=0.001, device=None):
        self.commission = commission
        self.slippage = slippage
        self.device = device or get_device()

    # ================================================================
    # 指标计算 (批量)
    # ================================================================

    def _rolling_mean(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """滑动均值 [batch, time] → [batch, time]"""
        # 使用 avg_pool1d，需要 [batch, 1, time] 格式
        pad = window - 1
        x_padded = torch.nn.functional.pad(x.unsqueeze(1), (pad, 0), mode='replicate')
        result = torch.nn.functional.avg_pool1d(x_padded, kernel_size=window, stride=1)
        return result.squeeze(1)

    def _ema(self, x: torch.Tensor, span: int) -> torch.Tensor:
        """
        指数移动平均 [batch, time] → [batch, time]

        使用向量化实现 (cumsum + 掩码)，完全避免 Python for 循环。
        数值精度对回测场景足够。
        """
        alpha = 2.0 / (span + 1)
        batch, time = x.shape
        device = x.device

        # 位置索引
        positions = torch.arange(time, dtype=torch.float32, device=device)

        # 创建下三角衰减矩阵: decay[i, j] = (1-alpha)^(i-j) if i >= j else 0
        # 但矩阵太大时改用分块计算
        if time <= 3000:
            # 直接矩阵乘法 (适用于常见时间序列长度)
            idx = positions.unsqueeze(0)  # [1, time]
            diff = idx.unsqueeze(2) - idx.unsqueeze(1)  # [time, time]
            diff = diff.squeeze(0)
            mask = (diff >= 0).float()
            decay = ((1 - alpha) ** diff.clamp(min=0)) * mask
            # 归一化权重
            col_sums = decay.sum(dim=0, keepdim=True).clamp(min=1e-10)
            decay = decay / col_sums
            # [batch, time] @ [time, time]
            return x @ decay
        else:
            # 超长序列: 使用 for 循环 (回退)
            result = torch.zeros_like(x)
            result[:, 0] = x[:, 0]
            for t in range(1, time):
                result[:, t] = alpha * x[:, t] + (1 - alpha) * result[:, t - 1]
            return result

    def _rolling_min(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """滑动最小值"""
        pad = window - 1
        x_padded = torch.nn.functional.pad(x.unsqueeze(1), (pad, 0), mode='replicate')
        # unfold 展开窗口
        windows = x_padded.unfold(2, window, 1)  # [batch, 1, time, window]
        return windows.min(dim=3).values.squeeze(1)

    def _rolling_max(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """滑动最大值"""
        pad = window - 1
        x_padded = torch.nn.functional.pad(x.unsqueeze(1), (pad, 0), mode='replicate')
        windows = x_padded.unfold(2, window, 1)
        return windows.max(dim=3).values.squeeze(1)

    def _rolling_std(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """滑动标准差"""
        pad = window - 1
        x_padded = torch.nn.functional.pad(x.unsqueeze(1), (pad, 0), mode='replicate')
        windows = x_padded.unfold(2, window, 1)  # [batch, 1, time, window]
        return windows.std(dim=3, unbiased=False).squeeze(1)

    def compute_indicators(self, prices: torch.Tensor, params: dict) -> dict:
        """
        批量计算技术指标

        Parameters:
        -----------
        prices : tensor [batch, time, 5]
            open, high, low, close, volume
        params : dict
            参数字典

        Returns:
        --------
        dict of {name: tensor [batch, time]}
        """
        close = prices[:, :, 3]  # [batch, time]
        high = prices[:, :, 1]
        low = prices[:, :, 2]
        volume = prices[:, :, 4]

        # ---- 移动平均线 ----
        ma_fast = self._rolling_mean(close, int(params.get('ma_fast', 5)))
        ma_slow = self._rolling_mean(close, int(params.get('ma_slow', 20)))
        ma_mid = self._rolling_mean(close, int(params.get('ma_mid', 10)))

        # ---- MACD ----
        ema_fast = self._ema(close, int(params.get('macd_fast', 12)))
        ema_slow = self._ema(close, int(params.get('macd_slow', 26)))
        macd = ema_fast - ema_slow
        macd_signal = self._ema(macd, int(params.get('macd_signal', 9)))
        macd_hist = macd - macd_signal

        # ---- RSI ----
        rsi_period = int(params.get('rsi_period', 14))
        delta = torch.zeros_like(close)
        delta[:, 1:] = close[:, 1:] - close[:, :-1]
        gain = torch.clamp(delta, min=0)
        loss = torch.clamp(-delta, min=0)
        avg_gain = self._rolling_mean(gain, rsi_period)
        avg_loss = self._rolling_mean(loss, rsi_period)
        rs = avg_gain / avg_loss.clamp(min=1e-10)
        rsi = 100 - (100 / (1 + rs))

        # ---- KDJ ----
        kdj_n = int(params.get('kdj_n', 9))
        low_n = self._rolling_min(low, kdj_n)
        high_n = self._rolling_max(high, kdj_n)
        denom = (high_n - low_n).clamp(min=1e-10)
        rsv = (close - low_n) / denom * 100
        k = self._ema(rsv, 3)
        d = self._ema(k, 3)
        j = 3 * k - 2 * d

        # ---- 布林带 ----
        bb_period = int(params.get('bb_period', 20))
        bb_std_mult = float(params.get('bb_std', 2.0))
        bb_mid = self._rolling_mean(close, bb_period)
        bb_std_val = self._rolling_std(close, bb_period)
        bb_upper = bb_mid + bb_std_mult * bb_std_val
        bb_lower = bb_mid - bb_std_mult * bb_std_val

        # ---- ATR ----
        atr_period = int(params.get('atr_period', 14))
        prev_close = torch.zeros_like(close)
        prev_close[:, 1:] = close[:, :-1]
        prev_close[:, 0] = close[:, 0]
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = torch.max(torch.max(tr1, tr2), tr3)
        atr = self._rolling_mean(tr, atr_period)

        # ---- 成交量均线 ----
        vol_ma5 = self._rolling_mean(volume, 5)
        vol_ma20 = self._rolling_mean(volume, 20)

        return {
            'close': close, 'high': high, 'low': low, 'volume': volume,
            'ma_fast': ma_fast, 'ma_slow': ma_slow, 'ma_mid': ma_mid,
            'macd': macd, 'macd_signal': macd_signal, 'macd_hist': macd_hist,
            'rsi': rsi, 'k': k, 'd': d, 'j': j,
            'bb_mid': bb_mid, 'bb_upper': bb_upper, 'bb_lower': bb_lower,
            'atr': atr, 'vol_ma5': vol_ma5, 'vol_ma20': vol_ma20,
        }

    # ================================================================
    # 信号生成 (批量)
    # ================================================================

    def generate_signals(self, ind: dict, params: dict,
                         mask: torch.Tensor) -> torch.Tensor:
        """
        批量生成买卖信号

        Returns: signals [batch, time]  (1=买, -1=卖, 0=持)
        """
        buy_threshold = float(params.get('buy_threshold', 2.0))
        sell_threshold = float(params.get('sell_threshold', 2.0))
        rsi_oversold = float(params.get('rsi_oversold', 30))
        rsi_overbought = float(params.get('rsi_overbought', 70))

        close = ind['close']
        batch, time = close.shape

        # ---- 买入信号得分 ----
        buy_score = torch.zeros(batch, time, device=close.device)

        # MA金叉: fast 上穿 slow
        ma_fast, ma_slow = ind['ma_fast'], ind['ma_slow']
        cross_up = (ma_fast[:, 1:] > ma_slow[:, 1:]) & (ma_fast[:, :-1] <= ma_slow[:, :-1])
        buy_score[:, 1:] += cross_up.float() * 1.0

        # 价格站上 MA 慢线
        buy_score += (close > ma_slow).float() * 0.3

        # MACD 金叉
        macd, macd_sig = ind['macd'], ind['macd_signal']
        macd_cross_up = (macd[:, 1:] > macd_sig[:, 1:]) & (macd[:, :-1] <= macd_sig[:, :-1])
        buy_score[:, 1:] += macd_cross_up.float() * 1.0

        # MACD 柱由负转正
        mh = ind['macd_hist']
        mh_turn = (mh[:, 1:] > 0) & (mh[:, :-1] <= 0)
        buy_score[:, 1:] += mh_turn.float() * 0.5

        # RSI 超卖
        buy_score += (ind['rsi'] < rsi_oversold).float() * 0.5

        # KDJ 金叉
        k, d = ind['k'], ind['d']
        kdj_up = (k[:, 1:] > d[:, 1:]) & (k[:, :-1] <= d[:, :-1])
        buy_score[:, 1:] += kdj_up.float() * 0.8

        # 布林下轨触及
        bb_lower = ind['bb_lower']
        bb_touch = (close <= bb_lower * 1.02) & (close > bb_lower * 0.98)
        buy_score += bb_touch.float() * 0.5

        # 放量
        vol_surge = ind['volume'] > ind['vol_ma5'] * 1.5
        buy_score += vol_surge.float() * 0.3

        # ---- 卖出信号得分 ----
        sell_score = torch.zeros(batch, time, device=close.device)

        # MA 死叉
        cross_down = (ma_fast[:, 1:] < ma_slow[:, 1:]) & (ma_fast[:, :-1] >= ma_slow[:, :-1])
        sell_score[:, 1:] += cross_down.float() * 1.0

        # 价格跌破 MA 慢线
        sell_score += (close < ma_slow).float() * 0.3

        # MACD 死叉
        macd_cross_down = (macd[:, 1:] < macd_sig[:, 1:]) & (macd[:, :-1] >= macd_sig[:, :-1])
        sell_score[:, 1:] += macd_cross_down.float() * 1.0

        # RSI 超买
        sell_score += (ind['rsi'] > rsi_overbought).float() * 0.5

        # KDJ 死叉
        kdj_down = (k[:, 1:] < d[:, 1:]) & (k[:, :-1] >= d[:, :-1])
        sell_score[:, 1:] += kdj_down.float() * 0.8

        # 布林上轨触及
        bb_upper = ind['bb_upper']
        bb_up_touch = (close >= bb_upper * 0.98) & (close < bb_upper * 1.02)
        sell_score += bb_up_touch.float() * 0.5

        # ---- 最终信号 ----
        signals = torch.zeros(batch, time, dtype=torch.int8, device=close.device)
        signals[buy_score >= buy_threshold] = 1
        signals[sell_score >= sell_threshold] = -1
        # 冲突时卖出优先
        conflict = (buy_score >= buy_threshold) & (sell_score >= sell_threshold)
        signals[conflict] = -1

        # Warmup 期清零
        warmup = min(60, max(10, time // 3))
        signals[:, :warmup] = 0

        # 无效数据区域清零
        signals[~mask] = 0

        return signals

    # ================================================================
    # 交易模拟 (向量化)
    # ================================================================

    def simulate_trades(self, close: torch.Tensor, high: torch.Tensor,
                        low: torch.Tensor, signals: torch.Tensor,
                        params: dict, mask: torch.Tensor) -> list:
        """
        批量交易模拟

        Parameters:
        -----------
        close/high/low/signals : tensor [batch, time]
        params : dict
        mask : tensor [batch, time]

        Returns:
        --------
        list[BacktestMetrics]  每只股票一个
        """
        stop_loss_pct = float(params.get('stop_loss_pct', 0.08))
        take_profit_pct = float(params.get('take_profit_pct', 0.20))
        position_size_pct = float(params.get('position_size_pct', 0.8))
        trailing_stop_pct = float(params.get('trailing_stop_pct', 0.05))
        min_holding = int(params.get('min_holding_days', 20))

        buy_cost = 1 + self.slippage + self.commission
        sell_cost = 1 - self.slippage - self.commission

        batch, time = close.shape
        results = []

        # 转为 numpy 用于逐股交易模拟 (交易逻辑含复杂分支，GPU 不一定更快)
        close_np = close.cpu().numpy()
        high_np = high.cpu().numpy()
        low_np = low.cpu().numpy()
        signals_np = signals.cpu().numpy()
        mask_np = mask.cpu().numpy()

        for b in range(batch):
            # 取有效数据范围
            valid = mask_np[b]
            if valid.sum() < 15:
                results.append(BacktestMetrics())
                continue

            c = close_np[b][valid]
            h = high_np[b][valid]
            lo = low_np[b][valid]
            sig = signals_np[b][valid]
            n = len(c)

            # 找买入信号
            buy_indices = np.where(sig == 1)[0]
            if len(buy_indices) == 0:
                results.append(BacktestMetrics())
                continue

            trades_pnl = []
            trades_holding = []

            for idx in range(len(buy_indices)):
                entry_idx = buy_indices[idx]
                entry_price = c[entry_idx] * buy_cost
                tp_price = entry_price * (1 + take_profit_pct)
                stop_price = entry_price * (1 - stop_loss_pct)

                end_idx = buy_indices[idx + 1] if idx + 1 < len(buy_indices) else n
                period_c = c[entry_idx:end_idx]
                period_h = h[entry_idx:end_idx]
                period_lo = lo[entry_idx:end_idx]
                period_sig = sig[entry_idx:end_idx]

                cummax = np.maximum.accumulate(period_h)
                trailing_stops = cummax * (1 - trailing_stop_pct)
                effective_stops = np.maximum(stop_price, trailing_stops)

                check_start = min(1, len(period_c) - 1)
                period_len = len(period_c)
                min_hold_end = min(min_holding, period_len)

                # 硬止损
                hit_stop = period_lo[check_start:] <= stop_price
                stop_triggers = np.where(hit_stop)[0] + check_start

                # 追踪止损
                if min_hold_end < period_len:
                    hit_trail = period_lo[min_hold_end:] <= trailing_stops[min_hold_end:]
                    trail_triggers = np.where(hit_trail)[0] + min_hold_end
                else:
                    trail_triggers = np.array([], dtype=int)

                # 止盈
                if min_hold_end < period_len:
                    hit_tp = period_h[min_hold_end:] >= tp_price
                    tp_triggers = np.where(hit_tp)[0] + min_hold_end
                else:
                    tp_triggers = np.array([], dtype=int)

                # 卖出信号
                if min_hold_end < period_len:
                    hit_sell = period_sig[min_hold_end:] == -1
                    sell_triggers = np.where(hit_sell)[0] + min_hold_end
                else:
                    sell_triggers = np.array([], dtype=int)

                exit_reason = None
                exit_offset = len(period_c) - 1

                if len(stop_triggers) > 0:
                    exit_reason = "stop_loss"
                    exit_offset = stop_triggers[0]
                elif len(trail_triggers) > 0:
                    exit_reason = "trailing_stop"
                    exit_offset = trail_triggers[0]
                elif len(tp_triggers) > 0:
                    exit_reason = "take_profit"
                    exit_offset = tp_triggers[0]
                elif len(sell_triggers) > 0:
                    exit_reason = "signal"
                    exit_offset = sell_triggers[0]

                if exit_offset >= len(period_c):
                    exit_offset = len(period_c) - 1

                if exit_reason == "stop_loss":
                    exit_price = stop_price
                elif exit_reason == "trailing_stop":
                    exit_price = trailing_stops[exit_offset]
                elif exit_reason == "take_profit":
                    exit_price = tp_price
                else:
                    exit_price = period_c[exit_offset] * sell_cost

                pnl_pct = (exit_price / entry_price - 1) * 100
                trades_pnl.append(pnl_pct)
                trades_holding.append(exit_offset)

            if not trades_pnl:
                results.append(BacktestMetrics())
                continue

            pnl_arr = np.array(trades_pnl, dtype=np.float64)
            holding_arr = np.array(trades_holding, dtype=np.float64)

            wins = pnl_arr[pnl_arr > 0]
            losses = pnl_arr[pnl_arr <= 0]
            win_rate = (len(wins) / len(pnl_arr)) * 100

            avg_win = np.mean(wins) if len(wins) > 0 else 0
            avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

            cumulative = np.cumprod(1 + pnl_arr / 100 * position_size_pct)
            total_return = (cumulative[-1] - 1) * 100

            years = n / 252
            if years > 0 and cumulative[-1] > 0:
                annualized_return = (cumulative[-1] ** (1 / years) - 1) * 100
            else:
                annualized_return = 0

            equity = np.concatenate([[1.0], cumulative])
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak * 100
            max_drawdown = abs(drawdown.min())

            if len(pnl_arr) > 1:
                daily_returns = pnl_arr / 100
                avg_ret = np.mean(daily_returns) * 252 / np.mean(holding_arr) if np.mean(holding_arr) > 0 else 0
                std_ret = np.std(daily_returns) * np.sqrt(252 / np.mean(holding_arr)) if np.mean(holding_arr) > 0 else 1
                sharpe = (avg_ret - 0.03) / std_ret if std_ret > 0 else 0
            else:
                sharpe = 0

            is_loss = pnl_arr <= 0
            max_consec = 0
            if np.any(is_loss):
                loss_streaks = np.diff(np.where(np.concatenate([[False], is_loss, [False]]))[0])
                max_consec = int(loss_streaks.max()) if len(loss_streaks) > 0 else 0

            results.append(BacktestMetrics(
                total_return=float(total_return),
                annualized_return=float(annualized_return),
                max_drawdown=float(max_drawdown),
                sharpe_ratio=float(sharpe),
                win_rate=float(win_rate),
                profit_factor=float(profit_factor),
                num_trades=len(pnl_arr),
                avg_holding_days=float(np.mean(holding_arr)),
                avg_return_per_trade=float(np.mean(pnl_arr)),
                max_consecutive_losses=max_consec,
            ))

        return results

    # ================================================================
    # 多参数批量评估 (GA 核心加速)
    # ================================================================

    def evaluate_population(self, stock_data_dict: dict,
                            params_list: list) -> list:
        """
        批量评估 GA 种群中所有个体

        对每组参数分别计算指标、信号、交易模拟。
        指标计算和信号生成在 GPU 上并行完成。

        Parameters:
        -----------
        stock_data_dict : dict
            {symbol: DataFrame} 股票数据
        params_list : list[dict]
            GA 种群中所有个体的参数

        Returns:
        --------
        list[tuple(fitness, details)]  每个个体的适应度
        """
        # 准备数据 (只做一次)
        prices, mask, symbols, lengths = prepare_stock_tensor(stock_data_dict)
        if prices is None:
            return [(0.0, {})] * len(params_list)

        prices = prices.to(self.device)
        mask = mask.to(self.device)

        results = []
        for params in params_list:
            try:
                # GPU 批量回测 (所有股票并行)
                indicators = self.compute_indicators(prices, params)
                signals = self.generate_signals(indicators, params, mask)
                metrics = self.simulate_trades(
                    indicators['close'], indicators['high'],
                    indicators['low'], signals, params, mask
                )
                # 过滤无效结果
                valid_metrics = [m for m in metrics if m.num_trades > 0]
                if valid_metrics:
                    fitness, details = self._compute_fitness(valid_metrics)
                else:
                    fitness, details = 0.0, {}
                results.append((fitness, details))
            except Exception:
                results.append((0.0, {}))

        return results

    # ================================================================
    # 主入口
    # ================================================================

    def backtest_batch(self, prices: torch.Tensor, mask: torch.Tensor,
                       params: dict) -> list:
        """
        同一参数批量回测多只股票

        Parameters:
        -----------
        prices : tensor [n_stocks, time, 5]
        mask : tensor [n_stocks, time]
        params : dict

        Returns:
        --------
        list[BacktestMetrics]
        """
        prices = prices.to(self.device)
        mask = mask.to(self.device)

        indicators = self.compute_indicators(prices, params)
        signals = self.generate_signals(indicators, params, mask)

        return self.simulate_trades(
            indicators['close'], indicators['high'], indicators['low'],
            signals, params, mask
        )

    def walk_forward_batch(self, prices: torch.Tensor, mask: torch.Tensor,
                           params: dict, train_days=504, test_days=180,
                           step_days=63) -> list:
        """
        批量 Walk-Forward 验证

        Returns: list[BacktestMetrics]  所有股票所有窗口的指标
        """
        n_stocks, total_len, _ = prices.shape

        if total_len < train_days + test_days:
            return []

        all_metrics = []
        idx = train_days

        while idx + test_days <= total_len:
            # 切片: 取 [idx:idx+test_days] 的数据
            window_prices = prices[:, idx:idx + test_days, :]
            window_mask = mask[:, idx:idx + test_days]

            # 过滤有效数据不足的股票
            valid_stocks = window_mask.sum(dim=1) >= 15
            if valid_stocks.any():
                metrics = self.backtest_batch(
                    window_prices[valid_stocks],
                    window_mask[valid_stocks],
                    params
                )
                all_metrics.extend(metrics)

            idx += step_days

        return all_metrics

    def evaluate_params_batch(self, stock_data_dict: dict, params: dict,
                              regime_labels: dict = None,
                              target_regime: str = None) -> tuple:
        """
        评估一组参数在多只股票上的表现 (兼容 FitnessEvaluator 接口)

        Returns: (fitness, details)
        """
        prices, mask, symbols, lengths = prepare_stock_tensor(stock_data_dict)
        if prices is None:
            return 0.0, {}

        prices = prices.to(self.device)
        mask = mask.to(self.device)

        if target_regime and regime_labels:
            # 按体制评估 (需要回 CPU 处理体制过滤)
            return self._evaluate_regime(
                prices, mask, symbols, params, regime_labels, target_regime
            )
        else:
            # 全局 Walk-Forward
            metrics = self.walk_forward_batch(prices, mask, params)
            if not metrics:
                return 0.0, {}
            return self._compute_fitness(metrics)

    def _evaluate_regime(self, prices, mask, symbols, params,
                         regime_labels, target_regime):
        """按体制评估 (回 CPU 处理体制窗口)"""
        all_metrics = []
        close_np = prices[:, :, 3].cpu().numpy()
        mask_np = mask.cpu().numpy()

        for i, sym in enumerate(symbols):
            if sym not in regime_labels:
                continue

            labels = regime_labels[sym]
            valid = mask_np[i]
            valid_len = valid.sum()

            if isinstance(labels, pd.Series):
                labels_arr = labels.values
            else:
                labels_arr = np.array(labels)

            # 对齐长度
            if len(labels_arr) > valid_len:
                labels_arr = labels_arr[-valid_len:]
            elif len(labels_arr) < valid_len:
                continue

            # 找体制窗口
            is_target = (labels_arr == target_regime)
            if is_target.sum() < 60:
                continue

            # 找连续窗口
            starts = []
            for j in range(len(is_target)):
                if is_target[j] and (j == 0 or not is_target[j - 1]):
                    starts.append(j)

            for start in starts:
                end = start
                while end < len(is_target) and is_target[end]:
                    end += 1

                if end - start < 15:
                    continue

                # 提取窗口数据 (在有效数据范围内偏移)
                offset = len(labels_arr) - valid_len
                w_start = start - offset
                w_end = end - offset

                if w_start < 0 or w_end > valid_len:
                    continue

                # 用原始 CPU backtester 处理单个窗口 (窗口通常很短)
                from vectorized_backtest import VectorizedBacktester
                cpu_bt = VectorizedBacktester(self.commission, self.slippage)

                # 构建 DataFrame
                p = prices[i].cpu().numpy()
                window_data = pd.DataFrame({
                    '开盘': p[w_start:w_end, 0],
                    '最高': p[w_start:w_end, 1],
                    '最低': p[w_start:w_end, 2],
                    '收盘': p[w_start:w_end, 3],
                    '成交量': p[w_start:w_end, 4],
                })

                try:
                    m = cpu_bt.backtest(window_data, params)
                    if m.num_trades > 0:
                        all_metrics.append(m)
                except Exception:
                    continue

        if not all_metrics:
            return 0.0, {}

        return self._compute_fitness(all_metrics)

    def _compute_fitness(self, metrics_list: list,
                         target_return=10.0, max_drawdown_limit=25.0):
        """计算适应度 (与 CPU 版逻辑一致)"""
        returns = [m.total_return for m in metrics_list]
        sharpes = [m.sharpe_ratio for m in metrics_list]
        drawdowns = [m.max_drawdown for m in metrics_list]
        win_rates = [m.win_rate for m in metrics_list]
        trades = [m.num_trades for m in metrics_list]
        holding_days = [m.avg_holding_days for m in metrics_list]

        target_met = sum(1 for r in returns if r >= target_return)
        target_rate = target_met / len(returns)

        median_ret = np.median(returns)
        ret_score = min(max(median_ret / 50, -1), 1)

        median_sharpe = np.median(sharpes)
        sharpe_score = min(max(median_sharpe / 2, -1), 1)

        median_dd = np.median(drawdowns)
        dd_penalty = max(0, 1 - median_dd / max_drawdown_limit)

        avg_trades = np.mean(trades)
        if avg_trades < 3:
            trade_penalty = avg_trades / 3
        elif avg_trades > 50:
            trade_penalty = max(0, 1 - (avg_trades - 50) / 50)
        else:
            trade_penalty = 1.0

        fitness = (
            0.40 * target_rate +
            0.25 * max(0, ret_score) +
            0.20 * max(0, sharpe_score) +
            0.15 * dd_penalty
        ) * trade_penalty

        details = {
            'target_rate': target_rate,
            'median_return': median_ret,
            'median_sharpe': median_sharpe,
            'median_drawdown': median_dd,
            'avg_trades': avg_trades,
            'avg_holding_days': np.mean(holding_days),
            'sample_count': len(metrics_list),
            'win_rate': np.mean(win_rates),
        }

        return fitness, details
