# -*- coding: utf-8 -*-
"""
遗传算法优化引擎 - Genetic Algorithm Optimization Engine

为不同市场体制分别优化技术分析参数，产出多套指标体系。
- 阶段1: 全局优化（通用基线参数）
- 阶段2: 按体制分别优化（体制专属参数）
- 阶段3: 混合策略优化（跨体制稳健参数）
- 多轮迭代：粗→中→细，逐步精炼
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from datetime import datetime
import json
import time
import copy

from parameter_space import ParameterSpace
from vectorized_backtest import VectorizedBacktester, BacktestMetrics
from historical_data_manager import HistoricalDataManager

# GPU 支持 (可选)
try:
    from gpu_backtest import GPUBacktester, prepare_stock_tensor, get_device
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# ==================== 数据结构 ====================

@dataclass
class IndicatorSystem:
    """一套指标体系"""
    name: str                       # 如 "牛市追涨型"
    description: str                # 适用场景描述
    applicable_regimes: list        # 适用体制: ['BULL', 'EARLY_BULL']
    params: dict                    # 完整参数集
    fitness_scores: dict            # 各体制下的fitness {regime: fitness}
    confidence: float               # 置信度 (0-1)
    sample_count: int               # 支撑该结论的样本数

    # 性能统计
    median_return: float = 0.0
    win_rate_above_10pct: float = 0.0
    median_sharpe: float = 0.0
    median_max_drawdown: float = 0.0
    median_holding_days: float = 0.0
    num_trades: float = 0.0


@dataclass
class Individual:
    """GA个体"""
    params: dict
    fitness: float = 0.0
    fitness_details: dict = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """优化结果"""
    regime: str                     # 体制名称, "GLOBAL" 表示全局
    best_individual: Individual
    best_system: IndicatorSystem
    all_systems: list               # 所有候选体系
    generations_run: int
    total_time_sec: float
    convergence_gen: int            # 在第几代收敛


# ==================== 适应度评估 ====================

class FitnessEvaluator:
    """
    适应度评估器

    对给定参数在多只股票上进行Walk-Forward回测，
    聚合结果计算适应度分数。
    """

    def __init__(self, backtester=None, target_return=10.0,
                 target_probability=0.80, max_drawdown_limit=25.0,
                 use_gpu=False):
        """
        Parameters:
        -----------
        target_return : float
            目标收益率(%)
        target_probability : float
            目标达标概率
        max_drawdown_limit : float
            最大回撤限制(%)
        use_gpu : bool
            是否使用 GPU 加速
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            self.gpu_backtester = GPUBacktester()
            self.backtester = backtester or VectorizedBacktester()  # fallback
            print(f"[INFO] GPU 模式已启用 (设备: {self.gpu_backtester.device})")
        else:
            self.backtester = backtester or VectorizedBacktester()
            self.gpu_backtester = None
        self.target_return = target_return
        self.target_probability = target_probability
        self.max_drawdown_limit = max_drawdown_limit

    def evaluate(self, params: dict, stock_data: dict,
                 regime_labels: dict = None,
                 target_regime: str = None,
                 train_window: int = 120,
                 sub_window: int = 252,
                 step: int = 63) -> tuple:
        """
        评估一组参数在多只股票上的表现（Phase 3 修复版本）

        统一 walk-forward：
        - train_window: warmup 长度（默认 120 = 全空间 ma_slow.high）
        - sub_window: 测试子窗口长度（默认 252 = 一年）
        - step: 步进（默认 63 = 季度）

        体制过滤时只在 regime 段内 walk-forward，自动避免跨边界窗口。
        """
        # GPU 加速路径
        if self.use_gpu and self.gpu_backtester and not target_regime:
            try:
                return self.gpu_backtester.evaluate_params_batch(
                    stock_data, params, regime_labels, target_regime
                )
            except Exception:
                pass

        all_metrics = []

        for symbol, data in stock_data.items():
            try:
                if target_regime and regime_labels and symbol in regime_labels:
                    labels = regime_labels[symbol]
                    metrics_list = self._walk_forward_by_regime(
                        data, labels, target_regime, params,
                        train_window=train_window,
                        sub_window=sub_window,
                        step=step,
                    )
                    all_metrics.extend(metrics_list)
                else:
                    # 全量 walk-forward（统一路径，不再用 backtester.walk_forward）
                    metrics_list = self._walk_forward(
                        data, params,
                        train_window=train_window,
                        sub_window=sub_window,
                        step=step,
                    )
                    all_metrics.extend(metrics_list)
            except Exception:
                continue

        if not all_metrics:
            return 0.0, {}

        return self._compute_fitness(all_metrics)

    def legacy_evaluate(self, params, stock_data, regime_labels=None, target_regime=None):
        """
        修复前的旧评估逻辑（M1 决策要求保留作对照）

        - train_days=504 仅作偏移量（不是真训练）
        - test_days=180（半年）
        - step_days=63
        - regime 短窗口用 _backtest_by_trade_filter
        """
        all_metrics = []
        for symbol, data in stock_data.items():
            try:
                if target_regime and regime_labels and symbol in regime_labels:
                    labels = regime_labels[symbol]
                    mask = labels == target_regime
                    if mask.sum() < 60:
                        continue
                    metrics_list = self._backtest_regime_windows_legacy(
                        data, params, labels, target_regime
                    )
                    all_metrics.extend(metrics_list)
                else:
                    metrics_list = self.backtester.walk_forward(
                        data, params, train_days=504, test_days=180, step_days=63
                    )
                    all_metrics.extend(metrics_list)
            except Exception:
                continue
        if not all_metrics:
            return 0.0, {}
        return self._compute_fitness_legacy(all_metrics)

    def _walk_forward(self, data, params, train_window=120, sub_window=252, step=63):
        """
        单只股票的 walk-forward（Phase 3 统一路径）

        - 跳过前 train_window 天（warmup）
        - 从 train_window 起，每 step 天取 sub_window 长度的测试窗口
        - 窗口不足 sub_window 长度时停止
        """
        if len(data) < train_window + sub_window:
            return []
        metrics_list = []
        idx = train_window
        while idx + sub_window <= len(data):
            test_data = data.iloc[idx:idx + sub_window]
            try:
                metrics = self.backtester.backtest(test_data, params)
                if metrics.num_trades > 0:
                    metrics_list.append(metrics)
            except Exception:
                pass
            idx += step
        return metrics_list

    def _walk_forward_by_regime(self, data, labels, target_regime, params,
                                train_window=120, sub_window=252, step=63):
        """
        在体制时段内做 walk-forward（Phase 3 R5 决策）

        - 找出 target_regime 的连续段
        - 段长 < sub_window → 跳过（窗口都凑不出来一个）
        - 段内 walk-forward，子窗口 = sub_window
        - 因为在段内做 walk-forward，窗口不会跨 regime 边界（R5 决策）
        """
        in_regime = (labels == target_regime).reset_index(drop=True)
        n = len(in_regime)
        if n < sub_window:
            return []

        # 找连续段
        segments = []  # list of (start, end)
        start = None
        for i in range(n):
            if in_regime.iloc[i]:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start >= sub_window:
                        segments.append((start, i))
                    start = None
        if start is not None and n - start >= sub_window:
            segments.append((start, n))

        if not segments:
            return []

        # 段内 walk-forward
        metrics_list = []
        for seg_start, seg_end in segments:
            seg_data = data.iloc[seg_start:seg_end].reset_index(drop=True)
            if len(seg_data) < train_window + sub_window:
                # 段太短，跳过 warmup 后直接 backtest 整个段
                try:
                    metrics = self.backtester.backtest(seg_data, params)
                    if metrics.num_trades > 0:
                        metrics_list.append(metrics)
                except Exception:
                    pass
                continue
            # 在段内做 walk-forward
            idx = train_window
            while idx + sub_window <= len(seg_data):
                test_data = seg_data.iloc[idx:idx + sub_window]
                try:
                    metrics = self.backtester.backtest(test_data, params)
                    if metrics.num_trades > 0:
                        metrics_list.append(metrics)
                except Exception:
                    pass
                idx += step
        return metrics_list

    def _backtest_regime_windows_legacy(self, data, params, labels, target_regime):
        """修复前的 regime 窗口回测（保留用于 M1 对照）"""
        metrics_list = []
        in_regime = labels == target_regime
        regime_total_days = in_regime.sum()

        max_consec = 0
        cur = 0
        for v in in_regime:
            if v:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 0

        if max_consec < 15 or regime_total_days < 60:
            return self._backtest_by_trade_filter(data, params, labels, target_regime)

        starts = []
        for i in range(len(in_regime)):
            if in_regime.iloc[i] and (i == 0 or not in_regime.iloc[i-1]):
                starts.append(i)

        for start in starts:
            end = start
            while end < len(in_regime) and in_regime.iloc[end]:
                end += 1
            if end - start < 15:
                continue
            window_data = data.iloc[start:end]
            if len(window_data) < 15:
                continue
            try:
                metrics = self.backtester.backtest(window_data, params)
                if metrics.num_trades > 0:
                    metrics_list.append(metrics)
            except Exception:
                continue
        return metrics_list

    def _backtest_by_trade_filter(self, data, params, labels, target_regime):
        """保留的旧 regime 短窗口处理逻辑（用于 legacy_evaluate）"""
        try:
            metrics, trade_details = self.backtester.backtest_with_details(data, params, legacy=True)
        except Exception:
            return []
        if not trade_details:
            return []
        filtered_trades = []
        for td in trade_details:
            if td.entry_idx < len(labels):
                entry_regime = labels.iloc[td.entry_idx]
                if entry_regime == target_regime:
                    filtered_trades.append(td)
        if not filtered_trades:
            return []
        filtered_metrics = self._compute_metrics_from_trades(filtered_trades, data, params)
        if filtered_metrics and filtered_metrics.num_trades > 0:
            return [filtered_metrics]
        return []

    def _compute_metrics_from_trades(self, trades, data, params):
        """保留的旧交易聚合逻辑（仅 legacy_evaluate 用）"""
        if not trades:
            return None
        pnl_arr = np.array([t.pnl_pct for t in trades], dtype=np.float64)
        holding_arr = np.array([t.holding_days for t in trades], dtype=np.float64)
        position_size_pct = float(params.get('position_size_pct', 0.8))
        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr <= 0]
        win_rate = (len(wins) / len(pnl_arr)) * 100 if len(pnl_arr) > 0 else 0
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        cumulative = np.cumprod(1 + pnl_arr / 100 * position_size_pct)
        total_return = (cumulative[-1] - 1) * 100 if len(cumulative) > 0 else 0
        total_days = len(data)
        years = total_days / 252
        annualized_return = (cumulative[-1] ** (1 / years) - 1) * 100 if years > 0 and cumulative[-1] > 0 else 0
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
        from vectorized_backtest import BacktestMetrics
        return BacktestMetrics(
            total_return=float(total_return), annualized_return=float(annualized_return),
            max_drawdown=float(max_drawdown), sharpe_ratio=float(sharpe),
            win_rate=float(win_rate), profit_factor=float(profit_factor),
            num_trades=len(pnl_arr), avg_holding_days=float(np.mean(holding_arr)),
            avg_return_per_trade=float(np.mean(pnl_arr)), max_consecutive_losses=max_consec,
        )

    def _compute_fitness(self, metrics_list: list) -> tuple:
        """
        从回测指标列表计算适应度（Phase 3 修复版，混合聚合）

        聚合策略（FIX_PLAN E4）：
        - target_rate: keep as ratio
        - return: mean
        - sharpe: median
        - drawdown: median

        fitness = 0.40 * target_rate + 0.25 * ret + 0.20 * sharpe + 0.15 * dd_penalty
        """
        returns = [m.total_return for m in metrics_list]
        sharpes = [m.sharpe_ratio for m in metrics_list]
        drawdowns = [m.max_drawdown for m in metrics_list]
        win_rates = [m.win_rate for m in metrics_list]
        trades = [m.num_trades for m in metrics_list]
        holding_days = [m.avg_holding_days for m in metrics_list]

        # 达标率: 收益>target_return 的比例
        target_met = sum(1 for r in returns if r >= self.target_return)
        target_rate = target_met / len(returns) if returns else 0

        # E4: return 改用 mean（让牛市大赚能反映）
        mean_ret = np.mean(returns) if returns else 0
        ret_score = min(max(mean_ret / 50, -1), 1)

        # E4: sharpe 保留 median
        median_sharpe = np.median(sharpes) if sharpes else 0
        sharpe_score = min(max(median_sharpe / 2, -1), 1)

        # E4: drawdown 保留 median
        median_dd = np.median(drawdowns) if drawdowns else 100
        dd_penalty = max(0, 1 - median_dd / self.max_drawdown_limit)

        avg_trades = np.mean(trades) if trades else 0
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
            'mean_return': float(mean_ret),
            'median_return': float(np.median(returns)) if returns else 0,
            'median_sharpe': float(median_sharpe),
            'median_drawdown': float(median_dd),
            'avg_trades': float(avg_trades),
            'avg_holding_days': float(np.mean(holding_days)) if holding_days else 0,
            'sample_count': len(metrics_list),
            'win_rate': float(np.mean(win_rates)) if win_rates else 0,
        }
        return fitness, details

    def _compute_fitness_legacy(self, metrics_list):
        """保留的旧 fitness 公式（用于 legacy_evaluate）"""
        returns = [m.total_return for m in metrics_list]
        sharpes = [m.sharpe_ratio for m in metrics_list]
        drawdowns = [m.max_drawdown for m in metrics_list]
        win_rates = [m.win_rate for m in metrics_list]
        trades = [m.num_trades for m in metrics_list]
        holding_days = [m.avg_holding_days for m in metrics_list]
        target_met = sum(1 for r in returns if r >= self.target_return)
        target_rate = target_met / len(returns) if returns else 0
        median_ret = np.median(returns) if returns else 0
        ret_score = min(max(median_ret / 50, -1), 1)
        median_sharpe = np.median(sharpes) if sharpes else 0
        sharpe_score = min(max(median_sharpe / 2, -1), 1)
        median_dd = np.median(drawdowns) if drawdowns else 100
        dd_penalty = max(0, 1 - median_dd / self.max_drawdown_limit)
        avg_trades = np.mean(trades) if trades else 0
        if avg_trades < 3:
            trade_penalty = avg_trades / 3
        elif avg_trades > 50:
            trade_penalty = max(0, 1 - (avg_trades - 50) / 50)
        else:
            trade_penalty = 1.0
        fitness = (
            0.40 * target_rate + 0.25 * max(0, ret_score) +
            0.20 * max(0, sharpe_score) + 0.15 * dd_penalty
        ) * trade_penalty
        details = {
            'target_rate': target_rate, 'median_return': median_ret,
            'median_sharpe': median_sharpe, 'median_drawdown': median_dd,
            'avg_trades': avg_trades,
            'avg_holding_days': np.mean(holding_days) if holding_days else 0,
            'sample_count': len(metrics_list),
            'win_rate': np.mean(win_rates) if win_rates else 0,
        }
        return fitness, details


# ==================== 遗传算法引擎 ====================

class GeneticAlgorithm:
    """
    遗传算法实现

    锦标赛选择(k=3) + 均匀交叉 + 高斯变异 + 自适应变异率
    """

    def __init__(self, population_size=100, max_generations=50,
                 tournament_size=3, crossover_rate=0.8,
                 mutation_rate=0.15, elite_ratio=0.1,
                 convergence_threshold=10, convergence_tolerance=0.005):
        """
        Parameters:
        -----------
        population_size : int
            种群大小
        max_generations : int
            最大代数
        tournament_size : int
            锦标赛大小
        crossover_rate : float
            交叉概率
        mutation_rate : float
            初始变异概率
        elite_ratio : float
            精英保留比例
        convergence_threshold : int
            连续N代无改善则收敛
        convergence_tolerance : float
            改善阈值
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.convergence_threshold = convergence_threshold
        self.convergence_tolerance = convergence_tolerance

    def run(self, evaluate_fn, seed_params=None, rng=None,
            verbose=True, gpu_backtester=None, stock_data=None) -> tuple:
        """
        运行遗传算法

        Parameters:
        -----------
        evaluate_fn : callable
            评估函数: (params: dict) -> (fitness: float, details: dict)
        seed_params : list[dict] or None
            种子参数（如上一轮最优解）
        rng : np.random.Generator or None
            随机数生成器
        verbose : bool
            是否打印进度
        gpu_backtester : GPUBacktester or None
            GPU 回测引擎 (如果提供，使用批量评估)
        stock_data : dict or None
            {symbol: DataFrame} 股票数据 (GPU 批量评估需要)

        Returns:
        --------
        (best_individual: Individual, all_individuals: list[Individual],
         convergence_gen: int)
        """
        if rng is None:
            rng = np.random.default_rng()

        # 初始化种群
        population = self._init_population(seed_params, rng)

        best_fitness = -np.inf
        best_individual = None
        no_improve_count = 0
        convergence_gen = 0

        # 是否使用 GPU 批量评估
        use_gpu_batch = (gpu_backtester is not None and stock_data is not None)

        for gen in range(self.max_generations):
            gen_start = time.time()

            if use_gpu_batch:
                # GPU 批量评估: 整个种群一次性评估
                unevaluated = [i for i, ind in enumerate(population) if ind.fitness == 0]
                if unevaluated:
                    params_list = [population[i].params for i in unevaluated]
                    batch_results = gpu_backtester.evaluate_population(
                        stock_data, params_list
                    )
                    for idx, (fitness, details) in zip(unevaluated, batch_results):
                        population[idx].fitness = fitness
                        population[idx].fitness_details = details
            else:
                # CPU 逐个评估
                for ind in population:
                    if ind.fitness == 0:  # 未评估过
                        fitness, details = evaluate_fn(ind.params)
                        ind.fitness = fitness
                        ind.fitness_details = details

            # 排序
            population.sort(key=lambda x: x.fitness, reverse=True)

            # 更新最优
            gen_best = population[0]
            if gen_best.fitness > best_fitness + self.convergence_tolerance:
                best_fitness = gen_best.fitness
                best_individual = copy.deepcopy(gen_best)
                no_improve_count = 0
                convergence_gen = gen
            else:
                no_improve_count += 1

            gen_time = time.time() - gen_start

            if verbose:
                top3 = [f"{p.fitness:.4f}" for p in population[:3]]
                print(f"  Gen {gen:3d} | Best: {gen_best.fitness:.4f} "
                      f"| Top3: {', '.join(top3)} "
                      f"| NoImprove: {no_improve_count} "
                      f"| {gen_time:.1f}s")

            # 收敛检查
            if no_improve_count >= self.convergence_threshold:
                if verbose:
                    print(f"  [收敛] 连续{self.convergence_threshold}代无改善，停止")
                break

            # 自适应变异率
            current_mutation = self.mutation_rate
            if no_improve_count > 3:
                # 长期无改善，增大变异率
                current_mutation = min(0.4, self.mutation_rate * (1 + no_improve_count * 0.05))

            # 产生下一代
            population = self._next_generation(
                population, evaluate_fn, current_mutation, rng
            )

        return best_individual, population, convergence_gen

    def _init_population(self, seed_params, rng):
        """初始化种群"""
        population = []

        # 加入种子参数
        if seed_params:
            for params in seed_params:
                ind = Individual(params=copy.deepcopy(params))
                population.append(ind)

        # 随机填充剩余
        while len(population) < self.population_size:
            params = ParameterSpace.random_sample(rng)
            ind = Individual(params=params)
            population.append(ind)

        return population

    def _next_generation(self, population, evaluate_fn, mutation_rate, rng):
        """产生下一代"""
        next_gen = []

        # 精英保留
        n_elite = max(1, int(len(population) * self.elite_ratio))
        for i in range(n_elite):
            elite = copy.deepcopy(population[i])
            next_gen.append(elite)

        # 产生剩余个体
        while len(next_gen) < self.population_size:
            # 锦标赛选择
            parent1 = self._tournament_select(population, rng)
            parent2 = self._tournament_select(population, rng)

            # 交叉
            if rng.random() < self.crossover_rate:
                child1_params, child2_params = ParameterSpace.crossover(
                    parent1.params, parent2.params, rng
                )
            else:
                child1_params = copy.deepcopy(parent1.params)
                child2_params = copy.deepcopy(parent2.params)

            # 变异
            child1_params = ParameterSpace.mutate(child1_params, mutation_rate, rng)
            child2_params = ParameterSpace.mutate(child2_params, mutation_rate, rng)

            next_gen.append(Individual(params=child1_params))
            if len(next_gen) < self.population_size:
                next_gen.append(Individual(params=child2_params))

        return next_gen

    def _tournament_select(self, population, rng):
        """锦标赛选择"""
        candidates = rng.choice(len(population), size=self.tournament_size, replace=False)
        best = max(candidates, key=lambda i: population[i].fitness)
        return population[best]


# ==================== 优化引擎 ====================

class OptimizationEngine:
    """
    多体制优化引擎

    阶段1: 全局优化 → 通用基线参数
    阶段2: 按体制分别优化 → 体制专属参数
    阶段3: 混合策略优化 → 跨体制稳健参数
    """

    def __init__(self, data_dir="./stock_data", n_workers=4, use_gpu=False):
        """
        Parameters:
        -----------
        data_dir : str
            数据目录
        n_workers : int
            并行工作线程数
        use_gpu : bool
            是否使用 GPU 加速
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.data_dir = data_dir
        self.data_manager = HistoricalDataManager(data_dir)
        self.backtester = VectorizedBacktester()
        self.evaluator = FitnessEvaluator(self.backtester, use_gpu=self.use_gpu)
        self.n_workers = n_workers

        self.results = {}       # {regime: OptimizationResult}
        self.stock_data = {}    # {symbol: DataFrame} (all)
        self.regime_labels = {} # {symbol: pd.Series} (all)
        # Phase 3: split-based views（按 train/val/test 过滤后的子集）
        self.splits = {
            'train': {'symbols': set(), 'regimes': None},
            'val':   {'symbols': set(), 'regimes': None},
            'test':  {'symbols': set(), 'regimes': None},
        }

    def load_data(self, symbols=None, min_history_days=180,
                  regime_labels_file=None, splits_dir=None):
        """
        加载股票数据和体制标签（Phase 3：支持 train/val/test 切分）

        Parameters
        ----------
        symbols : list or None
            股票代码列表；None 则使用股票池
        min_history_days : int
            最少历史天数（Phase 3 调整：180 = train_window 120 + 最小 test 60）
        regime_labels_file : str or None
            体制标签文件路径（推荐：data_dir/regime_labels.csv）
        splits_dir : str or None
            切分文件目录（含 train/val/test_symbols.txt + regime_labels_*.csv）
            如果提供，会进一步按 split 加载
        """
        # 加载体制标签
        if regime_labels_file and Path(regime_labels_file).exists():
            df = pd.read_csv(regime_labels_file, encoding='utf-8-sig')
            df['日期'] = pd.to_datetime(df['日期'])
            self._index_regime_labels = df
            print(f"[OK] 加载体制标签: {len(df)} 条")
        else:
            print("[WARN] 未提供体制标签文件，将跳过按体制优化")
            self._index_regime_labels = None

        # 获取股票池
        if symbols is None:
            symbols = self.data_manager.get_universe(
                min_history_days=min_history_days, min_rows=200
            )

        if not symbols:
            print("[ERROR] 无可用股票数据，请先运行 historical_data_manager.py batch")
            return

        print(f"[INFO] 加载 {len(symbols)} 只股票数据...")

        loaded = 0
        for sym in symbols:
            data = self.data_manager.load(sym)
            if len(data) >= 252:
                self.stock_data[sym] = data
                loaded += 1
                if self._index_regime_labels is not None:
                    self._assign_regime_labels(sym, data)

        print(f"[OK] 成功加载 {loaded} 只股票")

        # Phase 3: 加载 splits
        if splits_dir:
            self._load_splits(splits_dir)

    def _load_splits(self, splits_dir):
        """
        加载 train/val/test splits（来自 data_splitter.py 输出）

        文件格式：
        - splits/train_symbols.txt
        - splits/val_symbols.txt
        - splits/test_symbols.txt
        - regime_labels_train.csv
        - regime_labels_val.csv
        - regime_labels_test.csv
        """
        splits_path = Path(splits_dir) if not Path(splits_dir).is_absolute() else Path(splits_dir)
        # 也支持 data_dir 下的 splits/ 子目录
        if not splits_path.exists():
            alt = Path(self.data_dir) / 'splits'
            if alt.exists():
                splits_path = alt

        if not splits_path.exists():
            print(f"[WARN] splits 目录不存在: {splits_dir}")
            return

        for split_name in ['train', 'val', 'test']:
            sym_file = splits_path / f'{split_name}_symbols.txt'
            if not sym_file.exists():
                print(f"[WARN] 缺 {sym_file}")
                continue
            with open(sym_file, 'r', encoding='utf-8') as f:
                syms = set(line.strip() for line in f if line.strip())
            # 过滤实际有数据的股票
            actual_syms = syms & set(self.stock_data.keys())
            self.splits[split_name]['symbols'] = actual_syms
            print(f"[OK] {split_name}: {len(actual_syms)} 只股票 (来自 {sym_file.name})")

    def _get_split_data(self, split: str) -> tuple:
        """
        返回 (split_stock_data, split_regime_labels)

        如果 split 对应的 symbols 为空（splits 未加载），返回全集（兼容旧 API）
        """
        syms = self.splits.get(split, {}).get('symbols', set()) if self.splits else set()
        if not syms:
            return self.stock_data, self.regime_labels
        sub_data = {s: self.stock_data[s] for s in syms if s in self.stock_data}
        sub_labels = {s: self.regime_labels[s] for s in syms if s in self.regime_labels}
        return sub_data, sub_labels

    def _assign_regime_labels(self, symbol, data):
        """为股票数据分配体制标签(基于指数日期对齐) - 向量化版本"""
        idx_labels = self._index_regime_labels
        if idx_labels is None:
            return

        # 使用 pd.merge_asof 向量化查找最近的指数日期
        stock_df = data[['日期']].copy()
        idx_df = idx_labels[['日期', 'regime']].copy()

        # 统一datetime类型为datetime64[ns](避免us/ns不匹配)
        stock_df['日期'] = pd.to_datetime(stock_df['日期']).astype('datetime64[ns]')
        idx_df['日期'] = pd.to_datetime(idx_df['日期']).astype('datetime64[ns]')

        idx_df = idx_df.sort_values('日期')
        stock_df = stock_df.sort_values('日期')

        merged = pd.merge_asof(
            stock_df, idx_df,
            on='日期',
            direction='backward'
        )
        self.regime_labels[symbol] = merged['regime'].fillna('SIDEWAYS')

    def optimize_global(self, stock_sample=None, n_stocks=50,
                        ga_params=None, verbose=True, split='train') -> OptimizationResult:
        """
        阶段1: 全局优化（Phase 3: 默认在 train split 上跑）

        Parameters
        ----------
        stock_sample : list or None
            指定股票子集；None 则随机抽样
        n_stocks : int
            抽样股票数
        ga_params : dict or None
            GA 参数覆盖
        verbose : bool
            是否打印进度
        split : str
            'train'（默认）/ 'val' / 'test' / 'all'。GA 永远只看 train
        """
        print("\n" + "=" * 70)
        print(f"  阶段1: 全局优化 (Global Optimization, split={split})")
        print("=" * 70)

        # 抽样股票（从 split 子集里）
        sample_data = self._get_sample(stock_sample, n_stocks, split=split)
        if not sample_data:
            print("[ERROR] 无可用股票数据")
            return None

        print(f"  样本股票: {len(sample_data)} 只")

        ga_cfg = {
            'population_size': 80,
            'max_generations': 40,
            'convergence_threshold': 8,
        }
        if ga_params:
            ga_cfg.update(ga_params)

        def evaluate(params):
            return self.evaluator.evaluate(params, sample_data)

        ga = GeneticAlgorithm(**ga_cfg)
        start_time = time.time()

        if verbose:
            print(f"  GA配置: pop={ga_cfg['population_size']}, "
                  f"max_gen={ga_cfg['max_generations']}")
            if self.use_gpu:
                print(f"  GPU 加速: 启用")

        gpu_bt = self.evaluator.gpu_backtester if self.use_gpu else None

        best, all_inds, conv_gen = ga.run(
            evaluate, verbose=verbose,
            gpu_backtester=gpu_bt, stock_data=sample_data
        )
        total_time = time.time() - start_time

        system = IndicatorSystem(
            name="通用基线型",
            description=f"在所有市场体制下表现均衡的通用参数（{split} split）",
            applicable_regimes=['BULL', 'BEAR', 'SIDEWAYS', 'CRASH', 'RECOVERY'],
            params=best.params,
            fitness_scores={'GLOBAL': best.fitness},
            confidence=0.7,
            sample_count=best.fitness_details.get('sample_count', 0),
            median_return=best.fitness_details.get('mean_return', 0),  # Phase 3 改用 mean
            win_rate_above_10pct=best.fitness_details.get('target_rate', 0),
            median_sharpe=best.fitness_details.get('median_sharpe', 0),
            median_max_drawdown=best.fitness_details.get('median_drawdown', 0),
            median_holding_days=best.fitness_details.get('avg_holding_days', 0),
            num_trades=best.fitness_details.get('avg_trades', 0),
        )

        result = OptimizationResult(
            regime='GLOBAL',
            best_individual=best,
            best_system=system,
            all_systems=[system],
            generations_run=ga_cfg['max_generations'],
            total_time_sec=total_time,
            convergence_gen=conv_gen,
        )

        self.results['GLOBAL'] = result

        if verbose:
            self._print_result_summary(result)

        return result

    def optimize_by_regime(self, stock_sample=None, n_stocks=50,
                           ga_params=None, verbose=True, split='train') -> dict:
        """
        阶段2: 按体制分别优化（Phase 3: 分桶，默认在 train split 上跑）
        """
        print("\n" + "=" * 70)
        print(f"  阶段2: 按体制分别优化 (Per-Regime, split={split})")
        print("=" * 70)

        if not self.regime_labels:
            print("[ERROR] 无体制标签，请先加载体制标签文件")
            return {}

        sample_data = self._get_sample(stock_sample, n_stocks, split=split)
        if not sample_data:
            return {}

        # 找出数据量足够多的体制
        regime_counts = {}
        sample_labels = {s: self.regime_labels[s] for s in sample_data if s in self.regime_labels}
        for sym, labels in sample_labels.items():
            for regime in labels.unique():
                count = (labels == regime).sum()
                regime_counts[regime] = regime_counts.get(regime, 0) + count

        active_regimes = [r for r, c in regime_counts.items() if c >= 200]
        print(f"  活跃体制: {active_regimes}")
        print(f"  体制数据量: {regime_counts}")
        print(f"  样本股票: {len(sample_data)} 只")

        ga_cfg = {
            'population_size': 60,
            'max_generations': 30,
            'convergence_threshold': 6,
        }
        if ga_params:
            ga_cfg.update(ga_params)

        results = {}

        for regime in active_regimes:
            print(f"\n--- 优化体制: {regime} ---")

            regime_name_cn = {
                'BULL': '牛市', 'BEAR': '熊市', 'SIDEWAYS': '震荡',
                'CRASH': '暴跌', 'RECOVERY': '反弹'
            }.get(regime, regime)

            def evaluate(params, r=regime):
                return self.evaluator.evaluate(
                    params, sample_data, sample_labels, r
                )

            ga = GeneticAlgorithm(**ga_cfg)
            start_time = time.time()

            best, all_inds, conv_gen = ga.run(evaluate, verbose=verbose)
            total_time = time.time() - start_time

            system = IndicatorSystem(
                name=f"{regime_name_cn}专用型",
                description=f"专为{regime_name_cn}市场体制优化的参数（{split} split）",
                applicable_regimes=[regime],
                params=best.params,
                fitness_scores={regime: best.fitness},
                confidence=min(0.9, best.fitness * 1.2),
                sample_count=best.fitness_details.get('sample_count', 0),
                median_return=best.fitness_details.get('mean_return', 0),
                win_rate_above_10pct=best.fitness_details.get('target_rate', 0),
                median_sharpe=best.fitness_details.get('median_sharpe', 0),
                median_max_drawdown=best.fitness_details.get('median_drawdown', 0),
                median_holding_days=best.fitness_details.get('avg_holding_days', 0),
                num_trades=best.fitness_details.get('avg_trades', 0),
            )

            result = OptimizationResult(
                regime=regime,
                best_individual=best,
                best_system=system,
                all_systems=[system],
                generations_run=ga_cfg['max_generations'],
                total_time_sec=total_time,
                convergence_gen=conv_gen,
            )

            results[regime] = result
            self.results[regime] = result

            if verbose:
                self._print_result_summary(result)

        return results

    def optimize_robust(self, stock_sample=None, n_stocks=50,
                        ga_params=None, verbose=True, split='train') -> OptimizationResult:
        """
        阶段3: 稳健策略优化（Phase 3: 默认在 train split 上跑）

        适应度 = 0.7 * min(各体制 fitness) + 0.3 * avg(各体制 fitness)
        """
        print("\n" + "=" * 70)
        print(f"  阶段3: 稳健策略优化 (Robust, split={split})")
        print("=" * 70)

        sample_data = self._get_sample(stock_sample, n_stocks, split=split)
        if not sample_data:
            return None

        if self.regime_labels:
            all_regimes = set()
            sample_labels = {s: self.regime_labels[s] for s in sample_data if s in self.regime_labels}
            for labels in sample_labels.values():
                all_regimes.update(labels.unique())
            all_regimes = sorted(all_regimes)
        else:
            all_regimes = []
            sample_labels = {}

        print(f"  样本股票: {len(sample_data)} 只")
        print(f"  体制: {all_regimes}")

        ga_cfg = {
            'population_size': 80,
            'max_generations': 40,
            'convergence_threshold': 8,
        }
        if ga_params:
            ga_cfg.update(ga_params)

        def evaluate_robust(params):
            if not all_regimes or not sample_labels:
                return self.evaluator.evaluate(params, sample_data)

            regime_fitnesses = {}
            for regime in all_regimes:
                fitness, details = self.evaluator.evaluate(
                    params, sample_data, sample_labels, regime
                )
                if details.get('sample_count', 0) > 0:
                    regime_fitnesses[regime] = fitness

            if not regime_fitnesses:
                return 0.0, {}

            fitness_values = list(regime_fitnesses.values())
            min_fitness = min(fitness_values)
            avg_fitness = np.mean(fitness_values)
            robust_fitness = 0.7 * min_fitness + 0.3 * avg_fitness

            details = {
                'regime_fitnesses': regime_fitnesses,
                'min_fitness': min_fitness,
                'avg_fitness': avg_fitness,
                'regimes_used': list(regime_fitnesses.keys()),
                'regimes_skipped': [r for r in all_regimes if r not in regime_fitnesses],
            }
            return robust_fitness, details

        ga = GeneticAlgorithm(**ga_cfg)
        start_time = time.time()
        best, all_inds, conv_gen = ga.run(evaluate_robust, verbose=verbose)
        total_time = time.time() - start_time

        regime_scores = best.fitness_details.get('regime_fitnesses', {})
        regimes_used = best.fitness_details.get('regimes_used', [])
        regimes_skipped = best.fitness_details.get('regimes_skipped', [])

        if verbose:
            print(f"\n  使用体制: {regimes_used}")
            if regimes_skipped:
                print(f"  跳过体制(无数据): {regimes_skipped}")

        system = IndicatorSystem(
            name="通用稳健型",
            description=f"在所有市场体制下都不差的稳健参数（{split} split）",
            applicable_regimes=regimes_used if regimes_used else all_regimes,
            params=best.params,
            fitness_scores=regime_scores,
            confidence=0.6,
            sample_count=best.fitness_details.get('sample_count', 0),
            median_return=best.fitness_details.get('mean_return', 0),
            win_rate_above_10pct=best.fitness_details.get('target_rate', 0),
            median_sharpe=best.fitness_details.get('median_sharpe', 0),
            median_max_drawdown=best.fitness_details.get('median_drawdown', 0),
            median_holding_days=best.fitness_details.get('avg_holding_days', 0),
            num_trades=best.fitness_details.get('avg_trades', 0),
        )

        result = OptimizationResult(
            regime='ROBUST',
            best_individual=best,
            best_system=system,
            all_systems=[system],
            generations_run=ga_cfg['max_generations'],
            total_time_sec=total_time,
            convergence_gen=conv_gen,
        )
        self.results['ROBUST'] = result
        if verbose:
            self._print_result_summary(result)
        return result

    def iterative_optimize(self, stock_sample=None, n_stocks=50,
                           n_rounds=3, verbose=True, split='train') -> list:
        """
        多轮迭代优化（Phase 3: 默认在 train split 上跑）

        Round 1: 粗粒度 (大参数步长，小种群)
        Round 2: 中粒度
        Round 3: 细粒度 (小步长，大种群)
        """
        print("\n" + "=" * 70)
        print(f"  多轮迭代优化 ({n_rounds} 轮, split={split})")
        print("=" * 70)

        sample_data = self._get_sample(stock_sample, n_stocks, split=split)
        if not sample_data:
            return []

        # 各轮配置
        round_configs = [
            {'population_size': 40, 'max_generations': 15,
             'convergence_threshold': 5, 'mutation_rate': 0.25},
            {'population_size': 60, 'max_generations': 25,
             'convergence_threshold': 6, 'mutation_rate': 0.18},
            {'population_size': 100, 'max_generations': 40,
             'convergence_threshold': 8, 'mutation_rate': 0.12},
        ]

        all_round_results = []
        seed_params = None

        for round_idx in range(min(n_rounds, len(round_configs))):
            cfg = round_configs[round_idx]
            print(f"\n--- Round {round_idx + 1}/{n_rounds} ---")
            print(f"  pop={cfg['population_size']}, "
                  f"max_gen={cfg['max_generations']}, "
                  f"mut_rate={cfg['mutation_rate']}")

            def evaluate(params):
                return self.evaluator.evaluate(params, sample_data)

            ga = GeneticAlgorithm(
                population_size=cfg['population_size'],
                max_generations=cfg['max_generations'],
                convergence_threshold=cfg['convergence_threshold'],
                mutation_rate=cfg['mutation_rate'],
            )

            start_time = time.time()
            best, all_inds, conv_gen = ga.run(
                evaluate, seed_params=seed_params, verbose=verbose
            )
            total_time = time.time() - start_time

            result = {
                'round': round_idx + 1,
                'best_fitness': best.fitness,
                'best_params': best.params,
                'best_details': best.fitness_details,
                'convergence_gen': conv_gen,
                'time_sec': total_time,
            }
            all_round_results.append(result)

            # 将本轮TOP5作为下轮种子
            seed_params = [ind.params for ind in all_inds[:5]]

            if verbose:
                print(f"  最终适应度: {best.fitness:.4f}")
                print(f"  中位收益: {best.fitness_details.get('median_return', 0):.1f}%")
                print(f"  达标率: {best.fitness_details.get('target_rate', 0):.1%}")

        # 构建最终指标体系
        if all_round_results:
            final = all_round_results[-1]
            system = IndicatorSystem(
                name="迭代优化型",
                description=f"经过{n_rounds}轮迭代精炼的参数",
                applicable_regimes=['BULL', 'BEAR', 'SIDEWAYS', 'CRASH', 'RECOVERY'],
                params=final['best_params'],
                fitness_scores={'GLOBAL': final['best_fitness']},
                confidence=0.75,
                sample_count=final['best_details'].get('sample_count', 0),
                median_return=final['best_details'].get('median_return', 0),
                win_rate_above_10pct=final['best_details'].get('target_rate', 0),
                median_sharpe=final['best_details'].get('median_sharpe', 0),
                median_max_drawdown=final['best_details'].get('median_drawdown', 0),
                median_holding_days=final['best_details'].get('avg_holding_days', 0),
                num_trades=final['best_details'].get('avg_trades', 0),
            )

            result = OptimizationResult(
                regime='ITERATIVE',
                best_individual=Individual(
                    params=final['best_params'],
                    fitness=final['best_fitness'],
                    fitness_details=final['best_details'],
                ),
                best_system=system,
                all_systems=[system],
                generations_run=sum(r.get('convergence_gen', 0) for r in all_round_results),
                total_time_sec=sum(r['time_sec'] for r in all_round_results),
                convergence_gen=all_round_results[-1]['convergence_gen'],
            )
            self.results['ITERATIVE'] = result

        return all_round_results

    def _get_sample(self, stock_sample, n_stocks, split='train'):
        """
        获取股票样本（Phase 3: 默认从 train split 抽样）

        Parameters
        ----------
        stock_sample : list or None
            显式指定股票子集
        n_stocks : int
            抽样股票数
        split : str
            'train' / 'val' / 'test' / 'all'
        """
        # 先按 split 过滤 stock_data
        if split != 'all':
            split_syms = self.splits.get(split, {}).get('symbols', set())
            if not split_syms:
                # splits 未加载，fallback 到全集
                pool = dict(self.stock_data)
            else:
                pool = {s: self.stock_data[s] for s in split_syms if s in self.stock_data}
        else:
            pool = dict(self.stock_data)

        if stock_sample:
            return {s: pool[s] for s in stock_sample if s in pool}

        if not pool:
            return {}

        all_symbols = list(pool.keys())
        if len(all_symbols) <= n_stocks:
            return pool

        import random
        labeled = [s for s in all_symbols if s in self.regime_labels]
        unlabeled = [s for s in all_symbols if s not in self.regime_labels]

        sample = []
        n_labeled = min(len(labeled), int(n_stocks * 0.8))
        n_unlabeled = n_stocks - n_labeled

        if n_labeled > 0:
            sample.extend(random.sample(labeled, n_labeled))
        if n_unlabeled > 0 and unlabeled:
            sample.extend(random.sample(unlabeled, min(n_unlabeled, len(unlabeled))))

        return {s: pool[s] for s in sample}

    def _print_result_summary(self, result):
        """打印优化结果摘要"""
        system = result.best_system
        print(f"\n  结果: {system.name}")
        print(f"  适应度: {result.best_individual.fitness:.4f}")
        print(f"  中位收益率: {system.median_return:.1f}%")
        print(f"  达标率(>{self.evaluator.target_return}%): "
              f"{system.win_rate_above_10pct:.1%}")
        print(f"  中位夏普: {system.median_sharpe:.2f}")
        print(f"  中位最大回撤: {system.median_max_drawdown:.1f}%")
        print(f"  平均持有天数: {system.median_holding_days:.0f}")
        print(f"  收敛代数: {result.convergence_gen}")
        print(f"  耗时: {result.total_time_sec:.1f}s")

    # ======================================================================
    #   Phase 3 新增：selection-on-val + test-once
    # ======================================================================

    def selection_on_val(self, top_k: int = 10,
                         overfit_thresholds: list = None) -> dict:
        """
        在 val split 上对 GA-on-train 产出的 TOP-K 个体做 selection

        Parameters
        ----------
        top_k : int
            取 train GA 排名前 K 个个体
        overfit_thresholds : list[float] or None
            overfit_gap 阈值序列；默认 [0.2, 0.3, 0.4]（逐步放宽）

        Returns
        -------
        dict: {regime_name: {'best_individual', 'best_fitness', 'val_fitness',
                              'train_fitness', 'overfit_gap', 'all_evaluated'}}
        """
        if overfit_thresholds is None:
            overfit_thresholds = [0.2, 0.3, 0.4]

        print("\n" + "=" * 70)
        print(f"  Selection on Val (top_k={top_k}, overfit_gap thresholds={overfit_thresholds})")
        print("=" * 70)

        val_data, val_labels = self._get_split_data('val')
        if not val_data:
            print("[ERROR] val split 为空")
            return {}

        results = {}
        # 对每个已训练的 regime 做 selection
        for regime_name, opt_result in self.results.items():
            print(f"\n--- Selection for {regime_name} ---")

            # 找到这个 regime 的 GA 所有个体（按 train fitness 排序后取 top_k）
            # 简单实现：用 best_individual 作为代表，评估其在 val 上的表现
            # 更完整的实现：让 GA 返回所有个体，这里只取 best
            train_fitness = opt_result.best_individual.fitness
            best_params = opt_result.best_individual.params

            # 评估 val
            target_regime = regime_name if regime_name not in ('GLOBAL', 'ROBUST', 'ITERATIVE') else None
            val_fitness, val_details = self.evaluator.evaluate(
                best_params, val_data, val_labels, target_regime
            )

            overfit_gap = max(0, train_fitness - val_fitness)

            # 阈值过滤
            accepted = None
            for thr in overfit_thresholds:
                if overfit_gap <= thr:
                    accepted = thr
                    break

            if accepted is not None:
                print(f"  ✓ 通过阈值 {accepted} (gap={overfit_gap:.4f})")
                print(f"    train_f={train_fitness:.4f}, val_f={val_fitness:.4f}")
            else:
                print(f"  ✗ 所有阈值都未通过 (gap={overfit_gap:.4f})")
                print(f"    train_f={train_fitness:.4f}, val_f={val_fitness:.4f}")

            results[regime_name] = {
                'best_individual': opt_result.best_individual,
                'train_fitness': train_fitness,
                'val_fitness': val_fitness,
                'overfit_gap': overfit_gap,
                'accepted_threshold': accepted,
                'val_details': val_details,
            }

        return results

    def test_once(self, system_or_params, regime: str = None) -> dict:
        """
        在 test split 上对单个策略做一次性评估

        Parameters
        ----------
        system_or_params : IndicatorSystem or dict
            指标体系或直接传参数字典
        regime : str or None
            对 regime-specific 策略，指定体制以正确过滤

        Returns
        -------
        dict: test split 上的 fitness + details
        """
        test_data, test_labels = self._get_split_data('test')
        if not test_data:
            print("[ERROR] test split 为空")
            return {}

        if hasattr(system_or_params, 'params'):
            params = system_or_params.params
            target_regime = (system_or_params.applicable_regimes[0]
                             if regime is None and len(system_or_params.applicable_regimes) == 1
                             else regime)
        else:
            params = system_or_params
            target_regime = regime

        print(f"\n  [test-once] 评估 {'regime=' + str(target_regime) if target_regime else 'global'}")
        fitness, details = self.evaluator.evaluate(
            params, test_data, test_labels, target_regime
        )
        print(f"  test_fitness: {fitness:.4f}")
        if 'mean_return' in details:
            print(f"  mean_return:  {details['mean_return']:.2f}%")
        if 'median_sharpe' in details:
            print(f"  median_sharpe: {details['median_sharpe']:.3f}")
        if 'median_drawdown' in details:
            print(f"  median_drawdown: {details['median_drawdown']:.2f}%")

        return {
            'fitness': fitness,
            'details': details,
            'test_fitness': fitness,
        }

    def run_full_pipeline(self, n_stocks: int = 50,
                          ga_params: dict = None) -> dict:
        """
        完整流程：GA-on-train → selection-on-val → test-once

        Returns
        -------
        dict: 各阶段的完整结果
        """
        print("\n" + "#" * 70)
        print("#  Phase 3 完整流程: train → val → test")
        print("#" * 70)

        # 1. GA on train
        self.optimize_global(n_stocks=n_stocks, ga_params=ga_params, split='train')

        # 2. Selection on val
        selection_results = self.selection_on_val()

        # 3. Test once (用 selection 选中的 best)
        test_results = {}
        for regime_name, sel in selection_results.items():
            if sel['accepted_threshold'] is not None:
                test_results[regime_name] = self.test_once(
                    sel['best_individual'].params,
                    regime=regime_name if regime_name not in ('GLOBAL', 'ROBUST') else None
                )
            else:
                print(f"\n  [test-once] {regime_name} 跳过（未通过 overfit 阈值）")
                test_results[regime_name] = None

        return {
            'train_results': self.results,
            'selection_results': selection_results,
            'test_results': test_results,
        }

    def get_all_systems(self) -> list:
        """获取所有优化产出的指标体系"""
        systems = []
        for key, result in self.results.items():
            systems.extend(result.all_systems)
        return systems

    def save_results(self, filepath):
        """保存优化结果"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'systems': []
        }

        for key, result in self.results.items():
            system = result.best_system
            output['systems'].append({
                'name': system.name,
                'description': system.description,
                'applicable_regimes': system.applicable_regimes,
                'params': system.params,
                'fitness_scores': system.fitness_scores,
                'confidence': system.confidence,
                'median_return': system.median_return,
                'win_rate_above_10pct': system.win_rate_above_10pct,
                'median_sharpe': system.median_sharpe,
                'median_max_drawdown': system.median_max_drawdown,
                'median_holding_days': system.median_holding_days,
                'num_trades': system.num_trades,
                'sample_count': system.sample_count,
                'regime': result.regime,
                'convergence_gen': result.convergence_gen,
                'total_time_sec': result.total_time_sec,
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"[OK] 优化结果已保存: {filepath}")

    def load_results(self, filepath):
        """加载优化结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for sys_data in data.get('systems', []):
            system = IndicatorSystem(
                name=sys_data['name'],
                description=sys_data['description'],
                applicable_regimes=sys_data['applicable_regimes'],
                params=sys_data['params'],
                fitness_scores=sys_data['fitness_scores'],
                confidence=sys_data['confidence'],
                sample_count=sys_data['sample_count'],
                median_return=sys_data.get('median_return', 0),
                win_rate_above_10pct=sys_data.get('win_rate_above_10pct', 0),
                median_sharpe=sys_data.get('median_sharpe', 0),
                median_max_drawdown=sys_data.get('median_max_drawdown', 0),
                median_holding_days=sys_data.get('median_holding_days', 0),
                num_trades=sys_data.get('num_trades', 0),
            )

            regime = sys_data.get('regime', 'UNKNOWN')
            result = OptimizationResult(
                regime=regime,
                best_individual=Individual(
                    params=sys_data['params'],
                    fitness=max(sys_data.get('fitness_scores', {}).values(), default=0),
                ),
                best_system=system,
                all_systems=[system],
                generations_run=sys_data.get('convergence_gen', 0),
                total_time_sec=sys_data.get('total_time_sec', 0),
                convergence_gen=sys_data.get('convergence_gen', 0),
            )
            self.results[regime] = result

        print(f"[OK] 加载 {len(data.get('systems', []))} 套指标体系")


# ==================== CLI ====================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='遗传算法优化引擎')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # global: 全局优化
    p_global = subparsers.add_parser('global', help='全局优化')
    p_global.add_argument('--n-stocks', type=int, default=50, help='抽样股票数')
    p_global.add_argument('--pop', type=int, default=80, help='种群大小')
    p_global.add_argument('--gen', type=int, default=40, help='最大代数')
    p_global.add_argument('--regime-labels', default='stock_data/regime_labels.csv',
                         help='体制标签文件')
    p_global.add_argument('--save', default='stock_data/optimization_result.json',
                         help='保存路径')

    # regime: 按体制优化
    p_regime = subparsers.add_parser('regime', help='按体制分别优化')
    p_regime.add_argument('--n-stocks', type=int, default=50, help='抽样股票数')
    p_regime.add_argument('--pop', type=int, default=60, help='种群大小')
    p_regime.add_argument('--gen', type=int, default=30, help='最大代数')
    p_regime.add_argument('--regime-labels', default='stock_data/regime_labels.csv',
                         help='体制标签文件')
    p_regime.add_argument('--save', default='stock_data/optimization_result.json',
                         help='保存路径')

    # robust: 稳健优化
    p_robust = subparsers.add_parser('robust', help='稳健策略优化')
    p_robust.add_argument('--n-stocks', type=int, default=50, help='抽样股票数')
    p_robust.add_argument('--pop', type=int, default=80, help='种群大小')
    p_robust.add_argument('--gen', type=int, default=40, help='最大代数')
    p_robust.add_argument('--regime-labels', default='stock_data/regime_labels.csv',
                         help='体制标签文件')
    p_robust.add_argument('--save', default='stock_data/optimization_result.json',
                         help='保存路径')

    # full: 完整优化流程 (全局+体制+稳健+迭代)
    p_full = subparsers.add_parser('full', help='完整优化流程')
    p_full.add_argument('--n-stocks', type=int, default=50, help='抽样股票数')
    p_full.add_argument('--regime-labels', default='stock_data/regime_labels.csv',
                        help='体制标签文件')
    p_full.add_argument('--save', default='stock_data/optimization_result.json',
                        help='保存路径')

    # iterative: 多轮迭代优化
    p_iter = subparsers.add_parser('iterative', help='多轮迭代优化')
    p_iter.add_argument('--n-stocks', type=int, default=50, help='抽样股票数')
    p_iter.add_argument('--rounds', type=int, default=3, help='迭代轮数')
    p_iter.add_argument('--save', default='stock_data/optimization_result.json',
                        help='保存路径')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    engine = OptimizationEngine()

    if args.command in ('global', 'regime', 'robust', 'full', 'iterative'):
        labels_file = getattr(args, 'regime_labels', None)
        engine.load_data(regime_labels_file=labels_file)

    ga_params = {}
    if hasattr(args, 'pop'):
        ga_params['population_size'] = args.pop
    if hasattr(args, 'gen'):
        ga_params['max_generations'] = args.gen

    if args.command == 'global':
        engine.optimize_global(n_stocks=args.n_stocks, ga_params=ga_params)
        engine.save_results(args.save)

    elif args.command == 'regime':
        engine.optimize_by_regime(n_stocks=args.n_stocks, ga_params=ga_params)
        engine.save_results(args.save)

    elif args.command == 'robust':
        engine.optimize_robust(n_stocks=args.n_stocks, ga_params=ga_params)
        engine.save_results(args.save)

    elif args.command == 'iterative':
        engine.iterative_optimize(n_stocks=args.n_stocks, n_rounds=args.rounds)
        engine.save_results(args.save)

    elif args.command == 'full':
        print("=" * 70)
        print("  完整优化流程: 全局 → 按体制 → 稳健 → 迭代")
        print("=" * 70)

        engine.optimize_global(n_stocks=args.n_stocks)
        engine.optimize_by_regime(n_stocks=args.n_stocks)
        engine.optimize_robust(n_stocks=args.n_stocks)
        engine.iterative_optimize(n_stocks=args.n_stocks, n_rounds=2)

        engine.save_results(args.save)

        # 打印所有体系总结
        print("\n" + "=" * 70)
        print("  优化完成 — 指标体系总结")
        print("=" * 70)
        for system in engine.get_all_systems():
            print(f"\n  [{system.name}]")
            print(f"    {system.description}")
            print(f"    适用: {', '.join(system.applicable_regimes)}")
            print(f"    中位收益: {system.median_return:.1f}% | "
                  f"达标率: {system.win_rate_above_10pct:.1%} | "
                  f"夏普: {system.median_sharpe:.2f} | "
                  f"回撤: {system.median_max_drawdown:.1f}%")


if __name__ == "__main__":
    main()
