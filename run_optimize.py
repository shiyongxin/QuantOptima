# -*- coding: utf-8 -*-
"""
阶段2: 参数优化脚本 (离线运行, 不需要联网)

功能:
1. 加载股票数据和体制标签
2. 运行遗传算法多体制优化
3. 产出多套指标体系
4. 生成优化报告

运行:
    python run_optimize.py                        # 默认配置
    python run_optimize.py --preset quick         # 快速测试
    python run_optimize.py --preset thorough      # 深度优化
    python run_optimize.py --n-stocks 80 --gen 50 # 自定义参数

预计耗时: quick约30分钟, standard约4-6小时, thorough约8-12小时
"""
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'skills'))

from optimization_engine import OptimizationEngine
from optimization_config import (
    OptimizationConfig, quick_test_config, standard_config,
    thorough_config, production_config
)
from indicator_system_report import IndicatorSystemReport


PRESETS = {
    'quick': quick_test_config,
    'standard': standard_config,
    'thorough': thorough_config,
    'production': production_config,
}


def run_optimization(args):
    """运行优化"""
    # 加载配置
    if args.config:
        config = OptimizationConfig.load(args.config)
        print(f"[INFO] 使用配置文件: {args.config}")
    else:
        config = PRESETS[args.preset]()
        print(f"[INFO] 使用预设: {args.preset}")

    # 命令行覆盖
    if args.n_stocks:
        config.data.stock_sample_size = args.n_stocks
    if args.pop:
        config.ga.population_size = args.pop
    if args.gen:
        config.ga.max_generations = args.gen

    # 验证配置
    issues = config.validate()
    if issues:
        print("[WARN] 配置问题:")
        for issue in issues:
            print(f"  - {issue}")

    # 初始化引擎
    engine = OptimizationEngine(data_dir=args.data_dir, use_gpu=args.gpu)

    # 加载数据
    print("\n[INFO] 加载数据...")
    engine.load_data(
        min_history_days=config.data.min_history_days,
        regime_labels_file=args.regime_labels,
    )

    if not engine.stock_data:
        print("[ERROR] 无可用股票数据")
        print("  请先运行: python fetch_data.py")
        return

    ga_params = config.get_ga_params()
    n_stocks = config.data.stock_sample_size

    print(f"\n[INFO] 优化配置:")
    print(f"  股票样本: {n_stocks}")
    print(f"  种群大小: {ga_params['population_size']}")
    print(f"  最大代数: {ga_params['max_generations']}")
    print(f"  目标收益: {config.fitness.target_return}%")
    print(f"  目标概率: {config.fitness.target_probability:.0%}")

    # 根据模式运行
    mode = args.mode or config.mode
    total_start = time.time()

    if mode == 'global':
        engine.optimize_global(n_stocks=n_stocks, ga_params=ga_params)

    elif mode == 'regime':
        engine.optimize_by_regime(n_stocks=n_stocks, ga_params=ga_params)

    elif mode == 'robust':
        engine.optimize_robust(n_stocks=n_stocks, ga_params=ga_params)

    elif mode == 'iterative':
        engine.iterative_optimize(n_stocks=n_stocks, n_rounds=args.rounds)

    elif mode == 'full':
        print("\n" + "=" * 70)
        print("  完整优化流程: 全局 → 按体制 → 稳健 → 迭代")
        print("=" * 70)

        # Phase 1: 全局优化
        print("\n>>> Phase 1/4: 全局优化")
        engine.optimize_global(n_stocks=n_stocks, ga_params=ga_params)

        # Phase 2: 按体制优化
        print("\n>>> Phase 2/4: 按体制优化")
        regime_ga = ga_params.copy()
        regime_ga['population_size'] = max(40, ga_params['population_size'] // 2)
        regime_ga['max_generations'] = max(20, ga_params['max_generations'] // 2)
        engine.optimize_by_regime(n_stocks=n_stocks, ga_params=regime_ga)

        # Phase 3: 稳健优化
        print("\n>>> Phase 3/4: 稳健策略优化")
        engine.optimize_robust(n_stocks=n_stocks, ga_params=ga_params)

        # Phase 4: 多轮迭代
        print("\n>>> Phase 4/4: 多轮迭代精炼")
        engine.iterative_optimize(n_stocks=n_stocks, n_rounds=2)

    total_time = time.time() - total_start

    # 保存结果
    engine.save_results(args.output)

    # 生成报告
    report_dir = Path(args.data_dir) / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)

    report_gen = IndicatorSystemReport()
    report_gen.load_from_engine(engine)
    report_gen.save_text(report_dir / 'indicator_systems.txt')
    report_gen.save_markdown(report_dir / 'indicator_systems.md')
    report_gen.save_html(report_dir / 'indicator_systems.html')

    # 打印总结
    print("\n" + "=" * 70)
    print("  优化完成!")
    print("=" * 70)
    print(f"  总耗时: {total_time/60:.1f} 分钟")
    print(f"  产出体系: {len(engine.get_all_systems())} 套")
    print(f"  结果文件: {args.output}")
    print(f"  报告目录: {report_dir}")
    print()

    for system in engine.get_all_systems():
        regimes_cn = {
            'BULL': '牛市', 'BEAR': '熊市', 'SIDEWAYS': '震荡',
            'CRASH': '暴跌', 'RECOVERY': '反弹',
            'GLOBAL': '通用', 'ROBUST': '稳健', 'ITERATIVE': '迭代'
        }
        regimes = ', '.join(regimes_cn.get(r, r) for r in system.applicable_regimes)
        print(f"  [{system.name}]")
        print(f"    适用: {regimes}")
        print(f"    中位收益: {system.median_return:+.1f}% | "
              f"达标率: {system.win_rate_above_10pct:.1%} | "
              f"夏普: {system.median_sharpe:.2f}")
        print()

    print("=" * 70)
    print("  下一步: 将 optimization_result.json 复制回开发机器")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='阶段2: 参数优化')
    parser.add_argument('--preset', choices=['quick', 'standard', 'thorough', 'production'],
                       default='standard', help='预设配置')
    parser.add_argument('--config', help='自定义配置文件路径')
    parser.add_argument('--mode', choices=['global', 'regime', 'robust', 'iterative', 'full'],
                       help='优化模式 (覆盖配置文件)')
    parser.add_argument('--n-stocks', type=int, help='样本股票数')
    parser.add_argument('--pop', type=int, help='种群大小')
    parser.add_argument('--gen', type=int, help='最大代数')
    parser.add_argument('--rounds', type=int, default=2, help='迭代轮数')
    parser.add_argument('--data-dir', default='stock_data', help='数据目录')
    parser.add_argument('--regime-labels', default='stock_data/regime_labels.csv',
                       help='体制标签文件')
    parser.add_argument('--output', default='stock_data/optimization_result.json',
                       help='输出文件路径')
    parser.add_argument('--gpu', action='store_true',
                       help='使用 GPU (Apple Silicon MPS) 加速')

    args = parser.parse_args()
    run_optimization(args)


if __name__ == "__main__":
    main()
