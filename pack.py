# -*- coding: utf-8 -*-
"""
打包脚本 - 将所需模块复制到 optimization_package/ 目录

在项目根目录运行:
    python optimization_package/pack.py

运行后 optimization_package/ 可以整个复制到其他电脑
"""
import shutil
from pathlib import Path

# 需要复制的模块
SKILLS = [
    'market_regime_classifier.py',
    'historical_data_manager.py',
    'parameter_space.py',
    'vectorized_backtest.py',
    'optimization_engine.py',
    'multi_stock_backtester.py',
    'optimization_config.py',
    'daily_signal_generator.py',
    'indicator_system_report.py',
    'signal_generator.py',
    'technical_analyzer.py',
    'stock_data_fetcher.py',
    'advanced_indicators.py',
    'trend_indicators.py',
    'pattern_recognition.py',
    'risk_management.py',
    'backtest_framework.py',
]

def main():
    src_dir = Path('.claude/skills')
    dst_dir = Path('optimization_package/skills')
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for name in SKILLS:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            copied += 1
            print(f"  [OK] {name}")
        else:
            print(f"  [SKIP] {name} (不存在)")

    # 复制 stock_list.csv (股票名称映射)
    stock_list = src_dir / 'stock_list.csv'
    if stock_list.exists():
        shutil.copy2(stock_list, dst_dir / 'stock_list.csv')
        print(f"  [OK] stock_list.csv")

    print(f"\n打包完成: {copied} 个模块 → {dst_dir}")
    print(f"将 optimization_package/ 整个目录复制到目标电脑即可")

if __name__ == "__main__":
    main()
