#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_data.py — 一键增量更新 QuantOptima 所有股票到最新

用法:
    python update_data.py                 # 默认 12 天增量窗口
    python update_data.py --days 7        # 自定义增量窗口 (天)
    python update_data.py --check         # 只检查不更新
    python update_data.py --workers 1     # 强制单线程 (默认就是 1)
    python update_data.py --output-dir /tmp  # 把 symbol list 输出到指定目录

工作流:
    1. 扫描 stock_data/historical/*.parquet 找 last_date 分布
    2. 决定 start_date / end_date
    3. 写 /tmp/all_symbols.txt
    4. 调 HistoricalDataManager.batch_fetch() (单线程, 因 pyarrow+mini_racer 段错误)
    5. 验证: last_date 中位数应 <= 2 天 (周末/节假日)

⚠️ 关键决策: max_workers=1
   Python 3.14 + pyarrow 24 + ThreadPoolExecutor → mini_racer/V8 segfault
   已实测 (2026-07-05), 单线程比 6 线程还快 (24 min vs n/a 真值)
   见 ~/.hermes/skills/finance/quantoptima-incremental-data-refresh/SKILL.md
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 项目结构 (项目根 / skills/)
PROJECT_ROOT = Path(__file__).parent
SKILLS_DIR = PROJECT_ROOT / 'skills'
DATA_DIR = PROJECT_ROOT / 'stock_data'
HIST_DIR = DATA_DIR / 'historical'
META_FILE = DATA_DIR / 'metadata.json'
SYMBOL_LIST_FILE = Path('/tmp/all_symbols.txt')

sys.path.insert(0, str(SKILLS_DIR))

import pandas as pd
import pyarrow.parquet as pq


def scan_last_dates(max_sample=None):
    """扫描所有 parquet 文件的 last_date"""
    files = sorted(HIST_DIR.glob('*.parquet'))
    if max_sample:
        files = files[:max_sample]

    rows = []
    for f in files:
        try:
            tbl = pq.read_table(f, columns=['日期'])
            if tbl.num_rows == 0:
                continue
            last = pd.to_datetime(tbl.column('日期').to_pylist()[-1])
            first = pd.to_datetime(tbl.column('日期').to_pylist()[0])
            rows.append((f.stem, first, last))
        except Exception as e:
            print(f'  [WARN] {f.stem}: {e}')

    if not rows:
        return None, files

    df = pd.DataFrame(rows, columns=['symbol', 'first', 'last'])
    return df, files


def print_scan_summary(df):
    """打印扫描结果摘要"""
    today = pd.Timestamp.now().normalize()
    df['days_old'] = (today - df['last']).dt.days

    print('=' * 70)
    print(' 股票数据新鲜度报告')
    print('=' * 70)
    print(f'扫描股票数: {len(df)}')
    print(f'最早 last_date: {df["last"].min().date()}')
    print(f'最晚 last_date: {df["last"].max().date()}')
    print(f'\n距离今天天数分布:')
    print(df['days_old'].describe().round(0).to_string())
    print(f'\nlast_date 唯一取值 TOP 5:')
    print(df['last'].dt.date.value_counts().head(5).to_string())

    median_days = df['days_old'].median()
    print(f'\n中位数: {median_days:.0f} 天')
    if median_days <= 2:
        print('✅ 数据已是最新 (周末/节假日)')
    elif median_days <= 7:
        print('⚠️  数据略旧, 建议增量更新')
    else:
        print('❌ 数据明显滞后, 强烈建议全量重抓 (fetch_data.py --full)')
    return median_days


def write_symbol_list(output_dir='/tmp'):
    """把所有 symbol 写到文件, 给 batch_fetch 用"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'all_symbols.txt'
    symbols = sorted([f.stem for f in HIST_DIR.glob('*.parquet')])
    out.write_text('\n'.join(symbols))
    return out, symbols


def run_batch_fetch(symbols, start_date, end_date, workers=1, delay=0.1):
    """调 HistoricalDataManager.batch_fetch 做增量更新"""
    from historical_data_manager import HistoricalDataManager

    mgr = HistoricalDataManager(str(DATA_DIR))
    print(f'\n[INFO] 启动增量更新')
    print(f'       股票: {len(symbols)} 只')
    print(f'       窗口: {start_date} ~ {end_date}')
    print(f'       并发: {workers} (单线程)')
    print(f'       间隔: {delay}s')
    print(f'       预计耗时: {len(symbols) * delay / max(workers,1) / 60:.0f} 分钟 (理论)')
    print()

    t0 = time.time()
    mgr.batch_fetch(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        max_workers=workers,
        delay=delay,
        save_interval=50,
    )
    elapsed = time.time() - t0
    print(f'\n[OK] 增量更新完成, 实际耗时 {elapsed/60:.1f} 分钟')
    return elapsed


def verify_update(target_days=3):
    """验证: last_date 中位数应 <= target_days"""
    print('\n' + '=' * 70)
    print(' 验证更新结果')
    print('=' * 70)
    df, _ = scan_last_dates()
    if df is None:
        print('[FAIL] 无 parquet 文件')
        return False

    today = pd.Timestamp.now().normalize()
    df['days_old'] = (today - df['last']).dt.days
    median_days = df['days_old'].median()
    print(f'\n  中位数距今: {median_days:.0f} 天 (目标 <= {target_days})')

    if median_days <= target_days:
        print('  ✅ 通过, 数据已最新')
        return True
    else:
        print(f'  ❌ 未达标 ({median_days:.0f} > {target_days})')
        return False


def main():
    parser = argparse.ArgumentParser(
        description='QuantOptima 股票数据增量更新',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python update_data.py                 # 12 天增量窗口 (推荐)
  python update_data.py --days 7        # 7 天窗口 (更快)
  python update_data.py --check         # 只扫描不更新
  python update_data.py --check --sample 50   # 只看前 50 只诊断
        """,
    )
    parser.add_argument('--days', type=int, default=12,
                        help='增量窗口天数 (回溯 N 天, 默认 12)')
    parser.add_argument('--workers', type=int, default=1,
                        help='并发数 (默认 1, ⚠️ 不要改大)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='API 请求间隔秒数 (默认 0.1)')
    parser.add_argument('--check', action='store_true',
                        help='只检查数据新鲜度, 不执行更新')
    parser.add_argument('--skip-verify', action='store_true',
                        help='更新后跳过验证步骤')
    parser.add_argument('--output-dir', default='/tmp',
                        help='symbol list 输出目录')
    parser.add_argument('--sample', type=int, default=0,
                        help='扫描时只处理前 N 只 (0=全部)')
    args = parser.parse_args()

    print('=' * 70)
    print(f' QuantOptima 增量更新工具')
    print(f' {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)
    print(f' 数据目录: {DATA_DIR}')
    print(f' 增量窗口: {args.days} 天')
    print(f' 并发: {args.workers}')
    print()

    # Step 1: 扫描
    print('[Step 1/4] 扫描当前数据状态...')
    df, _ = scan_last_dates(max_sample=args.sample if args.sample > 0 else None)
    if df is None:
        print('[FAIL] 无历史数据, 请先跑 fetch_data.py --full')
        return 1
    print_scan_summary(df)

    if args.check:
        print('\n[INFO] --check 已设置, 不执行更新')
        return 0

    # Step 2: 准备 symbol list
    print(f'\n[Step 2/4] 写 symbol list 到 {args.output_dir}...')
    out, symbols = write_symbol_list(args.output_dir)
    print(f'       写入 {len(symbols)} 只 -> {out}')

    # Step 3: 跑增量
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y%m%d')
    print(f'\n[Step 3/4] 跑增量 (start={start_date} end={end_date})...')
    run_batch_fetch(symbols, start_date, end_date,
                    workers=args.workers, delay=args.delay)

    # Step 4: 验证
    if args.skip_verify:
        print('\n[Step 4/4] --skip-verify 已设置, 跳过')
        return 0

    print(f'\n[Step 4/4] 验证更新结果...')
    ok = verify_update(target_days=3)

    # 收尾
    print('\n' + '=' * 70)
    if ok:
        print(' ✅ 全部完成. 数据已最新.')
        print(' 下一步:')
        print('   python skills/bull_filter.py    # 跑多头过滤')
        print('   python skills/daily_signal_generator.py   # 出日报')
    else:
        print(' ⚠️  更新已完成但验证未达标. 检查:')
        print('   1. 网络是否可达 akshare API?')
        print('   2. 是否非交易日 (周末/节假日)?')
        print('   3. 重新跑: python update_data.py --skip-verify')
    print('=' * 70)

    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
