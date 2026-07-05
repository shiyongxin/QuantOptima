# -*- coding: utf-8 -*-
"""
纯多头过滤选股 — 不打分、只过/不过。

三条硬规则（缺一不可，缺一即排）:
  1. MA20 向上    — 今日 MA20 > 昨日 MA20
  2. 收盘价 > MA20 — 最新收盘价站在 MA20 上方
  3. MA20 > MA60  — 多头排列（中期趋势 > 长期趋势）

用法:
    python skills/bull_filter.py                        # 默认 universe=1832
    python skills/bull_filter.py --loose                # 宽松模式 (MA20 斜率改 5 日对比, MA60 > MA20*0.98)
    python skills/bull_filter.py --sector-filter 600    # 只看沪市主板
    python skills/bull_filter.py --exclude-st            # 排除 ST/退市风险
    python skills/bull_filter.py --output bull.csv      # 保存 CSV
    python skills/bull_filter.py --debug-symbol 000001   # 单股验证 (打印三个条件数值)

设计:
- 复用 HistoricalDataManager.get_universe() (默认 1832 只)
- 复用 mgr.load() / mgr.get_stock_name() 跟 stock_screener.py 保持一致
- 不依赖 GA 优化结果 — 这是独立的"硬过滤器"
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from historical_data_manager import HistoricalDataManager


def check_bull(data, loose=False):
    """
    3 个多头条件硬过滤

    Args:
        data:   DataFrame, 至少包含 '收盘' 列
        loose:  False = 严格（默认）；True = 宽松（MA20 斜率放宽到 5 日，MA20 与 MA60 允许小幅纠缠）

    Returns:
        dict with keys: pass (bool), close, ma20, ma60, ma20_now, ma20_yest, ma20_5d_ago,
                        cond_up, cond_above, cond_align, reason
    """
    if data is None or len(data) < 65:
        return {
            'pass': False,
            'reason': f'数据不足 (len={len(data) if data is not None else 0} < 65)',
            'cond_up': False, 'cond_above': False, 'cond_align': False,
        }

    close = data['收盘'].astype(float)

    # 计算 MA
    ma20_series = close.rolling(20).mean()
    ma60_series = close.rolling(60).mean()

    cur_close = float(close.iloc[-1])
    ma20_now = float(ma20_series.iloc[-1])
    ma60_now = float(ma60_series.iloc[-1])
    ma20_yest = float(ma20_series.iloc[-2])
    ma20_5d = float(ma20_series.iloc[-6]) if len(ma20_series) >= 6 else ma20_now

    # 条件 1: MA20 向上
    if loose:
        # 宽松: 5 日 MA20 抬升即可
        cond_up = ma20_now > ma20_5d
    else:
        # 严格: 今日 MA20 > 昨日 MA20（一天比一天高）
        cond_up = ma20_now > ma20_yest

    # 条件 2: 收盘价 > MA20
    cond_above = cur_close > ma20_now

    # 条件 3: MA20 > MA60 (多头排列)
    if loose:
        # 宽松: 允许 MA20 略低于 MA60 (纠缠区), MA20 > MA60 * 0.98
        cond_align = ma20_now > ma60_now * 0.98
    else:
        # 严格: MA20 真正站上 MA60
        cond_align = ma20_now > ma60_now

    all_pass = cond_up and cond_above and cond_align

    if not all_pass:
        fail_parts = []
        if not cond_up:
            fail_parts.append('MA20未向上')
        if not cond_above:
            fail_parts.append('股价未站上MA20')
        if not cond_align:
            fail_parts.append('MA20≤MA60 (空头)')
        reason = ';'.join(fail_parts)
    else:
        reason = '✓ 三条全部满足'

    # 涨跌幅
    change_1d = (cur_close / float(close.iloc[-2]) - 1) * 100 if len(close) > 1 else 0
    change_5d = (cur_close / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0
    change_20d = (cur_close / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0

    # MA20 斜率 (5 日对比)
    ma20_slope_5d = (ma20_now / ma20_5d - 1) * 100 if ma20_5d > 0 else 0

    return {
        'pass': all_pass,
        'reason': reason,
        'close': cur_close,
        'ma20': ma20_now,
        'ma60': ma60_now,
        'ma20_yest': ma20_yest,
        'ma20_5d_ago': ma20_5d,
        'ma20_slope_5d_pct': ma20_slope_5d,
        'cond_up': cond_up,
        'cond_above': cond_above,
        'cond_align': cond_align,
        'change_1d': change_1d,
        'change_5d': change_5d,
        'change_20d': change_20d,
    }


def action_label(close, ma20, ma60):
    """根据价位给操作建议 (不影响过滤结果, 仅供参考)"""
    if ma60 <= 0 or ma20 <= 0:
        return '?'
    pct_above_ma20 = (close / ma20 - 1) * 100
    if pct_above_ma20 > 10:
        return '远离MA20 追高风险'
    elif pct_above_ma20 > 5:
        return '强势区'
    elif pct_above_ma20 > 0:
        return '刚站稳MA20'
    else:
        return 'N/A'  # 不会到这里 (cond_above 已保证 > MA20)


def main():
    parser = argparse.ArgumentParser(
        description='纯多头过滤: MA20↑ ∧ 收盘>MA20 ∧ MA20>MA60'
    )
    parser.add_argument('--loose', action='store_true',
                        help='宽松模式: MA20 斜率改用 5 日对比, MA20≥MA60*0.98 视为多头')
    parser.add_argument('--sector-filter', default='', help='代码前缀过滤, 如 600/688/000')
    parser.add_argument('--exclude-st', action='store_true', help='排除 ST/*ST 股票')
    parser.add_argument('--output', default=None, help='CSV 输出文件')
    parser.add_argument('--data-dir', default='stock_data')
    parser.add_argument('--max-stocks', type=int, default=0,
                        help='最大扫描股票数 (0=全量 ~1832), 调试用')
    parser.add_argument('--debug-symbol', default='', help='对单只股票打印三个条件数值, 不进入扫描')
    parser.add_argument('--detail-all', action='store_true',
                        help='输出所有扫描结果 (包括被排除的), 用于审计')
    args = parser.parse_args()

    mgr = HistoricalDataManager(args.data_dir)

    # 单股调试模式
    if args.debug_symbol:
        sym = args.debug_symbol
        print(f'🔍 调试单股: {sym}')
        data = mgr.load(sym)
        if data is None or len(data) == 0:
            print(f'  ❌ 数据为空')
            return
        print(f'  数据范围: {data.index[0]} ~ {data.index[-1]} ({len(data)} 行)')
        result = check_bull(data, loose=args.loose)
        print(f'  最新收盘: {result["close"]:.2f}')
        print(f'  MA20:     {result["ma20"]:.2f} (昨: {result["ma20_yest"]:.2f}, 5日前: {result["ma20_5d_ago"]:.2f})')
        print(f'  MA60:     {result["ma60"]:.2f}')
        print(f'  MA20 5日斜率: {result["ma20_slope_5d_pct"]:+.2f}%')
        ma_above_pct = (result['close']/result['ma20']-1)*100
        ma_align_pct = (result['ma20']/result['ma60']-1)*100
        cu = 'YES' if result['cond_up'] else 'NO '
        ca = 'YES' if result['cond_above'] else 'NO '
        cl = 'YES' if result['cond_align'] else 'NO '
        pf = 'PASS' if result['pass'] else 'FAIL'
        print(f'  条件 1 (MA20↑):        {cu}')
        print(f'  条件 2 (收盘 > MA20):  {ca}  偏离 {ma_above_pct:+.2f}%')
        print(f'  条件 3 (MA20 > MA60):  {cl}  偏离 {ma_align_pct:+.2f}%')
        print(f'  最终: {pf} -- {result["reason"]}')
        return

    # 全市场扫描
    print('[INFO] 加载股票池...')
    universe = mgr.get_universe()
    if args.sector_filter:
        universe = [s for s in universe if s.startswith(args.sector_filter)]
    if args.max_stocks > 0:
        universe = universe[:args.max_stocks]
    print(f'[OK] 待扫描: {len(universe)} 只')
    print(f'      模式: {"宽松" if args.loose else "严格"}')

    print('[INFO] 开始过滤...')
    passed = []
    failed_detail = [] if args.detail_all else None

    for i, sym in enumerate(universe):
        if (i + 1) % 200 == 0:
            print(f'  进度: {i + 1}/{len(universe)} (通过 {len(passed)})')
        try:
            data = mgr.load(sym)
            if len(data) < 65:
                continue
            result = check_bull(data, loose=args.loose)

            name = mgr.get_stock_name(sym)
            # ST 过滤
            if args.exclude_st and ('ST' in name or '*ST' in name):
                continue

            row = {
                'symbol': sym,
                'name': name,
                **result,
            }
            if result['pass']:
                passed.append(row)
            elif args.detail_all:
                failed_detail.append(row)
        except Exception:
            continue

    # 排序: 按 MA20 距收盘的偏离度升序 (刚站上 MA20 的优先)
    passed.sort(key=lambda r: r['change_1d'], reverse=False)

    # 输出
    print()
    print('=' * 78)
    print(f'   纯多头过滤 ({datetime.now().strftime("%Y-%m-%d %H:%M")})')
    print('=' * 78)
    print(f'扫描: {len(universe)} 只 | 通过: {len(passed)} 只 | 通过率: {len(passed)/max(len(universe),1)*100:.1f}%')
    print(f'规则: MA20↑ ∧ 收盘>MA20 ∧ MA20>MA60 (模式: {"宽松" if args.loose else "严格"})')
    print()

    if not passed:
        print('❌ 没有任何股票满足全部 3 条')
        return

    # 表格
    print(f'【通过名单 · 共 {len(passed)} 只】')
    print('-' * 110)
    header = (f'{"代码":<7} {"名称":<10} {"价格":>7} {"1日":>6} {"5日":>6} {"20日":>6}  '
              f'{"MA20":>6} {"MA60":>6} {"MA20斜率":>8} {"距MA20":>7}  提示')
    print(header)
    print('-' * 110)

    for r in passed:
        pct_to_ma20 = (r['close'] / r['ma20'] - 1) * 100
        tip = action_label(r['close'], r['ma20'], r['ma60'])
        print(f'{r["symbol"]:<7} {r["name"]:<10} {r["close"]:>7.2f} '
              f'{r["change_1d"]:>+5.2f}% {r["change_5d"]:>+5.2f}% {r["change_20d"]:>+5.2f}%  '
              f'{r["ma20"]:>6.2f} {r["ma60"]:>6.2f} {r["ma20_slope_5d_pct"]:>+7.2f}% '
              f'{pct_to_ma20:>+6.2f}%  {tip}')
    print('-' * 110)

    # 详细失败模式统计
    print()
    print('【全部扫描结果统计】 (用于排除分布审计)')
    fail_stats = {'数据不足': 0, 'MA20未向上': 0, '股价未站上MA20': 0, 'MA20≤MA60': 0}
    if args.detail_all and failed_detail is not None:
        for r in failed_detail:
            for k in fail_stats:
                if k in r['reason']:
                    fail_stats[k] += 1
        for k, v in fail_stats.items():
            print(f'  {k}: {v} 只')
    else:
        # 即使不打印 detail, 也用 passed 反推粗估
        # (无法精确, 这里省略)
        print('  (加 --detail-all 可看到排除原因分布)')

    # CSV
    if args.output:
        df = pd.DataFrame(passed)
        df.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f'\n[OK] CSV 已保存: {args.output} ({len(passed)} 只)')

    return passed


if __name__ == '__main__':
    main()
