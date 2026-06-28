# -*- coding: utf-8 -*-
"""
综合评分选股 — 不依赖 BUY 信号，全量扫描 universe 每只股票的技术面，
输出综合得分 TOP N。

评分维度（合计 100 分）：
  趋势 40%：MA5/10/20/60 排列 + 收盘价 vs MA20 + MA20 斜率
  动量 25%：MACD 柱 + RSI + KDJ_K + KDJ_J
  量能 15%：5日量比 + 量价配合（涨+放量 > 涨+缩量 > 跌+缩量 > 跌+放量）
  风险 20%：BB 位置（0.5=中性）+ ATR%（越低越好）+ 20日波动率

用法：
    python skills/stock_screener.py                     # 默认 universe=1832, top=20
    python skills/stock_screener.py --top 30            # 选 30 只
    python skills/stock_screener.py --min-score 60      # 只看 >=60 分
    python skills/stock_screener.py --sector-filter "600"  # 只看沪市主板
    python skills/stock_screener.py --output picks.csv  # 导出 CSV
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np

from historical_data_manager import HistoricalDataManager
from daily_signal_generator import DailySignalGenerator


def score_trend(data):
    """
    趋势得分（0-40）：MA 排列 + 收盘价位置 + MA20 斜率
    """
    close = data['收盘'].astype(float)
    if len(close) < 60:
        return None

    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    cur = close.iloc[-1]

    score = 0

    # 1. MA 排列 (15 分)
    if ma5 > ma10 > ma20 > ma60:
        score += 15  # 全多头排列
    elif ma5 > ma10 > ma20:
        score += 12
    elif ma5 > ma20:
        score += 6
    elif ma5 < ma10 < ma20 < ma60:
        score += 0  # 全空头
    elif ma5 < ma10 < ma20:
        score += 3
    elif ma5 < ma20:
        score += 7
    else:
        score += 8  # 纠缠

    # 2. 收盘价 vs MA20 (10 分)
    if cur > ma20 * 1.05:
        score += 10  # 站上 MA20 5%+
    elif cur > ma20:
        score += 8
    elif cur > ma20 * 0.95:
        score += 5
    elif cur > ma20 * 0.90:
        score += 2
    else:
        score += 0  # 远离 MA20 下行

    # 3. MA20 斜率 (15 分) — 上升趋势加分
    if len(close) >= 40:
        ma20_now = close.rolling(20).mean().iloc[-1]
        ma20_prev = close.rolling(20).mean().iloc[-20]
        ma20_slope = (ma20_now - ma20_prev) / ma20_prev if ma20_prev > 0 else 0
        if ma20_slope > 0.05:
            score += 15
        elif ma20_slope > 0.02:
            score += 12
        elif ma20_slope > 0:
            score += 8
        elif ma20_slope > -0.02:
            score += 5
        elif ma20_slope > -0.05:
            score += 2
        else:
            score += 0

    return min(score, 40)


def score_momentum(data):
    """
    动量得分（0-25）：MACD 柱 + RSI + KDJ
    """
    close = data['收盘'].astype(float)
    if len(close) < 30:
        return None

    score = 0

    # 1. MACD 柱 (10 分) — 正值加分
    if len(close) >= 26:
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        macd_hist = (dif - dea).iloc[-1] * 2

        # 用价格标准化（避免高价股优势）
        norm_hist = macd_hist / close.iloc[-1] * 100
        if norm_hist > 1.0:
            score += 10
        elif norm_hist > 0.5:
            score += 8
        elif norm_hist > 0:
            score += 5
        elif norm_hist > -0.5:
            score += 2
        else:
            score += 0

    # 2. RSI (10 分)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss if loss.iloc[-1] > 0 else 100
    rsi = 100 - (100 / (1 + rs.iloc[-1]))
    if 50 < rsi < 70:
        score += 10  # 健康上涨区
    elif rsi >= 70:
        score += 6   # 超买（扣分避免追高）
    elif 40 < rsi <= 50:
        score += 7
    elif 30 < rsi <= 40:
        score += 4
    elif rsi <= 30:
        score += 5   # 超卖反弹机会
    else:
        score += 3

    # 3. KDJ_K (5 分)
    if len(close) >= 9:
        low9 = data['最低'].astype(float).rolling(9).min()
        high9 = data['最高'].astype(float).rolling(9).max()
        rsv = (close - low9) / (high9 - low9) * 100
        rsv = rsv.fillna(50)
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3 * k - 2 * d
        k_val = k.iloc[-1]
        if k_val > d.iloc[-1] and k_val < 80:
            score += 5  # KDJ 金叉且未超买
        elif k_val > d.iloc[-1] and k_val >= 80:
            score += 2
        elif k_val < d.iloc[-1] and k_val > 20:
            score += 2
        else:
            score += 3  # KDJ 死叉低位（超卖）

    return min(score, 25)


def score_volume(data):
    """
    量能得分（0-15）：5日量比 + 量价配合
    """
    vol = data['成交量'].astype(float)
    close = data['收盘'].astype(float)
    if len(close) < 30:
        return None

    score = 0

    # 1. 5日量比 (5 分)
    vol_5 = vol.tail(5).mean()
    vol_30 = vol.tail(30).mean()
    vol_ratio = vol_5 / vol_30 if vol_30 > 0 else 1.0
    if 1.2 < vol_ratio < 2.5:
        score += 5  # 健康放量
    elif vol_ratio >= 2.5:
        score += 3  # 巨量（可能是出货）
    elif vol_ratio >= 0.8:
        score += 4
    elif vol_ratio >= 0.5:
        score += 2  # 缩量
    else:
        score += 0  # 极度缩量

    # 2. 量价配合 (10 分)
    price_change_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100
    if price_change_5d > 2 and vol_ratio > 1.0:
        score += 10  # 上涨 + 放量 (最佳)
    elif price_change_5d > 0 and vol_ratio >= 0.8:
        score += 7   # 上涨 + 量平
    elif -2 < price_change_5d <= 2 and 0.5 < vol_ratio < 1.5:
        score += 5   # 横盘 + 量平
    elif price_change_5d <= -2 and vol_ratio < 1.0:
        score += 3   # 下跌 + 缩量（可能是洗盘）
    elif price_change_5d <= -2 and vol_ratio >= 1.0:
        score += 0   # 下跌 + 放量（出货信号）
    else:
        score += 4

    return min(score, 15)


def score_risk(data):
    """
    风险得分（0-20）：BB 位置 + ATR% + 20日波动率
    越低风险越高，得分越低
    """
    close = data['收盘'].astype(float)
    if len(close) < 20:
        return None

    bb_score = 0
    atr_score = 0
    vol_score = 0

    # 1. BB 位置 (5 分) — 越靠近 0.5 越安全
    if len(close) >= 20:
        ma20 = close.rolling(20).mean().iloc[-1]
        std20 = close.rolling(20).std().iloc[-1]
        bb_pos = (close.iloc[-1] - ma20) / (2 * std20) if std20 > 0 else 0.5
        bb_pos = max(0, min(1, bb_pos + 0.5))  # 映射到 0-1
        bb_score = max(0.0, min(5.0, 5 - abs(bb_pos - 0.5) * 10))

    # 2. ATR% (5 分) — 越低越好
    if len(data) >= 14:
        high = data['最高'].astype(float)
        low = data['最低'].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        atr_pct = atr / close.iloc[-1] * 100
        if atr_pct < 2:
            atr_score = 5
        elif atr_pct < 3:
            atr_score = 4
        elif atr_pct < 4:
            atr_score = 3
        elif atr_pct < 5:
            atr_score = 2
        else:
            atr_score = 0

    # 3. 20日波动率 (10 分)
    if len(close) >= 21:
        returns = close.pct_change().tail(20)
        vol = returns.std() * np.sqrt(252) * 100  # 年化波动率
        if vol < 20:
            vol_score = 10
        elif vol < 30:
            vol_score = 8
        elif vol < 40:
            vol_score = 6
        elif vol < 50:
            vol_score = 4
        else:
            vol_score = 2

    return bb_score + atr_score + vol_score


def score_stock(data):
    """综合评分一只股票"""
    if len(data) < 60:
        return None

    trend = score_trend(data)
    momentum = score_momentum(data)
    volume = score_volume(data)
    risk = score_risk(data)

    if any(s is None for s in [trend, momentum, volume, risk]):
        return None

    total = trend + momentum + volume + risk

    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest

    close = float(latest['收盘'])
    prev_close = float(prev['收盘'])

    return {
        'price': close,
        'change_1d': (close / prev_close - 1) * 100 if prev_close > 0 else 0,
        'change_5d': (close / data['收盘'].iloc[-6] - 1) * 100 if len(data) >= 6 else 0,
        'change_20d': (close / data['收盘'].iloc[-21] - 1) * 100 if len(data) >= 21 else 0,
        'score_total': total,
        'score_trend': trend,
        'score_momentum': momentum,
        'score_volume': volume,
        'score_risk': risk,
    }


def action_label(score):
    """根据总分给操作建议"""
    if score >= 75:
        return '🟢 强烈推荐'
    elif score >= 60:
        return '🟢 推荐'
    elif score >= 45:
        return '🟡 观望'
    elif score >= 30:
        return '🟠 谨慎'
    else:
        return '🔴 回避'


def main():
    parser = argparse.ArgumentParser(description='综合评分选股')
    parser.add_argument('--top', type=int, default=20, help='推荐 TOP N (默认 20)')
    parser.add_argument('--min-score', type=float, default=0, help='最低分过滤 (默认 0)')
    parser.add_argument('--sector-filter', default='', help='代码前缀过滤，如 600/688/000')
    parser.add_argument('--output', default=None, help='CSV 输出文件')
    parser.add_argument('--data-dir', default='stock_data')
    parser.add_argument('--max-stocks', type=int, default=0,
                        help='最大扫描股票数 (0=全量 ~1832)，调试用')
    args = parser.parse_args()

    mgr = HistoricalDataManager(args.data_dir)

    print('[INFO] 加载股票池...')
    universe = mgr.get_universe()
    if args.sector_filter:
        universe = [s for s in universe if s.startswith(args.sector_filter)]
    if args.max_stocks > 0:
        universe = universe[:args.max_stocks]
    print(f'[OK] 待扫描: {len(universe)} 只')

    print('[INFO] 开始评分...')
    results = []
    for i, sym in enumerate(universe):
        if (i + 1) % 100 == 0:
            print(f'  进度: {i + 1}/{len(universe)}')
        try:
            data = mgr.load(sym)
            if len(data) < 60:
                continue
            score = score_stock(data)
            if score is None:
                continue
            if score['score_total'] < args.min_score:
                continue
            results.append({
                'symbol': sym,
                'name': mgr.get_stock_name(sym),
                **score,
            })
        except Exception as e:
            continue

    # 排序
    results.sort(key=lambda r: r['score_total'], reverse=True)
    top_n = results[:args.top]

    # 输出
    print()
    print('=' * 72)
    print(f'   综合评分选股 TOP {len(top_n)} ({datetime.now().strftime("%Y-%m-%d")})')
    print('=' * 72)
    print(f'扫描范围: {len(universe)} 只 | 入选: {len(results)} 只 | TOP {args.top}')
    print()

    # 按分数段统计
    from collections import Counter
    dist = Counter()
    for r in results:
        if r['score_total'] >= 75: dist['>=75 强烈推荐'] += 1
        elif r['score_total'] >= 60: dist['60-74 推荐'] += 1
        elif r['score_total'] >= 45: dist['45-59 观望'] += 1
        elif r['score_total'] >= 30: dist['30-44 谨慎'] += 1
        else: dist['<30 回避'] += 1
    print('【全市场分布】')
    for k in ['>=75 强烈推荐', '60-74 推荐', '45-59 观望', '30-44 谨慎', '<30 回避']:
        if dist[k] > 0:
            print(f'  {k}: {dist[k]:>4} 只 ({dist[k]/len(results)*100:.1f}%)')
    print()

    print(f'【TOP {len(top_n)} 详细评分】')
    print('-' * 100)
    header = f'{"代码":<8} {"名称":<10} {"价格":>7} {"1日":>6} {"5日":>6} {"20日":>6} ' \
             f'{"总分":>5} {"趋势":>4} {"动量":>4} {"量能":>4} {"风险":>4} {"建议"}'
    print(header)
    print('-' * 100)

    for r in top_n:
        action = action_label(r['score_total'])
        print(f'{r["symbol"]:<8} {r["name"]:<10} {r["price"]:>7.2f} '
              f'{r["change_1d"]:>+5.2f}% {r["change_5d"]:>+5.2f}% {r["change_20d"]:>+5.2f}% '
              f'{r["score_total"]:>5.1f} {r["score_trend"]:>4} {r["score_momentum"]:>4} '
              f'{r["score_volume"]:>4} {r["score_risk"]:>4} {action}')

    print('-' * 100)

    # CSV 输出
    if args.output:
        df = pd.DataFrame(top_n)
        df['action'] = df['score_total'].apply(action_label)
        df.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f'[OK] CSV 已保存: {args.output}')

    return top_n


if __name__ == '__main__':
    main()