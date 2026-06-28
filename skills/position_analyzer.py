# -*- coding: utf-8 -*-
"""
持仓分析脚本

读取 stock_data/持仓.csv，对每只持仓股：
1. 计算当前盈亏%（最新收盘 vs 成本价）
2. 跑当前大盘体制下的指标体系，生成 BUY/HOLD/SELL 信号
3. 对比止损/止盈价，给出操作建议

用法：
    python skills/position_analyzer.py                    # 分析所有持仓
    python skills/position_analyzer.py --output report.txt  # 保存报告
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd

from historical_data_manager import HistoricalDataManager
from daily_signal_generator import DailySignalGenerator


def analyze_position(generator, mgr, row, regime):
    """分析单只持仓股票 — 忽略盈亏，只看技术面趋势"""
    symbol = str(row['股票代码']).zfill(6)
    name = row.get('股票名称', mgr.get_stock_name(symbol))
    shares = float(row['持股数量'])
    cost = float(row['成本价'])
    tp1 = float(row.get('止盈价1', 0))
    tp2 = float(row.get('止盈价2', 0))
    stop_loss = float(row.get('止损价', 0))
    status = row.get('状态', '')

    # 加载最新数据
    data = mgr.load(symbol)
    if len(data) < 60:
        return {
            'symbol': symbol, 'name': name, 'error': f'数据不足 ({len(data)} 行)',
        }

    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest
    close = float(latest['收盘'])
    prev_close = float(prev['收盘'])
    market_value = close * shares

    # ========== 近期趋势分析（30 天视角）==========
    last_30 = data.tail(30)
    last_5 = data.tail(5)
    last_60 = data.tail(60)

    # 多周期均线
    close_series = data['收盘'].astype(float)
    ma5 = close_series.rolling(5).mean().iloc[-1]
    ma10 = close_series.rolling(10).mean().iloc[-1]
    ma20 = close_series.rolling(20).mean().iloc[-1]
    ma60 = close_series.rolling(60).mean().iloc[-1] if len(close_series) >= 60 else ma20

    # 趋势状态（基于均线排列）
    if ma5 > ma10 > ma20 > ma60:
        trend = '强势上升'
        trend_icon = '🚀'
    elif ma5 > ma10 > ma20:
        trend = '上升'
        trend_icon = '📈'
    elif ma5 < ma10 < ma20 < ma60:
        trend = '强势下跌'
        trend_icon = '💀'
    elif ma5 < ma10 < ma20:
        trend = '下跌'
        trend_icon = '📉'
    else:
        trend = '震荡'
        trend_icon = '〰️'

    # 近期涨跌幅
    change_5d = (close / last_5['收盘'].iloc[0] - 1) * 100
    change_20d = (close / last_30['收盘'].iloc[0] - 1) * 100

    # 成交量趋势
    vol_recent_5 = last_5['成交量'].astype(float).mean()
    vol_baseline = last_60['成交量'].astype(float).mean() if len(data) >= 60 else vol_recent_5
    vol_ratio = vol_recent_5 / vol_baseline if vol_baseline > 0 else 1.0

    # ========== 指标体系信号 ==========
    try:
        params = generator.get_params_for_regime(regime)
        sig_result = generator.generate_for_stock(symbol, regime, params)
        signal = sig_result.signal if sig_result else 'N/A'
        sig_strength = sig_result.signal_strength if sig_result else 0
        reasons = sig_result.reasons if sig_result else []
        ma_trend = sig_result.ma_trend if sig_result else None
        macd_hist = sig_result.macd_hist if sig_result else None
        rsi = sig_result.rsi if sig_result else None
        kdj_k = sig_result.kdj_k if sig_result else None
        bb_pos = sig_result.bb_position if sig_result else None
        atr_pct = sig_result.atr_pct if sig_result else None
    except Exception as e:
        signal = 'ERR'
        sig_strength = 0
        reasons = [f'信号生成失败: {e}']
        ma_trend = macd_hist = rsi = kdj_k = bb_pos = atr_pct = None

    # ========== 操作策略（基于趋势，忽略盈亏）==========
    strategy_reasons = []

    # 1. 信号驱动
    if signal == 'BUY' and sig_strength > 0.5:
        strategy = '🟢 加仓'
        strategy_reasons.append(f'指标体系给出 BUY 信号 (强度 {sig_strength:+.2f})')
    elif signal == 'SELL' and abs(sig_strength) > 0.5:
        strategy = '🔴 减仓/清仓'
        strategy_reasons.append(f'指标体系给出 SELL 信号 (强度 {sig_strength:+.2f})')
    # 2. 趋势驱动
    elif '强势上升' in trend:
        if vol_ratio > 1.3:
            strategy = '🟢 持有/加仓'
            strategy_reasons.append(f'强势上升 + 放量 ({vol_ratio:.2f}x)')
        else:
            strategy = '🟢 持有'
            strategy_reasons.append(f'强势上升，量能正常 ({vol_ratio:.2f}x)')
    elif '强势下跌' in trend:
        strategy = '🔴 减仓'
        strategy_reasons.append('均线空头排列（强势下跌），建议降低敞口')
    elif '上升' in trend:
        if vol_ratio > 1.5:
            strategy = '🟢 持有/加仓'
            strategy_reasons.append(f'上升趋势 + 放量 ({vol_ratio:.2f}x)')
        else:
            strategy = '🟢 持有'
            strategy_reasons.append('短期均线多头排列')
    elif '下跌' in trend:
        if vol_ratio > 1.5:
            strategy = '🔴 减仓'
            strategy_reasons.append(f'下跌趋势 + 放量 ({vol_ratio:.2f}x)，加速下跌')
        else:
            strategy = '🟡 观望'
            strategy_reasons.append('短期均线空头排列但缩量，等待反弹信号')
    else:  # 震荡
        if signal == 'BUY':
            strategy = '🟢 持有'
            strategy_reasons.append('震荡市 + BUY 信号，逢低吸纳')
        elif signal == 'SELL':
            strategy = '🔴 减仓'
            strategy_reasons.append('震荡市 + SELL 信号，高位减仓')
        else:
            strategy = '🟡 观望'
            strategy_reasons.append('震荡市无明确信号')

    # 3. 特殊触发（不基于盈亏，基于价位 vs 止盈止损位）
    if tp2 and close >= tp2:
        strategy_reasons.append(f'现价 {close:.2f} ≥ 止盈2 {tp2}，可考虑分批兑现')
    if stop_loss and close <= stop_loss:
        strategy_reasons.append(f'现价 {close:.2f} ≤ 止损 {stop_loss}，技术面支持止损')

    # 4. RSI/KDJ 极端值
    if rsi is not None:
        if rsi >= 75:
            strategy_reasons.append(f'RSI={rsi:.1f} 超买区域')
        elif rsi <= 25:
            strategy_reasons.append(f'RSI={rsi:.1f} 超卖区域')

    return {
        'symbol': symbol,
        'name': name,
        'shares': shares,
        'cost': cost,  # 保留以备后用
        'price': close,
        'change_5d': change_5d,
        'change_20d': change_20d,
        'market_value': market_value,
        'trend': trend,
        'trend_icon': trend_icon,
        'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma60': ma60,
        'vol_ratio': vol_ratio,
        'signal': signal,
        'sig_strength': sig_strength,
        'ma_trend': ma_trend,
        'macd_hist': macd_hist,
        'rsi': rsi,
        'kdj_k': kdj_k,
        'bb_pos': bb_pos,
        'atr_pct': atr_pct,
        'strategy': strategy,
        'strategy_reasons': strategy_reasons,
        'system_reasons': reasons,
        'status': status,
    }


def format_report(results, regime, regime_cn, system_name, summary):
    """格式化为可读报告 — 基于技术面趋势，忽略盈亏"""
    lines = []
    lines.append('=' * 72)
    lines.append(f'   持仓策略分析 (基于近期技术面趋势) — {datetime.now().strftime("%Y-%m-%d")}')
    lines.append('=' * 72)
    lines.append('')
    lines.append(f'当前大盘: {regime_cn} ({regime})')
    lines.append(f'使用体系: {system_name}')
    lines.append(f'持仓股票: {len(results)} 只')
    lines.append(f'持仓总市值: ¥{summary["total_market"]:>12,.2f}')
    lines.append('')
    lines.append('⚠️  本报告基于近期 (5d/20d/60d) 趋势，忽略盈亏状态')
    lines.append('=' * 72)
    lines.append('')

    # 按操作策略分组
    groups = {
        '🟢 加仓/持有': [],
        '🟡 观望': [],
        '🔴 减仓/止损': [],
        '❌ 错误': [],
    }
    for r in results:
        if 'error' in r:
            groups['❌ 错误'].append(r)
        elif '🟢' in r.get('strategy', ''):
            groups['🟢 加仓/持有'].append(r)
        elif '🟡' in r.get('strategy', ''):
            groups['🟡 观望'].append(r)
        elif '🔴' in r.get('strategy', ''):
            groups['🔴 减仓/止损'].append(r)
        else:
            groups['🟡 观望'].append(r)

    for group_name, items in groups.items():
        if not items:
            continue
        lines.append('')
        lines.append(f'【{group_name}】 {len(items)} 只')
        lines.append('-' * 72)

        for r in items:
            if 'error' in r:
                lines.append(f'  ❌ {r["symbol"]} {r["name"]}: {r["error"]}')
                continue

            lines.append(f'  {r["symbol"]} {r["name"]}  {r["trend_icon"]} {r["trend"]}  '
                         f'→ {r["strategy"]}  [{r["status"]}]')
            lines.append(f'    现价: {r["price"]:.2f}  '
                         f'5日: {r["change_5d"]:+.2f}%  '
                         f'20日: {r["change_20d"]:+.2f}%  '
                         f'量比: {r["vol_ratio"]:.2f}x')

            # 均线
            lines.append(f'    均线: MA5={r["ma5"]:.2f}  MA10={r["ma10"]:.2f}  '
                         f'MA20={r["ma20"]:.2f}  MA60={r["ma60"]:.2f}')

            # 指标
            inds = []
            if r['rsi'] is not None: inds.append(f'RSI={r["rsi"]:.1f}')
            if r['macd_hist'] is not None: inds.append(f'MACD柱={r["macd_hist"]:+.3f}')
            if r['kdj_k'] is not None: inds.append(f'KDJ_K={r["kdj_k"]:.1f}')
            if r['bb_pos'] is not None: inds.append(f'BB位={r["bb_pos"]:.2f}')
            if r['atr_pct'] is not None: inds.append(f'ATR={r["atr_pct"]:.2f}%')
            if inds:
                lines.append(f'    指标: {"  ".join(inds)}')

            # 策略原因
            for reason in r['strategy_reasons']:
                lines.append(f'    ▸ {reason}')

    lines.append('')
    lines.append('=' * 72)
    lines.append(f'生成时间: {datetime.now().isoformat()}')
    lines.append('=' * 72)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='持仓分析')
    parser.add_argument('--positions', default='stock_data/持仓.csv',
                        help='持仓文件路径')
    parser.add_argument('--systems-file', default='stock_data/optimization_result.json',
                        help='优化结果文件')
    parser.add_argument('--regime-labels', default='stock_data/regime_labels.csv',
                        help='体制标签文件')
    parser.add_argument('--data-dir', default='stock_data',
                        help='数据目录')
    parser.add_argument('--output', default=None,
                        help='输出报告文件路径（默认打印到 stdout）')
    args = parser.parse_args()

    # 加载持仓
    positions_path = Path(args.positions)
    if not positions_path.exists():
        print(f'[ERROR] 持仓文件不存在: {positions_path}')
        sys.exit(1)
    positions_df = pd.read_csv(positions_path, encoding='utf-8-sig')
    print(f'[OK] 加载 {len(positions_df)} 条持仓')

    # 初始化生成器（自动检测体制）
    mgr = HistoricalDataManager(args.data_dir)
    generator = DailySignalGenerator(
        data_dir=args.data_dir,
        optimization_result_file=args.systems_file,
    )
    if Path(args.regime_labels).exists():
        generator.load_regime_labels(args.regime_labels)

    regime = generator.detect_current_regime()
    regime_cn = {'BULL': '牛市', 'BEAR': '熊市', 'SIDEWAYS': '横盘震荡',
                 'CRASH': '暴跌', 'RECOVERY': '反弹'}.get(regime, regime)
    system_name = generator.get_system_name_for_regime(regime)
    print(f'[INFO] 大盘: {regime_cn} ({regime}), 体系: {system_name}')

    # 分析每只持仓
    results = []
    for _, row in positions_df.iterrows():
        try:
            r = analyze_position(generator, mgr, row, regime)
        except Exception as e:
            r = {'symbol': str(row.get('股票代码', '?')), 'name': '?',
                 'error': f'分析失败: {e}'}
        results.append(r)

    # 汇总（基于技术面，不算盈亏）
    valid = [r for r in results if 'error' not in r]
    summary = {
        'total_market': sum(r['market_value'] for r in valid),
    }

    # 输出报告
    report = format_report(results, regime, regime_cn, system_name, summary)
    if args.output:
        Path(args.output).write_text(report, encoding='utf-8')
        print(f'[OK] 报告已保存: {args.output}')
    else:
        print()
        print(report)


if __name__ == '__main__':
    main()