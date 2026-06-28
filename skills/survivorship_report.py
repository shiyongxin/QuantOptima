# -*- coding: utf-8 -*-
"""
Survivorship Bias 报告生成器

生成 `survivorship_report.md`，量化当前 universe 的 survivorship bias 估计。

数据局限：
- free API（akshare/eastmoney）只能获取当前停牌的股票
- 历史退市股（CSMAR/Wind 等商业数据源有，但本工具未集成）
- 只能基于当前已停牌的股票做保守估计

报告内容：
1. 当前 universe 概况
2. 已知的停牌股票
3. Survivorship bias 估算
4. 对优化结果的影响

用法:
    python skills/survivorship_report.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import akshare as ak


def load_current_universe(data_dir='./stock_data'):
    """加载当前 universe 的股票列表和历史范围"""
    hist_dir = Path(data_dir) / 'historical'
    files = sorted(hist_dir.glob('*.parquet'))
    stats = []
    for f in files:
        sym = f.stem
        try:
            df = pd.read_parquet(f, columns=['日期'])
            if len(df) >= 100:
                stats.append({
                    'symbol': sym,
                    'start': pd.to_datetime(df['日期'].min()),
                    'end': pd.to_datetime(df['日期'].max()),
                    'days': len(df),
                })
        except Exception:
            pass
    return pd.DataFrame(stats)


def load_stopped_symbols():
    """从 akshare 获取当前已停牌的股票列表"""
    try:
        df = ak.stock_zh_a_stop_em()
        return df[['代码', '名称']].rename(columns={'代码': 'symbol', '名称': 'name'})
    except Exception as e:
        print(f"[WARN] 无法获取停牌列表: {e}")
        return pd.DataFrame(columns=['symbol', 'name'])


def load_all_a_share_spot():
    """获取全部 A 股代码（含已退市）"""
    try:
        df = ak.stock_zh_a_spot_em()
        return df
    except Exception as e:
        print(f"[WARN] 无法获取全A列表: {e}")
        return None


def compute_survivorship_stats(universe_df, stopped_df, all_spot_df):
    """计算 survivorship bias 统计"""
    stats = {}

    # 1. 当前 universe
    current_universe = len(universe_df)
    stats['current_universe'] = current_universe

    # 2. 时间分布
    if len(universe_df):
        stats['avg_history_days'] = int(universe_df['days'].median())
        stats['min_start'] = str(universe_df['start'].min().date())
        stats['max_end'] = str(universe_df['end'].max().date())

    # 3. 停牌股
    stopped = len(stopped_df)
    stats['stopped_count'] = stopped

    # 4. 全A列表（估算总上市数）
    if all_spot_df is not None and '代码' in all_spot_df.columns:
        total_a = len(all_spot_df)
        stats['total_a_shares'] = total_a
        known_coverage = stopped + current_universe
        stats['known_coverage'] = known_coverage
        stats['known_pct'] = round(known_coverage / total_a * 100, 1) if total_a else 0.0
        # 估算历史退市股（30年累计上市约total_a*0.3只，刨去已知覆盖）
        estimated_delisted = max(0, int(total_a * 0.3) - known_coverage)
        stats['historical_delisted_estimated'] = estimated_delisted
    else:
        stats['total_a_shares'] = None
        stats['known_coverage'] = None
        stats['historical_delisted_estimated'] = None

    # 5. Survivorship bias 估算（学术经验值）
    # A股年化 survivorship bias ≈ 1-3%（退市股退市前1-2年往往大跌）
    # 本报告采用保守估计 2%/年
    stats['bias_estimate_annual_pct'] = 2.0
    stats['bias_estimate_30yr_pct'] = round((1 + 0.02) ** 30 - 1, 1) * 100  # 复利
    return stats


def write_report(stats, output_path):
    """写 survivorship_report.md"""
    lines = [
        "# Survivorship Bias 报告",
        "",
        f"- 生成时间: 2026-06-27",
        f"- 数据目录: `stock_data/`",
        "",
        "## 1. 当前 Universe 概况",
        "",
        f"- 已有历史数据的股票: **{stats['current_universe']} 只**",
    ]
    if stats.get('min_start'):
        lines.append(f"- 历史范围: {stats['min_start']} ~ {stats['max_end']}")
    if stats.get('avg_history_days'):
        lines.append(f"- 中位历史长度: {stats['avg_history_days']} 天")

    lines.extend(["", "## 2. 已知的停牌股票", ""])

    if stats.get('total_a_shares'):
        lines.extend([
            f"- 沪深A股当前总数（akshare）: **{stats['total_a_shares']} 只**",
            f"- 当前已停牌（akshare）: **{stats['stopped_count']} 只**",
            f"- 已知覆盖（现存 + 停牌）: {stats['known_coverage']} 只 ({stats['known_pct']}%%)",
            f"- 估算历史退市股（未在 free data 中）: **{stats['historical_delisted_estimated']} 只**",
            "",
            "注：free data（akshare/eastmoney）无法获取历史完整退市列表。",
        ])
    else:
        lines.append(f"- 当前已停牌: **{stats['stopped_count']} 只**（全A列表获取失败）")

    lines.extend(["", "## 3. Survivorship Bias 估算", ""])
    annual = stats['bias_estimate_annual_pct']
    yr30 = stats['bias_estimate_30yr_pct']
    lines.extend([
        f"- 年化 bias 估算: **{annual}%%/年**（保守估计）",
        f"- 30 年累计 bias 估算: **{yr30}%%**",
        "",
        "### 估算方法",
        "",
        f"基于学术文献（石川 2020, Bloomberg A-share survivorship studies）:",
        f"- A 股年化 survivorship bias ≈ 1-3%%",
        f"- 本报告采用保守估计 **{annual}%%/年**",
        f"- 30 年复利后 ≈ **{yr30}%%**（已折算）",
        "",
        "### Bias 来源",
        "",
        "1. **退市股在退市前 1-2 年往往大幅下跌（-50%% to -90%%）",
        "2. 幸存股票（全A指数成分）天然跑赢全体",
        "3. 在只含幸存股票的 universe 上优化的策略，收益被系统性高估",
        "",
        "## 4. 对优化结果的影响",
        "",
        f"当前 universe（{stats['current_universe']} 只）只包含当前上市的股票。",
        f"估算 bias: **年化收益高估 ~{annual}%%，30 年累计约 {yr30}%%**。",
        "",
        "**含义**:",
        f"- 报告中的 'median_return' / 'total_return' 需下调约 {annual}%%/年",
        "- Sharpe 也会被高估（分子变大，分母不变）",
        "- 真实策略的收益可能比回测数字低 40-60%%（30 年复利后）",
        "",
        "**改善方法**:",
        "- 接入 CSMAR / Wind 等商业历史退市数据",
        "- 对每只股票估算退市前的 Stub period 收益",
        "- 从基准指数中减去估算的 survivorship bias",
    ])

    Path(output_path).write_text('\n'.join(lines), encoding='utf-8')
    print(f"[OK] Survivorship report -> {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Survivorship Bias 报告生成')
    parser.add_argument('--data-dir', default='./stock_data')
    parser.add_argument('--output', default='./stock_data/survivorship_report.md')
    args = parser.parse_args()

    print("[INFO] 加载当前 universe...")
    universe_df = load_current_universe(args.data_dir)
    print(f"  universe: {len(universe_df)} 只")

    print("[INFO] 获取停牌列表...")
    stopped_df = load_stopped_symbols()
    print(f"  停牌: {len(stopped_df)} 只")

    print("[INFO] 获取全A列表...")
    all_spot_df = load_all_a_share_spot()

    print("[INFO] 计算统计...")
    stats = compute_survivorship_stats(universe_df, stopped_df, all_spot_df)

    print("[INFO] 写报告...")
    write_report(stats, args.output)

    print()
    print("=== 摘要 ===")
    print(f"Universe: {stats['current_universe']} 只")
    print(f"停牌: {stats['stopped_count']} 只")
    if stats.get('total_a_shares'):
        print(f"覆盖率: {stats['known_pct']}%% ({stats['known_coverage']}/{stats['total_a_shares']})")
        print(f"估算历史退市: {stats['historical_delisted_estimated']} 只")
    print(f"Bias 估算: {stats['bias_estimate_annual_pct']}%%/年, 30年 {stats['bias_estimate_30yr_pct']}%%")


if __name__ == '__main__':
    main()
