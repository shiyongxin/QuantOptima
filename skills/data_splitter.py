# -*- coding: utf-8 -*-
"""
数据切分器 - Data Splitter

按大盘日历把指数体制标签和股票数据切成 train / val / test 三段，
验证每个 bucket 的 regime 覆盖度，生成 splits/{train,val,test}_symbols.txt
和 regime_labels_{train,val,test}.csv。

设计原则（见 FIX_PLAN.md Phase 0）：
- 大盘日历：所有股票共享同一时间轴
- 体制标签同时间窗生成：防跨边界泄漏
- 阈值校验：每个 bucket 至少含 N 天 BULL/BEAR/SIDEWAYS/CRASH/RECOVERY
- 原始数据保留：historical/*.parquet 不动

用法:
    from data_splitter import DataSplitter
    splitter = DataSplitter(data_dir='./stock_data')
    report = splitter.run()
    print(report)
"""
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd
import glob
import json


@dataclass
class BucketSpec:
    """一个时间桶的规格"""
    name: str                # 'train' / 'val' / 'test'
    start: str               # 'YYYY-MM-DD'
    end: str                 # 'YYYY-MM-DD'


@dataclass
class RegimeThresholds:
    """每个 regime 在每个 bucket 里必须达到的最小天数"""
    BULL: int = 300
    BEAR: int = 300
    SIDEWAYS: int = 500
    CRASH: int = 60
    RECOVERY: int = 60

    def get(self, regime: str) -> int:
        return getattr(self, regime, 0)


class DataSplitter:
    """
    数据切分器

    把指数体制标签按时间切三段，扫描 stock_data/historical/*.parquet 找出
    每个 bucket 里"有足够数据"的股票，写出 splits/*.txt 和 regime_labels_*.csv。
    """

    DEFAULT_BUCKETS = [
        BucketSpec('train', '2002-01-01', '2014-12-31'),  # 数据从 2002 起
        BucketSpec('val',   '2015-01-01', '2019-12-31'),
        BucketSpec('test',  '2020-01-01', '2026-12-31'),
    ]

    def __init__(self,
                 data_dir: str = './stock_data',
                 buckets: list[BucketSpec] = None,
                 thresholds: RegimeThresholds = None,
                 min_days_in_bucket: int = 180,
                 min_rows: int = 100):
        """
        Parameters
        ----------
        data_dir : str
            stock_data 根目录（含 historical/ 和 regime_labels.csv）
        buckets : list[BucketSpec] or None
            时间桶列表；None 用默认
        thresholds : RegimeThresholds or None
            体制覆盖度阈值；None 用默认
        min_days_in_bucket : int
            股票在该 bucket 内的最少有效天数（防 1 只股票贡献 5 天）
        min_rows : int
            股票 parquet 文件最少行数（防空/损坏文件）
        """
        self.data_dir = Path(data_dir)
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self.thresholds = thresholds or RegimeThresholds()
        self.min_days = min_days_in_bucket
        self.min_rows = min_rows

        self.historical_dir = self.data_dir / 'historical'
        self.regime_file = self.data_dir / 'regime_labels.csv'
        self.splits_dir = self.data_dir / 'splits'
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        """
        主入口：跑完整流程，返回 report dict
        """
        # 1. 加载体制标签
        regimes = self._load_regimes()

        # 2. 扫描所有股票的数据范围
        stock_ranges = self._scan_stock_ranges()

        # 3. 对每个 bucket 切分 regime 标签 + 选股
        report = {
            'buckets': {},
            'thresholds': self.thresholds.__dict__,
            'data_dir': str(self.data_dir),
            'regime_total_days': int(len(regimes)),
            'regime_date_range': [str(regimes['日期'].min().date()),
                                  str(regimes['日期'].max().date())],
        }

        for bucket in self.buckets:
            bucket_report = self._process_bucket(bucket, regimes, stock_ranges)
            report['buckets'][bucket.name] = bucket_report

        # 4. 写总报告
        self._write_report(report)
        return report

    def _load_regimes(self) -> pd.DataFrame:
        """加载 regime_labels.csv"""
        if not self.regime_file.exists():
            raise FileNotFoundError(
                f"未找到 {self.regime_file}。先跑 fetch_data.py 生成。"
            )
        df = pd.read_csv(self.regime_file, encoding='utf-8-sig')
        df['日期'] = pd.to_datetime(df['日期'])
        return df.sort_values('日期').reset_index(drop=True)

    def _scan_stock_ranges(self) -> dict[str, tuple]:
        """
        扫描 historical/ 下所有 parquet，返回 {symbol: (start_date, end_date, row_count)}

        读取只用 columns=['日期'] 加速。损坏文件跳过并记录。
        """
        ranges = {}
        skipped = []
        files = sorted(glob.glob(str(self.historical_dir / '*.parquet')))
        for f in files:
            symbol = Path(f).stem
            try:
                df = pd.read_parquet(f, columns=['日期'])
                if len(df) < self.min_rows:
                    skipped.append((symbol, f'rows={len(df)} < {self.min_rows}'))
                    continue
                df['日期'] = pd.to_datetime(df['日期'])
                ranges[symbol] = (df['日期'].min(), df['日期'].max(), len(df))
            except Exception as e:
                skipped.append((symbol, str(e)[:60]))

        if skipped:
            print(f"[WARN] 跳过 {len(skipped)} 个有问题的 parquet 文件")
            for sym, reason in skipped[:5]:
                print(f"  - {sym}: {reason}")
            if len(skipped) > 5:
                print(f"  ... 还有 {len(skipped) - 5} 个")

        return ranges

    def _process_bucket(self,
                        bucket: BucketSpec,
                        regimes: pd.DataFrame,
                        stock_ranges: dict) -> dict:
        """
        处理单个 bucket：切 regime 标签、选股、写文件、校验
        """
        s_dt = pd.Timestamp(bucket.start)
        e_dt = pd.Timestamp(bucket.end)

        # 切 regime 标签到该 bucket
        mask = (regimes['日期'] >= s_dt) & (regimes['日期'] <= e_dt)
        bucket_regimes = regimes[mask].reset_index(drop=True)

        # 计算每个 regime 的天数
        regime_days = {}
        for r in ['BULL', 'BEAR', 'SIDEWAYS', 'CRASH', 'RECOVERY']:
            regime_days[r] = int((bucket_regimes['regime'] == r).sum())

        # 选股：股票在 bucket 内至少有 min_days 有效天数
        selected = []
        for symbol, (s, e, n) in stock_ranges.items():
            # 股票与 bucket 时间轴求交
            overlap_start = max(s, s_dt)
            overlap_end = min(e, e_dt)
            if overlap_end < overlap_start:
                continue
            overlap_days = (overlap_end - overlap_start).days  # 粗估日历日
            # 用交易日估算更准：日历日 × 252/365
            overlap_trading_days = int(overlap_days * 252 / 365)
            if overlap_trading_days >= self.min_days:
                selected.append(symbol)

        selected.sort()

        # 写文件
        symbols_path = self.splits_dir / f'{bucket.name}_symbols.txt'
        labels_path = self.data_dir / f'regime_labels_{bucket.name}.csv'

        with open(symbols_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(selected) + '\n')
        bucket_regimes.to_csv(labels_path, index=False, encoding='utf-8-sig')

        # 阈值校验
        threshold_status = {}
        for r, days in regime_days.items():
            threshold = self.thresholds.get(r)
            ok = days >= threshold
            threshold_status[r] = {
                'days': days,
                'threshold': threshold,
                'pass': ok,
            }

        return {
            'start': bucket.start,
            'end': bucket.end,
            'regime_total_days': int(len(bucket_regimes)),
            'regime_days': regime_days,
            'threshold_status': threshold_status,
            'all_pass': all(s['pass'] for s in threshold_status.values()),
            'symbols_count': len(selected),
            'symbols_path': str(symbols_path),
            'labels_path': str(labels_path),
        }

    def _write_report(self, report: dict):
        """写 coverage_report.md + 控制台摘要"""
        md_lines = [
            '# 数据切分覆盖度报告',
            '',
            f'- 数据目录: `{report["data_dir"]}`',
            f'- 体制标签区间: {report["regime_date_range"][0]} ~ {report["regime_date_range"][1]}',
            f'- 体制标签总天数: {report["regime_total_days"]}',
            '',
            '## 阈值（每个 bucket 至少达到的天数）',
            '',
            '| 体制 | 阈值 |',
            '|---|---|',
        ]
        for r, t in report['thresholds'].items():
            md_lines.append(f'| {r} | {t} |')

        for bucket_name, br in report['buckets'].items():
            md_lines.extend([
                '',
                f'## {bucket_name}: {br["start"]} ~ {br["end"]}',
                '',
                f'- 区间交易日: {br["regime_total_days"]}',
                f'- 入选股票数: {br["symbols_count"]}',
                f'- 阈值校验: {"✅ 全通过" if br["all_pass"] else "❌ 有不达标"}',
                '',
                '| 体制 | 天数 | 阈值 | 通过 |',
                '|---|---|---|---|',
            ])
            for r, st in br['threshold_status'].items():
                mark = '✅' if st['pass'] else '❌'
                md_lines.append(
                    f'| {r} | {st["days"]} | {st["threshold"]} | {mark} |'
                )

            md_lines.extend([
                '',
                f'- 股票清单: `{br["symbols_path"]}`',
                f'- 体制标签: `{br["labels_path"]}`',
            ])

        # 总体结论
        any_fail = any(not br['all_pass'] for br in report['buckets'].values())
        md_lines.extend([
            '',
            '## 总体',
            '',
        ])
        if any_fail:
            md_lines.append(
                '⚠️ **存在阈值不达标的 bucket**。建议：'
                '1) 降低 CRASH/RECOVERY 阈值（这两个体制天然短周期）；'
                '2) 调整桶边界；3) 接受现状并在修复审计里写明 bias。'
            )
        else:
            md_lines.append('✅ 所有 bucket 都满足 regime 覆盖度阈值。')

        report_path = self.splits_dir / 'coverage_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines) + '\n')

        # 控制台摘要
        print(f"\n{'=' * 70}")
        print(f"  数据切分完成（详见 {report_path}）")
        print(f"{'=' * 70}")
        for bucket_name, br in report['buckets'].items():
            mark = '✅' if br['all_pass'] else '❌'
            print(f"\n  [{mark}] {bucket_name}: {br['start']} ~ {br['end']}")
            print(f"      区间天数: {br['regime_total_days']}, 股票数: {br['symbols_count']}")
            for r, st in br['threshold_status'].items():
                m = '✓' if st['pass'] else '✗'
                print(f"      {m} {r:10s}: {st['days']:4d} / {st['threshold']}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='数据切分器')
    parser.add_argument('--data-dir', default='./stock_data')
    parser.add_argument('--min-days', type=int, default=180,
                        help='股票在 bucket 内的最少有效天数（默认 180）')
    args = parser.parse_args()

    splitter = DataSplitter(data_dir=args.data_dir, min_days_in_bucket=args.min_days)
    splitter.run()
