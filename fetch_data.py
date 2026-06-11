# -*- coding: utf-8 -*-
"""
阶段1: 数据获取脚本 (需要联网)

功能:
1. 获取A股全部股票列表
2. 批量下载30年历史数据 (Parquet格式)
3. 获取大盘指数数据并生成体制标签
4. 数据验证

运行:
    python fetch_data.py                    # 默认获取500只
    python fetch_data.py --n 1000           # 获取1000只
    python fetch_data.py --symbols 000001 600519  # 指定股票
    python fetch_data.py --full             # 全量A股(~5000只)

预计耗时: 500只约2小时, 全量约8-12小时
"""
import sys
import argparse
from pathlib import Path

# 确保skills目录在路径中
sys.path.insert(0, str(Path(__file__).parent / 'skills'))

from historical_data_manager import HistoricalDataManager
from market_regime_classifier import MarketRegimeClassifier


def fetch_stock_data(args):
    """获取股票历史数据"""
    dm = HistoricalDataManager(args.data_dir)

    # 确定股票列表
    symbols = []
    if args.symbols:
        symbols = args.symbols
    elif args.full or args.n > 500:
        # 从akshare获取全部A股列表
        print("[INFO] 获取A股全部股票列表...")
        import akshare as ak
        try:
            df = ak.stock_zh_a_spot_em()
            symbols = [str(c).zfill(6) for c in df['代码'].tolist()]
            print(f"[OK] 获取到 {len(symbols)} 只A股")
        except Exception as e:
            print(f"[WARN] 获取股票列表失败: {e}")
            print("[INFO] 尝试从本地stock_list.csv加载...")
            stock_list = Path('skills/stock_list.csv')
            if stock_list.exists():
                import pandas as pd
                df = pd.read_csv(stock_list, encoding='utf-8-sig')
                col = 'code' if 'code' in df.columns else '股票代码'
                symbols = [str(c).zfill(6) for c in df[col].tolist()]
    else:
        # 从本地列表随机抽样
        stock_list = Path('skills/stock_list.csv')
        if stock_list.exists():
            import pandas as pd
            import random
            df = pd.read_csv(stock_list, encoding='utf-8-sig')
            col = 'code' if 'code' in df.columns else '股票代码'
            all_symbols = [str(c).zfill(6) for c in df[col].tolist()]
            n = min(args.n, len(all_symbols))
            symbols = random.sample(all_symbols, n)
        else:
            print("[ERROR] 无股票列表, 请用 --symbols 指定")
            return

    if not symbols:
        print("[ERROR] 无股票可获取")
        return

    print(f"\n{'='*60}")
    print(f"  数据获取计划")
    print(f"  股票数: {len(symbols)}")
    print(f"  日期范围: {args.start} ~ 今天")
    print(f"  并发数: {args.workers}")
    print(f"  请求间隔: {args.delay}s")
    print(f"  数据目录: {args.data_dir}")
    print(f"{'='*60}\n")

    # 批量获取
    dm.batch_fetch(
        symbols,
        start_date=args.start,
        max_workers=args.workers,
        delay=args.delay,
    )

    # 保存元数据
    dm._save_metadata()

    # 验证数据
    print("\n[INFO] 验证数据质量...")
    universe = dm.get_universe(min_rows=2000)
    print(f"[OK] 有足够历史的股票: {len(universe)} 只")

    return universe


def fetch_index_data(args):
    """获取大盘指数数据并生成体制标签"""
    print("\n" + "=" * 60)
    print("  获取大盘指数数据 + 体制分类")
    print("=" * 60)

    classifier = MarketRegimeClassifier(args.data_dir)

    # 加载沪深300指数
    index_data = classifier.load_index_data(
        index_code=args.index,
        start_date=args.start,
    )

    if index_data is None or len(index_data) == 0:
        print("[ERROR] 无法获取指数数据")
        # 尝试从已有的股票数据中构建
        print("[INFO] 尝试从股票数据估算大盘态势...")
        return

    # 体制分类
    print("\n[INFO] 进行体制分类...")
    regime_series = classifier.classify(method='adaptive')

    # 保存标签
    labels_file = Path(args.data_dir) / 'regime_labels.csv'
    classifier.save_labels(labels_file, regime_series)

    # 输出报告
    report = classifier.format_report(regime_series)
    report_file = Path(args.data_dir) / 'regime_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[OK] 体制报告: {report_file}")

    # 打印摘要
    print(report[:500])
    print("...")


def main():
    parser = argparse.ArgumentParser(description='阶段1: 数据获取')
    parser.add_argument('--symbols', nargs='+', help='指定股票代码')
    parser.add_argument('--n', type=int, default=500, help='随机获取N只 (默认500)')
    parser.add_argument('--full', action='store_true', help='全量A股')
    parser.add_argument('--start', default='19960101', help='开始日期')
    parser.add_argument('--index', default='000300', help='大盘指数代码')
    parser.add_argument('--data-dir', default='stock_data', help='数据目录')
    parser.add_argument('--workers', type=int, default=4, help='并发数')
    parser.add_argument('--delay', type=float, default=0.3, help='请求间隔(秒)')
    parser.add_argument('--skip-index', action='store_true', help='跳过指数数据获取')

    args = parser.parse_args()

    # Step 1: 获取股票数据
    print("\n" + "=" * 60)
    print("  阶段1: 股票历史数据获取")
    print("=" * 60)
    universe = fetch_stock_data(args)

    # Step 2: 获取指数数据 + 体制标签
    if not args.skip_index:
        fetch_index_data(args)

    # 完成
    print("\n" + "=" * 60)
    print("  数据获取完成!")
    print("=" * 60)
    print(f"  数据目录: {args.data_dir}")
    print(f"  下一步: python run_optimize.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
