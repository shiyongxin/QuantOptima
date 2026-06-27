# -*- coding: utf-8 -*-
"""
绕过 akshare 的 mini_racer Chromium crash，
用 eastmoney_fetcher 拉 30 年数据到 parquet。

akshare 在 Python 3.14 + mini_racer 0.14.1 上崩溃（V8 address pool 初始化失败），
eastmoney_fetcher 用纯 requests，无 JS 依赖，可正常工作。

输出格式跟 HistoricalDataManager 一致，data_splitter.py 直接可用。
"""
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent / 'skills'))

import pandas as pd
from eastmoney_fetcher import EastmoneyFetcher
from historical_data_manager import HistoricalDataManager

# 读 stock_list.csv
csv_path = Path('skills/stock_list.csv')
df_csv = pd.read_csv(csv_path, encoding='utf-8-sig')
col = 'code' if 'code' in df_csv.columns else '股票代码'
symbols = [str(c).zfill(6) for c in df_csv[col].tolist()]

print(f"[INFO] 从 {csv_path} 读取 {len(symbols)} 个股票代码")
print(f"[INFO] 日期范围: 19960101 ~ 20260627")
print(f"[INFO] 数据源: 东方财富 API (无 mini_racer 依赖)")
print(f"[INFO] 并发: 4, 间隔: 0.3s")
print()

fetcher = EastmoneyFetcher()
dm = HistoricalDataManager('./stock_data')

# 准备 metadata
if 'stocks' not in dm.metadata:
    dm.metadata['stocks'] = {}

success = 0
fail = 0
skipped = 0
start_time = time.time()


def fetch_one(symbol: str):
    try:
        df = fetcher.fetch_stock(symbol, start_date='19960101')
        if df is None or len(df) < 100:
            return symbol, 'short', 0
        # 标准化列名（eastmoney 已经是中文列名）
        df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量']].copy()
        df = df.sort_values('日期').reset_index(drop=True)
        # 写 parquet
        path = dm.historical_dir / f"{symbol}.parquet"
        df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)
        # 更新 metadata
        dm.metadata['stocks'][symbol] = {
            'first_date': str(df['日期'].iloc[0].date()) if hasattr(df['日期'].iloc[0], 'date') else str(df['日期'].iloc[0]),
            'last_date': str(df['日期'].iloc[-1].date()) if hasattr(df['日期'].iloc[-1], 'date') else str(df['日期'].iloc[-1]),
            'row_count': len(df),
            'updated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        return symbol, 'ok', len(df)
    except Exception as e:
        return symbol, f'fail: {str(e)[:80]}', 0


with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(fetch_one, sym): sym for sym in symbols}
    for i, fut in enumerate(as_completed(futures)):
        sym, status, n = fut.result()
        if status == 'ok':
            success += 1
        elif status == 'short':
            skipped += 1
        else:
            fail += 1
            print(f"  [FAIL] {sym}: {status}")
        if (i + 1) % 50 == 0 or (i + 1) == len(symbols):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(symbols) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(symbols)}] ok={success} skip={skipped} fail={fail} | "
                  f"{rate:.1f} stocks/s | ETA {eta/60:.0f} min")
        # 简易节流
        time.sleep(0.05)

# 保存 metadata
dm.metadata['last_update'] = time.strftime('%Y-%m-%dT%H:%M:%S')
dm.metadata['stock_count'] = len(dm.metadata['stocks'])
dm._save_metadata()

elapsed = time.time() - start_time
print()
print(f"=" * 60)
print(f"  30 年数据下载完成")
print(f"=" * 60)
print(f"  成功: {success}")
print(f"  跳过(<100天): {skipped}")
print(f"  失败: {fail}")
print(f"  耗时: {elapsed/60:.1f} 分钟")
print(f"  数据目录: {dm.historical_dir}")
print()
print(f"下一步: python skills/data_splitter.py")
