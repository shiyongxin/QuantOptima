# -*- coding: utf-8 -*-
"""
东方财富数据获取器 - Eastmoney Data Fetcher

基于东方财富 API 获取股票历史数据，作为 akshare 的备选方案。

功能:
- 获取股票历史 K 线数据
- 支持增量更新
- 自动重试机制
- 批量获取

用法:
    from eastmoney_fetcher import EastmoneyFetcher
    fetcher = EastmoneyFetcher()
    df = fetcher.fetch_stock('000001', start_date='20260101')
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional


class EastmoneyFetcher:
    """
    东方财富数据获取器

    使用东方财富 API 获取股票历史 K 线数据
    """

    def __init__(self, delay=0.3, max_retries=3):
        """
        Parameters:
        -----------
        delay : float
            请求间隔（秒）
        max_retries : int
            最大重试次数
        """
        self.delay = delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://quote.eastmoney.com',
        })

    def _get_market_code(self, symbol: str) -> str:
        """
        获取市场代码

        Parameters:
        -----------
        symbol : str
            股票代码，如 '000001', '600000'

        Returns:
        --------
        str : 市场代码 (0=深圳, 1=上海)
        """
        if symbol.startswith('6') or symbol.startswith('9'):
            return '1'  # 上海
        else:
            return '0'  # 深圳

    def fetch_stock(self, symbol: str, start_date: str = '19960101',
                    end_date: str = None, adjust: str = 'qfq') -> Optional[pd.DataFrame]:
        """
        获取单只股票历史数据

        Parameters:
        -----------
        symbol : str
            股票代码，如 '000001'
        start_date : str
            开始日期，格式 'YYYYMMDD'
        end_date : str
            结束日期，默认今天
        adjust : str
            复权方式: 'qfq'(前复权), 'hfq'(后复权), ''(不复权)

        Returns:
        --------
        pd.DataFrame or None : 股票数据
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        market_code = self._get_market_code(symbol)
        secid = f"{market_code}.{symbol}"

        # 复权代码
        fqt_map = {'qfq': '1', 'hfq': '2', '': '0'}
        fqt = fqt_map.get(adjust, '1')

        url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
        params = {
            'secid': secid,
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
            'klt': '101',  # 日K
            'fqt': fqt,
            'beg': start_date,
            'end': end_date,
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('data') is not None:
                        klines = data['data'].get('klines', [])
                        if klines:
                            return self._parse_klines(klines, symbol)
                time.sleep(self.delay)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"[ERROR] {symbol} 获取失败: {e}")

        return None

    def _parse_klines(self, klines: list, symbol: str) -> pd.DataFrame:
        """
        解析 K 线数据

        Parameters:
        -----------
        klines : list
            K 线数据列表
        symbol : str
            股票代码

        Returns:
        --------
        pd.DataFrame : 解析后的数据
        """
        records = []
        for line in klines:
            parts = line.split(',')
            if len(parts) >= 7:
                records.append({
                    '日期': parts[0],
                    '开盘': float(parts[1]),
                    '收盘': float(parts[2]),
                    '最高': float(parts[3]),
                    '最低': float(parts[4]),
                    '成交量': float(parts[5]),
                    '成交额': float(parts[6]),
                })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)
        return df

    def fetch_index(self, index_code: str = '000300',
                    start_date: str = '19960101',
                    end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取指数历史数据

        Parameters:
        -----------
        index_code : str
            指数代码，如 '000300'(沪深300), '000001'(上证指数)
        start_date : str
            开始日期
        end_date : str
            结束日期

        Returns:
        --------
        pd.DataFrame or None : 指数数据
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        # 指数市场代码
        if index_code.startswith('0'):
            market_code = '1'  # 上海指数
        else:
            market_code = '0'  # 深圳指数

        secid = f"{market_code}.{index_code}"

        url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
        params = {
            'secid': secid,
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
            'klt': '101',  # 日K
            'fqt': '1',
            'beg': start_date,
            'end': end_date,
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('data') is not None:
                        klines = data['data'].get('klines', [])
                        if klines:
                            return self._parse_klines(klines, index_code)
                time.sleep(self.delay)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"[ERROR] 指数 {index_code} 获取失败: {e}")

        return None

    def batch_fetch(self, symbols: list, start_date: str = '19960101',
                    end_date: str = None, callback=None) -> dict:
        """
        批量获取股票数据

        Parameters:
        -----------
        symbols : list
            股票代码列表
        start_date : str
            开始日期
        end_date : str
            结束日期
        callback : callable or None
            回调函数: callback(symbol, success, df)

        Returns:
        --------
        dict : {symbol: DataFrame}
        """
        results = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            df = self.fetch_stock(symbol, start_date, end_date)
            success = df is not None and len(df) > 0

            if success:
                results[symbol] = df

            if callback:
                callback(symbol, success, df)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{total}] {symbol} {'OK' if success else 'FAIL'}")

            time.sleep(self.delay)

        return results


def test_fetcher():
    """测试获取器"""
    fetcher = EastmoneyFetcher()

    print('=== 东方财富数据获取器测试 ===')
    print()

    # 测试1: 获取单只股票
    print('1. 测试获取股票数据 (000001)...')
    df = fetcher.fetch_stock('000001', start_date='20260601')
    if df is not None and len(df) > 0:
        print(f'   ✅ 成功: {len(df)} 条数据')
        print(f'   最新日期: {df.iloc[-1]["日期"]}')
        print(f'   最新收盘: {df.iloc[-1]["收盘"]}')
    else:
        print(f'   ❌ 失败')

    print()

    # 测试2: 获取指数
    print('2. 测试获取指数数据 (沪深300)...')
    df = fetcher.fetch_index('000300', start_date='20260601')
    if df is not None and len(df) > 0:
        print(f'   ✅ 成功: {len(df)} 条数据')
        print(f'   最新日期: {df.iloc[-1]["日期"]}')
        print(f'   最新收盘: {df.iloc[-1]["收盘"]}')
    else:
        print(f'   ❌ 失败')

    print()

    # 测试3: 批量获取
    print('3. 测试批量获取 (3只股票)...')
    symbols = ['000001', '000002', '600000']
    results = fetcher.batch_fetch(symbols, start_date='20260601')
    print(f'   成功: {len(results)}/{len(symbols)} 只')

    print()
    print('=== 测试完成 ===')


if __name__ == '__main__':
    test_fetcher()
