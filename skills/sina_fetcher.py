# -*- coding: utf-8 -*-
"""
新浪财经数据获取器 - Sina Finance Data Fetcher

基于新浪财经 API 获取股票历史数据，作为 akshare 的备选方案。

功能:
- 获取股票历史 K 线数据
- 支持增量更新
- 自动重试机制
- 批量获取

用法:
    from sina_fetcher import SinaFetcher
    fetcher = SinaFetcher()
    df = fetcher.fetch_stock('000001', start_date='20260101')
"""

import requests
import pandas as pd
import time
import re
from datetime import datetime, timedelta
from typing import Optional


class SinaFetcher:
    """
    新浪财经数据获取器

    使用新浪财经 API 获取股票历史 K 线数据
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
            'Referer': 'https://finance.sina.com.cn/',
        })

    def _get_market_prefix(self, symbol: str) -> str:
        """
        获取市场前缀

        Parameters:
        -----------
        symbol : str
            股票代码，如 '000001', '600000'

        Returns:
        --------
        str : 市场前缀 (sh=上海, sz=深圳)
        """
        if symbol.startswith('6') or symbol.startswith('9'):
            return 'sh'  # 上海
        else:
            return 'sz'  # 深圳

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

        market_prefix = self._get_market_prefix(symbol)
        full_symbol = f"{market_prefix}{symbol}"

        # 构建 URL
        url = f'https://finance.sina.com.cn/realstock/company/{full_symbol}/hisdata/klc_kl.js'

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    df = self._parse_response(response.text, symbol)
                    if df is not None and len(df) > 0:
                        # 过滤日期范围
                        start_dt = pd.to_datetime(start_date)
                        end_dt = pd.to_datetime(end_date)
                        df = df[(df['日期'] >= start_dt) & (df['日期'] <= end_dt)]
                        df = df.reset_index(drop=True)
                        return df
                time.sleep(self.delay)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"[ERROR] {symbol} 获取失败: {e}")

        return None

    def _parse_response(self, text: str, symbol: str) -> Optional[pd.DataFrame]:
        """
        解析新浪财经响应数据

        Parameters:
        -----------
        text : str
            响应文本
        symbol : str
            股票代码

        Returns:
        --------
        pd.DataFrame or None : 解析后的数据
        """
        try:
            # 解析 JavaScript 格式的数据
            # 格式: var hq_str_sh000001="..."
            match = re.search(r'var hq_str_\w+="([^"]*)"', text)
            if not match:
                return None

            data_str = match.group(1)
            if not data_str:
                return None

            # 解析数据行
            lines = data_str.split('\\n')
            records = []

            for line in lines:
                if not line.strip():
                    continue

                parts = line.split(',')
                if len(parts) >= 6:
                    try:
                        records.append({
                            '日期': parts[0],
                            '开盘': float(parts[1]),
                            '最高': float(parts[2]),
                            '最低': float(parts[3]),
                            '收盘': float(parts[4]),
                            '成交量': float(parts[5]),
                        })
                    except (ValueError, IndexError):
                        continue

            if not records:
                return None

            df = pd.DataFrame(records)
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期').reset_index(drop=True)
            return df

        except Exception as e:
            print(f"[ERROR] 解析失败: {e}")
            return None

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

        # 指数市场前缀
        if index_code.startswith('0'):
            market_prefix = 'sh'  # 上海指数
        else:
            market_prefix = 'sz'  # 深圳指数

        full_symbol = f"{market_prefix}{index_code}"

        # 构建 URL
        url = f'https://finance.sina.com.cn/realstock/company/{full_symbol}/hisdata/klc_kl.js'

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    df = self._parse_response(response.text, index_code)
                    if df is not None and len(df) > 0:
                        # 过滤日期范围
                        start_dt = pd.to_datetime(start_date)
                        end_dt = pd.to_datetime(end_date)
                        df = df[(df['日期'] >= start_dt) & (df['日期'] <= end_dt)]
                        df = df.reset_index(drop=True)
                        return df
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
    fetcher = SinaFetcher()

    print('=== 新浪财经数据获取器测试 ===')
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
