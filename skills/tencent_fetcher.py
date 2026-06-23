# -*- coding: utf-8 -*-
"""
腾讯财经数据获取器 - Tencent Finance Data Fetcher

基于腾讯财经 API 获取股票历史数据，作为 akshare 的备选方案。

功能:
- 获取股票历史 K 线数据
- 支持增量更新
- 自动重试机制
- 批量获取

用法:
    from tencent_fetcher import TencentFetcher
    fetcher = TencentFetcher()
    df = fetcher.fetch_stock('000001', start_date='20260101')
"""

import requests
import pandas as pd
import time
import json
from datetime import datetime, timedelta
from typing import Optional


class TencentFetcher:
    """
    腾讯财经数据获取器

    使用腾讯财经 API 获取股票历史 K 线数据
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
            'Referer': 'https://web.ifzq.gtimg.cn/',
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
        # 上海股票: 600xxx, 601xxx, 603xxx, 605xxx, 688xxx (科创板)
        # 深圳股票: 000xxx, 001xxx, 002xxx, 003xxx, 300xxx (创业板)
        # 上海指数: 000001 (上证指数), 000300 (沪深300)
        # 深圳指数: 399001 (深证成指), 399006 (创业板指)

        if symbol.startswith('6') or symbol.startswith('9'):
            return 'sh'  # 上海股票
        elif symbol.startswith('0'):
            return 'sz'  # 深圳股票
        elif symbol.startswith('3'):
            return 'sz'  # 深圳股票 (创业板)
        else:
            return 'sz'  # 默认深圳

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

        # 格式化日期
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')

        # 复权方式
        fq_map = {'qfq': 'qfq', 'hfq': 'hfq', '': ''}
        fq = fq_map.get(adjust, 'qfq')

        # 构建 URL
        url = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
        params = {
            '_var': f'kline_day{fq}',
            'param': f'{full_symbol},day,{start_str},{end_str},100,{fq}',
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    df = self._parse_response(response.text, symbol)
                    if df is not None and len(df) > 0:
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
        解析腾讯财经响应数据

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
            # 移除变量名 (格式: kline_dayqfq={...})
            if '=' in text:
                json_str = text.split('=', 1)[1]
            else:
                json_str = text

            data = json.loads(json_str)

            if not isinstance(data, dict):
                return None

            if data.get('code') != 0:
                return None

            if 'data' not in data:
                return None

            stock_data = data['data']

            if not isinstance(stock_data, dict):
                return None

            # 获取股票数据 (尝试不同的键名)
            stock = None
            market_prefix = self._get_market_prefix(symbol)
            full_symbol = f"{market_prefix}{symbol}"

            # 直接尝试完整符号
            if full_symbol in stock_data:
                stock = stock_data[full_symbol]
            else:
                # 尝试模糊匹配
                for key in stock_data.keys():
                    if symbol in key or key.endswith(symbol):
                        stock = stock_data[key]
                        break

            if stock is None:
                return None

            if not isinstance(stock, dict):
                return None

            # 获取日K线数据
            kline_key = None
            for key in ['day', 'qfqday', 'hfqday']:
                if key in stock:
                    kline_key = key
                    break

            if kline_key is None:
                return None

            days = stock[kline_key]

            if not days or not isinstance(days, list):
                return None

            # 解析数据
            records = []
            for day in days:
                if isinstance(day, list) and len(day) >= 6:
                    try:
                        records.append({
                            '日期': day[0],
                            '开盘': float(day[1]),
                            '最高': float(day[2]),
                            '最低': float(day[3]),
                            '收盘': float(day[4]),
                            '成交量': float(day[5]),
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

        # 格式化日期
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')

        # 构建 URL
        url = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
        params = {
            '_var': 'kline_dayqfq',
            'param': f'{full_symbol},day,{start_str},{end_str},100,qfq',
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    df = self._parse_response(response.text, index_code)
                    if df is not None and len(df) > 0:
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
    fetcher = TencentFetcher()

    print('=== 腾讯财经数据获取器测试 ===')
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
