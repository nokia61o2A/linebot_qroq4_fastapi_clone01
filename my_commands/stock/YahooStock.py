# YahooStock.py
import re
import time
import requests
from datetime import datetime

YF_QUOTE_API = "https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols}"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

def _normalize_symbol(symbol: str) -> str:
    """
    - 台股：純數字(可含末碼英字) → 補 .TW，如 2330 → 2330.TW、2881A → 2881A.TW
    - 指數 / 海外：原樣或大寫（NVDA、AAPL、^GSPC…）
    """
    s = symbol.strip().upper()
    if re.fullmatch(r"\d{4,6}[A-Z]?", s):
        return f"{s}.TW"
    return s

def _retry(times=3, delay=1.2):
    def deco(fn):
        def wrap(*args, **kwargs):
            last = None
            for i in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last = e
                    time.sleep(delay * (i + 1))
            raise last
        return wrap
    return deco


class YahooStock:
    """
    以 Yahoo Finance quote API 擷取股票快照資料。
    公開屬性：
      - stock_symbol: 使用者輸入（原樣）
      - symbol: 正規化後用於查詢的代碼（台股會補 .TW）
      - name: 股票名稱
      - now_price: 目前價（float）
      - change: 字串格式的漲跌 e.g. '+3.20 (+1.25%)'
      - currency: 幣別 e.g. 'TWD'
      - close_time: 文字化的市場時間（本機時區）
    """

    def __init__(self, stock_symbol: str):
        self.stock_symbol = stock_symbol
        self.symbol = _normalize_symbol(stock_symbol)

        self.name = None
        self.now_price = None
        self.change = None
        self.currency = None
        self.close_time = None

        try:
            info = self.fetch_stock_info()
            self.name = info.get("name")
            self.now_price = info.get("now_price")
            self.change = info.get("change")
            self.currency = info.get("currency")
            self.close_time = info.get("close_time")
        except Exception as e:
            # 保持介面相容：初始化失敗時不丟例外，但欄位為 None
            print(f"[YahooStock] 初始化失敗：{e}")

    @_retry(times=3, delay=1.0)
    def _query(self) -> dict:
        url = YF_QUOTE_API.format(symbols=self.symbol)
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=8)
        r.raise_for_status()
        data = r.json()
        result = (data or {}).get("quoteResponse", {}).get("result", [])
        if not result:
            raise ValueError(f"Yahoo quote API 無回傳結果：{self.symbol}")
        return result[0]

    def fetch_stock_info(self) -> dict:
        """
        呼叫 Yahoo quote API，整理成一致欄位。
        回傳 dict: name, now_price, change(字串), currency, close_time(字串)
        """
        q = self._query()

        name = q.get("longName") or q.get("shortName") or self.symbol
        price = q.get("regularMarketPrice")
        chg = q.get("regularMarketChange")
        chg_pct = q.get("regularMarketChangePercent")
        currency = q.get("currency")
        ts = q.get("regularMarketTime")  # epoch 秒

        # 組合變動字串
        change_str = None
        if chg is not None and chg_pct is not None:
            try:
                change_str = f"{chg:+.2f} ({chg_pct:+.2f}%)"
            except Exception:
                change_str = None

        # 轉成本機時區文字
        close_time_str = None
        if ts:
            try:
                close_time_str = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        return {
            "name": name,
            "now_price": price,
            "change": change_str,
            "currency": currency,
            "close_time": close_time_str,
        }

    # 讓 vars(self) 也好用（你的程式有印 vars(newprice_stock)）
    def __repr__(self) -> str:
        return (
            f"YahooStock(symbol='{self.symbol}', name='{self.name}', now_price={self.now_price}, "
            f"change='{self.change}', currency='{self.currency}', close_time='{self.close_time}')"
        )