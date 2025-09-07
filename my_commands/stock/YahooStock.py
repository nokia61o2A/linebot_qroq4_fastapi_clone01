# YahooStock.py
import re
import time
import requests
from datetime import datetime
from bs4 import BeautifulSoup

YF_QUOTE_API = "https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols}"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

def _normalize_symbol(symbol: str) -> str:
    """
    - 台股：純數字(可含末碼英字) → 補 .TW，如 2330 → 2330.TW、2881A → 2881A.TW、00937B → 00937B.TW
    - 指數/美股：原樣或大寫（NVDA、AAPL、^GSPC…）
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
            if last:
                raise last
        return wrap
    return deco


def _fetch_yahoo_html(symbol: str):
    """Yahoo Finance 頁面解析備援：嘗試抓 <fin-streamer> 即時價。"""
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}"
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        name = soup.select_one("h1.D\\(ib\\)")
        price = soup.select_one('fin-streamer[data-field="regularMarketPrice"]')
        chg_pct = soup.select_one('fin-streamer[data-field="regularMarketChangePercent"]')
        when = soup.find("div", string=lambda t: t and isinstance(t, str) and ("GMT" in t or "台北" in t))

        return {
            "name": name.get_text(strip=True) if name else symbol,
            "now_price": float(price.text.replace(",", "")) if price and price.text else None,
            "change": (chg_pct.text if chg_pct else None),
            "currency": None,
            "close_time": when.text.strip() if when else None,
        }
    except Exception:
        return None


def _fetch_twse(stock_no: str):
    """臺灣證交所收盤備援（ETF/債券ETF 很常在 Yahoo 缺資料）。"""
    try:
        # date 空字串 = 近一年
        url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=&stockNo={stock_no}"
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        r.raise_for_status()
        j = r.json()
        if j.get("stat") != "OK":
            return None
        rows = j.get("data") or []
        if not rows:
            return None
        d = rows[-1]  # 取最近一筆
        # [日期, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 漲跌價差, 交易筆數]
        name = j.get("title", "").replace("加權指數", "").strip() or stock_no
        price = d[6].replace(",", "")
        chg = d[7]
        return {
            "name": name,
            "now_price": float(price) if price not in ("", "--", "X") else None,
            "change": chg if chg not in ("", "--") else None,
            "currency": "TWD",
            "close_time": d[0],
        }
    except Exception:
        return None


class YahooStock:
    """
    以 Yahoo Finance 取得股票快照（API → HTML → TWSE 三層備援）。
      - stock_symbol: 使用者原輸入
      - symbol: 正規化代碼（台股補 .TW）
      - name, now_price, change, currency, close_time
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
            for k, v in info.items():
                setattr(self, k, v)
        except Exception as e:
            print(f"[YahooStock] 初始化失敗：{e}")

    @_retry(times=3, delay=1.0)
    def _query_api(self) -> dict:
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
        回傳 dict: name, now_price, change(字串), currency, close_time(字串)
        """
        # 第一層：Yahoo quote API
        try:
            q = self._query_api()
            name = q.get("longName") or q.get("shortName") or self.symbol
            price = q.get("regularMarketPrice")
            chg = q.get("regularMarketChange")
            chg_pct = q.get("regularMarketChangePercent")
            currency = q.get("currency")
            ts = q.get("regularMarketTime")

            change_str = None
            if chg is not None and chg_pct is not None:
                change_str = f"{chg:+.2f} ({chg_pct:+.2f}%)"

            close_time_str = None
            if ts:
                try:
                    close_time_str = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass

            if price is not None:
                return {
                    "name": name,
                    "now_price": float(price),
                    "change": change_str,
                    "currency": currency,
                    "close_time": close_time_str,
                }
        except Exception:
            pass

        # 第二層：Yahoo HTML
        html_info = _fetch_yahoo_html(self.symbol)
        if html_info and html_info.get("now_price") is not None:
            return html_info

        # 第三層：TWSE（只限台股 .TW）
        if self.symbol.endswith(".TW"):
            twse = _fetch_twse(self.symbol.replace(".TW", ""))
            if twse and twse.get("now_price") is not None:
                return twse

        # 全失敗
        return {"name": self.symbol, "now_price": None, "change": None, "currency": None, "close_time": None}

    def __repr__(self) -> str:
        return (
            f"YahooStock(symbol='{self.symbol}', name='{self.name}', now_price={self.now_price}, "
            f"change='{self.change}', currency='{self.currency}', close_time='{self.close_time}')"
        )