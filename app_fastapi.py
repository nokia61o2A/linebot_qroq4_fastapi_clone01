# app_fastapi.py  — 完整主程式（可直接部署）
# -*- coding: utf-8 -*-

# ========== 1) Imports ==========
import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, date

# --- 網路與資料處理 ---
import httpx           # 非同步 HTTP：用於 LINE、匯率 API
import requests        # 偶發同步備援（很少用）
import pandas as pd    # 指標與表格
import yfinance as yf  # 行情與新聞（Yahoo Finance 包裝）
# FastAPI 與 LINE SDK
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceUser, SourceGroup, SourceRoom
)
# LLM（OpenAI / Groq）
from groq import AsyncGroq, Groq
import openai

# 參考：FastAPI 與 LINE SDK、yfinance、httpx 的官方文件
# FastAPI: https://fastapi.tiangolo.com/
# LINE SDK: https://github.com/line/line-bot-sdk-python
# yfinance: https://pypi.org/project/yfinance/
# httpx: https://www.python-httpx.org/

# ========== 2) Setup ==========
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# --- 必要環境變數 ---
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise RuntimeError("缺少必要環境變數：請設定 BASE_URL / CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET / GROQ_API_KEY")

# --- API Client 初始化 ---
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
sync_groq_client = Groq(api_key=GROQ_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 上述初始化與模型名稱的用法，請見 OpenAI/Groq 官方文件
# OpenAI Chat: https://platform.openai.com/docs/api-reference/chat
# Groq Chat:   https://console.groq.com/docs/models

# ========== 3) FastAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """啟動時自動設定 LINE Webhook Endpoint。"""
    try:
        async with httpx.AsyncClient() as c:
            headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
            payload = {"endpoint": f"{BASE_URL}/callback"}
            r = await c.put(
                "https://api.line.me/v2/bot/channel/webhook/endpoint",
                headers=headers, json=payload, timeout=12.0
            )
            r.raise_for_status()
            logger.info(f"✅ Webhook 更新成功: {r.status_code}")
    except Exception as e:
        logger.error(f"Webhook 更新失敗: {e}", exc_info=True)
    yield

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.0.0")
router = APIRouter()

# 上述 FastAPI 壽命週期與 LINE Webhook 設置說明，見官方文件
# FastAPI lifespan: https://fastapi.tiangolo.com/advanced/events/
# LINE Webhook:     https://developers.line.biz/en/docs/messaging-api/building-bot/

# ========== 4) 共用工具 ==========
def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom):  return event.source.room_id
    return event.source.user_id

# --------- LLM：OpenAI→Groq 備援 ----------
def get_analysis_reply(messages: List[dict]) -> str:
    """
    messages 例：
    [{"role":"system","content":"..."},{"role":"user","content":"..."}]
    """
    # 1) OpenAI（若有設定）
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI 失敗（將改用 Groq）：{e}")

    # 2) Groq 主模型
    try:
        resp = sync_groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq 主模型失敗（改備援）：{e}")
        resp = sync_groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
        return resp.choices[0].message.content

# 上述呼叫方式與參數，請參考 OpenAI/Groq 官方 API 文件
# OpenAI chat params: https://platform.openai.com/docs/api-reference/chat/create
# Groq chat params:  https://console.groq.com/docs/models

# ========== 5) 匯率：JPY 近 7 日（免註冊） ==========
JPY_7D_SYMBOLS = ["TWD", "USD"]  # 可自行增減目標幣別

def _today_utc() -> date:
    return datetime.utcnow().date()

def _last_7d_range() -> Tuple[str, str]:
    end = _today_utc()
    start = end - timedelta(days=6)  # 含今天共 7 天
    return start.isoformat(), end.isoformat()

async def fetch_jpy_7d_exchangerate_host(client: httpx.AsyncClient, symbols=JPY_7D_SYMBOLS) -> pd.DataFrame:
    base = "JPY"
    start_date, end_date = _last_7d_range()
    url = (
        "https://api.exchangerate.host/timeseries"
        f"?base={base}&symbols={','.join(symbols)}&start_date={start_date}&end_date={end_date}"
    )
    r = await client.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    rates = data.get("rates")
    if not isinstance(rates, dict) or not rates:
        raise RuntimeError(f"exchangerate.host 回傳異常：{data}")
    rows = []
    for d in sorted(rates.keys()):
        row = {"date": d}
        for s in symbols:
            row[f"JPY->{s}"] = float(rates[d].get(s))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

async def fetch_jpy_7d_currency_api(client: httpx.AsyncClient, symbols=JPY_7D_SYMBOLS) -> pd.DataFrame:
    end = _today_utc()
    dates = [end - timedelta(days=i) for i in range(6, -1, -1)]
    rows = []
    for d in dates:
        dstr = d.isoformat()
        primary = f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{dstr}/v1/currencies/jpy.json"
        fallback = f"https://{dstr}.currency-api.pages.dev/v1/currencies/jpy.json"
        r = await client.get(primary, timeout=20)
        if r.status_code == 404 or not r.ok:
            r = await client.get(fallback, timeout=20)
        r.raise_for_status()
        obj = r.json()  # {"date":"YYYY-MM-DD","jpy":{"twd":..., "usd":...}}
        jpy_map = obj.get("jpy") or {}
        row = {"date": obj.get("date", dstr)}
        for s in symbols:
            val = jpy_map.get(s.lower())
            if val is None:
                raise RuntimeError(f"{dstr} 缺少 {s} 匯率")
            row[f"JPY->{s}"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)

async def get_jpy_7d_dataframe(symbols=JPY_7D_SYMBOLS) -> pd.DataFrame:
    async with httpx.AsyncClient() as client:
        try:
            df = await fetch_jpy_7d_exchangerate_host(client, symbols)
            df["source"] = "exchangerate.host"
            return df
        except Exception as e:
            logger.warning(f"exchangerate.host 失敗，改用 fawazahmed0：{e}")
            df = await fetch_jpy_7d_currency_api(client, symbols)
            df["source"] = "fawazahmed0/currency-api"
            return df

def format_jpy_7d_text(df: pd.DataFrame) -> str:
    syms = [c.replace("JPY->", "") for c in df.columns if c.startswith("JPY->")]
    lines = [f"近 7 日 1 JPY 對 { '、'.join(syms) } 匯率："]
    for _, row in df.sort_values("date").iterrows():
        parts = [f"JPY→{s} = {row[f'JPY->{s}']:.6f}" for s in syms]
        lines.append(f"{row['date']}  " + " ,  ".join(parts))
    lines.append(f"資料來源：{df['source'].iloc[0]}")
    return "\n".join(lines)

# 以上匯率端點與格式，請見官方文件
# exchangerate.host timeseries: https://exchangerate.host/documentation
# fawazahmed0/currency-api:    https://github.com/fawazahmed0/exchange-api#readme

# ========== 6) 股票查詢與分析 ==========
US_TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\.\-]{0,9}$")
TW_NUM_RE    = re.compile(r"^\d{3,6}$")

def normalize_ticker(user_text: str) -> Tuple[str, List[str]]:
    t = user_text.strip().upper().replace("．",".").replace("：",":")
    if t.startswith(("US:", "US.")):
        core = t.split(":",1)[-1].split(".",1)[-1]
        return (f"US:{core}", [core])
    if t.startswith(("OTC:", "OTC.")):
        core = re.sub(r"^(OTC[:\.])", "", t)
        return (f"OTC:{core}", [f"{core}.TWO"])
    if t.startswith(("TW:", "TW.")):
        core = re.sub(r"^(TW[:\.])", "", t)
        return (f"TW:{core}", [f"{core}.TW", f"{core}.TWO"])
    if TW_NUM_RE.match(t):
        return (f"TW:{t}", [f"{t}.TW", f"{t}.TWO"])
    if US_TICKER_RE.match(t):
        return (f"US:{t}", [t])
    return (t, [t])

def ta_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def ta_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss.replace(0, 1e-9))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_with_yf(ticker_try_list: List[str]) -> Tuple[str, yf.Ticker]:
    last_err = None
    for tk in ticker_try_list:
        try:
            y = yf.Ticker(tk)
            _ = y.fast_info  # 觸發查詢
            return tk, y
        except Exception as e:
            last_err = e
    raise RuntimeError(f"yfinance 取得失敗：{last_err}")

def analyze_stock_text(user_raw: str) -> str:
    disp, tries = normalize_ticker(user_raw)
    tk, y = fetch_with_yf(tries)

    hist = y.history(period="1y", interval="1d", auto_adjust=True)
    if hist.empty:
        raise RuntimeError(f"{tk} 沒有歷史資料")

    close = hist["Close"].dropna()
    last_dt = close.index.max().date()
    last_px = float(close.iloc[-1])

    sma20 = ta_sma(close, 20)
    sma50 = ta_sma(close, 50)
    rsi14 = ta_rsi(close, 14)

    def pct(from_days: int) -> Optional[float]:
        if len(close) <= from_days: return None
        past = float(close.iloc[-from_days-1])
        return (last_px / past - 1.0) * 100.0

    chg_5  = pct(5)
    chg_30 = pct(30)
    chg_90 = pct(90)

    fi = getattr(y, "fast_info", {})
    year_high = fi.get("year_high")
    year_low  = fi.get("year_low")
    market_cap = fi.get("market_cap")
    pe = fi.get("trailing_pe") or fi.get("pe_ratio")
    dy = fi.get("dividend_yield")

    lines = []
    lines.append(f"")
    lines.append(f"最後收盤：{last_dt}  價格：{last_px:,.2f}")
    if pd.notna(sma20.iloc[-1]): lines.append(f"SMA20：{sma20.iloc[-1]:,.2f}")
    if pd.notna(sma50.iloc[-1]): lines.append(f"SMA50：{sma50.iloc[-1]:,.2f}")
    if pd.notna(rsi14.iloc[-1]): lines.append(f"RSI14：{rsi14.iloc[-1]:.2f}")

    lines.append("── 報酬（價差，不含股息）")
    if chg_5  is not None:  lines.append(f"近 1 週：約 {chg_5:+.2f}%")
    if chg_30 is not None: lines.append(f"近 30 日：約 {chg_30:+.2f}%")
    if chg_90 is not None: lines.append(f"近 90 日：約 {chg_90:+.2f}%")

    lines.append("── 52 週參考")
    if (year_low is not None) and (year_high is not None):
        lines.append(f"52W 區間：約 {year_low:,.2f} ~ {year_high:,.2f}")

    lines.append("── 基本面（若可得）")
    if market_cap is not None: lines.append(f"市值：{market_cap:,}")
    if pe is not None:         lines.append(f"P/E：{pe}")
    if dy is not None:         lines.append(f"殖利率(%)：{dy}")

    try:
        ns = y.news[:3] if hasattr(y, "news") else []
        if ns:
            lines.append("── 最近新聞")
            for n in ns:
                ttl = n.get("title","").strip()
                url = n.get("link") or n.get("url")
                if ttl and url:
                    lines.append(f"• {ttl}\n  {url}")
    except Exception:
        pass

    lines.append("\n資料來源：Yahoo Finance / yfinance")
    return "\n".join(lines)

def is_stock_query(text: str) -> bool:
    t = text.strip()
    if t.upper().startswith(("US:", "US.", "TW:", "TW.", "OTC:", "OTC.")):
        return True
    if TW_NUM_RE.match(t):   # 2330
        return True
    if US_TICKER_RE.match(t):  # AAPL / NVDA
        return True
    return False

# 上述 .TW/.TWO 後綴與 yfinance 使用說明，見社群討論與文件
# yfinance: https://pypi.org/project/yfinance/
# .TW/.TWO 背景: https://github.com/ranaroussi/yfinance/discussions/1729

# ========== 7) LINE Webhook ==========
@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.exception("處理 callback 失敗")
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"status": "ok"})

# ========== 8) 事件處理 ==========
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10

@handler.add(MessageEvent, message=TextMessage)
def on_message_text(event: MessageEvent):
    uid = get_chat_id(event)
    text = (event.message.text or "").strip()

    # (A) 股票查詢：自動辨識
    if is_stock_query(text):
        try:
            reply = analyze_stock_text(text)
        except Exception as e:
            reply = f"抱歉，查詢失敗：{e}\n請確認代號是否正確，或稍後再試。"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
        return

    # (B) 匯率快捷：輸入「JPY7」可拿近 7 日 JPY 對 TWD、USD
    if text.upper() in ("JPY7", "JPY 7", "JPY近7日", "JPY七日"):
        try:
            df = asyncio.run(get_jpy_7d_dataframe())
            reply = format_jpy_7d_text(df)
        except Exception as e:
            reply = f"JPY 匯率取得失敗：{e}"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))
        return

    # (C) 一般對話 → LLM
    history = conversation_history.setdefault(uid, [])
    history.append({"role":"user","content":text})
    history[:] = history[-MAX_HISTORY_LEN:]

    sys = (
        "你是專業投資/工程助理，回答使用繁體中文（台灣用語）。"
        "若使用者輸入股票代碼，提醒可直接輸入代碼取得分析；若輸入 JPY7 可取近 7 日匯率。"
    )
    messages = [{"role":"system","content":sys}] + history
    try:
        ai_text = get_analysis_reply(messages)
    except Exception as e:
        ai_text = f"分析服務暫時無法使用：{e}"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=ai_text))

# ========== 9) 啟動 ==========
app.include_router(router)