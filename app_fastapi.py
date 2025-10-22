# app_fastapi.py
# -*- coding: utf-8 -*-

# ===================== 1) Imports =====================
import os
import re
import io
import random
import logging
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import requests
import httpx
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, AudioSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    QuickReply, QuickReplyButton, MessageAction, PostbackAction,
    PostbackEvent, FlexSendMessage, BubbleContainer, BoxComponent,
    TextComponent, ButtonComponent, SeparatorComponent, Sender
)

# AI：可用 Groq；OpenAI 僅供文字備援（不做 TTS）
from groq import Groq
import openai

# 免費 TTS
from gtts import gTTS

# 靜態上傳（語音檔）到 Cloudinary
import cloudinary
import cloudinary.uploader

import uvicorn

# ===================== 2) Setup & Env =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(asctime)s:%(message)s"
)
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# --- 必要環境變數 ---
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

if not CHANNEL_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("缺少 CHANNEL_ACCESS_TOKEN 或 CHANNEL_SECRET")

# --- AI Keys（文字用；TTS 不用 OpenAI）---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")  # 可留空

# --- Cloudinary ---
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

if CLOUDINARY_URL:
    cloudinary.config(cloudinary_url=CLOUDINARY_URL)
elif CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET
    )
else:
    logger.warning("⚠️ 未設定 Cloudinary，語音檔無法外部存取！請設定 CLOUDINARY_URL 或三段式變數。")

logger.info("✅ Cloudinary 配置成功")

# --- LINE SDK ---
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# --- AI Clients（文字）---
groq_client: Optional[Groq] = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logger.warning(f"Groq 初始化失敗：{e}")

openai_client: Optional[openai.OpenAI] = None
if OPENAI_API_KEY:
    try:
        if OPENAI_API_BASE:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
            logger.info(f"✅ OpenAI Client (base={OPENAI_API_BASE})")
        else:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            logger.info("✅ OpenAI Client")
    except Exception as e:
        logger.warning(f"OpenAI 初始化失敗：{e}")

# --- 常數 ---
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
}
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"

# --- 動態狀態 ---
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10

# TTS 狀態：是否開啟、語言（gTTS）
tts_enabled: Dict[str, bool] = {}
tts_lang: Dict[str, str] = {}  # 例：'zh-TW', 'ja', 'en'

# 翻譯模式：顯示頭像名稱（附語向）
translation_states: Dict[str, str] = {}  # e.g. "英文"/"日文"/"繁體中文"

# ===================== 3) Optional Modules (Lottery/Stock) =====================
LOTTERY_ENABLED = True
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    lottery_crawler = TaiwanLotteryCrawler()
    logger.info("✅ taiwanlottery 套件已載入")
except Exception as e:
    logger.error(f"⚠️ 無法載入 taiwanlottery 套件：{e}")
    LOTTERY_ENABLED = False
    lottery_crawler = None

STOCK_ENABLED = True
try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
except Exception as e:
    logger.error(f"⚠️ 股票模組載入失敗：{e}")
    STOCK_ENABLED = False

# ===================== 4) FastAPI =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 應用啟動（lifespan）")
    # 更新 LINE Webhook（僅正式 token 時）
    if BASE_URL and CHANNEL_TOKEN and CHANNEL_TOKEN != "dummy":
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
                payload = {"endpoint": f"{BASE_URL}/callback"}
                r = await c.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=payload)
                r.raise_for_status()
                logger.info(f"✅ Webhook 更新成功: {r.status_code}")
        except Exception as e:
            logger.warning(f"  ⚠️ Webhook 更新失敗：{e}")
    yield
    logger.info("👋 應用程式關閉")

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="3.1.0")
router = APIRouter()

# ===================== 5) Helpers =====================
def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup):
        return event.source.group_id
    if isinstance(event.source, SourceRoom):
        return event.source.room_id
    if isinstance(event.source, SourceUser):
        return event.source.user_id
    return "unknown"

def build_quick_reply(chat_id: Optional[str] = None) -> QuickReply:
    items = [
        QuickReplyButton(action=MessageAction(label="主選單", text="選單")),
        QuickReplyButton(action=MessageAction(label="台股大盤", text="台股大盤")),
        QuickReplyButton(action=MessageAction(label="美股大盤", text="美股大盤")),
        QuickReplyButton(action=MessageAction(label="黃金價格", text="金價")),
        QuickReplyButton(action=MessageAction(label="查 2330", text="2330")),
        QuickReplyButton(action=MessageAction(label="查 NVDA", text="NVDA")),
        QuickReplyButton(action=MessageAction(label="日圓匯率", text="JPY")),
        QuickReplyButton(action=MessageAction(label="TTS ON✅", text="TTS ON")),
        QuickReplyButton(action=MessageAction(label="TTS OFF", text="TTS OFF")),
        QuickReplyButton(action=PostbackAction(label="💖 AI 人設", data="menu:persona")),
        QuickReplyButton(action=PostbackAction(label="🏛️ 彩票選單", data="menu:lottery")),
        QuickReplyButton(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate")),
    ]
    return QuickReply(items=items)

def _ensure_qr_visible(messages: list, chat_id: Optional[str], sender: Optional[Sender]):
    """
    把 QuickReply 直接掛到最後一個可掛的訊息上，不新增任何占位文字。
    """
    if not messages:
        return
    qr = build_quick_reply(chat_id)
    # 優先掛在最後一則
    for m in reversed(messages):
        try:
            if getattr(m, "quick_reply", None) is None:
                setattr(m, "quick_reply", qr)
            return
        except Exception:
            continue
    # 如果都不支援 quick_reply，就作罷（實務上 Text/Audio/Flex 都支援）

def reply_messages(reply_token: str, messages: list, chat_id: Optional[str], sender: Optional[Sender]):
    _ensure_qr_visible(messages, chat_id, sender)
    try:
        line_bot_api.reply_message(reply_token, messages)
    except LineBotApiError as lbe:
        logger.error(f"LINE 回覆失敗：( {lbe.status_code} ) {lbe.error.message}")
        try:
            line_bot_api.reply_message(
                reply_token,
                TextSendMessage(text="抱歉，訊息傳送失敗。", quick_reply=build_quick_reply(chat_id), sender=sender)
            )
        except Exception:
            pass

def build_main_menu_flex() -> FlexSendMessage:
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(
            layout="vertical",
            contents=[TextComponent(text="AI 助理選單", weight="bold", size="lg")]
        ),
        body=BoxComponent(
            layout="vertical",
            spacing="md",
            contents=[
                TextComponent(text="選擇功能：", size="sm"),
                SeparatorComponent(margin="md"),
                ButtonComponent(action=PostbackAction(label="💹 金融查詢", data="menu:finance"), style="primary", color="#5E86C1"),
                ButtonComponent(action=PostbackAction(label="🏛️ 彩票分析", data="menu:lottery"), style="primary", color="#5EC186"),
                ButtonComponent(action=PostbackAction(label="💖 AI 角色", data="menu:persona"), style="secondary"),
                ButtonComponent(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate"), style="secondary"),
                ButtonComponent(action=PostbackAction(label="⚙️ 系統設定", data="menu:settings"), style="secondary"),
            ]
        )
    )
    return FlexSendMessage(alt_text="主選單", contents=bubble)

def build_submenu_flex(kind: str) -> FlexSendMessage:
    title, buttons = "子選單", []
    if kind == "finance":
        title, buttons = "💹 金融查詢", [
            ButtonComponent(action=MessageAction(label="台股", text="台股大盤")),
            ButtonComponent(action=MessageAction(label="美股", text="美股大盤")),
            ButtonComponent(action=MessageAction(label="黃金", text="金價")),
            ButtonComponent(action=MessageAction(label="日圓", text="JPY")),
            ButtonComponent(action=MessageAction(label="2330", text="2330")),
            ButtonComponent(action=MessageAction(label="NVDA", text="NVDA")),
        ]
    elif kind == "lottery":
        title, buttons = "🏛️ 彩票分析", [
            ButtonComponent(action=MessageAction(label="大樂透", text="大樂透")),
            ButtonComponent(action=MessageAction(label="威力彩", text="威力彩")),
            ButtonComponent(action=MessageAction(label="今彩539", text="539")),
        ]
    elif kind == "persona":
        title, buttons = "💖 AI 角色", [
            ButtonComponent(action=MessageAction(label="甜美女友", text="甜")),
            ButtonComponent(action=MessageAction(label="傲嬌女友", text="鹹")),
            ButtonComponent(action=MessageAction(label="萌系女友", text="萌")),
            ButtonComponent(action=MessageAction(label="酷系御姐", text="酷")),
            ButtonComponent(action=MessageAction(label="隨機", text="random")),
        ]
    elif kind == "translate":
        title, buttons = "🌐 翻譯工具", [
            ButtonComponent(action=MessageAction(label="翻英文", text="翻譯->英文")),
            ButtonComponent(action=MessageAction(label="翻日文", text="翻譯->日文")),
            ButtonComponent(action=MessageAction(label="翻繁中", text="翻譯->繁體中文")),
            ButtonComponent(action=MessageAction(label="結束", text="翻譯->結束")),
        ]
    elif kind == "settings":
        title, buttons = "⚙️ 系統設定", [
            ButtonComponent(action=MessageAction(label="開啟自動回答", text="開啟自動回答")),
            ButtonComponent(action=MessageAction(label="關閉自動回答", text="關閉自動回答")),
        ]
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[TextComponent(text=title, weight="bold", size="lg")]),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm")
    )
    return FlexSendMessage(alt_text=title, contents=bubble)

# ===================== 6) AI & Text =====================
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.2-90b-text")  # 新版可用
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

def get_analysis_reply(messages: List[dict]) -> str:
    # 先試 OpenAI（若有提供 base，可當免費轉發器）；否則 Groq；最後回復錯誤
    if openai_client:
        try:
            r = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                messages=messages,
                temperature=0.7,
                max_tokens=1800,
            )
            return r.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI 失敗：{e}")

    if groq_client:
        try:
            r = groq_client.chat.completions.create(
                model=GROQ_MODEL_PRIMARY, messages=messages, temperature=0.7, max_tokens=1800
            )
            return r.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq Primary 失敗：{e}")
            try:
                r = groq_client.chat.completions.create(
                    model=GROQ_MODEL_FALLBACK, messages=messages, temperature=0.8, max_tokens=1500
                )
                return r.choices[0].message.content
            except Exception as e2:
                logger.error(f"Groq Fallback 也失敗：{e2}")

    return "抱歉，AI 分析引擎目前不可用，請稍後再試。"

def analyze_sentiment(text: str) -> str:
    sys = "Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."
    msgs = [{"role":"system","content":sys},{"role":"user","content":text}]
    try:
        if groq_client:
            r = groq_client.chat.completions.create(
                model=GROQ_MODEL_FALLBACK, messages=msgs, temperature=0, max_tokens=10
            )
            out = (r.choices[0].message.content or "neutral").strip().lower()
            return out if out in ["positive","neutral","negative","angry"] else "neutral"
    except Exception:
        pass
    return "neutral"

# ===================== 7) Stocks & Finance =====================
_TW_CODE_RE = re.compile(r'^\d{4,6}[A-Za-z]?$')
_US_CODE_RE = re.compile(r'^[A-Za-z]{1,5}$')

def normalize_ticker(t: str) -> Tuple[str, str, str, bool]:
    t = t.strip().upper()
    if t in ["台股大盤", "大盤"]:
        return "^TWII", "^TWII", "^TWII", True
    if t in ["美股大盤", "美盤", "美股"]:
        return "^GSPC", "^GSPC", "^GSPC", True
    if _TW_CODE_RE.match(t):
        return f"{t}.TW", t, t, False
    if _US_CODE_RE.match(t) and t != "JPY":
        return t, t, t, False
    return t, t, t, False

def fetch_realtime_snapshot(yf_symbol: str, yahoo_slug: str) -> dict:
    snap = {"name": None, "now_price": None, "change": None, "currency": None, "close_time": None}
    try:
        tk = yf.Ticker(yf_symbol)
        info = {}
        hist = pd.DataFrame()
        try:
            info = tk.info or {}
        except Exception:
            pass
        try:
            hist = tk.history(period="2d", interval="1d")
        except Exception:
            pass

        snap["name"] = info.get("shortName") or info.get("longName") or yf_symbol
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        ccy = info.get("currency")
        if price:
            snap["now_price"] = f"{price:.2f}"
            snap["currency"] = ccy or ("TWD" if yf_symbol.endswith(".TW") else "USD")
        elif not hist.empty:
            p = float(hist["Close"].iloc[-1])
            snap["now_price"] = f"{p:.2f}"
            snap["currency"] = ccy or ("TWD" if yf_symbol.endswith(".TW") else "USD")

        if not hist.empty and len(hist) >= 2 and float(hist["Close"].iloc[-2]) != 0:
            chg = float(hist["Close"].iloc[-1]) - float(hist["Close"].iloc[-2])
            pct = chg / float(hist["Close"].iloc[-2]) * 100
            sign = "+" if chg >= 0 else ""
            snap["change"] = f"{sign}{chg:.2f} ({sign}{pct:.2f}%)"

        if not hist.empty:
            ts = hist.index[-1]
            snap["close_time"] = ts.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    if (not snap["now_price"] or not snap["name"]) and STOCK_ENABLED:
        try:
            ys = YahooStock(yahoo_slug)
            snap["name"] = ys.name or snap["name"] or yahoo_slug
            snap["now_price"] = ys.now_price or snap["now_price"]
            snap["change"] = ys.change or snap["change"]
            snap["currency"] = ys.currency or ("TWD" if yf_symbol.endswith(".TW") else snap["currency"])
            snap["close_time"] = ys.close_time or snap["close_time"]
        except Exception:
            pass
    return snap

stock_data_df: Optional[pd.DataFrame] = None
def load_stock_data() -> pd.DataFrame:
    global stock_data_df
    if stock_data_df is None:
        try:
            stock_data_df = pd.read_csv('name_df.csv')
        except FileNotFoundError:
            stock_data_df = pd.DataFrame(columns=['股號', '股名'])
    return stock_data_df

def get_stock_name(stock_id: str) -> Optional[str]:
    df = load_stock_data()
    res = df[df['股號'].astype(str).str.strip().str.upper() == str(stock_id).strip().upper()]
    if not res.empty:
        return res.iloc[0]['股名']
    return None

def is_stock_query(text: str) -> bool:
    t = text.strip().upper()
    return (
        t in ["台股大盤", "大盤", "美股大盤", "美盤", "美股"]
        or bool(_TW_CODE_RE.match(t))
        or (bool(_US_CODE_RE.match(t)) and t not in ["JPY"])
    )

def get_stock_report(user_input: str) -> str:
    yf_symbol, yahoo_slug, display_code, is_index = normalize_ticker(user_input)
    snapshot = fetch_realtime_snapshot(yf_symbol, yahoo_slug)

    price_data, news_data, value_part, dividend_part = "", "", "", ""
    if STOCK_ENABLED:
        try:
            input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol
            df = stock_price(input_code)
            price_data = str(df) if not df.empty else "N/A"
        except Exception as e:
            price_data = f"Err: {e}"

        try:
            nm = get_stock_name(yahoo_slug) or snapshot.get("name") or yahoo_slug
            nl = stock_news(nm)
            news_data = "\n".join(nl).replace("\u3000", " ")[:1024]
        except Exception as e:
            news_data = f"Err: {e}"

        if not is_index:
            try:
                input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol
                val = stock_fundamental(input_code)
                value_part = f"{val}\n" if val else ""
            except Exception as e:
                value_part = f"Err: {e}\n"
            try:
                input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol
                dvd = stock_dividend(input_code)
                dividend_part = f"{dvd}\n" if dvd else ""
            except Exception as e:
                dividend_part = f"Err: {e}\n"

    stock_link = (
        f"https://finance.yahoo.com/quote/{yf_symbol}"
        if yf_symbol.startswith("^") or not yf_symbol.endswith(".TW")
        else f"https://tw.stock.yahoo.com/quote/{yahoo_slug}"
    )
    content_msg = (
        f"分析報告:\n**代碼:** {display_code}, **名稱:** {snapshot.get('name')}\n"
        f"**價格:** {snapshot.get('now_price')} {snapshot.get('currency')}\n"
        f"**漲跌:** {snapshot.get('change')}\n"
        f"**時間:** {snapshot.get('close_time')}\n"
        f"**近期價:**\n{price_data}\n"
    )
    if value_part:
        content_msg += f"**基本面:**\n{value_part}"
    if dividend_part:
        content_msg += f"**配息:**\n{dividend_part}"
    if news_data:
        content_msg += f"**新聞:**\n{news_data}\n"

    content_msg += f"請寫出 {snapshot.get('name') or display_code} 近期趨勢分析，用繁體中文 Markdown，附連結：{stock_link}"
    system_prompt = "你是專業分析師。開頭列出股名(股號)/現價/漲跌/時間；分段說明走勢/基本面/技術面/消息面/風險/建議區間/停利目標/結論。資料不完整請保守說明。"
    msgs = [{"role":"system","content":system_prompt}, {"role":"user","content":content_msg}]
    return get_analysis_reply(msgs)

# ===================== 8) Lottery =====================
def _lotto_fallback_scrape(kind: str) -> str:
    try:
        if kind == "威力彩":
            url, pat = (
                "https://www.taiwanlottery.com/lotto/superlotto638/index.html",
                r"第\s*\d+\s*期.*?第一區.*?[:：\s]*([\d\s,]+?)\s*第二區.*?[:：\s]*(\d+)"
            )
        elif kind == "大樂透":
            url, pat = (
                "https://www.taiwanlottery.com/lotto/lotto649/index.html",
                r"第\s*\d+\s*期.*?(?:號碼|獎號).*?[:：\s]*([\d\s,]+?)(?:\s*特別號[:：\s]*(\d+))?"
            )
        elif kind == "539":
            url, pat = (
                "https://www.taiwanlottery.com/lotto/dailycash/index.html",
                r"第\s*\d+\s*期.*?(?:號碼|獎號).*?[:：\s]*([\d\s,]+)"
            )
        else:
            return f"不支援: {kind}"

        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        text = ' '.join(soup.stripped_strings)
        m = re.search(pat, text, re.DOTALL)
        if not m:
            return f"抱歉，找不到 {kind} 號碼。"

        if kind == "威力彩":
            first, second = re.sub(r'[,\s]+', ' ', m.group(1)).strip(), m.group(2)
            return f"{kind}: 一區 {first}；二區 {second}"
        elif kind == "大樂透":
            nums, special = re.sub(r'[,\s]+', ' ', m.group(1)).strip(), m.group(2)
            return f"{kind}: {nums}{'；特 ' + special if special else ''}"
        elif kind == "539":
            nums = re.sub(r'[,\s]+', ' ', m.group(1)).strip()
            return f"{kind}: {nums}"
    except Exception:
        return f"抱歉，{kind} 號碼擷取失敗。"

def get_lottery_analysis(lottery_type_input: str) -> str:
    kind = "威力彩" if "威力" in lottery_type_input else ("大樂透" if "大樂" in lottery_type_input else ("539" if "539" in lottery_type_input else lottery_type_input))
    latest_data_str = ""
    if LOTTERY_ENABLED and lottery_crawler:
        try:
            if kind == "威力彩":
                latest_data_str = str(lottery_crawler.super_lotto())
            elif kind == "大樂透":
                latest_data_str = str(lottery_crawler.lotto649())
            elif kind == "539":
                latest_data_str = str(lottery_crawler.daily_cash())
            else:
                return f"不支援 {kind}。"
        except Exception:
            latest_data_str = _lotto_fallback_scrape(kind)
    else:
        latest_data_str = _lotto_fallback_scrape(kind)

    prompt = (
        f"{kind} 近況/號碼：\n{latest_data_str}\n\n"
        "請用繁體中文寫出：\n"
        "1) 走勢重點(熱冷號)\n"
        "2) 選號建議(風險聲明)\n"
        "3) 三組推薦號碼\n"
        "分點條列精煉。"
    )
    messages = [{"role":"system","content":"你是資深彩券分析師。"},{"role":"user","content":prompt}]
    return get_analysis_reply(messages)

# ===================== 9) Gold & FX =====================
def _parse_bot_gold_text(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    text = " ".join(soup.stripped_strings)
    sell = buy = None
    listed_at = None
    m_sell = re.search(r"賣出價.*?([\d,]+\.?\d*)", text)
    m_buy = re.search(r"買入價.*?([\d,]+\.?\d*)", text)
    m_time = re.search(r"(?:掛牌時間|最後更新)[：:]\s*([0-9\/\-\s:]+)", text)

    if m_sell:
        try: sell = float(m_sell.group(1).replace(",", ""))
        except: pass
    if m_buy:
        try: buy = float(m_buy.group(1).replace(",", ""))
        except: pass
    if m_time:
        listed_at = m_time.group(1).strip()

    out = {}
    if sell is not None: out["sell_twd_per_g"] = sell
    if buy is not None: out["buy_twd_per_g"] = buy
    if listed_at: out["listed_at"] = listed_at
    return out

def get_gold_analysis() -> str:
    try:
        r = requests.get(BOT_GOLD_URL, headers=DEFAULT_HEADERS, timeout=10)
        r.raise_for_status()
        data = _parse_bot_gold_text(r.text)
        ts = data.get("listed_at") or "N/A"
        sell = float(data["sell_twd_per_g"])
        buy = float(data["buy_twd_per_g"])
        spread = sell - buy
        bias = "盤整" if spread <= 30 else ("偏寬" if spread <= 60 else "價差大")
        now = datetime.now().strftime("%H:%M")
        return (
            f"**金價（{now}）**\n"
            f"賣: **{sell:,.0f}** | 買: **{buy:,.0f}** | 價差: {spread:,.0f}（{bias}）\n"
            f"掛牌: {ts}\n來源: 台灣銀行"
        )
    except Exception as e:
        logger.error(f"黃金分析失敗：{e}")
        return "抱歉，目前無法取得黃金牌價。"

def get_currency_analysis(target_currency: str) -> str:
    try:
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        if data.get("result") != "success":
            return f"匯率 API 錯誤: {data.get('error-type','未知')}"
        rate = data["rates"].get("TWD")
        if rate is None:
            return "抱歉，API 無 TWD 匯率。"
        return f"即時：1 {target_currency.upper()} ≈ **{rate:.4f}** 新台幣"
    except Exception as e:
        logger.error(f"匯率分析失敗：{e}")
        return "抱歉，外匯資料暫無法取得。"

# ===================== 10) Free TTS (gTTS) =====================
def tts_make_and_upload(text: str, lang_code: str = "zh-TW") -> Optional[Tuple[str, int]]:
    """
    產生語音並上傳 Cloudinary。
    回傳 (url, duration_ms) 或 None
    """
    try:
        # gTTS 支援 zh-TW / ja / en 等
        tts = gTTS(text=text, lang=lang_code)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        # 上傳 cloudinary，設置 resource_type='video' 以支援 m4a/mp3
        up = cloudinary.uploader.upload(
            buf,
            resource_type="video",
            folder="linebot_tts",
            public_id=f"tts_{int(datetime.utcnow().timestamp())}_{random.randint(1000,9999)}",
            overwrite=True
        )
        url = up.get("secure_url")
        if not url:
            logger.error("TTS 上傳失敗：無 secure_url")
            return None
        # gTTS 無精確時長，粗估：每秒約 12～14 字（中文），簡化估 13 字/秒
        est_seconds = max(1, int(len(text) / 13) + 1)
        return url, est_seconds * 1000
    except Exception as e:
        logger.error(f"TTS 產生/上傳失敗：{e}")
        return None

# ===================== 11) Personas & Translate =====================
PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼", "greetings": "親愛的～我在這🌸", "emoji":"🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽", "greetings": "你又來啦？說吧😏", "emoji":"😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣", "greetings": "呀呼～(ﾉ>ω<)ﾉ", "emoji":"✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉", "greetings": "我在。說重點。", "emoji":"🧊⚡️"}
}
PERSONA_ALIAS = {"甜":"sweet", "鹹":"salty", "萌":"moe", "酷":"cool", "random":"random"}

LANGUAGE_MAP = {"英文": "English", "日文": "Japanese", "韓文": "Korean", "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"}

def set_user_persona(chat_id: str, key: str):
    key = random.choice(list(PERSONAS.keys())) if key == "random" else key
    key = "sweet" if key not in PERSONAS else key
    conversation_history.setdefault(chat_id, [])
    conversation_history[chat_id].append({"role":"system","content":f"persona:{key}"})
    return key

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    # 讀取最近一次 persona（預設 sweet）
    persona_key = "sweet"
    for m in reversed(conversation_history.get(chat_id, [])):
        if m.get("role")=="system" and str(m.get("content","")).startswith("persona:"):
            persona_key = m["content"].split(":",1)[1]
            break
    p = PERSONAS.get(persona_key, PERSONAS["sweet"])
    prompt = (
        f"你是「{p['title']}」。風格：{p['style']}\n"
        f"情緒：{sentiment}；調整語氣（開心→同樂；難過/生氣→共情安撫；中性→自然）。\n"
        f"用繁體中文，精煉自然，帶少量表情 {p['emoji']}。"
    )
    return prompt

# ===================== 12) LINE Handlers =====================
@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    body_decoded = body.decode("utf-8")
    if not handler:
        raise HTTPException(status_code=500, detail="Handler not initialized")
    try:
        handler.handle(body_decoded, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except LineBotApiError as lbe:
        logger.error(f"LINE API Error in callback: {lbe.status_code} {lbe.error.message}")
        return JSONResponse({"status": "ok but error logged"})
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse({"status": "ok"})

def _make_sender_for_translate(chat_id: str) -> Optional[Sender]:
    if chat_id in translation_states:
        lang = translation_states[chat_id]
        # 顯示成「翻譯模式(中->英)」等
        mapping = {"英文":"英","日文":"日","繁體中文":"中"}
        tail = mapping.get(lang, lang[:1])
        return Sender(name=f"翻譯模式(中->{tail})")
    return None

@handler.add(MessageEvent, message=TextMessage)
def on_message_text(event: MessageEvent):
    chat_id = get_chat_id(event)
    msg_raw = (event.message.text or "").strip()
    reply_token = event.reply_token

    if not msg_raw:
        return

    # 預設：每個 chat 初次皆關閉 TTS，語言 zh-TW
    tts_enabled.setdefault(chat_id, False)
    tts_lang.setdefault(chat_id, "zh-TW")

    # 翻譯模式頭像名稱
    sender = _make_sender_for_translate(chat_id)

    low = msg_raw.lower()

    # ====== 系統/選單 ======
    if low in ("menu", "選單", "主選單"):
        reply_messages(reply_token, [build_main_menu_flex()], chat_id, sender)
        return

    if low == "tts on":
        tts_enabled[chat_id] = True
        reply_messages(reply_token, [TextSendMessage(text="🎙️ 已開啟語音播報")], chat_id, sender)
        return

    if low == "tts off":
        tts_enabled[chat_id] = False
        reply_messages(reply_token, [TextSendMessage(text="🔇 已關閉語音播報")], chat_id, sender)
        return

    # ====== 功能路由 ======
    if msg_raw in ["大樂透", "威力彩", "539"]:
        report = get_lottery_analysis(msg_raw)
        messages = [TextSendMessage(text=report, sender=sender)]
        # TTS（簡短摘要）
        if tts_enabled.get(chat_id, False):
            tts_res = tts_make_and_upload(re.sub(r"\*\*|\#|\-|\>|\`", "", report)[:240], tts_lang.get(chat_id, "zh-TW"))
            if tts_res:
                url, dur = tts_res
                messages.append(AudioSendMessage(original_content_url=url, duration=dur))
        reply_messages(reply_token, messages, chat_id, sender)
        return

    if low in ("金價", "黃金"):
        out = get_gold_analysis()
        messages = [TextSendMessage(text=out, sender=sender)]
        if tts_enabled.get(chat_id, False):
            tts_res = tts_make_and_upload(re.sub(r"\*\*|\#|\-|\>|\`", "", out), tts_lang.get(chat_id, "zh-TW"))
            if tts_res:
                url, dur = tts_res
                messages.append(AudioSendMessage(original_content_url=url, duration=dur))
        reply_messages(reply_token, messages, chat_id, sender)
        return

    if low == "jpy":
        out = get_currency_analysis("JPY")
        messages = [TextSendMessage(text=out, sender=sender)]
        if tts_enabled.get(chat_id, False):
            tts_res = tts_make_and_upload(out, tts_lang.get(chat_id, "zh-TW"))
            if tts_res:
                url, dur = tts_res
                messages.append(AudioSendMessage(original_content_url=url, duration=dur))
        reply_messages(reply_token, messages, chat_id, sender)
        return

    if is_stock_query(msg_raw):
        report = get_stock_report(msg_raw)
        messages = [TextSendMessage(text=report, sender=sender)]
        if tts_enabled.get(chat_id, False):
            tts_res = tts_make_and_upload(re.sub(r"\*\*|\#|\-|\>|\`", "", report)[:240], tts_lang.get(chat_id, "zh-TW"))
            if tts_res:
                url, dur = tts_res
                messages.append(AudioSendMessage(original_content_url=url, duration=dur))
        reply_messages(reply_token, messages, chat_id, sender)
        return

    if msg_raw.startswith("翻譯->"):
        lang = msg_raw.split("->", 1)[1].strip()
        if lang == "結束":
            translation_states.pop(chat_id, None)
            reply_messages(reply_token, [TextSendMessage(text="✅ 翻譯模式結束")], chat_id, None)
        else:
            translation_states[chat_id] = lang
            reply_messages(reply_token, [TextSendMessage(text=f"🌐 開啟翻譯 → {lang}")], chat_id, _make_sender_for_translate(chat_id))
        return

    if msg_raw in PERSONA_ALIAS:
        key = PERSONA_ALIAS[msg_raw]
        set_user_persona(chat_id, key)
        pkey = "sweet"
        for m in reversed(conversation_history.get(chat_id, [])):
            if m.get("role")=="system" and str(m.get("content","")).startswith("persona:"):
                pkey = m["content"].split(":",1)[1]
                break
        p = PERSONAS[pkey]
        txt = f"💖 切換人設：{p['title']}\n{p['greetings']}"
        reply_messages(reply_token, [TextSendMessage(text=txt)], chat_id, sender)
        return

    # ====== 一般對話（含翻譯） ======
    history = conversation_history.get(chat_id, [])

    # 翻譯模式：把使用者文字翻譯到目標語，再用中文回應（或依你原邏輯）
    if chat_id in translation_states:
        target = translation_states[chat_id]
        sys = "You are a precise translation engine. Output ONLY the translated text, without intro."
        usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{msg_raw}"}}'
        translated = get_analysis_reply([{"role":"system","content":sys},{"role":"user","content":usr}])
        # 顯示翻譯結果
        messages = [TextSendMessage(text=translated, sender=_make_sender_for_translate(chat_id))]
        # TTS：語言跟隨翻譯目標（簡單映射）
        if tts_enabled.get(chat_id, False):
            lang_map = {"英文":"en","日文":"ja","繁體中文":"zh-TW"}
            tts_code = lang_map.get(target, "zh-TW")
            tts_res = tts_make_and_upload(translated, tts_code)
            if tts_res:
                url, dur = tts_res
                messages.append(AudioSendMessage(original_content_url=url, duration=dur))
        reply_messages(reply_token, messages, chat_id, _make_sender_for_translate(chat_id))
        return

    sentiment = analyze_sentiment(msg_raw)
    sys_prompt = build_persona_prompt(chat_id, sentiment)
    messages_ = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg_raw}]
    final_reply = get_analysis_reply(messages_)

    # 更新歷史
    history.extend([{"role":"user","content":msg_raw}, {"role":"assistant","content":final_reply}])
    conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]

    messages = [TextSendMessage(text=final_reply, sender=sender)]
    if tts_enabled.get(chat_id, False):
        tts_res = tts_make_and_upload(final_reply[:600], tts_lang.get(chat_id, "zh-TW"))
        if tts_res:
            url, dur = tts_res
            messages.append(AudioSendMessage(original_content_url=url, duration=dur))
    reply_messages(reply_token, messages, chat_id, sender)

@handler.add(PostbackEvent)
def on_postback(event: PostbackEvent):
    data = (event.postback.data or "").strip()
    kind = data[5:] if data.startswith("menu:") else None
    msgs = []
    if kind:
        msgs.append(build_submenu_flex(kind))
        msgs.append(TextSendMessage(text="請選擇 👇"))
    if msgs:
        reply_messages(event.reply_token, msgs, get_chat_id(event), _make_sender_for_translate(get_chat_id(event)))

# ===================== 13) Health & Root =====================
@router.get("/")
async def root():
    return PlainTextResponse("LINE Bot (FastAPI) running.", status_code=200)

@router.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

@router.get("/health/providers")
async def providers_health():
    return {
        "openai_ok": openai_client is not None,
        "groq_ok": groq_client is not None,
        "line_ok": line_bot_api is not None,
        "ts": datetime.utcnow().isoformat() + "Z",
    }

app.include_router(router)

# ===================== 14) Local Run =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)