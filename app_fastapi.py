# app_fastapi.py  (Full – TTS=gTTS only, no badge line)
# ================================================================
# - LINE Bot SDK v2（同步）
# - taiwanlottery 外部套件（失敗→官網備援）
# - 金價 / 外匯 / 股票：真實抓取
# - TTS：僅 gTTS(zh-tw) 輸出 mp3；Cloudinary 以 raw 上傳
# - 不顯示任何「語音：中文(zh-tw) · 引擎：...」附加行
# - 每則訊息固定帶 Quick Reply（含 TTS ON/OFF）
# - 翻譯模式：sender 顯示「翻譯模式(中->英)」
# ================================================================

import os
import re
import io
import random
import logging
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import httpx
import pandas as pd
import yfinance as yf

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

from groq import Groq
import openai
import uvicorn

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(asctime)s:%(message)s"
)
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# ---------- ENV ----------
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")  # 可接代理
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")

required = {
    "BASE_URL": BASE_URL,
    "CHANNEL_ACCESS_TOKEN": CHANNEL_TOKEN,
    "CHANNEL_SECRET": CHANNEL_SECRET,
    "GROQ_API_KEY": GROQ_API_KEY
}
missing = [k for k, v in required.items() if not v]
if missing:
    raise RuntimeError(f"❌ 缺少必要環境變數：{', '.join(missing)}")

# ---------- LINE v2 ----------
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# ---------- AI（聊天/分析仍可用 OpenAI 或 Groq；TTS 不用 OpenAI） ----------
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")
sync_groq = Groq(api_key=GROQ_API_KEY)

openai_client = None
if OPENAI_API_KEY:
    try:
        if OPENAI_API_BASE:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
            logger.info(f"✅ OpenAI Client (base={OPENAI_API_BASE})")
        else:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            logger.info("✅ OpenAI Client (official)")
    except Exception as e:
        logger.warning(f"OpenAI init failed: {e}")

# ---------- Cloudinary ----------
CLOUDINARY_CONFIGURED = False
cloudinary_uploader = None
if CLOUDINARY_URL:
    try:
        import re as _re
        import cloudinary
        import cloudinary.uploader
        m = _re.search(r"cloudinary://(?P<key>[^:]+):(?P<secret>[^@]+)@(?P<name>.+)", CLOUDINARY_URL)
        cloudinary.config(
            cloud_name=m.group("name"),
            api_key=m.group("key"),
            api_secret=m.group("secret"),
            secure=True
        )
        cloudinary_uploader = cloudinary.uploader
        CLOUDINARY_CONFIGURED = True
        logger.info("✅ Cloudinary 配置成功")
    except Exception as e:
        logger.error(f"Cloudinary 設定失敗：{e}")

# ---------- taiwanlottery ----------
LOTTERY_ENABLED = True
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    lottery_crawler = TaiwanLotteryCrawler()
    logger.info("✅ taiwanlottery 套件已載入")
except Exception as e:
    LOTTERY_ENABLED = False
    lottery_crawler = None
    logger.error(f"❌ 無法載入 taiwanlottery：{e}")

# ---------- 股票模組（若不存在則退化） ----------
STOCK_ENABLED = True
try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
    logger.info("✅ 股票模組載入")
except Exception as e:
    STOCK_ENABLED = False
    logger.error(f"⚠️ 股票模組載入失敗：{e}")

    def stock_price(id): return pd.DataFrame()
    def stock_news(hint): return ["（股票新聞模組未載入）"]
    def stock_fundamental(id): return "（基本面模組未載入）"
    def stock_dividend(id): return "（股利模組未載入）"
    class YahooStock:
        def __init__(self, id):
            self.name = id
            self.now_price = None
            self.change = None
            self.currency = None
            self.close_time = None

# ---------- 狀態 ----------
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
translation_states: Dict[str, str] = {}          # chat_id -> 目標語言（顯示名）
tts_switch: Dict[str, bool] = {}                 # chat_id -> True/False
_DEFAULT_TTS = True

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼", "emoji": "🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽", "emoji": "😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣", "emoji": "✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉", "emoji": "🧊⚡️"},
}
user_persona: Dict[str, str] = {}
PERSONA_ALIAS = {"甜":"sweet","鹹":"salty","萌":"moe","酷":"cool","random":"random"}

LANGUAGE_MAP = {
    "英文":"English","日文":"Japanese","韓文":"Korean","越南文":"Vietnamese",
    "繁體中文":"Traditional Chinese","中文":"Traditional Chinese"
}

# ---------- 常數 ----------
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
}
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"

TTS_LANG = "zh-tw"        # gTTS 語言碼
TTS_OUTPUT_FORMAT = "mp3" # 產出 mp3

_TW_CODE_RE = re.compile(r'^\d{4,6}[A-Za-z]?$')
_US_CODE_RE = re.compile(r'^[A-Za-z]{1,5}$')

# ---------- FastAPI ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 應用啟動（lifespan）")
    if BASE_URL and CHANNEL_TOKEN and CHANNEL_TOKEN != "dummy":
        try:
            async with httpx.AsyncClient() as c:
                headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
                payload = {"endpoint": f"{BASE_URL}/callback"}
                r = await c.put("https://api.line.me/v2/bot/channel/webhook/endpoint",
                                headers=headers, json=payload, timeout=10.0)
                r.raise_for_status()
                logger.info(f"✅ Webhook 更新成功: {r.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Webhook 更新失敗：{e}")
    yield
    logger.info("👋 應用關閉")

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="3.2.0")
router = APIRouter()

# =================== Helpers ===================
def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom): return event.source.room_id
    if isinstance(event.source, SourceUser): return event.source.user_id
    return "unknown"

def _parse_bot_gold_text(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    text = " ".join(soup.stripped_strings)

    sell = None
    buy = None
    listed_at = None

    m_sell = re.search(r"(?:本行賣出|賣出價)\s*([\d,]+\.?\d*)", text)
    m_buy  = re.search(r"(?:本行買進|買入價)\s*([\d,]+\.?\d*)", text)
    m_time = re.search(r"(?:掛牌時間|最後更新)[：:]\s*([0-9\/\-\s:]+)", text)

    if m_sell:
        try: sell = float(m_sell.group(1).replace(",", ""))
        except: pass
    if m_buy:
        try: buy = float(m_buy.group(1).replace(",", ""))
        except: pass
    if m_time: listed_at = m_time.group(1).strip()

    data = {}
    if sell is not None: data["sell_twd_per_g"] = sell
    if buy is not None:  data["buy_twd_per_g"] = buy
    if listed_at:        data["listed_at"] = listed_at
    return data

def normalize_ticker(t: str) -> Tuple[str, str, str, bool]:
    t = t.strip().upper()
    if t in ["台股大盤", "大盤"]: return "^TWII","^TWII","^TWII", True
    if t in ["美股大盤", "美盤", "美股"]: return "^GSPC","^GSPC","^GSPC", True
    if _TW_CODE_RE.match(t): return f"{t}.TW", t, t, False
    if _US_CODE_RE.match(t) and t != "JPY": return t, t, t, False
    return t, t, t, False

def fetch_realtime_snapshot(yf_symbol: str, yahoo_slug: str) -> dict:
    snap: dict = {"name": None, "now_price": None, "change": None, "currency": None, "close_time": None}
    try:
        tk = yf.Ticker(yf_symbol)
        info = {}
        hist = pd.DataFrame()
        try: info = tk.info or {}
        except Exception as e: logger.warning(f"yf.info fail: {e}")
        try: hist = tk.history(period="2d", interval="1d")
        except Exception as e: logger.warning(f"yf.history fail: {e}")

        name = info.get("shortName") or info.get("longName")
        snap["name"] = name or yf_symbol

        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        ccy = info.get("currency")
        if price:
            snap["now_price"] = f"{float(price):.2f}"
            snap["currency"] = ccy or ("TWD" if yf_symbol.endswith(".TW") else "USD")
        elif not hist.empty:
            price = float(hist["Close"].iloc[-1])
            snap["now_price"] = f"{price:.2f}"
            snap["currency"] = ccy or ("TWD" if yf_symbol.endswith(".TW") else "USD")

        if not hist.empty and len(hist) >= 2 and float(hist["Close"].iloc[-2]) != 0:
            chg = float(hist["Close"].iloc[-1]) - float(hist["Close"].iloc[-2])
            pct = chg / float(hist["Close"].iloc[-2]) * 100
            sign = "+" if chg >= 0 else ""
            snap["change"] = f"{sign}{chg:.2f} ({sign}{pct:.2f}%)"
        elif info.get('regularMarketChange') is not None and info.get('regularMarketChangePercent') is not None:
            chg = float(info['regularMarketChange']); pct = float(info['regularMarketChangePercent']) * 100
            sign = "+" if chg >= 0 else ""
            snap["change"] = f"{sign}{chg:.2f} ({sign}{pct:.2f}%)"

        if not hist.empty:
            ts = hist.index[-1]
            snap["close_time"] = ts.strftime("%Y-%m-%d %H:%M")
        elif info.get("regularMarketTime"):
            try:
                snap["close_time"] = datetime.fromtimestamp(info["regularMarketTime"]).strftime("%Y-%m-%d %H:%M")
            except Exception as e:
                logger.warning(f"timestamp parse fail: {e}")
    except Exception as e:
        logger.warning(f"yfinance fail: {e}")

    if (not snap["now_price"] or not snap["name"]) and STOCK_ENABLED and 'YahooStock' in globals():
        try:
            ys = YahooStock(yahoo_slug)
            snap["name"] = ys.name or snap["name"] or yahoo_slug
            snap["now_price"] = ys.now_price or snap["now_price"]
            snap["change"] = ys.change or snap["change"]
            snap["currency"] = ys.currency or ("TWD" if yf_symbol.endswith(".TW") else snap["currency"])
            snap["close_time"] = ys.close_time or snap["close_time"]
        except Exception as e:
            logger.error(f"YahooStock fallback fail: {e}")
    return snap

def is_stock_query(text: str) -> bool:
    t = text.strip().upper()
    return (
        t in ["台股大盤", "大盤", "美股大盤", "美盤", "美股"]
        or bool(_TW_CODE_RE.match(t))
        or (bool(_US_CODE_RE.match(t)) and t not in ["JPY"])
    )

# =================== AI ===================
def get_analysis_reply(messages: List[dict]) -> str:
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
            )
            return resp.choices[0].message.content
    # 若 OpenAI 不可用或失敗則走 Groq
        except Exception as e:
            logger.warning(f"OpenAI 失敗：{e}")

    try:
        resp = sync_groq.chat.completions.create(
            model=GROQ_MODEL_PRIMARY, messages=messages, temperature=0.7, max_tokens=2000
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq Primary 失敗：{e}")
        try:
            resp = sync_groq.chat.completions.create(
                model=GROQ_MODEL_FALLBACK, messages=messages, temperature=0.9, max_tokens=1500
            )
            return resp.choices[0].message.content
        except Exception as e2:
            logger.error(f"Groq Fallback 也失敗：{e2}")
            return "抱歉，AI 分析師目前連線不穩定，請稍後再試。"

def analyze_sentiment(text: str) -> str:
    msgs = [{"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
            {"role":"user","content":text}]
    try:
        resp = sync_groq.chat.completions.create(
            model=GROQ_MODEL_FALLBACK, messages=msgs, max_tokens=10, temperature=0
        )
        s = (resp.choices[0].message.content or "neutral").strip().lower()
        return s if s in ["positive","neutral","negative","angry"] else "neutral"
    except Exception:
        return "neutral"

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    key = user_persona.get(chat_id, "sweet")
    p = PERSONAS.get(key, PERSONAS["sweet"])
    return (f"你是「{p['title']}」。風格：{p['style']}。"
            f"使用繁體中文，簡潔自然，適度表情 {p['emoji']}。"
            f"使用者情緒：{sentiment}。")

def translate_text(text: str, target_lang_display: str) -> str:
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text."
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{text}"}}'
    try:
        resp = sync_groq.chat.completions.create(
            model=GROQ_MODEL_FALLBACK,
            messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
            max_tokens=len(text)*3 + 50,
            temperature=0.2
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"Groq 翻譯失敗：{e}")
        return "抱歉，翻譯功能暫時出錯。"

# =================== TTS（gTTS only） ===================
GTTS_AVAILABLE = False
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    pass

def _tts_gtts_bytes(text: str) -> Optional[bytes]:
    if not GTTS_AVAILABLE:
        return None
    try:
        clean = re.sub(r"[*_`~#]", "", (text or "").strip()) or "嗨，我在這裡。"
        tts = gTTS(text=clean, lang=TTS_LANG, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"gTTS 失敗：{e}")
        return None

def build_tts_audio_bytes(text: str) -> Optional[bytes]:
    return _tts_gtts_bytes(text)

# =================== Quick Reply ===================
def build_quick_reply(chat_id: Optional[str]) -> QuickReply:
    tts_on = tts_switch.get(chat_id, _DEFAULT_TTS)
    tts_on_label = "TTS ON✅" if tts_on else "TTS ON"
    tts_off_label = "TTS OFF" if tts_on else "TTS OFF✅"

    return QuickReply(items=[
        QuickReplyButton(action=MessageAction(label="主選單", text="選單")),
        QuickReplyButton(action=MessageAction(label="台股大盤", text="台股大盤")),
        QuickReplyButton(action=MessageAction(label="美股大盤", text="美股大盤")),
        QuickReplyButton(action=MessageAction(label="黃金價格", text="金價")),
        QuickReplyButton(action=MessageAction(label="查 2330", text="2330")),
        QuickReplyButton(action=MessageAction(label="查 NVDA", text="NVDA")),
        QuickReplyButton(action=MessageAction(label="日圓匯率", text="JPY")),
        QuickReplyButton(action=MessageAction(label=tts_on_label, text="TTS ON")),
        QuickReplyButton(action=MessageAction(label=tts_off_label, text="TTS OFF")),
        QuickReplyButton(action=PostbackAction(label="💖 AI 人設", data="menu:persona")),
        QuickReplyButton(action=PostbackAction(label="🎰 彩票選單", data="menu:lottery")),
        QuickReplyButton(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate")),
    ])

def _ensure_qr_visible(messages: List, chat_id: Optional[str], sender: Optional[Sender]):
    if not messages:
        return
    qr = build_quick_reply(chat_id)
    last = messages[-1]
    if isinstance(last, TextSendMessage):
        if getattr(last, "quick_reply", None) is None:
            last.quick_reply = qr  # type: ignore
    else:
        messages.append(TextSendMessage(text="·", quick_reply=qr, sender=sender))

def reply_messages(reply_token: str, messages: List, chat_id: Optional[str], sender: Optional[Sender]):
    _ensure_qr_visible(messages, chat_id, sender)
    try:
        line_bot_api.reply_message(reply_token, messages)
    except LineBotApiError as lbe:
        logger.error(f"LINE 回覆失敗：{lbe.status_code} {lbe.error.message}")
        try:
            line_bot_api.reply_message(reply_token,
                TextSendMessage(text="抱歉，訊息傳送失敗。", quick_reply=build_quick_reply(chat_id), sender=sender))
        except Exception:
            pass

# =================== 主選單 ===================
def build_main_menu_flex() -> FlexSendMessage:
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical",
                            contents=[TextComponent(text="AI 助理主選單", weight="bold", size="lg")]),
        body=BoxComponent(
            layout="vertical",
            spacing="md",
            contents=[
                SeparatorComponent(margin="md"),
                ButtonComponent(action=PostbackAction(label="💹 金融查詢", data="menu:finance"), style="primary"),
                ButtonComponent(action=PostbackAction(label="🎰 彩票分析", data="menu:lottery"), style="primary"),
                ButtonComponent(action=PostbackAction(label="💖 AI 角色", data="menu:persona"), style="secondary"),
                ButtonComponent(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate"), style="secondary"),
            ]
        )
    )
    return FlexSendMessage(alt_text="主選單", contents=bubble)

def build_submenu_flex(kind: str) -> FlexSendMessage:
    title, buttons = "子選單", []
    if kind == "finance":
        title, buttons = "💹 金融查詢", [
            ButtonComponent(action=MessageAction(label="台股大盤", text="台股大盤")),
            ButtonComponent(action=MessageAction(label="美股大盤", text="美股大盤")),
            ButtonComponent(action=MessageAction(label="黃金價格", text="金價")),
            ButtonComponent(action=MessageAction(label="日圓匯率", text="JPY")),
            ButtonComponent(action=MessageAction(label="查 2330", text="2330")),
            ButtonComponent(action=MessageAction(label="查 NVDA", text="NVDA")),
        ]
    elif kind == "lottery":
        title, buttons = "🎰 彩票分析", [
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
            ButtonComponent(action=MessageAction(label="翻成英文", text="翻譯->英文")),
            ButtonComponent(action=MessageAction(label="翻成日文", text="翻譯->日文")),
            ButtonComponent(action=MessageAction(label="翻成繁中", text="翻譯->繁體中文")),
            ButtonComponent(action=MessageAction(label="結束翻譯", text="翻譯->結束")),
        ]
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical",
                            contents=[TextComponent(text=title, weight="bold", size="lg")]),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm")
    )
    return FlexSendMessage(alt_text=title, contents=bubble)

# =================== 金價 / 外匯 / 股票 ===================
def get_gold_analysis() -> str:
    try:
        r = requests.get(BOT_GOLD_URL, headers=DEFAULT_HEADERS, timeout=10)
        r.raise_for_status()
        data = _parse_bot_gold_text(r.text)
        ts = data.get("listed_at") or "N/A"
        sell = float(data["sell_twd_per_g"])
        buy = float(data["buy_twd_per_g"])
        spread = sell - buy
        return (f"**金價（台灣銀行）**\n"
                f"- 掛牌時間：{ts}\n"
                f"- 賣出(1g)：{sell:,.0f} 元\n"
                f"- 買進(1g)：{buy:,.0f} 元\n"
                f"- 價差：{spread:,.0f} 元\n"
                f"來源：{BOT_GOLD_URL}")
    except Exception as e:
        logger.error(f"金價取得失敗：{e}")
        return "抱歉，目前無法取得金價。"

def get_currency_analysis(target_currency: str) -> str:
    try:
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        if data.get("result") != "success":
            return f"匯率 API 錯誤：{data.get('error-type','未知')}"
        rate = data["rates"].get("TWD")
        if rate is None:
            return "API 沒有回傳 TWD 匯率。"
        return f"即時：1 {target_currency.upper()} ≈ **{rate:.4f}** 新台幣"
    except Exception as e:
        logger.error(f"匯率取得失敗：{e}")
        return "抱歉，外匯資料暫時無法取得。"

def get_stock_report(user_input: str) -> str:
    yf_symbol, yahoo_slug, display_code, is_index = normalize_ticker(user_input)
    snap = fetch_realtime_snapshot(yf_symbol, yahoo_slug)

    price_data, news_data, value_part, dividend_part = "", "", "", ""
    if STOCK_ENABLED:
        try:
            input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol
            price_df = stock_price(input_code)
            price_data = str(price_df) if not price_df.empty else "N/A"
        except Exception as e:
            price_data = f"(price err: {e})"
        try:
            nm = snap.get("name") or yahoo_slug
            news_list = stock_news(nm)
            news_data = "\n".join(news_list).replace("\u3000"," ")[:1024]
        except Exception as e:
            news_data = f"(news err: {e})"
        if not is_index:
            try:
                val = stock_fundamental(input_code)
                value_part = f"{val}\n" if val else ""
            except Exception as e:
                value_part = f"(fund err: {e})\n"
            try:
                dvd = stock_dividend(input_code)
                dividend_part = f"{dvd}\n" if dvd else ""
            except Exception as e:
                dividend_part = f"(div err: {e})\n"

    stock_link = (
        f"https://finance.yahoo.com/quote/{yf_symbol}"
        if yf_symbol.startswith("^") or not yf_symbol.endswith(".TW")
        else f"https://tw.stock.yahoo.com/quote/{yahoo_slug}"
    )
    content_msg = (
        f"分析資料:\n**代碼:** {display_code}, **名稱:** {snap.get('name')}\n"
        f"**價格:** {snap.get('now_price')} {snap.get('currency')}\n"
        f"**漲跌:** {snap.get('change')}\n"
        f"**時間:** {snap.get('close_time')}\n"
        f"**近期價:**\n{price_data}\n"
    )
    if value_part: content_msg += f"**基本面:**\n{value_part}"
    if dividend_part: content_msg += f"**配息:**\n{dividend_part}"
    if news_data: content_msg += f"**新聞:**\n{news_data}\n"

    content_msg += f"請寫出 {snap.get('name') or display_code} 近期趨勢分析，用繁體中文 Markdown，附連結：{stock_link}"
    system_prompt = ("你是專業分析師。開頭列出股名(股號)/現價/漲跌/時間；分段說明走勢/基本面/技術面/消息面/風險/建議區間/停利目標/結論。")
    msgs = [{"role":"system","content":system_prompt}, {"role":"user","content":content_msg}]
    return get_analysis_reply(msgs)

# =================== 彩票 ===================
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
            return f"抱歉，找不到 {kind} 號碼（官方頁面可能改版）。"

        if kind == "威力彩":
            first, second = re.sub(r'[,\s]+', ' ', m.group(1)).strip(), m.group(2)
            return f"{kind}: 一區 {first}；二區 {second}"
        elif kind == "大樂透":
            nums, special = re.sub(r'[,\s]+', ' ', m.group(1)).strip(), m.group(2)
            return f"{kind}: {nums}{'；特 ' + special if special else ''}"
        elif kind == "539":
            nums = re.sub(r'[,\s]+', ' ', m.group(1)).strip()
            return f"{kind}: {nums}"
    except Exception as e:
        logger.error(f"Fallback 彩票爬蟲失敗：{e}")
        return f"抱歉，{kind} 號碼取不到（備援失敗）。"

def get_lottery_analysis(lottery_type_input: str) -> str:
    kind = "威力彩" if "威力" in lottery_type_input else (
        "大樂透" if "大樂" in lottery_type_input else (
            "539" if "539" in lottery_type_input else lottery_type_input
        )
    )

    latest_str = ""
    if LOTTERY_ENABLED and lottery_crawler:
        try:
            if kind == "威力彩":
                latest_str = str(lottery_crawler.super_lotto())
            elif kind == "大樂透":
                latest_str = str(lottery_crawler.lotto649())
            elif kind == "539":
                latest_str = str(lottery_crawler.daily_cash())
            else:
                return f"不支援 {kind}。"
        except Exception as e:
            logger.warning(f"taiwanlottery 失敗，改用備援：{e}")
            latest_str = _lotto_fallback_scrape(kind)
    else:
        latest_str = _lotto_fallback_scrape(kind)

    prompt = (
        f"{kind} 近況/號碼：\n{latest_str}\n\n"
        "請用繁體中文寫出：\n"
        "1) 走勢重點(熱冷號)\n"
        "2) 選號建議(風險聲明)\n"
        "3) 三組推薦號碼\n"
        "分點條列精煉。"
    )
    messages = [{"role":"system","content":"你是資深彩券分析師。"},{"role":"user","content":prompt}]
    return get_analysis_reply(messages)

# =================== 主處理 ===================
def _build_sender_for_chat(chat_id: str) -> Optional[Sender]:
    if chat_id in translation_states:
        lang = translation_states[chat_id]
        arrow = {"英文":"英","日文":"日","繁體中文":"中","韓文":"韓","越南文":"越"}.get(lang, lang)
        return Sender(name=f"翻譯模式(中->{arrow})")
    return None

def _append_tts_if_needed(msgs: List, chat_id: str, text_for_tts: str, sender: Optional[Sender]):
    """若 TTS 開啟，使用 gTTS 產生 mp3 → Cloudinary(raw) → 追加 Audio。"""
    enabled = tts_switch.get(chat_id, _DEFAULT_TTS)
    if not enabled or not CLOUDINARY_CONFIGURED:
        return

    audio_bytes = build_tts_audio_bytes(text_for_tts)
    if not audio_bytes:
        return
    try:
        upload_res = cloudinary_uploader.upload(
            io.BytesIO(audio_bytes),
            resource_type="raw",
            folder="line-bot-tts",
            filename_override="speech.mp3",
            public_id=None,
            overwrite=True,
            unique_filename=True
        )
        url = upload_res.get("secure_url")
        if url:
            # 粗估時長（60ms/字，3s~30s 範圍）
            est = max(3000, min(30000, len(text_for_tts) * 60))
            msgs.append(AudioSendMessage(original_content_url=url, duration=est))
    except Exception as e:
        logger.error(f"TTS 上傳失敗：{e}")

@handler.add(MessageEvent, message=TextMessage)
def on_message_text(event: MessageEvent):
    chat_id = get_chat_id(event)
    msg_raw = (event.message.text or "").strip()
    reply_token = event.reply_token
    if not msg_raw:
        return

    if chat_id not in tts_switch:
        tts_switch[chat_id] = _DEFAULT_TTS

    sender = _build_sender_for_chat(chat_id)
    low = msg_raw.lower()

    # TTS 開/關
    if low == "tts on":
        tts_switch[chat_id] = True
        reply_messages(reply_token, [TextSendMessage(text="🔊 已開啟語音播報", sender=sender)], chat_id, sender)
        return
    if low == "tts off":
        tts_switch[chat_id] = False
        reply_messages(reply_token, [TextSendMessage(text="🔇 已關閉語音播報", sender=sender)], chat_id, sender)
        return

    # 主選單
    if low in ("menu","選單","主選單"):
        reply_messages(reply_token, [build_main_menu_flex()], chat_id, sender)
        return

    # 翻譯模式切換
    if msg_raw.startswith("翻譯->"):
        lang = msg_raw.split("->",1)[1].strip()
        if lang == "結束":
            translation_states.pop(chat_id, None)
            reply_messages(reply_token, [TextSendMessage(text="✅ 已結束翻譯模式")], chat_id, None)
        else:
            translation_states[chat_id] = lang
            sender2 = _build_sender_for_chat(chat_id)
            reply_messages(reply_token, [TextSendMessage(text=f"🌐 開啟翻譯 → {lang}", sender=sender2)], chat_id, sender2)
        return

    # 金價
    if low in ("金價","黃金"):
        text = get_gold_analysis()
        msgs = [TextSendMessage(text=text, sender=sender)]
        _append_tts_if_needed(msgs, chat_id, text, sender)
        reply_messages(reply_token, msgs, chat_id, sender)
        return

    # 外匯（示範 JPY）
    if low == "jpy":
        text = get_currency_analysis("JPY")
        msgs = [TextSendMessage(text=text, sender=sender)]
        _append_tts_if_needed(msgs, chat_id, text, sender)
        reply_messages(reply_token, msgs, chat_id, sender)
        return

    # 彩票
    if msg_raw in ("大樂透","威力彩","539","今彩539"):
        text = get_lottery_analysis("539" if msg_raw=="今彩539" else msg_raw)
        msgs = [TextSendMessage(text=text, sender=sender)]
        _append_tts_if_needed(msgs, chat_id, text, sender)
        reply_messages(reply_token, msgs, chat_id, sender)
        return

    # 股票
    if is_stock_query(msg_raw):
        text = get_stock_report(msg_raw)
        msgs = [TextSendMessage(text=text, sender=sender)]
        _append_tts_if_needed(msgs, chat_id, text, sender)
        reply_messages(reply_token, msgs, chat_id, sender)
        return

    # 翻譯內容
    if chat_id in translation_states:
        out = translate_text(msg_raw, translation_states[chat_id])
        sender2 = _build_sender_for_chat(chat_id)
        msgs = [TextSendMessage(text=out, sender=sender2)]
        _append_tts_if_needed(msgs, chat_id, out, sender2)
        reply_messages(reply_token, msgs, chat_id, sender2)
        return

    # 一般聊天
    try:
        hist = conversation_history.get(chat_id, [])
        sent = analyze_sentiment(msg_raw)
        sys_prompt = build_persona_prompt(chat_id, sent)
        messages = [{"role":"system","content":sys_prompt}] + hist + [{"role":"user","content":msg_raw}]
        final_reply = get_analysis_reply(messages)
        hist.extend([{"role":"user","content":msg_raw},{"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = hist[-MAX_HISTORY_LEN*2:]
        msgs = [TextSendMessage(text=final_reply, sender=sender)]
        _append_tts_if_needed(msgs, chat_id, final_reply, sender)
        reply_messages(reply_token, msgs, chat_id, sender)
    except Exception as e:
        logger.error(f"一般聊天錯誤：{e}")
        reply_messages(reply_token, [TextSendMessage(text="抱歉我剛剛走神了，再說一次讓我補上！", sender=sender)], chat_id, sender)

@handler.add(PostbackEvent)
def on_postback(event: PostbackEvent):
    data = (event.postback.data or "").strip()
    chat_id = get_chat_id(event)
    sender = _build_sender_for_chat(chat_id)
    if data.startswith("menu:"):
        kind = data.split(":",1)[1]
        flex = build_submenu_flex(kind)
        reply_messages(event.reply_token, [flex, TextSendMessage(text="請選擇 👇", sender=sender)], chat_id, sender)
    else:
        reply_messages(event.reply_token, [TextSendMessage(text="收到你的選擇，正在處理中...", sender=sender)], chat_id, sender)

# =================== FastAPI Routes ===================
@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    body_decoded = body.decode("utf-8")
    try:
        handler.handle(body_decoded, signature)
        return JSONResponse({"status":"ok"})
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except LineBotApiError as lbe:
        logger.error(f"LINE API Error in callback: {lbe.status_code} {lbe.error.message}")
        return JSONResponse({"status":"ok-but-error"})
    except Exception as e:
        logger.error(f"Callback error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

@router.get("/")
async def root():
    return PlainTextResponse("LINE Bot running.", status_code=200)

@router.get("/healthz")
async def healthz():
    return {
        "status":"ok",
        "ts": datetime.utcnow().isoformat()+"Z",
        "providers":{
            "groq": sync_groq is not None,
            "openai": openai_client is not None,
            "cloudinary": CLOUDINARY_CONFIGURED,
            "lottery": LOTTERY_ENABLED,
            "stock": STOCK_ENABLED
        }
    }

app.include_router(router)

# =================== Local run ===================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Uvicorn on 0.0.0.0:{port}")
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)