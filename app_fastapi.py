# app_fastapi.py  v1.5.1  (Render-friendly, no-Redis, LINE SDK v3.19 同步 Handler 包裝 async)
# 變更：
# - 回退至 WebhookHandler（同步），用 asyncio.create_task 包裝 async 業務邏輯
# - /callback 不再 await handler.handle(...)；改同步呼叫
# - 其餘：翻譯/選單/股票/金價/JPY 人民幣/AI 人設/行內翻譯 全保留

import os
import re
import io
import random
import logging
import pkg_resources
import asyncio
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

# --- HTTP / 解析 ---
import requests
import httpx
from bs4 import BeautifulSoup

# --- 數據 / 金融 ---
import pandas as pd
import yfinance as yf

# --- FastAPI / LINE SDK v3 ---
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    AudioMessageContent,
    PostbackEvent,
)
# ✅ 使用同步的 WebhookHandler（3.19.0 沒有 AsyncWebhookHandler）
from linebot.v3.webhook import WebhookHandler

from linebot.v3.messaging import (
    Configuration, ApiClient, AsyncMessagingApi, ReplyMessageRequest,
    TextMessage, AudioMessage, ImageMessage, FlexMessage, FlexBubble, FlexBox,
    FlexText, FlexButton, QuickReply, QuickReplyItem, MessageAction, PostbackAction,
    BotInfoResponse,
)

# --- Cloudinary（可選） ---
import cloudinary
import cloudinary.uploader

# --- 語音 TTS（可選） ---
from gtts import gTTS

# --- LLM（Groq/OpenAI 皆可選） ---
from groq import AsyncGroq, Groq
import openai

# --- 圖表（可選） ---
try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
try:
    import mplfinance as mpf
    HAS_MPLFIN = True
except Exception:
    HAS_MPLFIN = False

# ====== 基本設定 ======
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

logger.info("Installed versions:")
for pkg in ["line-bot-sdk","fastapi","uvicorn","groq","openai","requests","pandas",
            "beautifulsoup4","httpx","yfinance","cloudinary","gTTS","matplotlib","mplfinance"]:
    try:
        version = pkg_resources.get_distribution(pkg).version
        logger.info(f"{pkg}: {version}")
    except pkg_resources.DistributionNotFound:
        logger.warning(f"{pkg}: not installed")

BASE_URL = os.getenv("BASE_URL")  # 用於自動更新 LINE Webhook
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto").lower()  # auto/openai/gtts

if not CHANNEL_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("缺少必要環境變數：CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET（LINE）")

# Cloudinary（可選）
if CLOUDINARY_URL:
    try:
        cloudinary.config(
            cloud_name=re.search(r"@(.+)", CLOUDINARY_URL).group(1),
            api_key=re.search(r"//(\d+):", CLOUDINARY_URL).group(1),
            api_secret=re.search(r":([A-Za-z0-9_-]+)@", CLOUDINARY_URL).group(1),
        )
        logger.info("Cloudinary OK")
    except Exception as e:
        logger.error(f"Cloudinary 設定失敗: {e}")
        CLOUDINARY_URL = None

# LINE / LLM
configuration = Configuration(access_token=CHANNEL_TOKEN)
async_api_client = ApiClient(configuration=configuration)
line_bot_api = AsyncMessagingApi(api_client=async_api_client)
handler = WebhookHandler(CHANNEL_SECRET)

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
sync_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.warning(f"初始化 OpenAI 失敗：{e}")

GROQ_MODEL_PRIMARY  = os.getenv("GROQ_MODEL_PRIMARY",  "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

# ====== 狀態 ======
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10

translation_states: Dict[str, str] = {}          # chat_id -> 顯示語名（中文）
translation_states_ttl: Dict[str, datetime] = {} # chat_id -> 到期時間
TRANSLATE_TTL_SECONDS = int(os.getenv("TRANSLATE_TTL_SECONDS", "7200"))  # 2h

auto_reply_status: Dict[str, bool] = {}
user_persona: Dict[str, str] = {}

PERSONAS = {
    "sweet": {"title":"甜美女友","style":"溫柔體貼，鼓勵安慰","greetings":"親愛的～我在這裡聽你說 🌸","emoji":"🌸💕😊"},
    "salty": {"title":"傲嬌女友","style":"機智吐槽，壞壞但有溫度","greetings":"你又來啦？說吧，哪裡卡住了。😏","emoji":"😏🙄"},
    "moe":   {"title":"萌系女友","style":"動漫語氣＋可愛顏文字","greetings":"呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ","emoji":"✨🎀"},
    "cool":  {"title":"酷系御姐","style":"冷靜精煉，關鍵建議","greetings":"我在。說重點。","emoji":"🧊⚡️"},
}
LANGUAGE_MAP = {
    "英文":"English","日文":"Japanese","韓文":"Korean","越南文":"Vietnamese",
    "繁體中文":"Traditional Chinese","中文":"Traditional Chinese",
    "en":"English","ja":"Japanese","jp":"Japanese","ko":"Korean","vi":"Vietnamese","zh":"Traditional Chinese"
}
PERSONA_ALIAS = {"甜":"sweet","鹹":"salty","萌":"moe","酷":"cool","random":"random"}

TRANSLATE_CMD = re.compile(
    r"^(?:翻譯|翻译|翻成)\s*(?:->|→|>)?\s*(英文|English|日文|Japanese|韓文|Korean|越南文|Vietnamese|繁體中文|中文)\s*$",
    re.IGNORECASE
)
INLINE_TRANSLATE = re.compile(
    r"^(en|eng|英文|ja|jp|日文|zh|繁中|中文)\s*[:：>]\s*(.+)$",
    re.IGNORECASE
)

def _now() -> datetime: 
    return datetime.utcnow()

def get_chat_id(event: MessageEvent) -> str:
    source = event.source
    stype = getattr(source, "type", None) or getattr(source, "_type", None)
    uid = getattr(source, "userId", None) or getattr(source, "user_id", None)
    gid = getattr(source, "groupId", None) or getattr(source, "group_id", None)
    rid = getattr(source, "roomId", None)  or getattr(source, "room_id", None)
    try:
        if hasattr(source, "to_dict"):
            d = source.to_dict() or {}
            stype = stype or d.get("type")
            uid = uid or d.get("userId")  or d.get("user_id")
            gid = gid or d.get("groupId") or d.get("group_id")
            rid = rid or d.get("roomId")  or d.get("room_id")
    except Exception:
        pass
    if gid: return f"group:{gid}"
    if rid: return f"room:{rid}"
    if uid: return f"user:{uid}"
    return f"{stype or 'unknown'}:{abs(hash(str(source)))%10_000_000}"

def _tstate_set(chat_id: str, lang_display: str):
    translation_states[chat_id] = lang_display
    translation_states_ttl[chat_id] = _now() + timedelta(seconds=TRANSLATE_TTL_SECONDS)
    logger.info(f"[TranslateMode] SET chat_id={chat_id} -> {lang_display} (ttl={TRANSLATE_TTL_SECONDS}s)")

def _tstate_get(chat_id: str) -> Optional[str]:
    exp = translation_states_ttl.get(chat_id)
    if exp and _now() > exp:
        translation_states.pop(chat_id, None)
        translation_states_ttl.pop(chat_id, None)
        logger.info(f"[TranslateMode] EXPIRE chat_id={chat_id}")
        return None
    return translation_states.get(chat_id)

def _tstate_clear(chat_id: str):
    translation_states.pop(chat_id, None)
    translation_states_ttl.pop(chat_id, None)
    logger.info(f"[TranslateMode] CLEAR chat_id={chat_id}")

def build_quick_reply() -> QuickReply:
    return QuickReply(items=[
        QuickReplyItem(action=MessageAction(label="主選單", text="選單")),
        QuickReplyItem(action=MessageAction(label="台股大盤", text="台股大盤")),
        QuickReplyItem(action=MessageAction(label="美股大盤", text="美股大盤")),
        QuickReplyItem(action=MessageAction(label="黃金價格", text="金價")),
        QuickReplyItem(action=MessageAction(label="查台積電", text="2330")),
        QuickReplyItem(action=MessageAction(label="查輝達", text="NVDA")),
        QuickReplyItem(action=MessageAction(label="查日圓", text="JPY")),
        QuickReplyItem(action=PostbackAction(label="💖 AI 人設", data="menu:persona")),
        QuickReplyItem(action=PostbackAction(label="🎰 彩票選單", data="menu:lottery")),
        QuickReplyItem(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate")),
        QuickReplyItem(action=MessageAction(label="結束翻譯", text="翻譯->結束")),
    ])

def build_main_menu() -> FlexMessage:
    items = [
        ("💹 金融查詢", PostbackAction(label="💹 金融查詢", data="menu:finance")),
        ("🎰 彩票分析", PostbackAction(label="🎰 彩票分析", data="menu:lottery")),
        ("💖 AI 角色扮演", PostbackAction(label="💖 AI 角色扮演", data="menu:persona")),
        ("🌐 翻譯工具", PostbackAction(label="🌐 翻譯工具", data="menu:translate")),
    ]
    buttons = [FlexButton(action=i[1], style="primary" if idx<2 else "secondary") for idx,i in enumerate(items)]
    bubble = FlexBubble(
        header=FlexBox(layout="vertical", contents=[FlexText(text="AI 助理主選單", weight="bold", size="lg")]),
        body=FlexBox(layout="vertical", spacing="md", contents=buttons),
    )
    return FlexMessage(alt_text="主選單", contents=bubble)

def build_submenu(kind: str) -> FlexMessage:
    menus = {
        "finance": ("💹 金融查詢", [
            ("台股大盤", MessageAction(label="台股大盤", text="台股大盤")),
            ("美股大盤", MessageAction(label="美股大盤", text="美股大盤")),
            ("黃金價格", MessageAction(label="黃金價格", text="金價")),
            ("日圓匯率", MessageAction(label="日圓匯率", text="JPY")),
            ("查 2330 台積電", MessageAction(label="查 2330 台積電", text="2330")),
            ("查 NVDA 輝達", MessageAction(label="查 NVDA 輝達", text="NVDA")),
        ]),
        "lottery": ("🎰 彩票分析", [
            ("大樂透", MessageAction(label="大樂透", text="大樂透")),
            ("威力彩", MessageAction(label="威力彩", text="威力彩")),
            ("今彩539", MessageAction(label="今彩539", text="539")),
        ]),
        "persona": ("💖 AI 角色扮演", [
            ("甜美女友", MessageAction(label="甜美女友", text="甜")),
            ("傲嬌女友", MessageAction(label="傲嬌女友", text="鹹")),
            ("萌系女友", MessageAction(label="萌系女友", text="萌")),
            ("酷系御姐", MessageAction(label="酷系御姐", text="酷")),
            ("隨機切換", MessageAction(label="隨機切換", text="random")),
        ]),
        "translate": ("🌐 翻譯工具", [
            ("翻成英文", MessageAction(label="翻成英文", text="翻譯->英文")),
            ("翻成日文", MessageAction(label="翻成日文", text="翻譯->日文")),
            ("翻成繁中", MessageAction(label="翻成繁中", text="翻譯->繁體中文")),
            ("結束翻譯模式", MessageAction(label="結束翻譯模式", text="翻譯->結束")),
        ]),
    }
    title, items = menus.get(kind, ("無效選單", []))
    rows, row = [], []
    for _, action in items:
        row.append(FlexButton(action=action, style="primary"))
        if len(row)==2:
            rows.append(FlexBox(layout="horizontal", spacing="sm", contents=row)); row=[]
    if row: rows.append(FlexBox(layout="horizontal", spacing="sm", contents=row))
    bubble = FlexBubble(
        header=FlexBox(layout="vertical", contents=[FlexText(text=title, weight="bold", size="lg")]),
        body=FlexBox(layout="vertical", spacing="md", contents=rows or [FlexText(text="（尚無項目）")]),
    )
    return FlexMessage(alt_text=title, contents=bubble)

async def reply_text_with_tts_and_extras(reply_token: str, text: str, extras: Optional[List]=None):
    if not text: text = "（無內容）"
    messages = [TextMessage(text=text, quick_reply=build_quick_reply())]
    if extras: messages.extend(extras)
    if CLOUDINARY_URL:
        try:
            audio_bytes = await text_to_speech_async(text)
            if audio_bytes:
                res = await run_in_threadpool(lambda: cloudinary.uploader.upload(
                    io.BytesIO(audio_bytes), resource_type="video", folder="line-bot-tts", format="mp3"))
                url = res.get("secure_url")
                if url:
                    est = max(3000, min(30000, len(text)*60))
                    messages.append(AudioMessage(original_content_url=url, duration=est))
        except Exception as e:
            logger.warning(f"TTS 附加失敗：{e}")
    await line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=messages))

# ====== LLM 包裝 ======
def get_analysis_reply(messages: List[dict]) -> str:
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=1500
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI 失敗：{e}")
    if sync_groq_client:
        try:
            resp = sync_groq_client.chat.completions.create(
                model=GROQ_MODEL_PRIMARY, messages=messages, temperature=0.7, max_tokens=2000
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq 主模型失敗：{e}")
            resp = sync_groq_client.chat.completions.create(
                model=GROQ_MODEL_FALLBACK, messages=messages, temperature=0.9, max_tokens=1500
            )
            return resp.choices[0].message.content
    return "（LLM 不可用）"

async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
    if not async_groq_client:
        return await run_in_threadpool(lambda: get_analysis_reply(messages))
    resp = await async_groq_client.chat.completions.create(
        model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    return resp.choices[0].message.content.strip()

async def analyze_sentiment(text: str) -> str:
    msgs = [{"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
            {"role":"user","content":text}]
    try:
        out = await groq_chat_async(msgs, max_tokens=10, temperature=0)
        return (out or "neutral").strip().lower()
    except Exception:
        return "neutral"

async def translate_text(text: str, target_lang_display: str) -> str:
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text with no extra words."
    clean = re.sub(r"[\u200B-\u200D\uFEFF]", "", text).strip()
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{clean}"}}'
    return await groq_chat_async([{"role":"system","content":sys},{"role":"user","content":usr}], 800, 0.2)

def set_user_persona(chat_id: str, key: str):
    key_mapped = PERSONA_ALIAS.get(key, key)
    if key_mapped == "random": key_mapped = random.choice(list(PERSONAS.keys()))
    if key_mapped not in PERSONAS: key_mapped = "sweet"
    user_persona[chat_id] = key_mapped
    return key_mapped

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    key = user_persona.get(chat_id, "sweet"); p = PERSONAS[key]
    return (f"你是一位「{p['title']}」。風格：{p['style']}。\n"
            f"使用者情緒：{sentiment}。\n"
            f"回覆請精煉自然，使用繁體中文，帶少量表情 {p['emoji']}.")

# ====== 金價/匯率/股票 ======
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"
DEFAULT_HEADERS = {"User-Agent":"Mozilla/5.0","Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}

def parse_bot_gold_text(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser"); text = soup.get_text(" ", strip=True)
    m_time = re.search(r"掛牌時間[:：]\s*([0-9]{4}/[0-9]{2}/[0-9]{2}\s+[0-9]{2}:[0-9]{2})", text)
    listed_at = m_time.group(1) if m_time else None
    m_sell = re.search(r"本行賣出\s*([0-9,]+(?:\.[0-9]+)?)", text)
    m_buy  = re.search(r"本行買進\s*([0-9,]+(?:\.[0-9]+)?)", text)
    if not (m_sell and m_buy): raise RuntimeError("找不到『本行賣出/本行買進』欄位")
    sell = float(m_sell.group(1).replace(",","")); buy = float(m_buy.group(1).replace(",",""))
    return {"listed_at":listed_at,"sell_twd_per_g":sell,"buy_twd_per_g":buy,"source":BOT_GOLD_URL}

def get_bot_gold_quote() -> str:
    try:
        r = requests.get(BOT_GOLD_URL, headers=DEFAULT_HEADERS, timeout=10); r.raise_for_status()
        data = parse_bot_gold_text(r.text)
        sell, buy = data["sell_twd_per_g"], data["buy_twd_per_g"]
        spread = sell - buy
        ts = data.get("listed_at") or "（頁面未標示）"
        return (f"**金價（台灣銀行）**\n- 掛牌時間：{ts}\n- 賣出(1g)：{sell:,.0f} 元\n- 買進(1g)：{buy:,.0f} 元\n"
                f"- 價差：{spread:,.0f} 元\n來源：{BOT_GOLD_URL}")
    except Exception as e:
        logger.error(f"金價流程失敗：{e}", exc_info=True)
        return "抱歉，目前無法取得金價。來源：https://rate.bot.com.tw/gold?Lang=zh-TW"

def get_fx_quote(base="JPY", quote="TWD") -> str:
    pair = f"{base}{quote}=X"
    try:
        tk = yf.Ticker(pair)
        info = getattr(tk, "fast_info", None)
        last = None
        if info and hasattr(info, "last_price"):
            last = info.last_price
        if not last:
            hist = tk.history(period="2d", interval="1d")
            if not hist.empty:
                last = float(hist["Close"].iloc[-1])
        if last:
            return f"即時近似：1 {base} ≈ {last:.5f} {quote}（資料源：Yahoo Finance）\nhttps://finance.yahoo.com/quote/{pair}"
    except Exception as e:
        logger.warning(f"yfinance FX 失敗：{e}")
    try:
        url = f"https://open.er-api.com/v6/latest/{base}"
        r = requests.get(url, timeout=10); r.raise_for_status()
        js = r.json()
        if js.get("result") == "success" and quote in js.get("rates", {}):
            rate = js["rates"][quote]
            return f"即時（API）：1 {base} ≈ {rate:.5f} {quote}\nhttps://open.er-api.com/v6/latest/{base}"
    except Exception as e:
        logger.error(f"ER-API 失敗：{e}")
    return "抱歉，外匯資料暫時無法取得。來源：https://finance.yahoo.com/ 、https://open.er-api.com/"

_TW_CODE_RE = re.compile(r'^\d{4,6}[A-Za-z]?$')
_US_CODE_RE = re.compile(r'^[A-Za-z]{1,5}$')

def normalize_ticker(user_text: str) -> Tuple[str, str, str, bool]:
    t = user_text.strip().upper()
    if t in ["台股大盤", "大盤", "^TWII"]:
        return "^TWII", "^TWII", "^TWII", True
    if t in ["美股大盤", "美股", "^GSPC", "SPX"]:
        return "^GSPC", "^GSPC", "^GSPC", True
    if _TW_CODE_RE.match(t):
        return f"{t}.TW", t, t, False
    if _US_CODE_RE.match(t) and t not in ["JPY"]:
        return t, t, t, False
    return t, t, t, False

def fetch_snapshot(yf_symbol: str) -> dict:
    snap: dict = {"name": None, "now": None, "chg": None, "ccy": None, "close_time": None}
    try:
        tk = yf.Ticker(yf_symbol)
        info = getattr(tk, "fast_info", None)
        hist = tk.history(period="2d", interval="1d")
        name = None
        try:
            name = tk.get_info().get("shortName")
        except Exception:
            pass
        snap["name"] = name or yf_symbol
        price, ccy = None, None
        if info and getattr(info, "last_price", None):
            price = info.last_price
            ccy = getattr(info, "currency", None)
        elif not hist.empty:
            price = float(hist["Close"].iloc[-1])
        if price:
            snap["now"] = f"{price:.2f}"
            snap["ccy"] = ccy or ("TWD" if yf_symbol.endswith(".TW") else "USD")
        if not hist.empty and len(hist) >= 2:
            chg = float(hist["Close"].iloc[-1]) - float(hist["Close"].iloc[-2])
            pct = chg / float(hist["Close"].iloc[-2]) * 100 if hist["Close"].iloc[-2] else 0.0
            sign = "+" if chg >= 0 else "-"
            snap["chg"] = f"{sign}{abs(chg):.2f} ({sign}{abs(pct):.2f}%)"
        if not hist.empty:
            ts = hist.index[-1]
            snap["close_time"] = ts.strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        logger.warning(f"yfinance snapshot 失敗：{e}")
    return snap

def generate_stock_chart_png(yf_symbol: str, period: str = "6mo", interval: str = "1d") -> Optional[bytes]:
    if not HAS_MPL:
        return None
    try:
        df = yf.download(yf_symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        buf = io.BytesIO()
        if HAS_MPLFIN:
            mpf.plot(df, type="candle", mav=(5, 20, 60), volume=True, style="yahoo",
                     tight_layout=True, savefig=dict(fname=buf, format="png"))
        else:
            plt.figure(figsize=(9, 5), dpi=200)
            plt.plot(df.index, df["Close"], label="Close")
            for w in (5, 20, 60):
                plt.plot(df.index, df["Close"].rolling(w).mean(), label=f"MA{w}")
            plt.title(f"{yf_symbol} Close & MAs"); plt.legend(); plt.tight_layout()
            plt.savefig(buf, format="png"); plt.close()
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.warning(f"生成股票圖失敗：{e}")
        return None

async def upload_image_to_cloudinary(image_bytes: bytes) -> Optional[str]:
    if not CLOUDINARY_URL: return None
    try:
        res = await run_in_threadpool(lambda: cloudinary.uploader.upload(
            io.BytesIO(image_bytes), resource_type="image", folder="line-bot-chart", format="png"
        ))
        return res.get("secure_url")
    except Exception as e:
        logger.error(f"Cloudinary 上傳圖片失敗: {e}")
        return None

async def get_stock_chart_url_async(user_input: str) -> Optional[str]:
    yf_symbol, _, _, _ = normalize_ticker(user_input)
    img = await run_in_threadpool(generate_stock_chart_png, yf_symbol)
    if not img:
        return None
    return await upload_image_to_cloudinary(img)

def build_stock_report(user_input: str) -> str:
    yf_symbol, _, display, _ = normalize_ticker(user_input)
    snap = fetch_snapshot(yf_symbol)
    title = snap.get("name") or display
    now  = snap.get("now") or "—"
    chg  = snap.get("chg") or "—"
    tstr = snap.get("close_time") or "—"
    link = f"https://finance.yahoo.com/quote/{yf_symbol}"
    return "\n".join([
        f"**{title}（{display}）**",
        f"- 現價：{now} {snap.get('ccy','')}",
        f"- 漲跌：{chg}",
        f"- 時間：{tstr}",
        f"更多：{link}",
    ])

def _create_tts_openai_sync(text: str) -> Optional[bytes]:
    if not openai_client: return None
    try:
        clean = re.sub(r"[*_`~#]", "", text)
        resp = openai_client.audio.speech.create(model="tts-1", voice="nova", input=clean)
        return resp.read()
    except Exception as e:
        logger.error(f"OpenAI TTS 失敗: {e}")
        return None

def _create_tts_gtts_sync(text: str) -> Optional[bytes]:
    try:
        clean = re.sub(r"[*_`~#]", "", text).strip() or "嗨，我在這裡。"
        tts = gTTS(text=clean, lang="zh-TW", tld="com.tw", slow=False)
        buf = io.BytesIO(); tts.write_to_fp(buf); buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"gTTS 失敗: {e}")
        return None

async def text_to_speech_async(text: str) -> Optional[bytes]:
    provider = TTS_PROVIDER
    if provider == "openai":
        return await run_in_threadpool(_create_tts_openai_sync, text)
    if provider == "gtts":
        return await run_in_threadpool(_create_tts_gtts_sync, text)
    if openai_client:
        b = await run_in_threadpool(_create_tts_openai_sync, text)
        if b: return b
    return await run_in_threadpool(_create_tts_gtts_sync, text)

# ====== 事件處理（用同步 wrapper -> async 邏輯） ======
# -- Text --
@handler.add(MessageEvent, message=TextMessageContent)
def _on_text_message(event: MessageEvent):
    asyncio.create_task(handle_text_message_async(event))

async def handle_text_message_async(event: MessageEvent):
    chat_id   = get_chat_id(event)
    msg_raw   = (event.message.text or "").strip()
    reply_tok = event.reply_token
    if not msg_raw: return

    try:
        bot_info: BotInfoResponse = await line_bot_api.get_bot_info()
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if chat_id not in auto_reply_status: auto_reply_status[chat_id] = True
    is_group_or_room = getattr(event.source, "type", "") in ("group","room")
    if is_group_or_room and not auto_reply_status.get(chat_id, True) and not msg_raw.startswith(f"@{bot_name}"):
        return

    msg = msg_raw
    if msg_raw.startswith(f"@{bot_name}"):
        msg = re.sub(f'^@{re.escape(bot_name)}\\s*','',msg_raw).strip()
    if not msg: return

    # 選單
    if msg in ("選單","主選單","menu","Menu"):
        await line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_tok, messages=[build_main_menu()]))
        return

    # 翻譯開關
    m = TRANSLATE_CMD.match(msg)
    if m:
        lang_token = m.group(1)
        rev = {"English":"英文","Japanese":"日文","Korean":"韓文","Vietnamese":"越南文","繁體中文":"繁體中文","中文":"繁體中文"}
        _tstate_set(chat_id, rev.get(lang_token, lang_token))
        await reply_text_with_tts_and_extras(reply_tok, f"🌐 已開啟翻譯 → {translation_states[chat_id]}，請直接輸入要翻的內容。")
        return

    if msg.startswith("翻譯->"):
        lang = msg.split("->",1)[1].strip()
        if lang == "結束":
            _tstate_clear(chat_id)
            await reply_text_with_tts_and_extras(reply_tok, "✅ 已結束翻譯模式")
        else:
            _tstate_set(chat_id, lang)
            await reply_text_with_tts_and_extras(reply_tok, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")
        return

    # 行內一次性翻譯
    im = INLINE_TRANSLATE.match(msg)
    if im:
        lang_key, text_to_translate = im.group(1).lower(), im.group(2)
        lang_display = {
            "en":"英文","eng":"英文","英文":"英文",
            "ja":"日文","jp":"日文","日文":"日文",
            "zh":"繁體中文","繁中":"繁體中文","中文":"繁體中文",
        }.get(lang_key, "英文")
        out = await translate_text(text_to_translate, lang_display)
        await reply_text_with_tts_and_extras(reply_tok, out)
        return

    # 翻譯模式最高優先
    current_lang = _tstate_get(chat_id)
    if current_lang:
        try:
            out = await translate_text(msg, current_lang)
            await reply_text_with_tts_and_extras(reply_tok, out)
        except Exception as e:
            logger.error(f"翻譯失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_tok, "抱歉，翻譯目前不可用。")
        return

    # 金價
    low = msg.lower()
    if low in ("金價","黃金","gold"):
        await reply_text_with_tts_and_extras(reply_tok, get_bot_gold_quote()); return

    # 日圓（觸發放寬）
    if low in ("jpy","日圓匯率","日圓","日幣","日元","yen"):
        await reply_text_with_tts_and_extras(reply_tok, get_fx_quote("JPY","TWD")); return

    # 股票/指數（寬鬆）
    if re.fullmatch(r"\^?[A-Z0-9.]{2,10}", msg) or msg.isdigit() or msg in ("台股大盤","美股大盤","大盤","美股"):
        report = build_stock_report(msg)
        extras = []
        try:
            chart_url = await get_stock_chart_url_async(msg)
            if chart_url:
                extras.append(ImageMessage(original_content_url=chart_url, preview_image_url=chart_url))
        except Exception as ce:
            logger.warning(f"附圖失敗（忽略）：{ce}")
        await reply_text_with_tts_and_extras(reply_tok, report, extras=extras)
        return

    # 人設
    if msg in PERSONA_ALIAS or low in PERSONA_ALIAS:
        key = PERSONA_ALIAS.get(msg, PERSONA_ALIAS.get(low, "sweet"))
        set_user_persona(chat_id, key)
        p = PERSONAS[user_persona[chat_id]]
        await reply_text_with_tts_and_extras(reply_tok, f"💖 已切換人設：{p['title']}\n\n{p['greetings']}")
        return

    # 彩票（示範）
    if msg in ("大樂透","威力彩","539","今彩539"):
        await reply_text_with_tts_and_extras(reply_tok, f"🎰 {msg} 功能示範版：暫提供趨勢建議，請以官方公告為準。\nhttps://www.taiwanlottery.com.tw/")
        return

    # 一般聊天
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]
        final_reply = await groq_chat_async(messages)
        history.extend([{"role":"user","content":msg},{"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
        await reply_text_with_tts_and_extras(reply_tok, final_reply)
    except Exception as e:
        logger.error(f"聊天回覆失敗: {e}", exc_info=True)
        await reply_text_with_tts_and_extras(reply_tok, "抱歉我剛剛走神了 😅 再說一次讓我補上！")

# -- Audio --
@handler.add(MessageEvent, message=AudioMessageContent)
def _on_audio_message(event: MessageEvent):
    asyncio.create_task(handle_audio_message_async(event))

async def handle_audio_message_async(event: MessageEvent):
    await reply_text_with_tts_and_extras(event.reply_token, "🎧 語音收到！目前此 demo 未開啟語音轉文字。")

# -- Postback --
@handler.add(PostbackEvent)
def _on_postback(event: PostbackEvent):
    asyncio.create_task(handle_postback_async(event))

async def handle_postback_async(event: PostbackEvent):
    data = event.postback.data or ""
    if data.startswith("menu:"):
        kind = data.split(":",1)[-1]
        await line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=event.reply_token, messages=[build_submenu(kind)])
        )

# ====== FastAPI ======
@asynccontextmanager
async def lifespan(app: FastAPI):
    if BASE_URL:
        async with httpx.AsyncClient() as c:
            for endpoint in ("https://api-data.line.me/v2/bot/channel/webhook/endpoint",
                             "https://api.line.me/v2/bot/channel/webhook/endpoint"):
                try:
                    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type":"application/json"}
                    payload = {"endpoint": f"{BASE_URL}/callback"}
                    r = await c.put(endpoint, headers=headers, json=payload, timeout=10.0)
                    r.raise_for_status()
                    logger.info(f"Webhook 更新成功: {endpoint} {r.status_code}")
                    break
                except Exception as e:
                    logger.warning(f"Webhook 更新失敗：{e}")
    yield

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.5.1")
router = APIRouter()

@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        # ✅ WebhookHandler 是同步函式，不能 await
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Callback 失敗：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse({"status":"ok"})

@router.get("/")
async def root():
    return PlainTextResponse("LINE Bot is running.", status_code=200)

@router.get("/healthz")
async def healthz():
    return PlainTextResponse("ok", status_code=200)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)