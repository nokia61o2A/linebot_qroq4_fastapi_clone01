# app_fastapi.py v1.4.6 (Async-native handler with StockGPT restored, TTS/STT, and quick menus)
# 變更摘要：
# - [NEW] 恢復並整合「台股大盤／美盤／個股代碼」專業分析（StockGPT）指令流
# - [NEW] 加入嚴謹的股號/美股代碼偵測與路由；一旦命中即「非閒聊」，直接輸出專業分析
# - [NEW] 兼容你既有 my_commands.stock.* 模組：price/news/fundamental/dividend/YahooStock
# - [NEW] 以「大盤/美盤/數字代碼/字母代碼」驅動，產生 markdown 報告 + 正確連結（Yahoo Finance）
# - [CHANGED] TTS/STT 流程保留；加入錯誤保護，避免擋住主流程
# - [CHANGED] QuickReply 增補金融常用鍵；維持原「大樂透」選單
# - [NEW] 以環境變數切換 OpenAI/Groq/gTTS；並保留 auto fallback
# - [NEW] 重要新增或修改處均以 # [NEW]/# [CHANGED] 註解
# - [CHANGED] v1.4.6：所有回覆（含 FlexMessage / AudioMessage）一律附上 Quick Reply（包含「主選單」）

import os
import re
import io
import random
import logging
import pkg_resources
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

# --- HTTP / 解析 ---
import requests
import httpx
from bs4 import BeautifulSoup

# --- 資料處理 / 金融 ---
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
from linebot.v3.webhook import WebhookParser
from linebot.v3.messaging import (
    Configuration, ApiClient, AsyncMessagingApi, ReplyMessageRequest,
    TextMessage, AudioMessage, ImageMessage, FlexMessage, FlexBubble, FlexBox,
    FlexText, FlexButton, QuickReply, QuickReplyItem, MessageAction, PostbackAction,
    BotInfoResponse, PushMessageRequest,
)

# --- Cloudinary（可選） ---
import cloudinary
import cloudinary.uploader

# --- 語音 TTS/STT（可選） ---
from gtts import gTTS

# --- LLM ---
from groq import AsyncGroq, Groq
import openai

# ====== 你既有的股票分析模組（沿用） ======
# [NEW]：以下模組需存在於你的專案目錄 my_commands/stock 下，與你貼上的版本一致
from my_commands.stock.stock_price import stock_price
from my_commands.stock.stock_news import stock_news
from my_commands.stock.stock_value import stock_fundamental
from my_commands.stock.stock_rate import stock_dividend
from my_commands.stock.YahooStock import YahooStock

# --- Matplotlib（可選） ---
try:
    import matplotlib
    matplotlib.use("Agg")
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
for pkg in ["line-bot-sdk", "fastapi", "uvicorn", "groq", "openai", "requests", "pandas",
            "beautifulsoup4", "httpx", "yfinance", "cloudinary", "gTTS", "matplotlib", "mplfinance"]:
    try:
        version = pkg_resources.get_distribution(pkg).version
        logger.info(f"{pkg}: {version}")
    except pkg_resources.DistributionNotFound:
        logger.warning(f"{pkg}: not installed")

BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto").lower()  # auto/openai/gtts

if not CHANNEL_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("缺少必要環境變數：CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET")

# Cloudinary 可選
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
parser = WebhookParser(CHANNEL_SECRET)

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
sync_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.warning(f"初始化 OpenAI 失敗：{e}")

# LLM 模型（聊天用途；股市分析本身不依賴 LLM 也可運行，只用於文字組織）
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

# ====== 狀態 ======
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10

translation_states: Dict[str, str] = {}
translation_states_ttl: Dict[str, datetime] = {}
TRANSLATE_TTL_SECONDS = int(os.getenv("TRANSLATE_TTL_SECONDS", "7200"))  # 2h

auto_reply_status: Dict[str, bool] = {}
user_persona: Dict[str, str] = {}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji": "🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji": "😏🙄"},
    "moe": {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji": "✨🎀"},
    "cool": {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議", "greetings": "我在。說重點。", "emoji": "🧊⚡️"},
}
LANGUAGE_MAP = {
    "英文": "English", "日文": "Japanese", "韓文": "Korean", "越南文": "Vietnamese",
    "繁體中文": "Traditional Chinese", "中文": "Traditional Chinese",
    "en": "English", "ja": "Japanese", "jp": "Japanese", "ko": "Korean", "vi": "Vietnamese", "zh": "Traditional Chinese"
}
PERSONA_ALIAS = {"甜": "sweet", "鹹": "salty", "萌": "moe", "酷": "cool", "random": "random"}

TRANSLATE_CMD = re.compile(
    r"^(?:翻譯|翻译|翻成)\s*(?:->|→|>)?\s*(英文|English|日文|Japanese|韓文|Korean|越南文|Vietnamese|繁體中文|中文)\s*$",
    re.IGNORECASE
)
INLINE_TRANSLATE = re.compile(
    r"^(en|eng|英文|ja|jp|日文|zh|繁中|中文)\s*[:：>]\s*(.+)$",
    re.IGNORECASE
)

# ====== 小工具 ======
def _now() -> datetime:
    return datetime.utcnow()

def get_chat_id(event: MessageEvent) -> str:
    source = event.source
    stype = getattr(source, "type", None) or getattr(source, "_type", None)
    uid = getattr(source, "userId", None) or getattr(source, "user_id", None)
    gid = getattr(source, "groupId", None) or getattr(source, "group_id", None)
    rid = getattr(source, "roomId", None) or getattr(source, "room_id", None)
    try:
        if hasattr(source, "to_dict"):
            d = source.to_dict() or {}
            stype = stype or d.get("type")
            uid = uid or d.get("userId") or d.get("user_id")
            gid = gid or d.get("groupId") or d.get("group_id")
            rid = rid or d.get("roomId") or d.get("room_id")
    except Exception:
        pass
    if gid:
        return f"group:{gid}"
    if rid:
        return f"room:{rid}"
    if uid:
        return f"user:{uid}"
    key_fallback = f"{stype or 'unknown'}:{abs(hash(str(source))) % 10_000_000}"
    return key_fallback

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
    val = translation_states.get(chat_id)
    logger.info(f"[TranslateMode] GET chat_id={chat_id} -> {val}")
    return val

def _tstate_clear(chat_id: str):
    translation_states.pop(chat_id, None)
    translation_states_ttl.pop(chat_id, None)
    logger.info(f"[TranslateMode] CLEAR chat_id={chat_id}")

def build_quick_reply() -> QuickReply:
    # [CHANGED] 增加台股/美股/金價等常用指令
    return QuickReply(items=[
        QuickReplyItem(action=MessageAction(label="主選單", text="選單")),
        QuickReplyItem(action=MessageAction(label="台股大盤", text="大盤")),
        QuickReplyItem(action=MessageAction(label="美股大盤", text="美盤")),
        QuickReplyItem(action=MessageAction(label="黃金價格", text="金價")),
        QuickReplyItem(action=MessageAction(label="查 2330", text="2330")),
        QuickReplyItem(action=MessageAction(label="查 NVDA", text="NVDA")),
        QuickReplyItem(action=MessageAction(label="日圓匯率", text="JPY")),
        QuickReplyItem(action=MessageAction(label="大樂透", text="大樂透")),
        QuickReplyItem(action=PostbackAction(label="💖 AI 人設", data="menu:persona")),
        QuickReplyItem(action=PostbackAction(label="🎰 彩票選單", data="menu:lottery")),
        QuickReplyItem(action=MessageAction(label="結束翻譯", text="翻譯->結束")),
    ])

def build_main_menu() -> FlexMessage:
    items = [
        ("💹 金融查詢", PostbackAction(label="💹 金融查詢", data="menu:finance")),
        ("🎰 彩票分析", PostbackAction(label="🎰 彩票分析", data="menu:lottery")),
        ("💖 AI 角色扮演", PostbackAction(label="💖 AI 角色扮演", data="menu:persona")),
        ("🌐 翻譯工具", PostbackAction(label="🌐 翻譯工具", data="menu:translate")),
    ]
    buttons = [FlexButton(action=i[1], style="primary" if idx < 2 else "secondary") for idx, i in enumerate(items)]
    bubble = FlexBubble(
        header=FlexBox(layout="vertical", contents=[FlexText(text="AI 助理主選單", weight="bold", size="lg")]),
        body=FlexBox(layout="vertical", spacing="md", contents=buttons),
    )
    # [CHANGED] v1.4.6：FlexMessage 也加上 Quick Reply（含主選單）
    return FlexMessage(alt_text="主選單", contents=bubble, quick_reply=build_quick_reply())

def build_submenu(kind: str) -> FlexMessage:
    menus = {
        "finance": ("💹 金融查詢", [
            ("台股大盤", MessageAction(label="台股大盤", text="大盤")),
            ("美股大盤", MessageAction(label="美股大盤", text="美盤")),
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
        if len(row) == 2:
            rows.append(FlexBox(layout="horizontal", spacing="sm", contents=row))
            row = []
    if row:
        rows.append(FlexBox(layout="horizontal", spacing="sm", contents=row))
    bubble = FlexBubble(
        header=FlexBox(layout="vertical", contents=[FlexText(text=title, weight="bold", size="lg")]),
        body=FlexBox(layout="vertical", spacing="md", contents=rows or [FlexText(text="（尚無項目）")]),
    )
    # [CHANGED] v1.4.6：Flex 子選單也帶 Quick Reply
    return FlexMessage(alt_text=title, contents=bubble, quick_reply=build_quick_reply())

async def reply_text_with_tts_and_extras(reply_token: str, text: str, extras: Optional[List] = None):
    if not text:
        text = "（無內容）"
    # TextMessage 本來就有 Quick Reply
    messages = [TextMessage(text=text, quick_reply=build_quick_reply())]
    if extras:
        # [CHANGED] v1.4.6：確保 extras 內的訊息（如 AudioMessage）也帶 Quick Reply
        patched = []
        for m in extras:
            try:
                # 若該訊息型別支援 quick_reply 屬性，則補上（LINE v3 訊息物件皆支援）
                setattr(m, "quick_reply", build_quick_reply())
            except Exception:
                pass
            patched.append(m)
        messages.extend(patched)
    if CLOUDINARY_URL:
        try:
            audio_bytes = await text_to_speech_async(text)
            if audio_bytes:
                res = await run_in_threadpool(lambda: cloudinary.uploader.upload(
                    io.BytesIO(audio_bytes), resource_type="video", folder="line-bot-tts", format="mp3"))
                url = res.get("secure_url")
                if url:
                    est = max(3000, min(30000, len(text) * 60))
                    # [CHANGED] v1.4.6：TTS AudioMessage 也加 Quick Reply
                    messages.append(AudioMessage(original_content_url=url, duration=est, quick_reply=build_quick_reply()))
        except Exception as e:
            logger.warning(f"TTS 附加失敗：{e}")
    await line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=messages))

# ====== LLM 包裝（僅用於一般聊天或少量文字重寫） ======
def get_analysis_reply(messages: List[dict]) -> str:
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=1500
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI 失敗：{e}")
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

async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
    if not async_groq_client:
        return await run_in_threadpool(lambda: get_analysis_reply(messages))
    resp = await async_groq_client.chat.completions.create(
        model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    return resp.choices[0].message.content.strip()

async def analyze_sentiment(text: str) -> str:
    msgs = [{"role": "system", "content": "Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
            {"role": "user", "content": text}]
    try:
        out = await groq_chat_async(msgs, max_tokens=10, temperature=0)
        return (out or "neutral").strip().lower()
    except Exception:
        return "neutral"

async def translate_text(text: str, target_lang_display: str) -> str:
    target = LANGUAGE_MAP.get(target_lang_display.lower(), target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text with no extra words."
    clean = re.sub(r"[\u200B-\u200D\uFEFF]", "", text).strip()
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{clean}"}}'
    return await groq_chat_async([{"role": "system", "content": sys}, {"role": "user", "content": usr}], 800, 0.2)

def set_user_persona(chat_id: str, key: str):
    key_mapped = PERSONA_ALIAS.get(key, key)
    if key_mapped == "random":
        key_mapped = random.choice(list(PERSONAS.keys()))
    if key_mapped not in PERSONAS:
        key_mapped = "sweet"
    user_persona[chat_id] = key_mapped
    return key_mapped

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    key = user_persona.get(chat_id, "sweet")
    p = PERSONAS[key]
    return (f"你是一位「{p['title']}」。風格：{p['style']}。\n"
            f"使用者情緒：{sentiment}。\n"
            f"回覆請精煉自然，使用繁體中文，帶少量表情 {p['emoji']}.")

# ====== 金價（沿用） ======
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}

def parse_bot_gold_text(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    m_time = re.search(r"掛牌時間[:：]\s*([0-9]{4}/[0-9]{2}/[0-9]{2}\s+[0-9]{2}:[0-9]{2})", text)
    listed_at = m_time.group(1) if m_time else None
    m_sell = re.search(r"本行賣出\s*([0-9,]+(?:\.[0-9]+)?)", text)
    m_buy = re.search(r"本行買進\s*([0-9,]+(?:\.[0-9]+)?)", text)
    if not (m_sell and m_buy):
        raise RuntimeError("找不到『本行賣出/本行買進』欄位")
    sell = float(m_sell.group(1).replace(",", ""))
    buy = float(m_buy.group(1).replace(",", ""))
    return {"listed_at": listed_at, "sell_twd_per_g": sell, "buy_twd_per_g": buy, "source": BOT_GOLD_URL}

def get_bot_gold_quote() -> dict:
    r = requests.get(BOT_GOLD_URL, headers=DEFAULT_HEADERS, timeout=10)
    r.raise_for_status()
    return parse_bot_gold_text(r.text)

# ====== 彩票分析（沿用） ======
def get_lottery_analysis(lottery_type: str) -> str:
    prompt = f"你是一位專業的{lottery_type}分析師，請根據近期數據提供趨勢分析和3組隨機號碼建議，使用繁體中文。"
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": f"分析 {lottery_type}"}]
    return get_analysis_reply(messages)

# ====== 語音處理 ======
def _transcribe_with_openai_sync(audio_bytes: bytes, filename: str = "audio.m4a") -> Optional[str]:
    if not openai_client:
        return None
    try:
        f = io.BytesIO(audio_bytes)
        f.name = filename
        resp = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
        return (resp.text or "").strip() or None
    except Exception as e:
        logger.warning(f"OpenAI STT 失敗：{e}")
        return None

def _transcribe_with_groq_sync(audio_bytes: bytes, filename: str = "audio.m4a") -> Optional[str]:
    if not sync_groq_client:
        return None
    try:
        f = io.BytesIO(audio_bytes)
        f.name = filename
        resp = sync_groq_client.audio.transcriptions.create(file=f, model="whisper-large-v3")
        return (resp.text or "").strip() or None
    except Exception as e:
        logger.warning(f"Groq STT 失敗：{e}")
        return None

async def speech_to_text_async(audio_bytes: bytes) -> Optional[str]:
    text = await run_in_threadpool(_transcribe_with_openai_sync, audio_bytes)
    if text:
        return text
    return await run_in_threadpool(_transcribe_with_groq_sync, audio_bytes)

def _create_tts_openai_sync(text: str) -> Optional[bytes]:
    if not openai_client:
        return None
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
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
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
        if b:
            return b
    return await run_in_threadpool(_create_tts_gtts_sync, text)

# ====== StockGPT：偵測與分析主流程 ======
# [NEW] 台股代碼：4~6 位數字，可帶結尾 1 字母；美股代碼：1~5 英文字母
TW_TICKER_RE = re.compile(r"^\d{4,6}[A-Za-z]?$")
US_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}$")

def _is_stock_query(text: str) -> bool:
    t = text.strip()
    if t in ("大盤", "台股大盤", "台灣大盤", "美盤", "美股大盤", "美股"):
        return True
    if TW_TICKER_RE.match(t):
        return True
    # 避免把常見英文單字誤判成美股代碼，加入白名單再判
    if US_TICKER_RE.match(t) and t.upper() not in {"MENU", "NVDA"} - set():  # NVDA 仍允許
        return True
    return False

def _normalize_ticker_and_name(user_text: str) -> Tuple[str, str, str]:
    """
    依輸入回傳 (ticker, display_name, yahoo_link)
    - 大盤 → ^TWII
    - 美盤/美股 → ^GSPC
    - 其餘：直接使用代碼；YahooStock 會補全中文名
    """
    raw = user_text.strip()
    if raw in ("大盤", "台股大盤", "台灣大盤"):
        return "^TWII", "台灣大盤", "https://tw.finance.yahoo.com/quote/%5ETWII/"
    if raw in ("美盤", "美股大盤", "美股"):
        return "^GSPC", "美國大盤", "https://tw.finance.yahoo.com/quote/%5EGSPC/"
    ticker = raw.upper()
    link = f"https://tw.stock.yahoo.com/quote/{ticker}" if TW_TICKER_RE.match(ticker) else f"https://tw.finance.yahoo.com/quote/{ticker}"
    return ticker, ticker, link

def _safe_to_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)

def _remove_full_width_spaces(data):
    if isinstance(data, list):
        return [_remove_full_width_spaces(item) for item in data]
    if isinstance(data, str):
        return data.replace('\u3000', ' ')
    return data

def _truncate_text(data, max_length=1024):
    if isinstance(data, list):
        return [_truncate_text(item, max_length) for item in data]
    if isinstance(data, str):
        return data[:max_length]
    return data

def build_stock_prompt_block(stock_id: str, stock_name_hint: str) -> Tuple[str, dict]:
    """
    組裝分析用文字區塊；同時回傳一份原始資料 dict 方便除錯
    """
    debug_payload = {}
    # 即時資訊
    ys = YahooStock(stock_id)
    debug_payload["yahoo_stock"] = {k: _safe_to_str(v) for k, v in vars(ys).items()}

    # 價格（by日）
    price_df = stock_price(stock_id)
    debug_payload["price"] = _safe_to_str(price_df)

    # 新聞（去全形空格 + 1024 截斷）
    news = _remove_full_width_spaces(stock_news(stock_name_hint))
    news = _truncate_text(news, 1024)
    debug_payload["news"] = _safe_to_str(news)

    # 基本面/配息（大盤不取）
    fund_text = None
    div_text = None
    if stock_id not in ["^TWII", "^GSPC"]:
        try:
            fv = stock_fundamental(stock_id)
            fund_text = _safe_to_str(fv) if fv is not None else "（無法取得）"
        except Exception as e:
            fund_text = f"（基本面錯誤：{e}）"
        try:
            dv = stock_dividend(stock_id)
            div_text = _safe_to_str(dv) if dv is not None else "（無法取得）"
        except Exception as e:
            div_text = f"（配息錯誤：{e}）"
    debug_payload["fundamental"] = fund_text
    debug_payload["dividend"] = div_text

    # 組裝分析文字
    blk = []
    blk.append(f"**股票代碼:** {stock_id}, **股票名稱:** {ys.name}")
    blk.append(f"**即時資訊(vars):** {vars(ys)}")
    blk.append(f"近期價格資訊:\n{price_df}")
    if stock_id not in ["^TWII", "^GSPC"]:
        blk.append(f"每季營收資訊:\n{fund_text}")
        blk.append(f"配息資料:\n{div_text}")
    blk.append(f"近期新聞資訊:\n{news}")
    content = "\n".join(_safe_to_str(s) for s in blk)
    return content, debug_payload

def render_stock_report(stock_id: str, stock_link: str, content_block: str) -> str:
    """
    以 Markdown 生成最終報告結構（非閒聊）
    """
    sys = (
        "你現在是一位專業的證券分析師。請基於近期的股價走勢、基本面、新聞與籌碼概念進行綜合分析，"
        "輸出條列清楚、數字精確、可讀性高的報告。\n"
        "請包含：\n"
        "- 股名(股號) / 現價(與漲跌幅) / 資料時間\n"
        "- 股價走勢\n- 基本面分析\n- 技術面重點\n- 消息面\n- 籌碼面\n"
        "- 建議買進區間（例：100–110 元）\n- 預計停利點（%）\n- 建議部位（張數）\n"
        "- 總結：目前偏多/偏空/觀望\n"
        f"最後請附上正確連結：[股票資訊連結]({stock_link})。\n"
        "回應語言：繁體中文（台灣），格式：Markdown。"
    )
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": content_block}
    ]
    try:
        out = get_analysis_reply(messages)
    except Exception as e:
        out = f"（分析模型不可用）原始資料如下，請自行判讀：\n\n{content_block}\n\n連結：{stock_link}"
    return out

# ====== 事件處理 ======
async def handle_text_message(event: MessageEvent):
    chat_id = get_chat_id(event)
    msg_raw = (event.message.text or "").strip()
    reply_tok = event.reply_token
    if not msg_raw:
        return

    try:
        bot_info: BotInfoResponse = await line_bot_api.get_bot_info()
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True
    is_group_or_room = getattr(event.source, "type", "") in ("group", "room")
    if is_group_or_room and not auto_reply_status.get(chat_id, True) and not msg_raw.startswith(f"@{bot_name}"):
        return

    msg = msg_raw
    if msg_raw.startswith(f"@{bot_name}"):
        msg = re.sub(f'^@{re.escape(bot_name)}\\s*', '', msg_raw).strip()
    if not msg:
        return

    # ===== A. 翻譯模式指令 =====
    m = TRANSLATE_CMD.match(msg)
    if m:
        lang_token = m.group(1)
        rev = {"english": "英文", "japanese": "日文", "korean": "韓文", "vietnamese": "越南文", "繁體中文": "繁體中文", "中文": "繁體中文"}
        lang_display = rev.get(lang_token.lower(), lang_token)
        _tstate_set(chat_id, lang_display)
        await reply_text_with_tts_and_extras(reply_tok, f"🌐 已開啟翻譯 → {lang_display}，請直接輸入要翻的內容。")
        return

    if msg.startswith("翻譯->"):
        lang = msg.split("->", 1)[1].strip()
        if lang == "結束":
            _tstate_clear(chat_id)
            await reply_text_with_tts_and_extras(reply_tok, "✅ 已結束翻譯模式")
        else:
            _tstate_set(chat_id, lang)
            await reply_text_with_tts_and_extras(reply_tok, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")
        return

    im = INLINE_TRANSLATE.match(msg)
    if im:
        lang_key, text_to_translate = im.group(1).lower(), im.group(2)
        lang_display = {
            "en": "英文", "eng": "英文", "英文": "英文",
            "ja": "日文", "jp": "日文", "日文": "日文",
            "zh": "繁體中文", "繁中": "繁體中文", "中文": "繁體中文",
        }.get(lang_key, "英文")
        out = await translate_text(text_to_translate, lang_display)
        await reply_text_with_tts_and_extras(reply_tok, out)
        return

    current_lang = _tstate_get(chat_id)
    if current_lang:
        try:
            out = await translate_text(msg, current_lang)
            await reply_text_with_tts_and_extras(reply_tok, out)
        except Exception as e:
            logger.error(f"翻譯失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_tok, "抱歉，翻譯目前不可用。")
        return

    # ===== B. 主選單 / 子選單 =====
    low = msg.lower()
    if low in ("menu", "選單", "主選單"):
        # [CHANGED] v1.4.6：build_main_menu() 已內建 quick_reply
        await line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_tok, messages=[build_main_menu()]))
        return

    # ===== C. 金價/彩票 =====
    if msg in ("金價", "黃金"):
        try:
            data = get_bot_gold_quote()
            ts, sell, buy = data.get("listed_at") or "（未標示）", data["sell_twd_per_g"], data["buy_twd_per_g"]
            spread = sell - buy
            txt = (f"**金價（台灣銀行）**\n- 掛牌時間：{ts}\n- 賣出(1g)：{sell:,.0f} 元\n- 買進(1g)：{buy:,.0f} 元\n"
                   f"- 價差：{spread:,.0f} 元\n來源：{BOT_GOLD_URL}")
            await reply_text_with_tts_and_extras(reply_tok, txt)
        except Exception as e:
            await reply_text_with_tts_and_extras(reply_tok, "抱歉，目前無法取得金價。")
        return

    if msg in ("大樂透", "威力彩", "539"):
        try:
            report = await run_in_threadpool(get_lottery_analysis, msg)
            await reply_text_with_tts_and_extras(reply_tok, report)
        except Exception as e:
            logger.error(f"彩票分析流程失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_tok, f"抱歉，分析 {msg} 時發生錯誤。")
        return

    # ===== D. 【重點】股票查詢（非閒聊） =====
    if _is_stock_query(msg):
        try:
            ticker, name_hint, link = _normalize_ticker_and_name(msg)
            # 收集資料 + 組 prompt
            content_block, debug_payload = await run_in_threadpool(build_stock_prompt_block, ticker, name_hint)
            # 呼叫 LLM（或降級為原始資料）
            report = await run_in_threadpool(render_stock_report, ticker, link, content_block)
            await reply_text_with_tts_and_extras(reply_tok, report)
        except Exception as e:
            logger.error(f"[StockGPT] 失敗：{e}", exc_info=True)
            await reply_text_with_tts_and_extras(
                reply_tok,
                f"抱歉，取得 {msg} 的分析時發生錯誤：{e}\n請稍後再試或換個代碼。"
            )
        return

    # ===== E. 其餘：一般聊天（保留，但不影響股票分析流） =====
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": msg}]
        final_reply = await groq_chat_async(messages)
        history.extend([{"role": "user", "content": msg}, {"role": "assistant", "content": final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN * 2:]
        await reply_text_with_tts_and_extras(reply_tok, final_reply)
    except Exception as e:
        logger.error(f"聊天回覆失敗: {e}", exc_info=True)
        await reply_text_with_tts_and_extras(reply_tok, "抱歉我剛剛走神了 😅 再說一次讓我補上！")

async def handle_audio_message(event: MessageEvent):
    reply_tok = event.reply_token
    try:
        content_stream = await line_bot_api.get_message_content(event.message.id)
        audio_in = await content_stream.read()

        text = await speech_to_text_async(audio_in)
        if not text:
            await reply_text_with_tts_and_extras(reply_tok, "🎧 語音收到！目前語音轉文字失敗，請稍後再試。")
            return

        # 先回覆 STT 內容
        await line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=reply_tok,
                messages=[TextMessage(text=f"🎧 我聽到了：\n{text}", quick_reply=build_quick_reply())]  # [CHANGED] 已含 Quick Reply
            )
        )

        # 使用 TTS 回覆錄音
        audio_bytes = await text_to_speech_async(f"你說了：{text}")
        if audio_bytes and CLOUDINARY_URL:
            try:
                res = await run_in_threadpool(lambda: cloudinary.uploader.upload(
                    io.BytesIO(audio_bytes), resource_type="video", folder="line-bot-tts", format="mp3"))
                url = res.get("secure_url")
                if url:
                    est = max(3000, min(30000, len(text) * 60))
                    # [CHANGED] v1.4.6：第二段 AudioMessage 也帶 Quick Reply，確保該回覆本身也有主選單
                    await line_bot_api.reply_message(
                        ReplyMessageRequest(
                            reply_token=reply_tok,
                            messages=[AudioMessage(original_content_url=url, duration=est, quick_reply=build_quick_reply())]
                        )
                    )
            except Exception as e:
                logger.warning(f"TTS 附加失敗（忽略）：{e}")

    except Exception as e:
        logger.error(f"處理語音訊息失敗: {e}", exc_info=True)
        await reply_text_with_tts_and_extras(reply_tok, "抱歉，語音處理失敗，請稍後再試。")

async def handle_postback(event: PostbackEvent):
    data = event.postback.data or ""
    if data.startswith("menu:"):
        kind = data.split(":", 1)[-1]
        # [CHANGED] build_submenu() 已內建 quick_reply
        await line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=event.reply_token, messages=[build_submenu(kind)])
        )

async def handle_events(events):
    for event in events:
        if isinstance(event, MessageEvent):
            if isinstance(event.message, TextMessageContent):
                await handle_text_message(event)
            elif isinstance(event.message, AudioMessageContent):
                await handle_audio_message(event)
        elif isinstance(event, PostbackEvent):
            await handle_postback(event)

# ====== FastAPI ======
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [CHANGED] 啟動時嘗試更新 LINE Webhook（可選）
    if BASE_URL:
        async with httpx.AsyncClient() as c:
            for endpoint in ("https://api-data.line.me/v2/bot/channel/webhook/endpoint",
                            "https://api.line.me/v2/bot/channel/webhook/endpoint"):
                try:
                    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
                    payload = {"endpoint": f"{BASE_URL}/callback"}
                    r = await c.put(endpoint, headers=headers, json=payload, timeout=10.0)
                    r.raise_for_status()
                    logger.info(f"Webhook 更新成功: {endpoint} {r.status_code}")
                    break
                except Exception as e:
                    logger.warning(f"Webhook 更新失敗：{e}")
    yield

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.4.6")  # [CHANGED] bump version
router = APIRouter()

@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        events = parser.parse(body.decode("utf-8"), signature)
        await handle_events(events)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Callback 失敗：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse({"status": "ok"})

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