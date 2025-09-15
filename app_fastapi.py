# app_fastapi.py
# ========== 1) Imports ==========
import os
import re
import io
import random
import logging
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime

# --- 數據處理與爬蟲 ---
import requests
from bs4 import BeautifulSoup
import httpx
import pandas as pd
import yfinance as yf

# --- FastAPI 與 LINE Bot SDK v3 ---
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    AudioMessageContent,
    PostbackEvent,
    WebhookHandler,  # 修正：使用 WebhookHandler 替代 AsyncWebhookHandler
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    AsyncMessagingApi,
    ReplyMessageRequest,
    TextMessage,
    AudioMessage,
    ImageMessage,
    FlexMessage,
    FlexBubble,
    FlexBox,
    FlexText,
    FlexButton,
    QuickReply,
    QuickReplyItem,
    MessageAction,
    PostbackAction,
    BotInfoResponse,
)

# --- Cloudinary（上傳音訊/圖片） ---
import cloudinary
import cloudinary.uploader

# --- gTTS（免費 TTS 後備） ---
from gtts import gTTS

# --- AI 相關 ---
from groq import AsyncGroq, Groq
import openai

# --- 圖表（可選，無則自動跳過） ---
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


# ========== 2) Setup ==========
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# --- 環境變數 ---
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto").lower()  # auto / openai / gtts

if not CHANNEL_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("缺少必要環境變數：CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET")

# --- Cloudinary 設定 ---
if CLOUDINARY_URL:
    try:
        cloudinary.config(
            cloud_name=re.search(r"@(.+)", CLOUDINARY_URL).group(1),
            api_key=re.search(r"//(\d+):", CLOUDINARY_URL).group(1),
            api_secret=re.search(r":([A-Za-z0-9_-]+)@", CLOUDINARY_URL).group(1),
        )
        logger.info("✅ Cloudinary 設定成功！")
    except Exception as e:
        logger.error(f"Cloudinary 設定失敗: {e}")
        CLOUDINARY_URL = None
else:
    logger.warning("未設定 CLOUDINARY_URL，TTS/圖表將無法上傳。")

# --- LINE API 用戶端 ---
configuration = Configuration(access_token=CHANNEL_TOKEN)
async_api_client = ApiClient(configuration=configuration)
line_bot_api = AsyncMessagingApi(api_client=async_api_client)
handler = WebhookHandler(CHANNEL_SECRET)  # 修正：使用 WebhookHandler

# --- AI 客戶端 ---
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
sync_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.warning(f"初始化 OpenAI 失敗：{e}")
else:
    logger.warning("未設定 OPENAI_API_KEY，OpenAI STT/TTS 將停用（改用 Groq/gTTS 後備）。")

# Groq 模型（避免使用已下架的 3.1-70b）
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

# --- 自訂模組（可無則降級） ---
LOTTERY_ENABLED = True
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()
    logger.info("✅ 已載入 TaiwanLotteryCrawler / CaiyunfangweiCrawler")
except Exception as e:
    logger.warning(f"無法載入自訂彩券模組：{e}（將使用後備解析）")
    LOTTERY_ENABLED = False

STOCK_ENABLED = True
try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
except Exception as e:
    logger.warning(f"無法載入自訂股票模組：{e}（僅顯示快照/圖表）")
    STOCK_ENABLED = False

# --- 狀態字典與常數 ---
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}
auto_reply_status: Dict[str, bool] = {}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji": "🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji": "😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji": "✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議", "greetings": "我在。說重點。", "emoji": "🧊⚡️"},
}
LANGUAGE_MAP = {
    "英文": "English", "日文": "Japanese", "韓文": "Korean",
    "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"
}
PERSONA_ALIAS = {"甜":"sweet", "鹹":"salty", "萌":"moe", "酷":"cool", "random":"random"}


# ========== 3) FastAPI Lifespan（啟動時設定 Webhook） ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    if BASE_URL:
        async with httpx.AsyncClient() as c:
            for endpoint in (
                "https://api-data.line.me/v2/bot/channel/webhook/endpoint",
                "https://api.line.me/v2/bot/channel/webhook/endpoint",
            ):
                try:
                    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
                    payload = {"endpoint": f"{BASE_URL}/callback"}
                    r = await c.put(endpoint, headers=headers, json=payload, timeout=10.0)
                    r.raise_for_status()
                    logger.info(f"✅ Webhook 更新成功: {endpoint} {r.status_code}")
                    break
                except Exception as e:
                    logger.warning(f"Webhook 更新失敗（嘗試 {endpoint}）: {e}")
    else:
        logger.warning("未設定 BASE_URL，略過 Webhook 更新。")
    yield


app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.3.1")
router = APIRouter()


# ========== 4) Helpers ==========
def get_chat_id(event: MessageEvent) -> str:
    """
    兼容 v2/v3 屬性命名：userId / user_id、groupId / group_id、roomId / room_id
    避免翻譯模式的狀態用不同 key 造成「看起來開了卻沒翻」的情況。
    """
    source = event.source
    stype = getattr(source, "type", "")
    if stype == "group":
        return getattr(source, "groupId", None) or getattr(source, "group_id", None) or "group:unknown"
    if stype == "room":
        return getattr(source, "roomId", None) or getattr(source, "room_id", None) or "room:unknown"
    # user
    return getattr(source, "userId", None) or getattr(source, "user_id", None) or "user:unknown"

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
    ])

def build_main_menu() -> FlexMessage:
    items = [
        ("💹 金融查詢", PostbackAction(label="💹 金融查詢", data="menu:finance")),
        ("🎰 彩票分析", PostbackAction(label="🎰 彩票分析", data="menu:lottery")),
        ("💖 AI 角色扮演", PostbackAction(label="💖 AI 角色扮演", data="menu:persona")),
        ("🌐 翻譯工具", PostbackAction(label="🌐 翻譯工具", data="menu:translate")),
    ]
    buttons = [
        FlexButton(action=items[0][1], style="primary"),
        FlexButton(action=items[1][1], style="primary"),
        FlexButton(action=items[2][1], style="secondary"),
        FlexButton(action=items[3][1], style="secondary"),
    ]
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
    rows = []
    row = []
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
    return FlexMessage(alt_text=title, contents=bubble)

async def reply_text_with_tts_and_extras(reply_token: str, text: str, extras: Optional[List]=None):
    """統一回覆：文字 +（可選）附加訊息 +（可選）TTS 音訊"""
    if not text:
        text = "（無內容）"
    messages = [TextMessage(text=text, quick_reply=build_quick_reply())]
    if extras:
        messages.extend(extras)
    # 附加 TTS 音訊（如可用）
    if CLOUDINARY_URL:
        try:
            audio_bytes = await text_to_speech_async(text)
            if audio_bytes:
                public_audio_url = await upload_audio_to_cloudinary(audio_bytes)
                if public_audio_url:
                    est_dur = max(3000, min(30000, len(text) * 60))
                    messages.append(AudioMessage(original_content_url=public_audio_url, duration=est_dur))
        except Exception as e:
            logger.warning(f"TTS 附加失敗（忽略）：{e}")
    await line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=messages))


# ========== 5) AI & 分析 ==========
def get_analysis_reply(messages: List[dict]) -> str:
    # 先試 OpenAI（可選）
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI 失敗：{e}")

    # 再試 Groq 主力
    try:
        resp = sync_groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq 主模型失敗：{e}")
        try:
            resp = sync_groq_client.chat.completions.create(
                model=GROQ_MODEL_FALLBACK,
                messages=messages,
                temperature=0.9,
                max_tokens=1500,
            )
            return resp.choices[0].message.content
        except Exception as ee:
            logger.error(f"所有 AI API 都失敗：{ee}")
            return "抱歉，AI 分析師目前連線不穩定，請稍後再試。"

async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
    if not async_groq_client:
        return await run_in_threadpool(lambda: get_analysis_reply(messages))
    resp = await async_groq_client.chat.completions.create(
        model=GROQ_MODEL_FALLBACK,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ========== 6) 金融工具 ==========
# ---- 6.1 台銀金價（穩定文字解析）----
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def parse_bot_gold_text(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    m_time = re.search(r"掛牌時間[:：]\s*([0-9]{4}/[0-9]{2}/[0-9]{2}\s+[0-9]{2}:[0-9]{2})", text)
    listed_at = m_time.group(1) if m_time else None

    m_sell = re.search(r"本行賣出\s*([0-9,]+(?:\.[0-9]+)?)", text)
    m_buy  = re.search(r"本行買進\s*([0-9,]+(?:\.[0-9]+)?)", text)
    if not (m_sell and m_buy):
        raise RuntimeError("找不到『本行賣出/本行買進』欄位")

    sell = float(m_sell.group(1).replace(",", ""))
    buy  = float(m_buy.group(1).replace(",", ""))

    return {"listed_at": listed_at, "sell_twd_per_g": sell, "buy_twd_per_g": buy, "source": BOT_GOLD_URL}

def get_bot_gold_quote() -> dict:
    r = requests.get(BOT_GOLD_URL, headers=DEFAULT_HEADERS, timeout=10)
    r.raise_for_status()
    return parse_bot_gold_text(r.text)

def get_gold_analysis() -> str:
    try:
        data = get_bot_gold_quote()
        ts = data.get("listed_at") or "（頁面未標示）"
        sell, buy = data["sell_twd_per_g"], data["buy_twd_per_g"]
        spread = sell - buy
        bias = "盤整" if spread <= 30 else ("偏寬" if spread <= 60 else "價差偏大")
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        return (
            f"**金價快報（台灣銀行）**\n"
            f"- 掛牌時間：{ts}\n"
            f"- 本行賣出（1克）：**{sell:,.0f} 元**\n"
            f"- 本行買進（1克）：**{buy:,.0f} 元**\n"
            f"- 買賣價差：{spread:,.0f} 元（{bias}）\n"
            f"\n資料來源：{BOT_GOLD_URL}\n（更新於 {now}）"
        )
    except Exception as e:
        logger.error(f"金價流程失敗：{e}", exc_info=True)
        return "抱歉，目前無法從台灣銀行取得黃金牌價。稍後再試一次 🙏"

# ---- 6.2 匯率 ----
def get_currency_analysis(target_currency: str) -> str:
    try:
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("result") == "success":
            rate = data["rates"].get("TWD")
            if rate is None:
                return "抱歉，API 中找不到 TWD 的匯率資訊。"
            return f"即時：1 {target_currency.upper()} ≈ {rate:.5f} 新台幣"
        else:
            return f"抱歉，獲取匯率資料失敗：{data.get('error-type', '未知錯誤')}"
    except Exception as e:
        logger.error(f"處理 {target_currency} 匯率時發生錯誤: {e}", exc_info=True)
        return "抱歉，外匯資料暫時無法取得。"

# ---- 6.3 股票 ----
_TW_CODE_RE = re.compile(r'^\d{4,6}[A-Za-z]?$')     # 2330 / 006208 / 00937B / 1101B
_US_CODE_RE = re.compile(r'^[A-Za-z]{1,5}$')        # NVDA / AAPL / QQQ

def normalize_ticker(user_text: str) -> Tuple[str, str, str, bool]:
    """
    回傳: (yfinance_symbol, yahoo_tw_slug, display_code, is_index)
    - 台股數字代碼（含尾碼字母）加上 .TW 給 yfinance
    - Yahoo 台股頁面 slug 用原始碼
    - 指數：^TWII / ^GSPC
    """
    t = user_text.strip().upper()
    if t in ["台股大盤", "大盤", "^TWII"]:
        return "^TWII", "^TWII", "^TWII", True
    if t in ["美股大盤", "美盤", "美股", "^GSPC"]:
        return "^GSPC", "^GSPC", "^GSPC", True
    if _TW_CODE_RE.match(t):
        return f"{t}.TW", t, t, False
    if _US_CODE_RE.match(t) and t != "JPY":
        return t, t, t, False
    return t, t, t, False

def fetch_realtime_snapshot(yf_symbol: str, yahoo_slug: str) -> dict:
    snap: dict = {"name": None, "now_price": None, "change": None, "currency": None, "close_time": None}
    try:
        tk = yf.Ticker(yf_symbol)
        info = getattr(tk, "fast_info", None)
        hist = tk.history(period="2d", interval="1d")

        # 名稱
        name = None
        try:
            name = tk.get_info().get("shortName")
        except Exception:
            pass
        snap["name"] = name or yf_symbol

        # 價格 & 幣別
        price, ccy = None, None
        if info and getattr(info, "last_price", None):
            price = info.last_price
            ccy = getattr(info, "currency", None)
        elif not hist.empty:
            price = float(hist["Close"].iloc[-1])
            ccy = getattr(info, "currency