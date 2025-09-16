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
    WebhookParser,  # 改用 Parser（v3 正確做法）
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

# v3 解析用 Parser（取代 AsyncWebhookHandler/WebhookHandler）
parser = WebhookParser(CHANNEL_SECRET)

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


app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.3.2")
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
            ccy = getattr(info, "currency", None)
        if price:
            snap["now_price"] = f"{price:.2f}"
            snap["currency"] = ccy or ("TWD" if yf_symbol.endswith(".TW") else "USD")

        # 變動
        if not hist.empty and len(hist) >= 2:
            chg = float(hist["Close"].iloc[-1]) - float(hist["Close"].iloc[-2])
            pct = chg / float(hist["Close"].iloc[-2]) * 100 if hist["Close"].iloc[-2] else 0.0
            sign = "+" if chg >= 0 else "-"
            snap["change"] = f"{sign}{abs(chg):.2f} ({sign}{abs(pct):.2f}%)"

        # 時間
        if not hist.empty:
            ts = hist.index[-1]
            snap["close_time"] = ts.strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        logger.warning(f"yfinance 取得 {yf_symbol} 失敗：{e}")

    # 後備：YahooStock（若可用）
    if (not snap["now_price"] or not snap["name"]) and 'YahooStock' in globals():
        try:
            ys = YahooStock(yahoo_slug)
            snap["name"] = ys.name or snap["name"] or yahoo_slug
            snap["now_price"] = ys.now_price or snap["now_price"]
            snap["change"] = ys.change or snap["change"]
            snap["currency"] = ys.currency or ("TWD" if yf_symbol.endswith(".TW") else snap["currency"])
            snap["close_time"] = ys.close_time or snap["close_time"]
        except Exception as e:
            logger.error(f"YahooStock 取得 {yahoo_slug} 失敗：{e}")

    return snap

# 圖片上傳 + 圖表產生（可選）
def _upload_image_sync(image_bytes: bytes) -> Optional[dict]:
    if not CLOUDINARY_URL:
        return None
    try:
        return cloudinary.uploader.upload(
            io.BytesIO(image_bytes),
            resource_type="image",
            folder="line-bot-chart",
            format="png"
        )
    except Exception as e:
        logger.error(f"Cloudinary 上傳圖片失敗: {e}")
        return None

async def upload_image_to_cloudinary(image_bytes: bytes) -> Optional[str]:
    res = await run_in_threadpool(_upload_image_sync, image_bytes)
    return res.get("secure_url") if res else None

def generate_stock_chart_png(yf_symbol: str, period: str = "6mo", interval: str = "1d") -> Optional[bytes]:
    if not HAS_MPL:
        return None
    try:
        df = yf.download(yf_symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None

        buf = io.BytesIO()
        if HAS_MPLFIN:
            mpf.plot(
                df, type="candle", mav=(5, 20, 60), volume=True, style="yahoo",
                tight_layout=True, savefig=dict(fname=buf, format="png")
            )
        else:
            plt.figure(figsize=(9, 5), dpi=200)
            plt.plot(df.index, df["Close"], label="Close")
            for w in (5, 20, 60):
                plt.plot(df.index, df["Close"].rolling(w).mean(), label=f"MA{w}")
            plt.title(f"{yf_symbol} Close & MAs")
            plt.legend()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            plt.close()

        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.warning(f"生成股票圖失敗：{e}")
        return None

async def get_stock_chart_url_async(user_input: str) -> Optional[str]:
    yf_symbol, _, _, _ = normalize_ticker(user_input)
    img = await run_in_threadpool(generate_stock_chart_png, yf_symbol)
    if not img or not CLOUDINARY_URL:
        return None
    return await upload_image_to_cloudinary(img)

# 進階報告（若有你自訂模組）
stock_data_df: Optional[pd.DataFrame] = None
def load_stock_data() -> pd.DataFrame:
    global stock_data_df
    if stock_data_df is None:
        try:
            stock_data_df = pd.read_csv('name_df.csv')
        except FileNotFoundError:
            logger.error("`name_df.csv` not found. Stock name lookup disabled.")
            stock_data_df = pd.DataFrame(columns=['股號', '股名'])
    return stock_data_df

def get_stock_name(stock_id_without_suffix: str) -> Optional[str]:
    df = load_stock_data()
    res = df[df['股號'].astype(str).str.upper() == stock_id_without_suffix.upper()]
    return res.iloc[0]['股名'] if not res.empty else None

def get_stock_analysis(user_input: str) -> str:
    yf_symbol, yahoo_slug, display_code, is_index = normalize_ticker(user_input)
    snapshot = fetch_realtime_snapshot(yf_symbol, yahoo_slug)

    price_data = ""
    news_data = ""
    value_part = ""
    dividend_part = ""
    if STOCK_ENABLED:
        try:
            price_data = str(stock_price(yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol))
        except Exception as e:
            logger.warning(f"price_data 失敗：{e}")
        try:
            nm = get_stock_name(yahoo_slug) or snapshot.get("name") or yahoo_slug
            news_data = str(stock_news(nm)).replace("\u3000", " ")[:1024]
        except Exception as e:
            logger.warning(f"news_data 失敗：{e}")
        if not is_index:
            try:
                val = stock_fundamental(yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol)
                value_part = f"{val}\n" if val else ""
            except Exception as e:
                logger.warning(f"fundamental 失敗：{e}")
            try:
                dvd = stock_dividend(yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol)
                dividend_part = f"{dvd}\n" if dvd else ""
            except Exception as e:
                logger.warning(f"dividend 失敗：{e}")

    stock_link = (
        f"https://finance.yahoo.com/quote/{yf_symbol}"
        if yf_symbol.startswith("^") or yf_symbol.endswith(".TW") or _US_CODE_RE.match(yf_symbol)
        else f"https://tw.stock.yahoo.com/quote/{yahoo_slug}"
    )

    content_msg = (
        f"你是專業的證券分析師，依據以下資料撰寫完整報告：\n"
        f"- 股票代碼：{display_code}\n- 名稱：{snapshot.get('name')}\n"
        f"- 即時快照：{snapshot}\n"
        f"- 近期價格資訊：\n{price_data}\n"
    )
    if value_part:    content_msg += f"- 每季營收資訊：\n{value_part}"
    if dividend_part: content_msg += f"- 配息資料：\n{dividend_part}"
    if news_data:     content_msg += f"- 近期新聞：\n{news_data}\n"
    content_msg += f"請以嚴謹專業、繁體中文、Markdown 格式撰寫，最後附連結：{stock_link}"

    system_prompt = (
        "你是專業的台股/美股分析師。開頭列：股名(股號)、現價/漲跌幅、資料時間；"
        "分段：股價走勢、基本面、技術面、消息面、風險、建議區間與停利目標；最後綜合結論。"
    )
    msgs = [{"role":"system","content":system_prompt}, {"role":"user","content":content_msg}]
    return get_analysis_reply(msgs)


# ========== 7) 彩票分析 ==========
def _lotto_fallback_scrape(kind: str) -> str:
    """當自訂爬蟲不可用時，從台彩官網以文字方式粗略擷取最新號碼（易受改版影響）。"""
    try:
        if kind == "威力彩":
            url = "https://www.taiwanlottery.com/lotto/superlotto638/index.html"
            pat = r"第一區(?:號碼)?[:：]\s*([0-9\s]+)\s*第二區(?:號碼)?[:：]\s*([0-9]{1,2})"
        elif kind == "大樂透":
            url = "https://www.taiwanlottery.com/lotto/lotto649/index.html"
            pat = r"(?:中獎號碼|開出順序)[:：]\s*([0-9\s]+)"
        elif kind == "539":
            url = "https://www.taiwanlottery.com/lotto/dailycash/index.html"
            pat = r"(?:中獎號碼|開出順序)[:：]\s*([0-9\s]+)"
        else:
            return f"不支援彩種：{kind}"

        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        r.raise_for_status()
        text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
        m = re.search(pat, text)
        if not m:
            return f"抱歉，暫時找不到 {kind} 最新號碼。"
        if kind == "威力彩":
            first, second = m.group(1), m.group(2)
            return f"{kind} 最新號碼：第一區 {first.strip()}；第二區 {second}"
        else:
            nums = m.group(1)
            return f"{kind} 最新號碼：{nums.strip()}"
    except Exception as e:
        logger.error(f"後備彩票爬取失敗：{e}", exc_info=True)
        return f"抱歉，{kind} 近期號碼暫時取不到。"

def get_lottery_analysis(lottery_type_input: str) -> str:
    kind = "威力彩" if "威力" in lottery_type_input else (
           "大樂透" if "大樂" in lottery_type_input else (
           "539" if "539" in lottery_type_input else lottery_type_input))

    latest_data_str = ""
    if LOTTERY_ENABLED:
        try:
            if kind == "威力彩":
                latest_data_str = str(TaiwanLotteryCrawler().super_lotto())
            elif kind == "大樂透":
                latest_data_str = str(TaiwanLotteryCrawler().lotto649())
            elif kind == "539":
                latest_data_str = str(TaiwanLotteryCrawler().daily_cash())
            else:
                return f"不支援 {kind}。"
        except Exception as e:
            logger.warning(f"自訂彩票爬蟲失敗，改用後備：{e}")
            latest_data_str = _lotto_fallback_scrape(kind)
    else:
        latest_data_str = _lotto_fallback_scrape(kind)

    # 可選：財神方位
    cai_part = ""
    try:
        if 'caiyunfangwei_crawler' in globals():
            cai = caiyunfangwei_crawler.get_caiyunfangwei()
            cai_part = f"今天日期：{cai.get('今天日期','')}\n今日歲次：{cai.get('今日歲次','')}\n財神方位：{cai.get('財神方位','')}\n"
    except Exception:
        cai_part = ""

    prompt = (
        f"你是一位資深彩券分析師。以下是 {kind} 的近況/最新號碼資料：\n"
        f"{latest_data_str}\n\n{cai_part}"
        "請用繁體中文寫出：\n"
        "1) 近期走勢重點（熱門/冷門/奇偶大小分佈）\n"
        "2) 選號建議與注意事項（理性與風險聲明）\n"
        "3) 提供三組推薦號碼（符合彩種格式，並由小到大排序）\n"
        "請以條列方式、精煉呈現。"
    )
    messages = [{"role":"system","content":"你是資深彩券分析師。"}, {"role":"user","content":prompt}]
    return get_analysis_reply(messages)


# ========== 8) 對話 / 翻譯 / 心情 ==========
async def analyze_sentiment(text: str) -> str:
    msgs = [
        {"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
        {"role":"user","content":text}
    ]
    try:
        out = await groq_chat_async(msgs, max_tokens=10, temperature=0)
        return (out or "neutral").strip().lower()
    except Exception:
        return "neutral"

async def translate_text(text: str, target_lang_display: str) -> str:
    """
    嚴格輸出翻譯文本，不加多餘說明。
    target_lang_display 可為「英文/日文/繁體中文...」，會映射到英文語名給模型。
    """
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text with no extra words."
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{text}"}}'
    return await groq_chat_async([{"role":"system","content":sys},{"role":"user","content":usr}], 800, 0.2)

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
            f"使用者情緒：{sentiment}（開心→分享喜悅；生氣/難過→先共情安撫再建議；中性→自然聊天）。\n"
            f"回覆請精煉自然，使用繁體中文，帶少量表情 {p['emoji']}。")


# ========== 9) TTS / STT（音訊處理） ==========
def _upload_audio_sync(audio_bytes: bytes) -> Optional[dict]:
    if not CLOUDINARY_URL: return None
    try:
        return cloudinary.uploader.upload(
            io.BytesIO(audio_bytes), resource_type="video", folder="line-bot-tts", format="mp3"
        )
    except Exception as e:
        logger.error(f"Cloudinary 上傳失敗: {e}")
        return None

async def upload_audio_to_cloudinary(audio_bytes: bytes) -> Optional[str]:
    response = await run_in_threadpool(_upload_audio_sync, audio_bytes)
    return response.get("secure_url") if response else None

def _create_tts_with_openai_sync(text: str) -> Optional[bytes]:
    if not openai_client: return None
    try:
        clean = re.sub(r"[*_`~#]", "", text)
        resp = openai_client.audio.speech.create(model="tts-1", voice="nova", input=clean)
        return resp.read()
    except Exception as e:
        logger.error(f"OpenAI TTS 生成失敗: {e}", exc_info=True)
        return None

def _create_tts_with_gtts_sync(text: str) -> Optional[bytes]:
    try:
        clean = re.sub(r"[*_`~#]", "", text).strip() or "嗨，我在這裡。"
        tts = gTTS(text=clean, lang="zh-TW", tld="com.tw", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"gTTS 生成失敗: {e}", exc_info=True)
        return None

async def text_to_speech_async(text: str) -> Optional[bytes]:
    provider = TTS_PROVIDER
    async def try_openai(): return await run_in_threadpool(_create_tts_with_openai_sync, text)
    async def try_gtts():   return await run_in_threadpool(_create_tts_with_gtts_sync, text)
    if provider == "openai": return await try_openai()
    if provider == "gtts":   return await try_gtts()
    if openai_client:
        b = await try_openai()
        if b: return b
    return await try_gtts()

# STT
def _transcribe_with_openai_sync(audio_bytes: bytes, filename: str = "audio.m4a") -> Optional[str]:
    if not openai_client: return None
    try:
        f = io.BytesIO(audio_bytes); f.name = filename
        resp = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
        return (resp.text or "").strip() or None
    except Exception as e:
        logger.warning(f"OpenAI STT 失敗：{e}")
        return None

def _transcribe_with_groq_sync(audio_bytes: bytes, filename: str = "audio.m4a") -> Optional[str]:
    if not sync_groq_client: return None
    try:
        f = io.BytesIO(audio_bytes); f.name = filename
        resp = sync_groq_client.audio.transcriptions.create(file=f, model="whisper-large-v3")
        return (resp.text or "").strip() or None
    except Exception as e:
        logger.warning(f"Groq STT 失敗：{e}")
        return None

async def speech_to_text_async(audio_bytes: bytes) -> Optional[str]:
    text = await run_in_threadpool(_transcribe_with_openai_sync, audio_bytes)
    if text: return text
    return await run_in_threadpool(_transcribe_with_groq_sync, audio_bytes)


# ========== 10) LINE Event Handlers（函式化，供 Parser 呼叫） ==========
async def on_text_message(event: MessageEvent):
    chat_id, msg_raw, reply_token = get_chat_id(event), event.message.text.strip(), event.reply_token

    # 取得 bot 顯示名稱（供 @bot 判斷）
    try:
        bot_info: BotInfoResponse = await line_bot_api.get_bot_info()
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if not msg_raw:
        return

    # 預設群組自動回覆開啟
    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True

    # 群組/聊天室：若關閉自動回覆，必須 @bot 才回
    is_group_or_room = getattr(event.source, "type", "") in ("group", "room")
    if is_group_or_room and not auto_reply_status.get(chat_id, True) and not msg_raw.startswith(f"@{bot_name}"):
        return

    # 去除 @botname 前綴
    msg = msg_raw
    if msg_raw.startswith(f"@{bot_name}"):
        msg = re.sub(f'^@{re.escape(bot_name)}\\s*', '', msg_raw).strip()
    if not msg:
        return

    low = msg.lower()

    # === 路由 ===
    # 主選單
    if low in ("menu", "選單", "主選單"):
        await line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=[build_main_menu()]))
        return

    # 彩票
    if msg in ("大樂透", "威力彩", "539"):
        try:
            report = await run_in_threadpool(get_lottery_analysis, msg)
            await reply_text_with_tts_and_extras(reply_token, report)
        except Exception as e:
            logger.error(f"彩票分析流程失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")
        return

    # 金價
    if low in ("金價", "黃金"):
        try:
            out = await run_in_threadpool(get_gold_analysis)
            await reply_text_with_tts_and_extras(reply_token, out)
        except Exception as e:
            logger.error(f"金價分析流程失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_token, "抱歉，金價分析服務暫時無法使用。")
        return

    # 匯率（簡化：僅 JPY；你可自行擴充 USD/EUR）
    if low == "jpy":
        try:
            out = await run_in_threadpool(get_currency_analysis, "JPY")
            await reply_text_with_tts_and_extras(reply_token, out)
        except Exception as e:
            logger.error(f"日圓分析流程失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_token, "抱歉，日圓匯率分析服務暫時無法使用。")
        return

    # 翻譯模式切換（開/關）
    if low.startswith("翻譯->"):
        lang = msg.split("->", 1)[1].strip()
        if lang == "結束":
            translation_states.pop(chat_id, None)
            await reply_text_with_tts_and_extras(reply_token, "✅ 已結束翻譯模式")
        else:
            translation_states[chat_id] = lang
            await reply_text_with_tts_and_extras(reply_token, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")
        return

    # ✅ 只要翻譯模式開著，就優先翻譯（避免被其它分支攔截）
    if chat_id in translation_states:
        try:
            out = await translate_text(msg, translation_states[chat_id])
            await reply_text_with_tts_and_extras(reply_token, out)
        except Exception as e:
            logger.error(f"翻譯失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_token, "抱歉，翻譯目前不可用。")
        return

    # 股票/指數
    if re.fullmatch(r"\^?[A-Z0-9.]{2,10}", msg) or msg.isdigit() or msg in ("台股大盤", "美股大盤", "大盤", "美股"):
        try:
            text = await run_in_threadpool(get_stock_analysis, msg)
            extras = []
            try:
                chart_url = await get_stock_chart_url_async(msg)
                if chart_url:
                    extras.append(ImageMessage(original_content_url=chart_url, preview_image_url=chart_url))
            except Exception as ce:
                logger.warning(f"附圖失敗（忽略）：{ce}")
            await reply_text_with_tts_and_extras(reply_token, text, extras=extras)
        except Exception as e:
            logger.error(f"股票分析流程失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")
        return

    # 自動回覆設定（僅群組/聊天室有意義）
    if low in ("開啟自動回答", "關閉自動回答"):
        is_on = low == "開啟自動回答"
        auto_reply_status[chat_id] = is_on
        text = "✅ 已開啟自動回答" if is_on else "❌ 已關閉自動回答（群組需 @我 才回）"
        await reply_text_with_tts_and_extras(reply_token, text)
        return

    # 人設切換（注意：因為翻譯模式分支已提前處理，不會誤觸）
    if msg in PERSONA_ALIAS or low in PERSONA_ALIAS:
        key = set_user_persona(chat_id, PERSONA_ALIAS.get(msg, PERSONA_ALIAS.get(low, "sweet")))
        p = PERSONAS[user_persona[chat_id]]
        txt = f"💖 已切換人設：{p['title']}\n\n{p['greetings']}"
        await reply_text_with_tts_and_extras(reply_token, txt)
        return

    # 一般聊天（人設 + 情緒）
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]
        final_reply = await groq_chat_async(messages)
        history.extend([{"role":"user","content":msg}, {"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
        await reply_text_with_tts_and_extras(reply_token, final_reply)
    except Exception as e:
        logger.error(f"AI 回覆失敗: {e}", exc_info=True)
        await reply_text_with_tts_and_extras(reply_token, "抱歉我剛剛走神了 😅 再說一次讓我補上！")


async def on_audio_message(event: MessageEvent):
    reply_token = event.reply_token
    try:
        content_stream = await line_bot_api.get_message_content(event.message.id)
        audio_in = await content_stream.read()

        text = await speech_to_text_async(audio_in)
        if not text:
            raise RuntimeError("語音轉文字失敗")

        sentiment = await analyze_sentiment(text)
        sys_prompt = build_persona_prompt(get_chat_id(event), sentiment)
        final_reply_text = await groq_chat_async(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}]
        )

        await reply_text_with_tts_and_extras(
            reply_token,
            f"🎧 我聽到了：\n{text}\n\n—\n{final_reply_text}"
        )
    except Exception as e:
        logger.error(f"處理語音訊息失敗: {e}", exc_info=True)
        await reply_text_with_tts_and_extras(reply_token, "抱歉，我沒聽清楚，可以再說一次嗎？")


async def on_postback(event: PostbackEvent):
    data = event.postback.data or ""
    if data.startswith("menu:"):
        kind = data.split(":", 1)[-1]
        await line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=event.reply_token, messages=[build_submenu(kind)])
        )


# ========== 11) FastAPI Routes ==========
@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body_bytes = await request.body()
    body_text = body_bytes.decode("utf-8")

    try:
        events = parser.parse(body_text, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Webhook 解析失敗：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Parse error")

    # 逐一處理事件（v3 沒有 handler.add，因此自行分派）
    for event in events:
        try:
            if isinstance(event, MessageEvent) and isinstance(event.message, TextMessageContent):
                await on_text_message(event)
            elif isinstance(event, MessageEvent) and isinstance(event.message, AudioMessageContent):
                await on_audio_message(event)
            elif isinstance(event, PostbackEvent):
                await on_postback(event)
            else:
                # 其他事件暫不處理
                pass
        except Exception as e:
            logger.error(f"事件處理失敗：{e}", exc_info=True)

    return JSONResponse({"status": "ok"})

@router.get("/")
async def root():
    return PlainTextResponse("LINE Bot is running.", status_code=200)

@router.get("/healthz")
async def healthz():
    return PlainTextResponse("ok", status_code=200)

app.include_router(router)


# ========== 12) Local run ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)