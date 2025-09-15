# ========== 1) Imports ==========
import os
import re
import io
import random
import logging
from typing import Dict, List, Optional, Tuple
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
    AsyncWebhookHandler,
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    AsyncMessagingApi,
    ReplyMessageRequest,
    TextMessage,
    AudioMessage,
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

# --- 雲端儲存 (Cloudinary) ---
import cloudinary
import cloudinary.uploader

# --- gTTS（免費 TTS，做為 fallback） ---
from gtts import gTTS

# --- AI 相關 ---
from groq import AsyncGroq, Groq
import openai

# ========== 2) Setup ==========
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# --- 環境變數 ---
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # 建議設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 可選
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
    logger.warning("未設定 CLOUDINARY_URL，TTS 語音訊息將無法傳送。")

# --- LINE Bot v3 用戶端初始化（async） ---
configuration = Configuration(access_token=CHANNEL_TOKEN)
async_api_client = ApiClient(configuration=configuration)
line_bot_api = AsyncMessagingApi(api_client=async_api_client)
handler = AsyncWebhookHandler(CHANNEL_SECRET)

# --- Groq / OpenAI ---
async_groq_client: Optional[AsyncGroq] = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
sync_groq_client: Optional[Groq] = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.warning(f"初始化 OpenAI 失敗：{e}")
else:
    logger.warning("未設定 OPENAI_API_KEY，OpenAI TTS/STT 將停用（將以 gTTS/Groq 為主）。")

# --- Groq 模型（使用未下架版本） ---
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama3-70b-8192")

# --- 自訂模組（可失敗則 fallback） ---
LOTTERY_ENABLED = True
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()
    logger.info("✅ 已載入 TaiwanLotteryCrawler / CaiyunfangweiCrawler")
except Exception as e:
    logger.warning(f"無法載入彩券模組：{e}（將使用後備抓取）")
    LOTTERY_ENABLED = False
    caiyunfangwei_crawler = None

STOCK_ENABLED = True
try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
except Exception as e:
    logger.warning(f"無法載入股票延伸模組：{e}（只提供基礎快照）")
    STOCK_ENABLED = False
    YahooStock = None  # 顯式標示

# --- 狀態與常數 ---
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}
auto_reply_status: Dict[str, bool] = {}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji": "🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji": "😏🙄"},
    "moe": {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji": "✨🎀"},
    "cool": {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議", "greetings": "我在。說重點。", "emoji": "🧊⚡️"},
}
LANGUAGE_MAP = {
    "英文": "English",
    "日文": "Japanese",
    "韓文": "Korean",
    "越南文": "Vietnamese",
    "繁體中文": "Traditional Chinese",
}

# ========== 3) FastAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """啟動時嘗試更新 LINE Webhook（api-data / api 各試一次）"""
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
        logger.warning("未設定 BASE_URL，跳過 Webhook 更新。")
    yield

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.3.0")
router = APIRouter()

# ========== 4) Helpers ==========
def get_chat_id(event: MessageEvent) -> str:
    """用 source.type 取得 chat id（不綁定類別，版本較穩）"""
    s = event.source
    if getattr(s, "type", "") == "group":
        return s.group_id
    if getattr(s, "type", "") == "room":
        return s.room_id
    return s.user_id

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

def reply_with_quick_bar(reply_token: str, text: str):
    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=reply_token,
            messages=[TextMessage(text=text, quick_reply=build_quick_reply())],
        )
    )

def build_flex_menu(title: str, items_data: List[Tuple[str, object]], alt_text: str) -> FlexMessage:
    buttons = [FlexButton(action=action) for _, action in items_data]
    bubble = FlexBubble(
        header=FlexBox(layout="vertical", contents=[FlexText(text=title, weight="bold", size="lg")]),
        body=FlexBox(layout="vertical", spacing="md", contents=buttons),
    )
    return FlexMessage(alt_text=alt_text, contents=bubble)

def build_main_menu() -> FlexMessage:
    items = [
        ("💹 金融查詢", PostbackAction(label="💹 金融查詢", data="menu:finance")),
        ("🎰 彩票分析", PostbackAction(label="🎰 彩票分析", data="menu:lottery")),
        ("💖 AI 角色扮演", PostbackAction(label="💖 AI 角色扮演", data="menu:persona")),
        ("🌐 翻譯工具", PostbackAction(label="🌐 翻譯工具", data="menu:translate")),
    ]
    return build_flex_menu("AI 助理主選單", items, "主選單")

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
    return build_flex_menu(title, [(lbl, act) for lbl, act in items], title)

# ========== 5) AI & 寫作 ==========
def _groq_complete(messages: List[dict], temp=0.7, max_tokens=1500) -> str:
    last_err = None
    if sync_groq_client:
        for model in (GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK):
            try:
                resp = sync_groq_client.chat.completions.create(
                    model=model, messages=messages, temperature=temp, max_tokens=max_tokens
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                logger.warning(f"Groq({model}) 失敗：{e}")
    return f"（AI 模組暫時連線不穩定）{(' ' + str(last_err)) if last_err else ''}"

def get_analysis_reply(messages: List[dict]) -> str:
    # 先試 OpenAI（若有），再退到 Groq
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, max_tokens=1500, temperature=0.7
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI 失敗：{e}")
    return _groq_complete(messages)

async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
    if not async_groq_client:
        return await run_in_threadpool(lambda: get_analysis_reply(messages))
    resp = await async_groq_client.chat.completions.create(
        model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    return resp.choices[0].message.content.strip()

# ========== 6) 金融工具 ==========
# ---- 6.1 金價（文字解析，抗 DOM 改版）----
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
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

    return {"listed_at": listed_at, "sell_twd_per_g": sell, "buy_twd_per_g": buy}

def get_gold_analysis() -> str:
    try:
        r = requests.get(BOT_GOLD_URL, headers=DEFAULT_HEADERS, timeout=10)
        r.raise_for_status()
        data = parse_bot_gold_text(r.text)

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
            return f"最新：1 {target_currency.upper()} ≈ {rate:.5f} 新台幣"
        else:
            return f"抱歉，獲取匯率資料失敗：{data.get('error-type', '未知錯誤')}"
    except Exception as e:
        logger.error(f"處理 {target_currency} API 資料時發生錯誤: {e}", exc_info=True)
        return "抱歉，處理外匯資料時發生內部錯誤，請稍後再試。"

# ---- 6.3 股票（簡版快照 + 可選延伸）----
_TW_CODE_RE = re.compile(r'^\d{4,6}[A-Za-z]?$')     # 2330 / 00937B / 1101B
_US_CODE_RE = re.compile(r'^[A-Za-z]{1,5}$')        # NVDA / AAPL / QQQ

def normalize_ticker(user_text: str) -> Tuple[str, str, str, bool]:
    t = user_text.strip().upper()
    if t in ["台股大盤", "大盤"]:
        return "^TWII", "^TWII", "^TWII", True
    if t in ["美股大盤", "美盤", "美股"]:
        return "^GSPC", "^GSPC", "^GSPC", True
    if _TW_CODE_RE.match(t):  # 台股
        return f"{t}.TW", t, t, False
    if _US_CODE_RE.match(t) and t != "JPY":  # 美股
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

    # 後備：YahooStock（若有）
    if (not snap["now_price"] or not snap["name"]) and YahooStock:
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

def get_stock_analysis(stock_id_input: str) -> str:
    try:
        yf_symbol, yahoo_slug, display_code, _ = normalize_ticker(stock_id_input)
        stock = yf.Ticker(yf_symbol)
        info = getattr(stock, "fast_info", None)
        name = None
        try:
            name = stock.get_info().get("longName") or stock.get_info().get("shortName")
        except Exception:
            pass
        name = name or display_code

        # 即時價
        snap = fetch_realtime_snapshot(yf_symbol, yahoo_slug)
        line1 = f"**{name}（{display_code}）**"
        line2 = f"- 即時：{snap.get('now_price','N/A')} {snap.get('currency','')}"
        if snap.get("change"):
            line2 += f"　{snap['change']}"
        if snap.get("close_time"):
            line2 += f"（{snap['close_time']}）"
        return f"{line1}\n{line2}"
    except Exception as e:
        logger.error(f"股票查詢失敗：{e}", exc_info=True)
        return f"查詢 {stock_id_input} 失敗：{e}"

# ========== 7) 彩票分析 ==========
def _lotto_fallback_scrape(kind: str) -> str:
    """
    後備：直接抓台彩官網文字並以 regex 擷取最新號碼（容易受改版影響，但可救急）
    """
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
    kind = "威力彩" if "威力" in lottery_type_input else ("大樂透" if "大樂" in lottery_type_input else ("539" if "539" in lottery_type_input else lottery_type_input))
    latest_data_str = ""

    # 1) 優先：你的自訂爬蟲
    if LOTTERY_ENABLED:
        try:
            if kind == "威力彩":   latest_data_str = str(lottery_crawler.super_lotto())
            elif kind == "大樂透": latest_data_str = str(lottery_crawler.lotto649())
            elif kind == "539":    latest_data_str = str(lottery_crawler.daily_cash())
            else: return f"不支援 {kind}。"
        except Exception as e:
            logger.warning(f"自訂彩票爬蟲失敗，改用後備：{e}")
            latest_data_str = _lotto_fallback_scrape(kind)
    else:
        # 2) 後備：官網文字
        latest_data_str = _lotto_fallback_scrape(kind)

    # 可選：財神方位（若存在）
    cai_part = ""
    try:
        if caiyunfangwei_crawler:
            cai = caiyunfangwei_crawler.get_caiyunfangwei()
            cai_part = f"今天日期：{cai.get('今天日期','')}\n今日歲次：{cai.get('今日歲次','')}\n財神方位：{cai.get('財神方位','')}\n"
    except Exception:
        cai_part = ""

    prompt = (
        f"你是一位資深彩券分析師。以下是 {kind} 近況/最新號碼資料：\n"
        f"{latest_data_str}\n\n{cai_part}"
        "請用繁體中文寫出：\n"
        "1) 近期走勢重點（高機率區間/熱冷號）\n"
        "2) 選號建議與注意事項（理性與風險聲明）\n"
        "3) 提供三組推薦號碼（依彩種格式呈現，號碼由小到大排序）\n"
        "文字請精煉、條列。"
    )
    messages = [{"role":"system","content":"你是資深彩券分析師。"}, {"role":"user","content":prompt}]
    return get_analysis_reply(messages)

# ========== 8) 翻譯與人設 ==========
async def analyze_sentiment(text: str) -> str:
    msgs = [
        {"role": "system", "content": "Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
        {"role": "user", "content": text},
    ]
    out = await groq_chat_async(msgs, max_tokens=10, temperature=0)
    return (out or "neutral").strip().lower()

async def translate_text(text: str, target_lang_display: str) -> str:
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text."
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{text}"}}'
    return await groq_chat_async([{"role": "system", "content": sys}, {"role": "user", "content": usr}], 800, 0.2)

def set_user_persona(chat_id: str, key: str):
    key_map = {"甜":"sweet", "鹹":"salty", "萌":"moe", "酷":"cool", "random":"random"}
    k = key_map.get(key, key)
    if k == "random":
        k = random.choice(list(PERSONAS.keys()))
    if k not in PERSONAS:
        k = "sweet"
    user_persona[chat_id] = k
    return k

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    k = user_persona.get(chat_id, "sweet")
    p = PERSONAS[k]
    return (
        f"你是一位「{p['title']}」。風格：{p['style']}\n"
        f"使用者情緒：{sentiment}；請調整語氣（開心→一起開心；難過/生氣→先共情再建議；中性→自然聊天）。\n"
        f"回覆使用繁體中文，精煉自然，帶少量表情 {p['emoji']}。"
    )

# ========== 9) TTS / STT ==========
def _upload_audio_sync(audio_bytes: bytes) -> Optional[dict]:
    if not CLOUDINARY_URL:
        return None
    try:
        return cloudinary.uploader.upload(
            io.BytesIO(audio_bytes), resource_type="video", folder="line-bot-tts", format="mp3"
        )
    except Exception as e:
        logger.error(f"Cloudinary 上傳失敗: {e}")
        return None

async def upload_audio_to_cloudinary(audio_bytes: bytes) -> Optional[str]:
    res = await run_in_threadpool(_upload_audio_sync, audio_bytes)
    return res.get("secure_url") if res else None

def _create_tts_with_openai_sync(text: str) -> Optional[bytes]:
    if not openai_client:
        return None
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
    if TTS_PROVIDER == "openai":
        return await run_in_threadpool(_create_tts_with_openai_sync, text)
    if TTS_PROVIDER == "gtts":
        return await run_in_threadpool(_create_tts_with_gtts_sync, text)
    # auto
    if openai_client:
        b = await run_in_threadpool(_create_tts_with_openai_sync, text)
        if b:
            return b
    return await run_in_threadpool(_create_tts_with_gtts_sync, text)

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

# ========== 10) LINE Event Handlers ==========
@handler.add(MessageEvent, message=TextMessageContent)
async def handle_text_message(event: MessageEvent):
    chat_id = get_chat_id(event)
    msg_raw = (event.message.text or "").strip()
    reply_token = event.reply_token

    # 群組需要 @bot 名稱才回
    try:
        bot_info: BotInfoResponse = await line_bot_api.get_bot_info()
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if not msg_raw:
        return

    if getattr(event.source, "type", "") in ("group", "room") and not msg_raw.startswith(f"@{bot_name}"):
        # 若你想預設自動回覆，可移除此條件
        pass

    # 移除 @bot 前綴
    msg = re.sub(f'^@{re.escape(bot_name)}\\s*', "", msg_raw)
    if not msg:
        return

    low = msg.lower()

    try:
        # 主選單
        if low in ("menu", "選單", "主選單"):
            await line_bot_api.reply_message(
                ReplyMessageRequest(reply_token=reply_token, messages=[build_main_menu()])
            )
            return

        # 彩券
        if msg in ("大樂透", "威力彩", "539"):
            report = await run_in_threadpool(get_lottery_analysis, msg)
            reply_with_quick_bar(reply_token, report)
            return

        # 金價
        if low in ("金價", "黃金"):
            out = await run_in_threadpool(get_gold_analysis)
            reply_with_quick_bar(reply_token, out)
            return

        # 匯率
        if low.upper() in ("JPY", "USD", "EUR"):
            out = await run_in_threadpool(get_currency_analysis, low.upper())
            reply_with_quick_bar(reply_token, out)
            return

        # 股票
        if re.fullmatch(r"\^?[A-Z0-9.]{2,10}", msg) or msg.isdigit() or msg in ("台股大盤", "美股大盤", "大盤", "美股"):
            out = await run_in_threadpool(get_stock_analysis, msg)
            reply_with_quick_bar(reply_token, out)
            return

        # 人設
        if msg in ("甜", "鹹", "萌", "酷", "random"):
            key = set_user_persona(chat_id, msg)
            p = PERSONAS[user_persona[chat_id]]
            reply_with_quick_bar(reply_token, f"💖 已切換人設：{p['title']}\n{p['greetings']}")
            return

        # 翻譯模式開關
        if low.startswith("翻譯->"):
            lang = msg.split("->", 1)[1].strip()
            if lang == "結束":
                translation_states.pop(chat_id, None)
                reply_with_quick_bar(reply_token, "✅ 已結束翻譯模式")
            else:
                translation_states[chat_id] = lang
                reply_with_quick_bar(reply_token, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")
            return

        # 翻譯中的訊息
        if chat_id in translation_states:
            out = await translate_text(msg, translation_states[chat_id])
            reply_with_quick_bar(reply_token, out)
            return

        # 一般聊天（情緒 + 人設）
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": msg}]
        final_reply = await groq_chat_async(messages)
        history.extend([{"role": "user", "content": msg}, {"role": "assistant", "content": final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN * 2 :]
        reply_with_quick_bar(reply_token, final_reply)

    except Exception as e:
        logger.error(f"指令處理失敗: {e}", exc_info=True)
        reply_with_quick_bar(reply_token, "抱歉，處理時發生錯誤 😵")

@handler.add(MessageEvent, message=AudioMessageContent)
async def handle_audio_message(event: MessageEvent):
    reply_token = event.reply_token
    try:
        # 下載語音 -> 轉文字
        content_stream = await line_bot_api.get_message_content(event.message.id)
        audio_in = await content_stream.read()
        text = await speech_to_text_async(audio_in)
        if not text:
            raise RuntimeError("語音轉文字失敗")

        # 生成回覆 + TTS
        sentiment = await analyze_sentiment(text)
        sys_prompt = build_persona_prompt(get_chat_id(event), sentiment)
        final_reply_text = await groq_chat_async(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}]
        )

        messages_to_send = [
            TextMessage(text=f"🎧 我聽到了：\n{text}\n\n—\n{final_reply_text}", quick_reply=build_quick_reply())
        ]

        if final_reply_text and CLOUDINARY_URL:
            audio_out = await text_to_speech_async(final_reply_text)
            if audio_out:
                public_audio_url = await upload_audio_to_cloudinary(audio_out)
                if public_audio_url:
                    est_dur = max(3000, min(30000, len(final_reply_text) * 60))
                    messages_to_send.append(
                        AudioMessage(original_content_url=public_audio_url, duration=est_dur)
                    )
                    logger.info("✅ 成功上傳 TTS 語音並加入回覆佇列。")

        await line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=messages_to_send)
        )

    except Exception as e:
        logger.error(f"處理語音訊息失敗: {e}", exc_info=True)
        await line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text="抱歉，我沒聽清楚，可以再說一次嗎？")])
        )

@handler.add(PostbackEvent)
async def handle_postback(event: PostbackEvent):
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
    body = await request.body()
    try:
        await handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Callback 處理失敗：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse({"status": "ok"})

@router.get("/")
async def root():
    return PlainTextResponse("LINE Bot is running.")

@router.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

app.include_router(router)

# ========== 12) Local run ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)