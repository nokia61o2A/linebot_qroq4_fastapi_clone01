# app_fastapi.py
# ========== 1) Imports ==========
import os
import re
import io
import random
import logging
import asyncio
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from datetime import datetime

# --- 數據處理與爬蟲 ---
import requests
from bs4 import BeautifulSoup
import httpx
import pandas as pd
import yfinance as yf

# --- FastAPI 與 LINE Bot SDK v2 ---
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

# --- 雲端儲存 (Cloudinary) ---
import cloudinary
import cloudinary.uploader

# --- gTTS（免費 TTS） ---
from gtts import gTTS

# --- LINE Bot SDK v3 Imports ---
# from linebot.v3 import WebhookHandler <- 舊的同步 Handler，我們不再使用
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    AudioMessageContent,
    PostbackEvent,
    AsyncWebhookHandler,  # <--- 【修改點 1】: 引入 AsyncWebhookHandler
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

# --- AI 相關 ---
from groq import AsyncGroq, Groq
import openai

<<<<<<< HEAD
# --- 自訂模組（錯誤處理） ---
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    LOTTERY_ENABLED = True
except ImportError:
    logging.warning("無法載入彩票模組，彩票功能將停用。")
    LOTTERY_ENABLED = False

try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
    STOCK_ENABLED = True
except ImportError as e:
    logging.warning(f"無法載入股票模組，股票功能將停用。錯誤: {e}")
    STOCK_ENABLED = False

=======
>>>>>>> fixlottery
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
    logger.warning("未設定 CLOUDINARY_URL，TTS 語音訊息將無法傳送。")

# --- API 用戶端初始化 ---
configuration = Configuration(access_token=CHANNEL_TOKEN)
async_api_client = ApiClient(configuration=configuration)
line_bot_api = AsyncMessagingApi(api_client=async_api_client)
handler = AsyncWebhookHandler(CHANNEL_SECRET)  # <--- 【修改點 2】: 使用 AsyncWebhookHandler 建立 handler

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
sync_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

openai_client = None
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("未設定 OPENAI_API_KEY，語音轉文字與 OpenAI TTS 將停用（將以 gTTS 為主）。")

<<<<<<< HEAD
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

if LOTTERY_ENABLED:
=======
# Groq 模型（改用未下架版本）
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

# --- 【靈活載入】自訂模組（可無則降級爬蟲） ---
LOTTERY_ENABLED = True
try:
    # 你專案中的自訂爬蟲（建議優先用）
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
>>>>>>> fixlottery
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()
    logger.info("✅ 已載入自訂 TaiwanLotteryCrawler / CaiyunfangweiCrawler")
except Exception as e:
    logger.warning(f"無法載入自訂彩票模組：{e}，將使用後備解析。")
    LOTTERY_ENABLED = False  # 若要強制啟用，也可設 True，會走 fallback 爬蟲

# 股票相關（價格、新聞、基本面、配息、Yahoo 爬蟲）
STOCK_ENABLED = True
try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
except Exception as e:
    logger.warning(f"無法載入股票模組：{e}；將只顯示基本快照。")
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
<<<<<<< HEAD
=======
LANGUAGE_MAP = {"英文": "English", "日文": "Japanese", "韓文": "Korean", "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"}
PERSONA_ALIAS = {"甜":"sweet", "鹹":"salty", "萌":"moe", "酷":"cool", "random":"random"}
>>>>>>> fixlottery

# ========== 3) FastAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    嘗試兩個官方端點：api-data 與 api（避免區域性限制）
    """
    if BASE_URL:
        async with httpx.AsyncClient() as c:
            for endpoint in (
                "https://api-data.line.me/v2/bot/channel/webhook/endpoint",
                "https://api.line.me/v2/bot/channel/webhook/endpoint",
            ):
                try:
                    headers = {
                        "Authorization": f"Bearer {CHANNEL_TOKEN}",
                        "Content-Type": "application/json",
                    }
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

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.2.0")
router = APIRouter()

# ========== 4) Helpers ==========
def get_chat_id(event: MessageEvent) -> str:
    """
    透過判斷 source.type 取得 chat ID（不依賴型別檢查，較穩定）
    """
    source = event.source
    if getattr(source, "type", "") == "group":
        return source.group_id
    if getattr(source, "type", "") == "room":
        return source.room_id
    return source.user_id

<<<<<<< HEAD
# ---------- 金價抓取（強化版，對應台銀文字內容） ----------
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
=======
def build_quick_reply() -> QuickReply:
    return QuickReply(items=[
        QuickReplyButton(action=MessageAction(label="主選單", text="選單")),
        QuickReplyButton(action=MessageAction(label="台股大盤", text="台股大盤")),
        QuickReplyButton(action=MessageAction(label="美股大盤", text="美股大盤")),
        QuickReplyButton(action=MessageAction(label="黃金價格", text="金價")),
        QuickReplyButton(action=MessageAction(label="查台積電", text="2330")),
        QuickReplyButton(action=MessageAction(label="查輝達", text="NVDA")),
        QuickReplyButton(action=MessageAction(label="查日圓", text="JPY")),
        QuickReplyButton(action=PostbackAction(label="💖 AI 人設", data="menu:persona")),
        QuickReplyButton(action=PostbackAction(label="🎰 彩票選單", data="menu:lottery")),
    ])
>>>>>>> fixlottery

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

    return {
        "listed_at": listed_at,
        "sell_twd_per_g": sell,
        "buy_twd_per_g": buy,
        "source": BOT_GOLD_URL,
    }

def get_bot_gold_quote() -> dict:
    r = requests.get(BOT_GOLD_URL, headers=DEFAULT_HEADERS, timeout=10)
    r.raise_for_status()
    return parse_bot_gold_text(r.text)

def format_gold_report(data: dict) -> str:
    ts = data.get("listed_at") or "（頁面未標示）"
    sell = data["sell_twd_per_g"]
    buy = data["buy_twd_per_g"]
    spread = sell - buy
    bias = "盤整" if spread <= 30 else ("偏寬" if spread <= 60 else "價差偏大")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return (
        f"**金價快報（台灣銀行）**\n"
        f"- 資料時間：{ts}\n"
        f"- 本行賣出（1克）：**{sell:,.0f} 元**\n"
        f"- 本行買進（1克）：**{buy:,.0f} 元**\n"
        f"- 買賣價差：{spread:,.0f} 元（{bias}）\n"
        f"\n資料來源：{BOT_GOLD_URL}\n（更新於 {now}）"
    )

<<<<<<< HEAD
def get_gold_analysis() -> str:
    try:
        data = get_bot_gold_quote()
        return format_gold_report(data)
    except Exception as e:
        logger.error(f"金價流程失敗：{e}", exc_info=True)
        return "抱歉，目前無法從台灣銀行取得黃金牌價。稍後再試一次 🙏"

# ---------- 匯率 ----------
def get_currency_analysis(target_currency: str) -> str:
    try:
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("result") == "success":
            rate = data["rates"].get("TWD")
            if rate is None:
                return "抱歉，API中找不到 TWD 的匯率資訊。"
            return f"最新：1 {target_currency.upper()} ≈ {rate:.5f} 新台幣"
        else:
            return f"抱歉，獲取匯率資料失敗：{data.get('error-type', '未知錯誤')}"
    except Exception as e:
        logger.error(f"處理 {target_currency} API 資料時發生錯誤: {e}", exc_info=True)
        return "抱歉，處理外匯資料時發生內部錯誤，請稍後再試。"

# ---------- 標準聊天/分析 ----------
def get_analysis_reply(messages: List[dict]) -> str:
    try:
        if openai_client:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, max_tokens=1500, temperature=0.7
=======
def build_main_menu_flex() -> FlexSendMessage:
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[TextComponent(text="AI 助理主選單", weight="bold", size="lg")]),
        body=BoxComponent(
            layout="vertical", spacing="md",
            contents=[
                TextComponent(text="請選擇功能分類：", size="sm"),
                SeparatorComponent(margin="md"),
                ButtonComponent(action=PostbackAction(label="💹 金融查詢", data="menu:finance"), style="primary", color="#5E86C1"),
                ButtonComponent(action=PostbackAction(label="🎰 彩票分析", data="menu:lottery"), style="primary", color="#5EC186"),
                ButtonComponent(action=PostbackAction(label="💖 AI 角色扮演", data="menu:persona"), style="secondary"),
                ButtonComponent(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate"), style="secondary"),
                ButtonComponent(action=PostbackAction(label="⚙️ 系統設定", data="menu:settings"), style="secondary"),
            ]
        )
    )
    return FlexSendMessage(alt_text="主選單", contents=bubble)

def build_submenu_flex(kind: str) -> FlexSendMessage:
    title = "子選單"
    buttons = []
    if kind == "finance":
        title = "💹 金融查詢"
        buttons = [
            ButtonComponent(action=MessageAction(label="台股大盤", text="台股大盤")),
            ButtonComponent(action=MessageAction(label="美股大盤", text="美股大盤")),
            ButtonComponent(action=MessageAction(label="黃金價格", text="金價")),
            ButtonComponent(action=MessageAction(label="日圓匯率", text="JPY")),
            ButtonComponent(action=MessageAction(label="查 2330 台積電", text="2330")),
            ButtonComponent(action=MessageAction(label="查 NVDA 輝達", text="NVDA")),
        ]
    elif kind == "lottery":
        title = "🎰 彩票分析"
        buttons = [
            ButtonComponent(action=MessageAction(label="大樂透", text="大樂透")),
            ButtonComponent(action=MessageAction(label="威力彩", text="威力彩")),
            ButtonComponent(action=MessageAction(label="今彩539", text="539")),
        ]
    elif kind == "persona":
        title = "💖 AI 角色扮演"
        buttons = [
            ButtonComponent(action=MessageAction(label="甜美女友", text="甜")),
            ButtonComponent(action=MessageAction(label="傲嬌女友", text="鹹")),
            ButtonComponent(action=MessageAction(label="萌系女友", text="萌")),
            ButtonComponent(action=MessageAction(label="酷系御姐", text="酷")),
            ButtonComponent(action=MessageAction(label="隨機切換", text="random")),
        ]
    elif kind == "translate":
        title = "🌐 翻譯工具"
        buttons = [
            ButtonComponent(action=MessageAction(label="翻成英文", text="翻譯->英文")),
            ButtonComponent(action=MessageAction(label="翻成日文", text="翻譯->日文")),
            ButtonComponent(action=MessageAction(label="翻成繁中", text="翻譯->繁體中文")),
            ButtonComponent(action=MessageAction(label="結束翻譯模式", text="翻譯->結束")),
        ]
    elif kind == "settings":
        title = "⚙️ 系統設定"
        buttons = [
            ButtonComponent(action=MessageAction(label="開啟自動回答 (群組)", text="開啟自動回答")),
            ButtonComponent(action=MessageAction(label="關閉自動回答 (群組)", text="關閉自動回答")),
        ]
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[TextComponent(text=title, weight="bold", size="lg")]),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm")
    )
    return FlexSendMessage(alt_text=title, contents=bubble)

# ========== 5) AI & 分析 ==========
def get_analysis_reply(messages: List[dict]) -> str:
    """先試 OpenAI（若有），失敗改用 Groq。"""
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
>>>>>>> fixlottery
            )
            return resp.choices[0].message.content
        raise Exception("OpenAI client not initialized.")
    except Exception as openai_err:
        logger.warning(f"OpenAI API 失敗: {openai_err}")
        try:
            if not sync_groq_client:
                raise Exception("Groq client not initialized.")
            resp = sync_groq_client.chat.completions.create(
                model=GROQ_MODEL_PRIMARY, messages=messages, max_tokens=2000, temperature=0.7
            )
            return resp.choices[0].message.content
        except Exception as groq_err:
            logger.warning(f"Groq 主要模型失敗: {groq_err}")
            try:
                if not sync_groq_client:
                    raise Exception("Groq client not initialized.")
                resp = sync_groq_client.chat.completions.create(
                    model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=1500, temperature=0.9
                )
                return resp.choices[0].message.content
            except Exception as fallback_err:
                logger.error(f"所有 AI API 都失敗: {fallback_err}")
                return "（分析模組暫時連線不穩定）"

async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
<<<<<<< HEAD
    if not async_groq_client:
        return await run_in_threadpool(lambda: get_analysis_reply(messages))
    resp = await async_groq_client.chat.completions.create(
        model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    return resp.choices[0].message.content.strip()

# ---------- 彩票 ----------
def get_lottery_analysis(lottery_type_input: str) -> str:
    if not LOTTERY_ENABLED:
        return "彩票模組未啟用。"
    
    lt = lottery_type_input.lower()
    if "威力" in lt:
        data = lottery_crawler.super_lotto()
        lottery_name = "威力彩"
    elif "大樂" in lt:
        data = lottery_crawler.lotto649()
        lottery_name = "大樂透"
    elif "539" in lt:
        data = lottery_crawler.daily_cash()
        lottery_name = "今彩539"
    else:
        return f"不支援 {lottery_type_input}。"

    extra_info = ""
    try:
        info = caiyunfangwei_crawler.get_caiyunfangwei()
        extra_info = (
            f'***財神方位提示***\n'
            f'國歷：{info.get("今天日期", "未知")}\n'
            f'農曆：{info.get("今日歲次", "未知")}\n'
            f'今日財神方位：**{info.get("財神方位", "未知")}**\n\n'
        )
    except Exception as e:
        logger.warning(f"無法獲取財神方位資訊: {e}")
        extra_info = "財神方位資訊暫時無法獲取。\n\n"

    prompt = (
        f"你是一位專業的樂透彩分析師，請基於以下「{lottery_name}」的最近幾期開獎號碼資料，撰寫一份詳細的趨勢分析報告，並遵循以下指示：\n\n"
        f"1.  **開頭資訊**：請先顯示我提供的「財神方位提示」。\n"
        f"2.  **數據來源**：清楚列出最近幾期的開獎號碼。\n"
        f"   - 資料:\n{data}\n\n"
        f"3.  **趨勢分析**：\n"
        f"   - 分析並列出「最熱門的號碼」(Hot Numbers) 和「最冷門的號碼」(Cold Numbers)。\n"
        f"   - 根據號碼分佈（例如大小、奇偶比例）提供簡要的趨勢觀察。\n\n"
        f"4.  **推薦號碼**：\n"
        f"   - 根據你的專業分析，提供三組推薦號碼。\n"
        f"   - 號碼組合必須符合「{lottery_name}」的遊戲規則（例如：大樂透為6個號碼，威力彩為6+1個號碼）。\n"
        f"   - 每組號碼請由小到大排序。\n\n"
        f"5.  **結語**：最後，請附上一句20字以內、具有勵志感的發財吉祥話。\n\n"
        f"請務必使用台灣用語的繁體中文回覆。"
    )
    
    # 呼叫 AI 模型進行分析
    return get_analysis_reply(
        [{"role": "system", "content": f"你是一位專業且詳細的「{lottery_name}」彩券分析師。"}, {"role": "user", "content": prompt}]
    )

# ---------- 股票（簡版） ----------
def get_stock_analysis(stock_id_input: str) -> str:
    if not STOCK_ENABLED:
        return "股票模組未啟用。"
    try:
        stock = yf.Ticker(f"{stock_id_input}.TW" if stock_id_input.isdigit() else stock_id_input)
        info = stock.info
        name = info.get("longName", stock_id_input)
        price = info.get("currentPrice", "N/A")
        prev_close = info.get("previousClose", "N/A")
        return f"**{name} ({stock_id_input})**\n- 即時股價: {price}\n- 昨日收盤: {prev_close}"
    except Exception as e:
        logger.error(f"股票查詢失敗：{e}", exc_info=True)
        return f"查詢 {stock_id_input} 失敗：{e}"

# --- UI & 對話 Helpers ---
=======
    resp = await async_groq_client.chat(completions_create_kwargs={
        "model": GROQ_MODEL_FALLBACK,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    })
    # 上面寫法可避免版本差異；若你的 groq 套件不支援此語法，改用：
    # resp = await async_groq_client.chat.completions.create(...)
    # 兩者擇一即可。
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        # 若使用舊語法
        return (resp.choices[0].message.content or "").strip()

# ========== 6) 金融工具 ==========
# ---- 6.1 黃金（穩定文字解析，避免 DOM 改版炸裂）----
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def _parse_bot_gold_text(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    # 掛牌時間：2025/09/14 09:31
    m_time = re.search(r"掛牌時間[:：]\s*([0-9]{4}/[0-9]{2}/[0-9]{2}\s+[0-9]{2}:[0-9]{2})", text)
    listed_at = m_time.group(1) if m_time else None

    # 本行賣出/買進（單位通常是 1 克）
    m_sell = re.search(r"本行賣出\s*([0-9,]+(?:\.[0-9]+)?)", text)
    m_buy  = re.search(r"本行買進\s*([0-9,]+(?:\.[0-9]+)?)", text)
    if not (m_sell and m_buy):
        raise RuntimeError("找不到『本行賣出/本行買進』欄位")

    sell = float(m_sell.group(1).replace(",", ""))
    buy  = float(m_buy.group(1).replace(",", ""))

    return {"listed_at": listed_at, "sell_twd_per_g": sell, "buy_twd_per_g": buy}

def get_gold_analysis() -> str:
    logger.info("開始執行黃金價格分析…")
    try:
        r = requests.get(BOT_GOLD_URL, headers=DEFAULT_HEADERS, timeout=10)
        r.raise_for_status()
        data = _parse_bot_gold_text(r.text)

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
        logger.error(f"黃金價格流程失敗: {e}", exc_info=True)
        return "抱歉，目前無法從台灣銀行取得黃金牌價，稍後再試 🙏"

# ---- 6.2 匯率 ----
def get_currency_analysis(target_currency: str):
    logger.info(f"開始執行 {target_currency} 匯率分析…")
    try:
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        if data.get("result") != "success":
            return f"抱歉，獲取匯率資料失敗：{data.get('error-type','未知錯誤')}"
        rate = data["rates"].get("TWD")
        if rate is None:
            return f"抱歉，API 無 TWD 匯率。"
        return f"即時：1 {target_currency.upper()} ≈ {rate:.5f} 新台幣"
    except Exception as e:
        logger.error(f"匯率分析錯誤: {e}", exc_info=True)
        return "抱歉，外匯資料暫時無法取得。"

# ---- 6.3 股票 ----
_TW_CODE_RE = re.compile(r'^\d{4,6}[A-Za-z]?$')     # 2330 / 00937B / 1101B
_US_CODE_RE = re.compile(r'^[A-Za-z]{1,5}$')        # NVDA / AAPL / QQQ

def normalize_ticker(user_text: str) -> Tuple[str, str, str, bool]:
    """
    - 回傳: (yfinance_symbol, yahoo_tw_slug, display_code, is_index)
    - 台股數字代碼（含尾碼字母）加上 .TW 供 yfinance 使用
    - Yahoo 台股頁面 slug 使用「原始大寫代碼」（不加 .TW）
    - 指數：台股大盤/^TWII、美股大盤/^GSPC
    """
    t = user_text.strip().upper()
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

    # 後備：YahooStock（若你專案有）
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

def get_stock_report(user_input: str) -> str:
    yf_symbol, yahoo_slug, display_code, is_index = normalize_ticker(user_input)
    snapshot = fetch_realtime_snapshot(yf_symbol, yahoo_slug)

    # 擴充資料（若你有自訂模組）
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
        f"你現在是一位專業的證券分析師, 依據以下資料寫一份分析報告：\n"
        f"**股票代碼:** {display_code}, **股票名稱:** {snapshot.get('name')}\n"
        f"**即時報價:** {snapshot}\n"
        f"**近期價格資訊:**\n{price_data}\n"
    )
    if value_part:    content_msg += f"**每季營收資訊：**\n{value_part}"
    if dividend_part: content_msg += f"**配息資料：**\n{dividend_part}"
    if news_data:     content_msg += f"**近期新聞資訊：**\n{news_data}\n"
    content_msg += (
        f"請以嚴謹專業的角度寫出 {snapshot.get('name') or display_code} 近期趨勢，"
        f"用繁體中文、Markdown 格式，最後附上連結：{stock_link}"
    )

    system_prompt = (
        "你是專業的台股/美股分析師。請在開頭列出：股名(股號)、現價與漲跌幅、資料時間；"
        "接著分段說明：股價走勢、基本面、技術面、消息面、風險、建議區間與停利目標，最後給綜合結論。"
    )
    msgs = [{"role":"system","content":system_prompt}, {"role":"user","content":content_msg}]
    return get_analysis_reply(msgs)

# ========== 7) 彩票分析 ==========
def _lotto_fallback_scrape(kind: str) -> str:
    """
    當自訂 TaiwanLotteryCrawler 無法使用時的極簡後備方案：
    直接抓台彩官網頁面文字並以 regex 粗略擷取最新一期號碼。
    （若頁面再改版，這段容易失效；建議優先使用 TaiwanLotteryCrawler）
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

    # 1) 優先用你的自訂爬蟲
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
        # 2) 後備：簡單頁面解析
        latest_data_str = _lotto_fallback_scrape(kind)

    # 可選：財運方位（若載入成功）
    cai_part = ""
    try:
        if 'caiyunfangwei_crawler' in globals():
            cai = caiyunfangwei_crawler.get_caiyunfangwei()
            cai_part = f"今天日期：{cai.get('今天日期','')}\n今日歲次：{cai.get('今日歲次','')}\n財神方位：{cai.get('財神方位','')}\n"
    except Exception:
        cai_part = ""

    # 交給 LLM 產出趨勢與建議
    prompt = (
        f"你是一位資深彩券分析師。以下是 {kind} 近況/最新號碼資料：\n"
        f"{latest_data_str}\n\n{cai_part}"
        "請用繁體中文寫出：\n"
        "1) 近期走勢重點（高機率區間/熱冷號）\n"
        "2) 選號建議與注意事項（理性與風險聲明）\n"
        "3) 提供三組推薦號碼（依彩種格式呈現）\n"
        "文字請精煉、分點條列。"
    )
    messages = [{"role":"system","content":"你是資深彩券分析師。"}, {"role":"user","content":prompt}]
    return get_analysis_reply(messages)

# ========== 8) 對話與翻譯 ==========
>>>>>>> fixlottery
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
    key_mapped = {
        "甜": "sweet", "鹹": "salty", "萌": "moe", "酷": "cool", "random": "random",
    }.get(key, key)

    if key_mapped == "random":
        key_mapped = random.choice(list(PERSONAS.keys()))
    if key_mapped not in PERSONAS:
        key_mapped = "sweet"

    user_persona[chat_id] = key_mapped
    return key_mapped

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    key = user_persona.get(chat_id, "sweet")
    p = PERSONAS[key]
    return (
        f"你是一位「{p['title']}」。風格：{p['style']}\n"
        f"使用者情緒：{sentiment}；請調整語氣（開心→一起開心；難過/生氣→先共情、安撫再給建議；中性→自然聊天）。\n"
        f"回覆使用繁體中文，精煉自然，帶少量表情 {p['emoji']}。"
    )

<<<<<<< HEAD
# --- Quick Reply / Flex ---
def build_quick_reply() -> QuickReply:
    actions = [
        MessageAction(label="主選單", text="選單"),
        MessageAction(label="台股大盤", text="^TWII"),
        MessageAction(label="查台積電", text="2330"),
        PostbackAction(label="💖 AI 人設", data="menu:persona"),
    ]
    return QuickReply(items=[QuickReplyItem(action=a) for a in actions])

def build_flex_menu(title, items_data, alt_text):
    buttons = []
    for label, action_obj, _ in items_data:
        buttons.append(FlexButton(action=action_obj))
    bubble = FlexBubble(
        header=FlexBox(layout="vertical", contents=[FlexText(text=title, weight="bold", size="lg")]),
        body=FlexBox(layout="vertical", spacing="md", contents=buttons),
    )
    return FlexMessage(alt_text=alt_text, contents=bubble)

def build_main_menu():
    items = [
        ("💹 金融查詢", PostbackAction(label="💹 金融查詢", data="menu:finance"), "finance"),
        ("🎰 彩票分析", PostbackAction(label="🎰 彩票分析", data="menu:lottery"), "lottery"),
        ("💖 AI 角色扮演", PostbackAction(label="💖 AI 角色扮演", data="menu:persona"), "persona"),
        ("🌐 翻譯工具", PostbackAction(label="🌐 翻譯工具", data="menu:translate"), "translate"),
    ]
    return build_flex_menu("AI 助理主選單", items, "主選單")

def build_submenu(kind: str):
    menus = {
        "finance": ("💹 金融查詢", [
            ("台股大盤", MessageAction(label="台股大盤", text="^TWII"), ""),
            ("美股 S&P500", MessageAction(label="美股 S&P500", text="^GSPC"), ""),
            ("黃金價格", MessageAction(label="黃金價格", text="金價"), ""),
            ("日圓匯率", MessageAction(label="日圓匯率", text="JPY"), ""),
        ]),
        "lottery": ("🎰 彩票分析", [
            ("大樂透", MessageAction(label="大樂透", text="大樂透"), ""),
            ("威力彩", MessageAction(label="威力彩", text="威力彩"), ""),
            ("今彩539", MessageAction(label="今彩539", text="539"), ""),
        ]),
        "persona": ("💖 AI 角色扮演", [
            ("甜美女友", MessageAction(label="甜美女友", text="甜"), ""),
            ("傲嬌女友", MessageAction(label="傲嬌女友", text="鹹"), ""),
            ("萌系女友", MessageAction(label="萌系女友", text="萌"), ""),
            ("酷系御姐", MessageAction(label="酷系御姐", text="酷"), ""),
        ]),
        "translate": ("🌐 翻譯工具", [
            ("翻成英文", MessageAction(label="翻成英文", text="翻譯->英文"), ""),
            ("翻成日文", MessageAction(label="翻成日文", text="翻譯->日文"), ""),
            ("結束翻譯", MessageAction(label="結束翻譯", text="翻譯->結束"), ""),
        ]),
    }
    title, items = menus.get(kind, ("無效選單", []))
    return build_flex_menu(title, items, title)

# ========== 5) 上傳與 TTS ==========
def _upload_audio_sync(audio_bytes: bytes) -> Optional[dict]:
    if not CLOUDINARY_URL: return None
=======
# ========== 9) LINE Handlers ==========
@handler.add(MessageEvent, message=TextMessage)
def on_message_text(event: MessageEvent):
>>>>>>> fixlottery
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
    async def try_gtts(): return await run_in_threadpool(_create_tts_with_gtts_sync, text)
    if provider == "openai": return await try_openai()
    if provider == "gtts": return await try_gtts()
    if openai_client:
        b = await try_openai()
        if b: return b
    return await try_gtts()

# ---------- STT（語音轉文字） ----------
def _transcribe_with_openai_sync(audio_bytes: bytes, filename: str = "audio.m4a") -> Optional[str]:
    if not openai_client: return None
    try:
        f = io.BytesIO(audio_bytes)
        f.name = filename
        resp = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
        return (resp.text or "").strip() or None
    except Exception as e:
        logger.warning(f"OpenAI STT 失敗：{e}")
        return None

def _transcribe_with_groq_sync(audio_bytes: bytes, filename: str = "audio.m4a") -> Optional[str]:
    if not sync_groq_client: return None
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
    if text: return text
    return await run_in_threadpool(_transcribe_with_groq_sync, audio_bytes)

# ========== 6) LINE Event Handlers ==========
@handler.add(MessageEvent, message=TextMessageContent)
async def handle_text_message(event: MessageEvent):
    chat_id, msg, reply_token = get_chat_id(event), event.message.text.strip(), event.reply_token
    try:
        bot_info: BotInfoResponse = await line_bot_api.get_bot_info()
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if hasattr(event.source, "group_id") or hasattr(event.source, "room_id"):
        if not msg.startswith(f"@{bot_name}"):
            return
        msg = re.sub(f'^@{re.escape(bot_name)}\\s*', "", msg)

    if not msg: return

<<<<<<< HEAD
    final_reply_text, low = "", msg.lower()
    try:
        if low in ("menu", "選單"):
            await line_bot_api.reply_message(
                ReplyMessageRequest(reply_token=reply_token, messages=[build_main_menu()])
            )
            return
        elif low in ("大樂透", "威力彩", "539"):
            final_reply_text = get_lottery_analysis(low)
        elif low in ("金價", "黃金"):
            final_reply_text = get_gold_analysis()
        elif low.upper() in ("JPY", "USD", "EUR"):
            final_reply_text = get_currency_analysis(low)
        elif re.fullmatch(r"\^?[A-Z0-9.]{2,10}", msg) or msg.isdigit():
            final_reply_text = get_stock_analysis(msg.upper())
        elif low in ("甜", "鹹", "萌", "酷", "random"):
            key = set_user_persona(chat_id, low)
            p = PERSONAS[key]
            final_reply_text = f"💖 已切換人設：{p['title']}\n{p['greetings']}"
        elif low.startswith("翻譯->"):
            lang = low.split("->", 1)[1].strip()
            if lang == "結束":
                translation_states.pop(chat_id, None)
                final_reply_text = "✅ 已結束翻譯模式"
            else:
                translation_states[chat_id] = lang
                final_reply_text = f"🌐 已開啟翻譯 → {lang}"
        elif chat_id in translation_states:
            final_reply_text = await translate_text(msg, translation_states[chat_id])
        else:
            sentiment = await analyze_sentiment(msg)
            sys_prompt = build_persona_prompt(chat_id, sentiment)
            history = conversation_history.setdefault(chat_id, [])
            messages = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": msg}]
            final_reply_text = await groq_chat_async(messages)
            history.extend([{"role": "user", "content": msg}, {"role": "assistant", "content": final_reply_text}])
            conversation_history[chat_id] = history[-MAX_HISTORY_LEN * 2 :]
=======
    if is_group and not auto_reply_status.get(chat_id, True) and not msg_raw.startswith(f"@{bot_name}"):
        return

    msg = msg_raw[len(f"@{bot_name}"):].strip() if msg_raw.startswith(f"@{bot_name}") else msg_raw
    if not msg:
        return

    low = msg.lower()

    # --- 功能路由 ---
    if low in ("menu", "選單", "主選單"):
        return line_bot_api.reply_message(reply_token, build_main_menu_flex())

    # 彩票
    if msg in ["大樂透", "威力彩", "539"]:
        try:
            report = await run_in_threadpool(get_lottery_analysis, msg)
            return reply_with_quick_bar(reply_token, report)
        except Exception as e:
            logger.error(f"彩票分析流程失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")

    # 金價
    if low in ("金價", "黃金"):
        try:
            out = await run_in_threadpool(get_gold_analysis)
            return reply_with_quick_bar(reply_token, out)
        except Exception as e:
            logger.error(f"黃金分析流程失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "抱歉，金價分析服務暫時無法使用。")

    # 匯率
    if low == "jpy":
        try:
            out = await run_in_threadpool(get_currency_analysis, "JPY")
            return reply_with_quick_bar(reply_token, out)
        except Exception as e:
            logger.error(f"日圓分析流程失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "抱歉，日圓匯率分析服務暫時無法使用。")

    # 股票
    if is_stock_query(msg):
        try:
            report = await run_in_threadpool(get_stock_report, msg)
            return reply_with_quick_bar(reply_token, report)
        except Exception as e:
            logger.error(f"股票分析流程失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")

    # 自動回覆設定
    if low in ("開啟自動回答", "關閉自動回答"):
        is_on = low == "開啟自動回答"
        auto_reply_status[chat_id] = is_on
        text = "✅ 已開啟自動回答" if is_on else "❌ 已關閉自動回答（群組需 @我 才回）"
        return reply_with_quick_bar(reply_token, text)

    # 翻譯模式
    if low.startswith("翻譯->"):
        lang = msg.split("->", 1)[1].strip()
        if lang == "結束":
            translation_states.pop(chat_id, None)
            return reply_with_quick_bar(reply_token, "✅ 已結束翻譯模式")
        translation_states[chat_id] = lang
        return reply_with_quick_bar(reply_token, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")

    # 人設切換
    if msg in PERSONA_ALIAS:
        key = set_user_persona(chat_id, PERSONA_ALIAS[msg])
        p = PERSONAS[user_persona[chat_id]]
        txt = f"💖 已切換人設：{p['title']}\n\n{p['greetings']}"
        return reply_with_quick_bar(reply_token, txt)

    # 翻譯內容
    if chat_id in translation_states:
        out = await translate_text(msg, translation_states[chat_id])
        return reply_with_quick_bar(reply_token, out)

    # 一般聊天（人設 + 情緒）
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]
        final_reply = get_analysis_reply(messages)  # 同步即可
        history.extend([{"role":"user","content":msg}, {"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
        return reply_with_quick_bar(reply_token, final_reply)
>>>>>>> fixlottery
    except Exception as e:
        logger.error(f"指令 '{msg}' 處理失敗: {e}", exc_info=True)
        final_reply_text = "抱歉，處理時發生錯誤 😵"

<<<<<<< HEAD
    messages_to_send = [TextMessage(text=final_reply_text, quick_reply=build_quick_reply())]
    if final_reply_text and CLOUDINARY_URL:
        audio_bytes = await text_to_speech_async(final_reply_text)
        if audio_bytes:
            public_audio_url = await upload_audio_to_cloudinary(audio_bytes)
            if public_audio_url:
                est_dur = max(3000, min(30000, len(final_reply_text) * 60))
                messages_to_send.append(
                    AudioMessage(original_content_url=public_audio_url, duration=est_dur)
                )
                logger.info("✅ 成功上傳 TTS 語音並加入回覆佇列。")
    await line_bot_api.reply_message(
        ReplyMessageRequest(reply_token=reply_token, messages=messages_to_send)
    )

@handler.add(MessageEvent, message=AudioMessageContent)
async def handle_audio_message(event: MessageEvent):
    reply_token = event.reply_token
    try:
        content_stream = await line_bot_api.get_message_content(event.message.id)
        audio_in = await content_stream.read()
        text = await speech_to_text_async(audio_in)
        if not text: raise RuntimeError("語音轉文字失敗")

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
    data = event.postback.data
    if data.startswith("menu:"):
        kind = data.split(":", 1)[-1]
        await line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=event.reply_token, messages=[build_submenu(kind)])
        )

# ========== 7) FastAPI Routes ==========
=======
# ========== 10) FastAPI Routes ==========
>>>>>>> fixlottery
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

app.include_router(router)

<<<<<<< HEAD
# ========== 8) Local run ==========
=======
# ========== 11) Local run ==========
>>>>>>> fixlottery
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)