# app_fastapi.py (Version 2.0.0 - Based on User's V2 SDK Example + Fixes)
# ========== 1) Imports ==========
import os
import re
import random
import logging # --- 繁體中文解：提前匯入 logging ---
import asyncio
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime

# --- 數據處理與爬蟲 ---
import requests
from bs4 import BeautifulSoup
import httpx
import pandas as pd
import yfinance as yf
import mplfinance as mpf # --- 繁體中文解：確保 mplfinance 已匯入 ---

# --- FastAPI 與 LINE Bot SDK v2 ---
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool # --- 繁體中文解：V2 handler 需要這個 ---

# --- 繁體中文解：使用 V2 SDK 的匯入 ---
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    QuickReply, QuickReplyButton, MessageAction,
    PostbackAction, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent,
    TextComponent, ButtonComponent, SeparatorComponent
)

# --- AI 相關 ---
from groq import AsyncGroq, Groq
import openai

# ========== 2) Setup ==========
# --- 繁體中文解：[修正] 將 logger 初始化移到最前面 ---
logger = logging.getLogger("uvicorn.error") # 使用 uvicorn 的 logger 以確保輸出
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s') # 基本設定

# --- 環境變數 ---
BASE_URL = os.getenv("BASE_URL")
# --- 繁體中文解：[修正] 讀取 CHANNEL_ACCESS_TOKEN (符合 Render 設定) ---
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # 可不設，會自動改用 Groq

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    # --- 繁體中文解：在拋出錯誤前先記錄日誌 ---
    logger.critical("❌ 缺少必要環境變數：BASE_URL / CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET / GROQ_API_KEY")
    raise RuntimeError("缺少必要環境變數：BASE_URL / CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET / GROQ_API_KEY")
else:
    logger.info("✅ 必要環境變數已載入")

# --- API 用戶端初始化 (V2 SDK) ---
try:
    line_bot_api = LineBotApi(CHANNEL_TOKEN)
    handler = WebhookHandler(CHANNEL_SECRET)
    logger.info("✅ LINE Bot API (V2) 初始化成功")
except Exception as e:
    logger.critical(f"❌ LINE Bot API 初始化失敗: {e}", exc_info=True)
    # 在 Render 上，若這裡失敗，部署通常會中止，但本地運行需要處理
    line_bot_api = None
    handler = None

# --- AI Client 初始化 ---
async_groq_client = None
sync_groq_client = None
if GROQ_API_KEY:
    try:
        async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
        sync_groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✅ Groq API Client 初始化成功")
    except Exception as e:
        logger.error(f"❌ Groq API Client 初始化失敗: {e}")
else:
    logger.warning("⚠️ 未設定 GROQ_API_KEY")


openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info("✅ OpenAI Client 初始化成功")
    except Exception as e:
        logger.warning(f"⚠️ 初始化 OpenAI 失敗：{e}")
else:
    logger.info("ℹ️ 未設定 OPENAI_API_KEY，將僅使用 Groq")


# Groq 模型（改用未下架版本）
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama3-70b-8192") # 注意: llama-3.3 可能尚未普遍可用
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama3-8b-8192") # 注意: llama-3.1-8b-instant 可能是更好的選擇

# --- 【靈活載入】自訂模組（可無則降級爬蟲） ---
LOTTERY_ENABLED = True
try:
    # 你專案中的自訂爬蟲（建議優先用）
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()
    logger.info("✅ 已載入自訂 TaiwanLotteryCrawler / CaiyunfangweiCrawler")
except Exception as e:
    # --- 繁體中文解：[修正] 使用已定義的 logger ---
    logger.warning(f"⚠️ 無法載入自訂彩票模組：{e}，將使用後備解析。")
    LOTTERY_ENABLED = False # 若要強制啟用，也可設 True，會走 fallback 爬蟲
    lottery_crawler = None
    caiyunfangwei_crawler = None

# 股票相關（價格、新聞、基本面、配息、Yahoo 爬蟲）
STOCK_ENABLED = True
try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental # 假設是 stock_value.py
    from my_commands.stock.stock_rate import stock_dividend     # 假設是 stock_rate.py
    from my_commands.stock.YahooStock import YahooStock
    logger.info("✅ 已載入自訂股票模組 (my_commands.stock)")
except Exception as e:
    # --- 繁體中文解：[修正] 使用已定義的 logger ---
    logger.warning(f"⚠️ 無法載入股票模組：{e}；將只顯示基本快照。")
    STOCK_ENABLED = False
    # --- 繁體中文解：定義備援函數/類別，即使匯入失敗也能運行 ---
    def stock_price(id): logger.error(f"股票模組未載入，無法執行 stock_price({id})"); return pd.DataFrame()
    def stock_news(hint): logger.error(f"股票模組未載入，無法執行 stock_news({hint})"); return ["股票模組未載入"]
    def stock_fundamental(id): logger.error(f"股票模組未載入，無法執行 stock_fundamental({id})"); return "股票模組未載入"
    def stock_dividend(id): logger.error(f"股票模組未載入，無法執行 stock_dividend({id})"); return "股票模組未載入"
    class YahooStock:
        def __init__(self, id): logger.error(f"股票模組未載入，無法建立 YahooStock({id})"); self.name=id; self.now_price=None; self.change=None; self.currency=None; self.close_time=None


# --- 狀態字典與常數 ---
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}
auto_reply_status: Dict[str, bool] = {}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji":"🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji":"😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji":"✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議", "greetings": "我在。說重點。", "emoji":"🧊⚡️"}
}
LANGUAGE_MAP = {"英文": "English", "日文": "Japanese", "韓文": "Korean", "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"}
PERSONA_ALIAS = {"甜":"sweet", "鹹":"salty", "萌":"moe", "酷":"cool", "random":"random"}

# ========== 3) FastAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時更新 LINE Webhook
    logger.info("應用程式啟動 (lifespan)...")
    if BASE_URL and CHANNEL_TOKEN != "dummy": # Dummy check
        try:
            async with httpx.AsyncClient() as c:
                headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
                payload = {"endpoint": f"{BASE_URL}/callback"}
                r = await c.put(
                    "https://api.line.me/v2/bot/channel/webhook/endpoint",
                    headers=headers, json=payload, timeout=10.0
                )
                r.raise_for_status()
                logger.info(f"✅ Webhook 更新成功: {r.status_code}")
        except Exception as e:
            logger.error(f"⚠️ Webhook 更新失敗: {e}", exc_info=True)
    else:
        logger.warning("⚠️ Webhook 未更新：未設定 BASE_URL 或 CHANNEL_ACCESS_TOKEN (Mock 模式)")

    # --- 繁體中文解：可以加入 Groq/OpenAI 的啟動健檢 (非必要) ---
    # (省略健檢程式碼，參照 V3 版本)

    logger.info("Lifespan 啟動程序完成，應用程式準備就緒。")
    yield
    logger.info("應用程式關閉 (lifespan)...")


# --- 繁體中文解：使用你的 V2 版本號 ---
app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.2.0-v2-logfix")
router = APIRouter()

# ========== 4) Helpers (V2 SDK Style) ==========
def get_chat_id(event: MessageEvent) -> str:
    # --- 繁體中文解：使用你的 V2 Source 判斷 ---
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom):  return event.source.room_id
    if isinstance(event.source, SourceUser): return event.source.user_id # V2 用 user_id
    logger.warning(f"未知的 event source type: {type(event.source)}")
    return "unknown_source"


def build_quick_reply() -> QuickReply:
    # --- 繁體中文解：使用你的 V2 QuickReply 定義 ---
    logger.debug("建立 QuickReply 按鈕")
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

def reply_with_quick_bar(reply_token: str, text: str):
    # --- 繁體中文解：使用你的 V2 回覆方式 ---
    if not line_bot_api:
        logger.error("LINE Bot API 未初始化，無法回覆")
        print(f"[MOCK] Reply with Quick Bar: {text}")
        return
    try:
        logger.debug(f"準備回覆 (含 QuickReply): {text[:50]}...")
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=text, quick_reply=build_quick_reply())
        )
        logger.debug("回覆 (含 QuickReply) 成功")
    except Exception as e:
        logger.error(f"回覆 (含 QuickReply) 失敗: {e}", exc_info=True)


def build_main_menu_flex() -> FlexSendMessage:
    # --- 繁體中文解：使用你的 V2 FlexMessage 定義 ---
    logger.debug("建立主選單 FlexMessage")
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
     # --- 繁體中文解：使用你的 V2 FlexMessage 定義 ---
    logger.debug(f"建立子選單 FlexMessage (kind={kind})")
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
    logger.debug(f"呼叫 get_analysis_reply (OpenAI優先), messages count: {len(messages)}")
    if openai_client:
        try:
            logger.debug("嘗試使用 OpenAI...")
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini", # 確保模型名稱正確
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
            )
            reply = resp.choices[0].message.content
            logger.debug(f"OpenAI 成功回覆，長度: {len(reply)}")
            return reply
        except Exception as e:
            logger.warning(f"⚠️ OpenAI 失敗：{e}")

    # --- Fallback to Groq ---
    if not sync_groq_client:
        logger.error("❌ Groq Client 未初始化，無法回覆")
        return "抱歉，AI 分析引擎目前無法連線。"

    try:
        logger.debug(f"嘗試使用 Groq 主模型: {GROQ_MODEL_PRIMARY}")
        resp = sync_groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=messages,
            temperature=0.7,
            max_tokens=2000, # 可以設大一點
        )
        reply = resp.choices[0].message.content
        logger.debug(f"Groq 主模型成功回覆，長度: {len(reply)}")
        return reply
    except Exception as e:
        logger.warning(f"⚠️ Groq 主模型失敗：{e}")
        try:
            logger.debug(f"嘗試使用 Groq 備援模型: {GROQ_MODEL_FALLBACK}")
            resp = sync_groq_client.chat.completions.create(
                model=GROQ_MODEL_FALLBACK,
                messages=messages,
                temperature=0.9, # 備援可以活潑一點
                max_tokens=1500,
            )
            reply = resp.choices[0].message.content
            logger.debug(f"Groq 備援模型成功回覆，長度: {len(reply)}")
            return reply
        except Exception as ee:
            logger.error(f"❌ 所有 AI API 都失敗：{ee}", exc_info=True)
            return "抱歉，AI 分析師目前連線不穩定，請稍後再試。"

async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
    # --- 繁體中文解：[修正] 使用新版 groq 的異步呼叫方式 ---
    logger.debug(f"呼叫 groq_chat_async, messages count: {len(messages)}")
    if not async_groq_client:
        logger.error("❌ Async Groq Client 未初始化，無法回覆")
        return "抱歉，AI 聊天引擎目前無法連線。"
    try:
        resp = await async_groq_client.chat.completions.create(
             model=GROQ_MODEL_FALLBACK, # 異步通常用較快的模型
             messages=messages,
             max_tokens=max_tokens,
             temperature=temperature
        )
        reply = (resp.choices[0].message.content or "").strip()
        logger.debug(f"Groq 異步成功回覆，長度: {len(reply)}")
        return reply
    except Exception as e:
        logger.error(f"❌ Groq 異步呼叫失敗: {e}", exc_info=True)
        return "抱歉，AI 聊天暫時出錯了。"


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
    logger.info("呼叫：get_gold_analysis()")
    try:
        r = requests.get(BOT_GOLD_URL, headers=DEFAULT_HEADERS, timeout=10)
        r.raise_for_status()
        data = _parse_bot_gold_text(r.text)
        logger.debug(f"金價原始資料: {data}")

        ts = data.get("listed_at") or "（頁面未標示）"
        sell, buy = data["sell_twd_per_g"], data["buy_twd_per_g"]
        spread = sell - buy
        bias = "盤整" if spread <= 30 else ("偏寬" if spread <= 60 else "價差偏大")
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        report = (
            f"**金價快報（台灣銀行）**\n"
            f"- 掛牌時間：{ts}\n"
            f"- 本行賣出（1克）：**{sell:,.0f} 元**\n"
            f"- 本行買進（1克）：**{buy:,.0f} 元**\n"
            f"- 買賣價差：{spread:,.0f} 元（{bias}）\n"
            f"\n資料來源：{BOT_GOLD_URL}\n（更新於 {now}）"
        )
        logger.info("金價分析成功")
        return report
    except Exception as e:
        logger.error(f"❌ 黃金價格流程失敗: {e}", exc_info=True)
        return "抱歉，目前無法從台灣銀行取得黃金牌價，稍後再試 🙏"

# ---- 6.2 匯率 ----
def get_currency_analysis(target_currency: str):
    logger.info(f"呼叫：get_currency_analysis(target_currency={target_currency})")
    try:
        # --- 繁體中文解：使用你的 V2 範例中的 API ---
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        logger.debug(f"匯率 API 回應: {data}")
        if data.get("result") != "success":
            error_msg = f"抱歉，獲取匯率資料失敗：{data.get('error-type','未知錯誤')}"
            logger.error(error_msg)
            return error_msg
        rate = data["rates"].get("TWD")
        if rate is None:
            logger.error("匯率 API 回應中無 TWD 資料")
            return f"抱歉，API 無 TWD 匯率。"

        report = f"即時：1 {target_currency.upper()} ≈ {rate:.5f} 新台幣"
        logger.info("匯率分析成功")
        return report
    except Exception as e:
        logger.error(f"❌ 匯率分析錯誤: {e}", exc_info=True)
        return "抱歉，外匯資料暫時無法取得。"

# ---- 6.3 股票 ----
_TW_CODE_RE = re.compile(r'^\d{4,6}[A-Za-z]?$')     # 2330 / 00937B / 1101B
_US_CODE_RE = re.compile(r'^[A-Za-z]{1,5}$')       # NVDA / AAPL / QQQ

def normalize_ticker(user_text: str) -> Tuple[str, str, str, bool]:
    """
    - 回傳: (yfinance_symbol, yahoo_tw_slug, display_code, is_index)
    - 台股數字代碼（含尾碼字母）加上 .TW 供 yfinance 使用
    - Yahoo 台股頁面 slug 使用「原始大寫代碼」（不加 .TW）
    - 指數：台股大盤/^TWII、美股大盤/^GSPC
    """
    t = user_text.strip().upper()
    logger.debug(f"標準化股票代碼: {t}")
    if t in ["台股大盤", "大盤"]:
        return "^TWII", "^TWII", "^TWII", True
    if t in ["美股大盤", "美盤", "美股"]:
        return "^GSPC", "^GSPC", "^GSPC", True

    if _TW_CODE_RE.match(t):
        return f"{t}.TW", t, t, False
    # --- 繁體中文解：避免將 JPY 誤判為美股 ---
    if _US_CODE_RE.match(t) and t != "JPY":
        return t, t, t, False
    # --- 繁體中文解：若都不匹配，仍回傳原始值，讓後續處理 ---
    logger.warning(f"無法明確識別的股票/指數代碼: {t}")
    return t, t, t, False # 預設非指數

def fetch_realtime_snapshot(yf_symbol: str, yahoo_slug: str) -> dict:
    logger.debug(f"呼叫 fetch_realtime_snapshot (yf: {yf_symbol}, slug: {yahoo_slug})")
    snap: dict = {"name": None, "now_price": None, "change": None, "currency": None, "close_time": None}
    try:
        tk = yf.Ticker(yf_symbol)
        # --- 繁體中文解：嘗試更可靠的方式獲取資訊 ---
        info = {}
        try: info = tk.info or {} # .info 通常包含較多資訊
        except Exception as info_e: logger.warning(f"yf tk.info 失敗 for {yf_symbol}: {info_e}")

        hist = tk.history(period="2d", interval="1d") # 獲取昨日收盤價計算漲跌

        # 名稱
        name = info.get("shortName") or info.get("longName")
        snap["name"] = name or yf_symbol # 備援使用代碼

        # 價格 & 幣別 (優先使用 regularMarketPrice)
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose") # 多重備援
        ccy = info.get("currency")
        if price:
            snap["now_price"] = f"{price:.2f}"
            snap["currency"] = ccy or ("TWD" if yf_symbol.endswith(".TW") else "USD") # 預設幣別
        elif not hist.empty: # 再度備援：使用歷史收盤價
             price = float(hist["Close"].iloc[-1])
             snap["now_price"] = f"{price:.2f}"
             snap["currency"] = ccy or ("TWD" if yf_symbol.endswith(".TW") else "USD")

        # 變動 (基於歷史資料較可靠)
        if not hist.empty and len(hist) >= 2 and hist["Close"].iloc[-2] != 0: # 避免除以零
            chg = float(hist["Close"].iloc[-1]) - float(hist["Close"].iloc[-2])
            pct = chg / float(hist["Close"].iloc[-2]) * 100
            sign = "+" if chg >= 0 else "" # 正號可省略或保留
            snap["change"] = f"{sign}{chg:.2f} ({sign}{pct:.2f}%)"
        elif info.get('regularMarketChange') is not None and info.get('regularMarketChangePercent') is not None: # 備援：使用 info 的漲跌
             chg = info['regularMarketChange']
             pct = info['regularMarketChangePercent'] * 100
             sign = "+" if chg >= 0 else ""
             snap["change"] = f"{sign}{chg:.2f} ({sign}{pct:.2f}%)"


        # 時間 (使用歷史資料的時間)
        if not hist.empty:
            ts = hist.index[-1]
            snap["close_time"] = ts.strftime("%Y-%m-%d %H:%M") # 通常是收盤時間
        elif info.get("regularMarketTime"):
             # info 的時間可能是 Unix timestamp，需要轉換
             try: snap["close_time"] = datetime.fromtimestamp(info["regularMarketTime"]).strftime("%Y-%m-%d %H:%M")
             except: pass


    except Exception as e:
        logger.warning(f"⚠️ yfinance 取得 {yf_symbol} 失敗：{e}")

    # 後備：YahooStock（若你有載入且 yfinance 失敗）
    if (not snap["now_price"] or not snap["name"]) and STOCK_ENABLED and 'YahooStock' in globals():
        logger.debug(f"yfinance 失敗，嘗試使用 YahooStock 後備 for {yahoo_slug}")
        try:
            ys = YahooStock(yahoo_slug) # 假設你的 YahooStock 能處理 slug
            snap["name"] = ys.name or snap["name"] or yahoo_slug
            snap["now_price"] = ys.now_price or snap["now_price"]
            snap["change"] = ys.change or snap["change"]
            snap["currency"] = ys.currency or ("TWD" if yf_symbol.endswith(".TW") else snap["currency"])
            snap["close_time"] = ys.close_time or snap["close_time"]
            logger.debug("YahooStock 後備成功")
        except Exception as e:
            logger.error(f"❌ YahooStock 取得 {yahoo_slug} 失敗：{e}")

    logger.debug(f"Snapshot 結果: {snap}")
    return snap

# --- 繁體中文解：你的 V2 範例中有這個，加回來 ---
stock_data_df: Optional[pd.DataFrame] = None
def load_stock_data() -> pd.DataFrame:
    global stock_data_df
    if stock_data_df is None:
        try:
            # --- 繁體中文解：請確保你的專案根目錄有 name_df.csv ---
            stock_data_df = pd.read_csv('name_df.csv')
            logger.info("✅ 成功載入 name_df.csv")
        except FileNotFoundError:
            logger.error("❌ `name_df.csv` not found. Stock name lookup disabled.")
            stock_data_df = pd.DataFrame(columns=['股號', '股名']) # 建立空表避免錯誤
    return stock_data_df

def get_stock_name(stock_id_without_suffix: str) -> Optional[str]:
    df = load_stock_data()
    # --- 繁體中文解：確保比較時型別一致 ---
    res = df[df['股號'].astype(str).str.strip().str.upper() == str(stock_id_without_suffix).strip().upper()]
    if not res.empty:
        name = res.iloc[0]['股名']
        logger.debug(f"從 name_df.csv 找到 {stock_id_without_suffix} -> {name}")
        return name
    logger.debug(f"在 name_df.csv 中找不到 {stock_id_without_suffix}")
    return None


def get_stock_report(user_input: str) -> str:
    logger.info(f"呼叫：get_stock_report(user_input={user_input})")
    yf_symbol, yahoo_slug, display_code, is_index = normalize_ticker(user_input)
    snapshot = fetch_realtime_snapshot(yf_symbol, yahoo_slug)

    # 擴充資料（若你有自訂模組且已啟用）
    price_data = ""
    news_data = ""
    value_part = ""
    dividend_part = ""
    if STOCK_ENABLED:
        logger.debug("股票模組已啟用，嘗試獲取詳細資料...")
        try:
            # --- 繁體中文解：決定傳入哪個代碼給你的函數 ---
            input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol
            logger.debug(f"呼叫 stock_price({input_code})")
            price_df = stock_price(input_code) # 假設你的函數能處理
            price_data = str(price_df) if not price_df.empty else "無法取得價格資料"
        except Exception as e:
            logger.warning(f"⚠️ stock_price 失敗：{e}")
            price_data = f"錯誤: {e}"
        try:
            # --- 繁體中文解：嘗試從 CSV 或 Snapshot 獲取名稱給 news ---
            nm = get_stock_name(yahoo_slug) or snapshot.get("name") or yahoo_slug
            logger.debug(f"呼叫 stock_news({nm})")
            news_list = stock_news(nm) # 假設返回 list of strings
            news_data = "\n".join(news_list).replace("\u3000", " ")[:1024] # 合併並清理
        except Exception as e:
            logger.warning(f"⚠️ stock_news 失敗：{e}")
            news_data = f"錯誤: {e}"

        if not is_index: # 指數沒有基本面和配息
            try:
                input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol
                logger.debug(f"呼叫 stock_fundamental({input_code})")
                val = stock_fundamental(input_code)
                value_part = f"{val}\n" if val else ""
            except Exception as e:
                logger.warning(f"⚠️ stock_fundamental 失敗：{e}")
                value_part = f"錯誤: {e}\n"
            try:
                input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol
                logger.debug(f"呼叫 stock_dividend({input_code})")
                dvd = stock_dividend(input_code)
                dividend_part = f"{dvd}\n" if dvd else ""
            except Exception as e:
                logger.warning(f"⚠️ stock_dividend 失敗：{e}")
                dividend_part = f"錯誤: {e}\n"
    else:
        logger.warning("⚠️ 股票模組未啟用，僅顯示快照")


    # --- 繁體中文解：使用你的 V2 範例中的連結邏輯 ---
    stock_link = (
        f"https://finance.yahoo.com/quote/{yf_symbol}"
        if yf_symbol.startswith("^") or not yf_symbol.endswith(".TW") # 指數或美股用 Yahoo Finance
        else f"https://tw.stock.yahoo.com/quote/{yahoo_slug}" # 台股用 Yahoo TW
    )

    # --- 建立給 AI 的 Prompt ---
    content_msg = (
        f"你現在是一位專業的證券分析師, 依據以下資料寫一份分析報告：\n"
        f"**股票代碼:** {display_code}, **股票名稱:** {snapshot.get('name')}\n"
        # --- 繁體中文解：提供更清晰的報價資訊 ---
        f"**目前價格:** {snapshot.get('now_price')} {snapshot.get('currency')}\n"
        f"**今日漲跌:** {snapshot.get('change')}\n"
        f"**資料時間:** {snapshot.get('close_time')}\n"
        f"**近期價格資訊:**\n{price_data}\n"
    )
    if value_part:    content_msg += f"**基本面/營收資訊：**\n{value_part}"
    if dividend_part: content_msg += f"**配息資料：**\n{dividend_part}"
    if news_data:     content_msg += f"**近期新聞資訊：**\n{news_data}\n"
    content_msg += (
        f"請以嚴謹專業的角度寫出 {snapshot.get('name') or display_code} 近期趨勢，"
        f"用繁體中文、Markdown 格式，最後**務必**附上這個連結：{stock_link}" # 強調連結
    )

    system_prompt = (
        "你是專業的台股/美股分析師。請在開頭列出：股名(股號)、現價與漲跌幅、資料時間；"
        "接著分段說明：股價走勢、基本面、技術面、消息面、風險、建議區間與停利目標，最後給綜合結論。"
        "如果資料不完整或有錯誤，請保守說明。" # 增加保守說明提示
    )
    msgs = [{"role":"system","content":system_prompt}, {"role":"user","content":content_msg}]

    logger.info("準備呼叫 AI 進行股票分析...")
    # --- 繁體中文解：使用 get_analysis_reply (同步函數) ---
    analysis_result = get_analysis_reply(msgs)
    logger.info("股票分析完成")
    return analysis_result


# ========== 7) 彩票分析 ==========
def _lotto_fallback_scrape(kind: str) -> str:
    """
    當自訂 TaiwanLotteryCrawler 無法使用時的極簡後備方案：
    直接抓台彩官網頁面文字並以 regex 粗略擷取最新一期號碼。
    （若頁面再改版，這段容易失效；建議優先使用 TaiwanLotteryCrawler）
    """
    logger.warning(f"使用後備彩票爬蟲 for {kind}")
    try:
        if kind == "威力彩":
            url = "https://www.taiwanlottery.com/lotto/superlotto638/index.html"
            # --- 繁體中文解：更寬鬆的比對模式 ---
            pat = r"第\s*\d+\s*期\s*開獎結果.*?第一區(?:中獎)?號碼(?:依大小順序排列)?[:：\s]*([\d\s,]+?)\s*第二區(?:中獎)?號碼[:：\s]*(\d+)"
        elif kind == "大樂透":
            url = "https://www.taiwanlottery.com/lotto/lotto649/index.html"
            pat = r"第\s*\d+\s*期\s*開獎結果.*?(?:中獎號碼|獎號)(?:依大小順序排列)?[:：\s]*([\d\s,]+?)(?:\s*特別號[:：\s]*(\d+))?"
        elif kind == "539":
            url = "https://www.taiwanlottery.com/lotto/dailycash/index.html"
            pat = r"第\s*\d+\s*期\s*開獎結果.*?(?:中獎號碼|獎號)(?:依大小順序排列)?[:：\s]*([\d\s,]+)"
        else:
            return f"不支援彩種：{kind}"

        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        r.raise_for_status()
        # --- 繁體中文解：指定 parser, 清理多餘空格 ---
        soup = BeautifulSoup(r.content, "html.parser") # 使用 r.content 處理編碼
        text = ' '.join(soup.stripped_strings) # 清理空格
        logger.debug(f"後備爬蟲取得文字 (前 200 字): {text[:200]}")

        m = re.search(pat, text, re.DOTALL) # re.DOTALL 讓 . 匹配換行
        if not m:
            logger.error(f"後備爬蟲正則表達式匹配失敗 for {kind}")
            return f"抱歉，暫時找不到 {kind} 最新號碼 (Fallback regex failed)。"

        if kind == "威力彩":
            first = re.sub(r'[,\s]+', ' ', m.group(1)).strip() # 清理數字間的空格和逗號
            second = m.group(2)
            return f"{kind} 最新號碼：第一區 {first}；第二區 {second}"
        elif kind == "大樂透":
            nums = re.sub(r'[,\s]+', ' ', m.group(1)).strip()
            special = m.group(2)
            if special:
                return f"{kind} 最新號碼：{nums}；特別號 {special}"
            else:
                return f"{kind} 最新號碼：{nums}"
        elif kind == "539":
            nums = re.sub(r'[,\s]+', ' ', m.group(1)).strip()
            return f"{kind} 最新號碼：{nums}"

    except Exception as e:
        logger.error(f"❌ 後備彩票爬取失敗：{e}", exc_info=True)
        return f"抱歉，{kind} 近期號碼暫時取不到 (Fallback exception)。"


def get_lottery_analysis(lottery_type_input: str) -> str:
    logger.info(f"呼叫：get_lottery_analysis(lottery_type_input={lottery_type_input})")
    kind = "威力彩" if "威力" in lottery_type_input else ("大樂透" if "大樂" in lottery_type_input else ("539" if "539" in lottery_type_input else lottery_type_input))
    latest_data_str = ""

    # 1) 優先用你的自訂爬蟲
    if LOTTERY_ENABLED and lottery_crawler:
        try:
            logger.debug(f"嘗試使用自訂爬蟲獲取 {kind} 資料...")
            if kind == "威力彩":   latest_data_str = str(lottery_crawler.super_lotto())
            elif kind == "大樂透": latest_data_str = str(lottery_crawler.lotto649())
            elif kind == "539":    latest_data_str = str(lottery_crawler.daily_cash())
            else: return f"不支援 {kind}。"
            logger.info("自訂爬蟲成功獲取資料")
        except Exception as e:
            logger.warning(f"⚠️ 自訂彩票爬蟲失敗，改用後備：{e}")
            latest_data_str = _lotto_fallback_scrape(kind)
    else:
        # 2) 後備：簡單頁面解析
        logger.warning("自訂彩票模組未啟用或未載入，使用後備爬蟲")
        latest_data_str = _lotto_fallback_scrape(kind)

    # 可選：財運方位（若載入成功）
    cai_part = ""
    if caiyunfangwei_crawler:
        try:
            logger.debug("嘗試獲取財運方位...")
            cai = caiyunfangwei_crawler.get_caiyunfangwei() # 假設你的函數是同步的
            cai_part = f"今天日期：{cai.get('今天日期','')}\n今日歲次：{cai.get('今日歲次','')}\n財神方位：{cai.get('財神方位','')}\n"
            logger.info("財運方位獲取成功")
        except Exception as e:
            logger.warning(f"⚠️ 無法獲取財運方位: {e}")
            cai_part = "" # 失敗則不顯示

    # 交給 LLM 產出趨勢與建議
    prompt = (
        f"你是一位資深彩券分析師。以下是 {kind} 近況/最新號碼資料：\n"
        f"{latest_data_str}\n\n{cai_part}" # 加入財運（如果有的話）
        "請用繁體中文寫出：\n"
        "1) 近期走勢重點（高機率區間/熱冷號）\n"
        "2) 選號建議與注意事項（理性與風險聲明）\n"
        "3) 提供三組推薦號碼（依彩種格式呈現）\n"
        "文字請精煉、分點條列。"
    )
    messages = [{"role":"system","content":"你是資深彩券分析師。"}, {"role":"user","content":prompt}]

    logger.info("準備呼叫 AI 進行彩票分析...")
    # --- 繁體中文解：使用 get_analysis_reply (同步函數) ---
    analysis_result = get_analysis_reply(messages)
    logger.info("彩票分析完成")
    return analysis_result

# ========== 8) 對話與翻譯 ==========
async def analyze_sentiment(text: str) -> str:
    # --- 繁體中文解：使用異步 Groq 進行快速情緒分析 ---
    logger.debug(f"呼叫 analyze_sentiment for: {text[:30]}...")
    msgs = [
        {"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
        {"role":"user","content":text}
    ]
    try:
        out = await groq_chat_async(msgs, max_tokens=10, temperature=0) # 低溫求精確
        result = (out or "neutral").strip().lower()
        logger.debug(f"情緒分析結果: {result}")
        # --- 繁體中文解：增加結果驗證 ---
        if result not in ["positive", "neutral", "negative", "angry"]:
             logger.warning(f"情緒分析返回意外結果: {result}, 使用 neutral 作為備援")
             return "neutral"
        return result
    except Exception as e:
        logger.error(f"❌ 情緒分析失敗: {e}", exc_info=True)
        return "neutral" # 失敗時回傳中性

async def translate_text(text: str, target_lang_display: str) -> str:
    logger.debug(f"呼叫 translate_text to {target_lang_display} for: {text[:30]}...")
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display) # 轉換為英文代碼
    sys = "You are a precise translation engine. Output ONLY the translated text, without any introductory phrases or explanations." # 更嚴格的指令
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{text}"}}' # JSON 格式可能更穩定
    try:
        # --- 繁體中文解：使用異步 Groq 進行翻譯 ---
        translated_text = await groq_chat_async([{"role":"system","content":sys},{"role":"user","content":usr}], max_tokens=len(text)*3 + 50, temperature=0.2) # 根據原文長度調整 max_tokens, 低溫求精確
        logger.debug(f"翻譯結果: {translated_text[:50]}...")
        return translated_text
    except Exception as e:
        logger.error(f"❌ 翻譯失敗: {e}", exc_info=True)
        return "抱歉，翻譯功能暫時出錯了。"


def set_user_persona(chat_id: str, key: str):
    logger.debug(f"呼叫 set_user_persona for {chat_id[:10]}... with key={key}")
    # --- 繁體中文解：使用你的 V2 隨機邏輯 ---
    if key == "random": key = random.choice(list(PERSONAS.keys()))
    if key not in PERSONAS: key = "sweet" # 預設回甜美
    user_persona[chat_id] = key
    logger.info(f"人設切換成功: {chat_id[:10]}... -> {key}")
    return key

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    key = user_persona.get(chat_id, "sweet") # 預設甜美
    p = PERSONAS[key]
    prompt = (f"你是一位「{p['title']}」。風格：{p['style']}\n"
              f"使用者情緒：{sentiment}；請調整語氣（開心→一起開心；難過/生氣→先共情、安撫再給建議；中性→自然聊天）。\n"
              f"回覆使用繁體中文，精煉自然，帶少量表情 {p['emoji']}。")
    logger.debug(f"建構人設 Prompt (key={key}, sentiment={sentiment}): {prompt[:50]}...")
    return prompt

# ========== 9) LINE Handlers (V2 SDK Style) ==========
@handler.add(MessageEvent, message=TextMessage)
def on_message_text(event: MessageEvent):
    # --- 繁體中文解：V2 handler 是同步的，需要在內部運行異步函數 ---
    logger.info(f"收到 V2 TextMessage Event from {get_chat_id(event)[:10]}...")
    try:
        # --- 繁體中文解：使用 asyncio.run 執行異步的 handle_message_async ---
        # --- 繁體中文解：注意：在某些環境下 (如已運行的事件循環)，可能需要改用 asyncio.create_task ---
        asyncio.run(handle_message_async(event))
        logger.info("異步 TextMessage 處理完成")
    except Exception as e:
        logger.error(f"❌ V2 on_message_text 頂層錯誤: {e}", exc_info=True)
        # --- 繁體中文解：嘗試發送錯誤訊息 (可能失敗) ---
        try:
             reply_with_quick_bar(event.reply_token, "抱歉，處理您的訊息時發生內部錯誤。")
        except:
             pass

@handler.add(PostbackEvent)
def on_postback(event: PostbackEvent):
    # --- 繁體中文解：Postback 通常是同步處理即可 ---
    logger.info(f"收到 V2 Postback Event from {get_chat_id(event)[:10]}..., data: {event.postback.data}")
    data = (event.postback.data or "").strip()
    if data.startswith("menu:"):
        kind = data.split(":", 1)[-1]
        logger.info(f"匹配到 Postback 選單: {kind}")
        try:
            # --- 繁體中文解：V2 直接回覆 FlexMessage 和 TextSendMessage (含 QuickReply) ---
            line_bot_api.reply_message(
                event.reply_token,
                [build_submenu_flex(kind), TextSendMessage(text="請選擇一項服務 👇", quick_reply=build_quick_reply())]
            )
            logger.info("Postback 子選單回覆成功")
        except Exception as e:
            logger.error(f"❌ Postback 回覆失敗: {e}", exc_info=True)
    else:
        logger.warning(f"⚠️ 未處理的 Postback data: {data}")

def is_stock_query(text: str) -> bool:
    # --- 繁體中文解：使用你的 V2 判斷邏輯 ---
    t = text.strip().upper()
    if t in ["台股大盤", "大盤", "美股大盤", "美盤", "美股"]:
        return True
    if _TW_CODE_RE.match(t):  # 2330 / 00937B / 1101B ...
        return True
    if _US_CODE_RE.match(t) and t not in ["JPY"]: # 排除 JPY
        return True
    return False

# --- 繁體中文解：這是處理訊息的核心異步函數 ---
async def handle_message_async(event: MessageEvent):
    chat_id = get_chat_id(event)
    # --- 繁體中文解：確保 event.message 是 TextMessage ---
    if not isinstance(event.message, TextMessage):
        logger.warning(f"收到非文字訊息，忽略: {type(event.message)}")
        return
    msg_raw = event.message.text.strip()
    reply_token = event.reply_token
    is_group = not isinstance(event.source, SourceUser)

    logger.info(f"處理文字訊息: '{msg_raw[:50]}...' from {chat_id[:10]}...")

    try:
        # --- 繁體中文解：獲取 Bot 名稱 (同步 API) ---
        bot_info = await run_in_threadpool(line_bot_api.get_bot_info)
        bot_name = bot_info.display_name
        logger.debug(f"Bot name: {bot_name}")
    except Exception as e:
        logger.warning(f"⚠️ 獲取 Bot info 失敗: {e}")
        bot_name = "AI 助手" # 預設名稱

    if not msg_raw:
        logger.debug("空訊息，忽略")
        return

    # --- 繁體中文解：群組自動回覆邏輯 ---
    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True # 預設開啟

    mentioned = msg_raw.startswith(f"@{bot_name}")
    should_reply_in_group = is_group and (auto_reply_status.get(chat_id, True) or mentioned)

    if is_group and not should_reply_in_group:
        logger.debug("群組中且未提及 Bot 且自動回覆關閉，忽略")
        return

    # --- 繁體中文解：移除 @ 提及 ---
    msg = msg_raw[len(f"@{bot_name}"):].strip() if mentioned else msg_raw
    if not msg: # 如果移除 @ 後變空訊息
        logger.debug("移除 @ 後訊息為空，忽略")
        return

    low = msg.lower()

    # --- 功能路由 ---
    try:
        if low in ("menu", "選單", "主選單"):
            logger.info("分支：主選單")
            # --- 繁體中文解：V2 直接回覆 FlexMessage ---
            return line_bot_api.reply_message(reply_token, build_main_menu_flex())

        # 彩票
        if msg in ["大樂透", "威力彩", "539"]:
            logger.info(f"分支：彩票分析 ({msg})")
            report = await run_in_threadpool(get_lottery_analysis, msg)
            return reply_with_quick_bar(reply_token, report)

        # 金價
        if low in ("金價", "黃金"):
            logger.info("分支：金價查詢")
            out = await run_in_threadpool(get_gold_analysis)
            return reply_with_quick_bar(reply_token, out)

        # 匯率 (JPY)
        if low == "jpy":
            logger.info("分支：日圓匯率查詢")
            out = await run_in_threadpool(get_currency_analysis, "JPY")
            return reply_with_quick_bar(reply_token, out)

        # 股票
        if is_stock_query(msg):
            logger.info(f"分支：股票查詢 ({msg})")
            report = await run_in_threadpool(get_stock_report, msg)
            return reply_with_quick_bar(reply_token, report)

        # 自動回覆設定
        if low in ("開啟自動回答", "關閉自動回答"):
            logger.info(f"分支：自動回覆設定 ({low})")
            is_on = low == "開啟自動回答"
            auto_reply_status[chat_id] = is_on
            text = "✅ 已開啟自動回答 (群組訊息都會回)" if is_on else "❌ 已關閉自動回答 (群組需 @我 才回)"
            return reply_with_quick_bar(reply_token, text)

        # 翻譯模式
        if msg.startswith("翻譯->"): # 注意：這裡用 msg (可能已被移除@)
            lang = msg.split("->", 1)[1].strip()
            logger.info(f"分支：翻譯模式切換 ({lang})")
            if lang == "結束":
                translation_states.pop(chat_id, None)
                return reply_with_quick_bar(reply_token, "✅ 已結束翻譯模式")
            else:
                translation_states[chat_id] = lang
                return reply_with_quick_bar(reply_token, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")

        # 人設切換
        if msg in PERSONA_ALIAS: # 注意：這裡用 msg
            logger.info(f"分支：人設切換 ({msg})")
            key_alias = msg
            key = set_user_persona(chat_id, PERSONA_ALIAS[key_alias])
            p = PERSONAS[user_persona[chat_id]] # 確保用最新的 key
            txt = f"💖 已切換人設：{p['title']}\n\n{p['greetings']}"
            return reply_with_quick_bar(reply_token, txt)

        # 翻譯內容
        if chat_id in translation_states:
            logger.info(f"分支：執行翻譯 (-> {translation_states[chat_id]})")
            out = await translate_text(msg, translation_states[chat_id])
            return reply_with_quick_bar(reply_token, out)

        # 一般聊天（人設 + 情緒）
        logger.info("分支：一般聊天 (Groq/OpenAI)")
        history = conversation_history.get(chat_id, [])
        logger.debug("分析情緒...")
        sentiment = await analyze_sentiment(msg)
        logger.debug("建構 Prompt...")
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]

        logger.info("呼叫 AI 進行聊天回覆...")
        # --- 繁體中文解：使用同步的 get_analysis_reply，但在異步函數中需要 run_in_threadpool ---
        final_reply = await run_in_threadpool(get_analysis_reply, messages)

        # --- 繁體中文解：更新歷史紀錄 ---
        history.extend([{"role":"user","content":msg}, {"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:] # 保留最新 N 筆
        logger.debug("聊天歷史已更新")

        return reply_with_quick_bar(reply_token, final_reply)

    except Exception as e:
        logger.error(f"❌ handle_message_async 內部錯誤: {e}", exc_info=True)
        try:
             # --- 繁體中文解：嘗試回覆通用錯誤訊息 ---
             reply_with_quick_bar(reply_token, "抱歉，處理您的請求時發生了未預期的錯誤 😵‍💫")
        except Exception as reply_e:
             logger.error(f"❌ 連錯誤訊息都無法回覆: {reply_e}")


# ========== 10) FastAPI Routes ==========
@router.post("/callback")
async def callback(request: Request):
    # --- 繁體中文解：V2 的 handler.handle 是同步的，需用 run_in_threadpool ---
    logger.info("收到 /callback 請求 (V2)")
    if not handler:
        logger.critical("❌ WebhookHandler 未初始化，無法處理請求")
        raise HTTPException(status_code=500, detail="WebhookHandler not initialized")

    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    body_decoded = body.decode("utf-8")
    logger.debug(f"Callback V2 - Signature: {signature[:10]}..., Body size: {len(body_decoded)}")

    try:
        # --- 繁體中文解：在異步路由中執行同步的 handler.handle ---
        await run_in_threadpool(handler.handle, body_decoded, signature)
        logger.info("✅ Callback V2 處理完成")
    except InvalidSignatureError:
        logger.error(f"❌ Invalid signature 驗證失敗 (Signature: {signature})，請檢查 CHANNEL_SECRET 是否正確。")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"❌ Callback V2 處理失敗：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse({"status": "ok"})

@router.get("/")
async def root():
    logger.debug("收到 / (root) 請求")
    return PlainTextResponse("LINE Bot (V2 SDK) is running.", status_code=200)

@router.get("/healthz")
async def healthz():
    # logger.debug("Received /healthz request") # 通常不需要記錄這個
    return PlainTextResponse("ok")

# --- 繁體中文解：保留你的健康檢查 API ---
@router.get("/health/providers")
async def providers_health():
    # --- 繁體中文解：這裡的 OK 狀態需要在 lifespan 中實際檢查才能更新 ---
    # --- 繁體中文解：為簡化，暫時只回傳基本資訊 ---
    logger.info("收到 /health/providers 請求")
    # --- 繁體中文解：注意，我們沒有像 V3 版本那樣在 lifespan 中設定 OPENAI_OK / GROQ_OK ---
    # --- 繁體中文解：這裡僅表示 Client 是否初始化 ---
    return {
        "openai_client_initialized": openai_client is not None,
        "groq_client_initialized": sync_groq_client is not None and async_groq_client is not None,
        "line_api_initialized": line_bot_api is not None,
        "ts": datetime.utcnow().isoformat() + "Z",
    }


app.include_router(router)

# ========== 11) Local run ==========
if __name__ == "__main__":
    # --- 繁體中文解：使用你的 V2 範例中的啟動方式 ---
    # import uvicorn # uvicorn 已在頂部匯入
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"準備啟動 Uvicorn (app_fastapi:app) 於 0.0.0.0:{port}")
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)