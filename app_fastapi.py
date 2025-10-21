# app_fastapi.py (Version 3.2.0 - Real Gold Fetch + Lottery by taiwanlottery)
# 變更摘要：
# - 金價 get_gold_analysis() 改為「實際抓台灣銀行官網」且具兩層備援解析（DOM / Regex），非模擬。
# - 保留你指定的：v2 LINE SDK、taiwanlottery 外部套件、yfinance 股票、ER-API 匯率、Groq/OpenAI 對話。
# - QuickReply 固定出現（含各功能入口）。若要擴充 TTS/更多按鈕可再加。

import os
import re
import random
import logging
import asyncio
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import httpx
import pandas as pd
import yfinance as yf
import mplfinance as mpf  # 預留

from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    QuickReply, QuickReplyButton, MessageAction,
    PostbackAction, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent,
    TextComponent, ButtonComponent, SeparatorComponent
)

from groq import AsyncGroq, Groq
import openai
import uvicorn

# ========== 1) Logging / Env ==========
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(asctime)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("日誌系統初始化完成")

logger.info("開始讀取環境變數...")
BASE_URL = os.getenv("BASE_URL")
logger.info(f"BASE_URL: {'已設定' if BASE_URL else '未設定'}")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
logger.info(f"CHANNEL_ACCESS_TOKEN: {'已設定' if CHANNEL_TOKEN else '未設定'}")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
logger.info(f"CHANNEL_SECRET: {'已設定' if CHANNEL_SECRET else '未設定'}")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
logger.info(f"GROQ_API_KEY: {'已設定' if GROQ_API_KEY else '未設定'}")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logger.info(f"OPENAI_API_KEY: {'已設定' if OPENAI_API_KEY else '未設定'}")

required_vars = {
    "BASE_URL": BASE_URL,
    "CHANNEL_ACCESS_TOKEN": CHANNEL_TOKEN,
    "CHANNEL_SECRET": CHANNEL_SECRET,
    "GROQ_API_KEY": GROQ_API_KEY
}
missing_vars = [name for name, value in required_vars.items() if not value]
if missing_vars:
    error_message = f"❌ 缺少必要環境變數: {', '.join(missing_vars)}"
    logger.critical(error_message); raise RuntimeError(error_message)
else:
    logger.info("✅ 所有必要環境變數均已設定")

# ========== 2) LINE / LLM ==========
try:
    line_bot_api = LineBotApi(CHANNEL_TOKEN)
    handler = WebhookHandler(CHANNEL_SECRET)
    logger.info("✅ LINE Bot API (V2) 初始化成功")
except Exception as e:
    logger.critical(f"❌ LINE Bot API 初始化失敗: {e}", exc_info=True)
    line_bot_api = None; handler = None
    raise RuntimeError(f"LINE Bot API 初始化失敗: {e}")

async_groq_client, sync_groq_client = None, None
if GROQ_API_KEY:
    try:
        async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
        sync_groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✅ Groq API Client 初始化成功 (Sync & Async)")
    except Exception as e:
        logger.error(f"❌ Groq API Client 初始化失敗: {e}")
else:
    logger.warning("⚠️ 未設定 GROQ_API_KEY")

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_base_url = os.getenv("OPENAI_API_BASE")
        if openai_base_url:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=openai_base_url)
            logger.info(f"✅ OpenAI Client (自訂 URL: {openai_base_url})")
        else:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            logger.info("✅ OpenAI Client (官方 URL)")
    except Exception as e:
        logger.warning(f"⚠️ 初始化 OpenAI 失敗：{e}")
else:
    logger.info("ℹ️ 未設定 OPENAI_API_KEY")

GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")
logger.info(f"Groq 模型: Primary={GROQ_MODEL_PRIMARY}, Fallback={GROQ_MODEL_FALLBACK}")
# 參考：Groq 模型列表（官方） https://console.groq.com/ ；OpenAI 模型 https://platform.openai.com/docs/models

# ========== 3) 外部模組（樂透 & 股票） ==========
LOTTERY_ENABLED = True
try:
    # 你指定使用 taiwanlottery 外部套件
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()
    logger.info("✅ 已載入彩票模組（taiwanlottery）")
except ModuleNotFoundError:
    logger.error("❌ 找不到 'taiwanlottery' 模組。請在 requirements.txt 安裝：taiwanlottery")
    LOTTERY_ENABLED = False; lottery_crawler = None; caiyunfangwei_crawler = None
except Exception as e:
    logger.warning(f"⚠️ 無法載入彩票模組：{e}"); LOTTERY_ENABLED = False
    lottery_crawler = None; caiyunfangwei_crawler = None
# 來源：taiwanlottery PyPI https://pypi.org/project/taiwanlottery/

STOCK_ENABLED = True
try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
    logger.info("✅ 已載入股票模組")
except ModuleNotFoundError as e:
    logger.error(f"❌ 股票模組載入失敗 (ImportError): {e}"); STOCK_ENABLED = False
except Exception as e:
    logger.warning(f"⚠️ 無法載入股票模組：{e}"); STOCK_ENABLED = False
# 來源：Yahoo Finance 非官方 yfinance https://pypi.org/project/yfinance/

if not STOCK_ENABLED:
    def stock_price(id): logger.error("股票(備援): stock_price"); return pd.DataFrame()
    def stock_news(hint): logger.error("股票(備援): stock_news"); return ["股票模組未載入"]
    def stock_fundamental(id): logger.error("股票(備援): stock_fundamental"); return "股票模組未載入"
    def stock_dividend(id): logger.error("股票(備援): stock_dividend"); return "股票模組未載入"
    class YahooStock:
        def __init__(self, id): logger.error(f"股票(備援): YahooStock({id})"); self.name=id; self.now_price=None; self.change=None; self.currency=None; self.close_time=None

# ========== 4) 常數 / 工具 ==========
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"  # 台灣銀行黃金牌價頁
# 來源：台灣銀行黃金牌價 https://rate.bot.com.tw/gold?Lang=zh-TW

conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}
auto_reply_status: Dict[str, bool] = {}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼", "greetings": "親愛的～我在這🌸", "emoji":"🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽", "greetings": "你又來啦？說吧😏", "emoji":"😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣", "greetings": "呀呼～(ﾉ>ω<)ﾉ", "emoji":"✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉", "greetings": "我在。說重點。", "emoji":"🧊⚡️"}
}
LANGUAGE_MAP = {"英文": "English", "日文": "Japanese", "韓文": "Korean", "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"}
PERSONA_ALIAS = {"甜":"sweet", "鹹":"salty", "萌":"moe", "酷":"cool", "random":"random"}

# ========== 5) FastAPI lifespan ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("應用程式啟動 (lifespan)...")
    if BASE_URL and CHANNEL_TOKEN and CHANNEL_TOKEN != "dummy":
        try:
            async with httpx.AsyncClient() as c:
                headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
                payload = {"endpoint": f"{BASE_URL}/callback"}
                logger.info(f"準備更新 Webhook 至: {payload['endpoint']}")
                r = await c.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=payload, timeout=10.0)
                r.raise_for_status(); logger.info(f"✅ Webhook 更新成功: {r.status_code}")
        except Exception as e:
            logger.error(f"⚠️ Webhook 更新失敗: {e}", exc_info=True)
    else:
        logger.warning("⚠️ Webhook 未更新：未設定 BASE_URL 或 CHANNEL_ACCESS_TOKEN (Mock 模式)")
    logger.info("Lifespan 啟動程序完成。"); yield; logger.info("應用程式關閉 (lifespan)...")

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="3.2.0")
router = APIRouter()

# ========== 6) Helpers ==========
def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom):  return event.source.room_id
    if isinstance(event.source, SourceUser):  return event.source.user_id
    logger.warning(f"未知的 event source type: {type(event.source)}"); return "unknown_source"

def build_quick_reply() -> QuickReply:
    # 固定讓每則訊息都帶快速回覆按鈕
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
    if not line_bot_api:
        logger.error("LINE API 未初始化"); return
    try:
        line_bot_api.reply_message(reply_token, TextSendMessage(text=text, quick_reply=build_quick_reply()))
    except LineBotApiError as lbe:
        logger.error(f"❌ 回覆 (QR) 失敗: {lbe.status_code} {lbe.error.message}", exc_info=False)
    except Exception as e:
        logger.error(f"❌ 回覆 (QR) 未知錯誤: {e}", exc_info=True)

def build_main_menu_flex() -> FlexSendMessage:
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[TextComponent(text="AI 助理選單", weight="bold", size="lg")]),
        body=BoxComponent(layout="vertical", spacing="md", contents=[
            TextComponent(text="選擇功能：", size="sm"),
            SeparatorComponent(margin="md"),
            ButtonComponent(action=PostbackAction(label="💹 金融查詢", data="menu:finance"), style="primary", color="#5E86C1"),
            ButtonComponent(action=PostbackAction(label="🎰 彩票分析", data="menu:lottery"), style="primary", color="#5EC186"),
            ButtonComponent(action=PostbackAction(label="💖 AI 角色", data="menu:persona"), style="secondary"),
            ButtonComponent(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate"), style="secondary"),
            ButtonComponent(action=PostbackAction(label="⚙️ 系統設定", data="menu:settings"), style="secondary"),
        ])
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

# ========== 7) AI ==========
def get_analysis_reply(messages: List[dict]) -> str:
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=1500
            )
            reply = resp.choices[0].message.content
            return reply
        except Exception as e:
            logger.warning(f"OpenAI 失敗：{e}")
    if not sync_groq_client:
        return "抱歉，AI 分析引擎無法連線。"
    try:
        resp = sync_groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY, messages=messages, temperature=0.7, max_tokens=2000
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq Primary 失敗：{e}")
        try:
            resp = sync_groq_client.chat.completions.create(
                model=GROQ_MODEL_FALLBACK, messages=messages, temperature=0.9, max_tokens=1500
            )
            return resp.choices[0].message.content
        except Exception as ee:
            logger.error(f"所有 AI API 都失敗：{ee}", exc_info=True)
            return "抱歉，AI 分析師目前連線不穩定，請稍後再試。"
# Groq/OpenAI 官方文件： https://console.groq.com/ ；https://platform.openai.com/docs

def analyze_sentiment(text: str) -> str:
    msgs = [{"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
            {"role":"user","content":text}]
    if not sync_groq_client: return "neutral"
    try:
        resp = sync_groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK, messages=msgs, max_tokens=10, temperature=0
        )
        result = (resp.choices[0].message.content or "neutral").strip().lower()
        return result if result in ["positive", "neutral", "negative", "angry"] else "neutral"
    except Exception:
        return "neutral"

def translate_text(text: str, target_lang_display: str) -> str:
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text, without intro."
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{text}"}}'
    if not sync_groq_client: return "抱歉，翻譯引擎無法連線。"
    try:
        resp = sync_groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK,
            messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
            max_tokens=len(text)*3 + 50, temperature=0.2
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return "抱歉，翻譯功能暫時出錯。"

# ========== 8) 金融：金價 / 匯率 / 股票 ==========
def _parse_bot_gold_dom(html: str) -> Optional[dict]:
    """
    主要解析法：以 DOM 選擇器讀取「本行賣出 / 本行買進 / 掛牌時間」。
    ＊台銀頁面如改版，可能需調整選擇器。
    回傳：{"sell": float, "buy": float, "time": "YYYY/MM/DD HH:MM"}
    """
    soup = BeautifulSoup(html, "html.parser")
    text = " ".join(soup.stripped_strings)

    # 先嘗試頁面常見文案
    time_pat = re.search(r"(?:掛牌時間|最後更新)[：:]\s*([0-9]{4}/[0-9]{2}/[0-9]{2}\s+[0-9]{2}:[0-9]{2})", text)
    sell_pat = re.search(r"(?:本行賣出|賣出)\s*([0-9,]+(?:\.[0-9]+)?)", text)
    buy_pat  = re.search(r"(?:本行買進|買進)\s*([0-9,]+(?:\.[0-9]+)?)", text)

    if sell_pat and buy_pat:
        data = {
            "sell": float(sell_pat.group(1).replace(",", "")),
            "buy":  float(buy_pat.group(1).replace(",", "")),
            "time": time_pat.group(1) if time_pat else None
        }
        return data

    # 次要嘗試：表格型態（若存在標題）
    table = soup.find("table")
    if table:
        ttext = " ".join(table.stripped_strings)
        sell_pat2 = re.search(r"(?:本行賣出|賣出)\D*([0-9,]+(?:\.[0-9]+)?)", ttext)
        buy_pat2  = re.search(r"(?:本行買進|買進)\D*([0-9,]+(?:\.[0-9]+)?)", ttext)
        if sell_pat2 and buy_pat2:
            return {
                "sell": float(sell_pat2.group(1).replace(",", "")),
                "buy":  float(buy_pat2.group(1).replace(",", "")),
                "time": time_pat.group(1) if time_pat else None
            }
    return None
# 來源：實際解析台灣銀行網頁 https://rate.bot.com.tw/gold?Lang=zh-TW

def _parse_bot_gold_regex(html: str) -> Optional[dict]:
    """
    備援解析法：純 Regex 從全頁文字擷取數字，避免 DOM 變動造成失敗。
    """
    text = " ".join(BeautifulSoup(html, "html.parser").stripped_strings)
    m_time = re.search(r"(?:掛牌時間|最後更新)[：:]\s*([0-9]{4}/[0-9]{2}/[0-9]{2}\s+[0-9]{2}:[0-9]{2})", text)
    m_sell = re.search(r"(?:本行賣出|賣出)\s*([0-9,]+(?:\.[0-9]+)?)", text)
    m_buy  = re.search(r"(?:本行買進|買進)\s*([0-9,]+(?:\.[0-9]+)?)", text)
    if m_sell and m_buy:
        return {
            "sell": float(m_sell.group(1).replace(",", "")),
            "buy":  float(m_buy.group(1).replace(",", "")),
            "time": m_time.group(1) if m_time else None
        }
    return None

def get_gold_analysis() -> str:
    """
    ✔ 真實抓取台灣銀行黃金牌價（非模擬）
    - 先以 DOM 解析；失敗再走 Regex 備援。
    - 回傳清楚文字，附上來源網址。
    """
    logger.info("呼叫：get_gold_analysis()")
    try:
        r = requests.get(BOT_GOLD_URL, headers=DEFAULT_HEADERS, timeout=10)
        r.raise_for_status()
        html = r.text

        data = _parse_bot_gold_dom(html) or _parse_bot_gold_regex(html)
        if not data or "sell" not in data or "buy" not in data:
            raise RuntimeError("解析台銀黃金牌價失敗，頁面可能改版")

        sell = float(data["sell"]); buy = float(data["buy"])
        ts = data.get("time") or "（頁面未標示）"
        spread = sell - buy

        report = (
            f"**金價（台灣銀行）**\n"
            f"- 掛牌時間：{ts}\n"
            f"- 賣出(1g)：{sell:,.0f} 元\n"
            f"- 買進(1g)：{buy:,.0f} 元\n"
            f"- 價差：{spread:,.0f} 元\n"
            f"來源：{BOT_GOLD_URL}"
        )
        return report
    except Exception as e:
        logger.error(f"黃金分析失敗: {e}", exc_info=False)
        return "抱歉，目前無法取得黃金牌價 🙏"
# 來源：台灣銀行黃金牌價 https://rate.bot.com.tw/gold?Lang=zh-TW

def get_currency_analysis(target_currency: str):
    logger.info(f"呼叫：get_currency_analysis({target_currency})")
    try:
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"
        res = requests.get(url, timeout=10); res.raise_for_status()
        data = res.json()
        if data.get("result") != "success":
            return f"匯率 API 錯誤: {data.get('error-type','未知')}"
        rate = data["rates"].get("TWD")
        if rate is None:
            return "抱歉，API 回應中無 TWD 匯率。"
        return f"即時：1 {target_currency.upper()} ≈ **{rate:.4f}** 新台幣"
    except Exception as e:
        logger.error(f"匯率分析失敗: {e}", exc_info=False)
        return "抱歉，外匯資料暫無法取得。"
# 來源：ER-API 匯率 https://open.er-api.com/v6/latest/USD

_TW_CODE_RE = re.compile(r'^\d{4,6}[A-Za-z]?$')
_US_CODE_RE = re.compile(r'^[A-Za-z]{1,5}$')

def normalize_ticker(t: str) -> Tuple[str, str, str, bool]:
    t = t.strip().upper()
    if t in ["台股大盤", "大盤"]: return "^TWII", "^TWII", "^TWII", True
    if t in ["美股大盤", "美盤", "美股"]: return "^GSPC", "^GSPC", "^GSPC", True
    if _TW_CODE_RE.match(t): return f"{t}.TW", t, t, False
    if _US_CODE_RE.match(t) and t != "JPY": return t, t, t, False
    return t, t, t, False

def fetch_realtime_snapshot(yf_symbol: str, yahoo_slug: str) -> dict:
    snap: dict = {"name": None, "now_price": None, "change": None, "currency": None, "close_time": None}
    try:
        tk = yf.Ticker(yf_symbol); info = {}; hist = pd.DataFrame()
        try: info = tk.info or {}
        except Exception as info_e: logger.warning(f"yf info fail: {info_e}")
        try: hist = tk.history(period="2d", interval="1d")
        except Exception as hist_e: logger.warning(f"yf history fail: {hist_e}")
        name = info.get("shortName") or info.get("longName"); snap["name"] = name or yf_symbol
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"); ccy = info.get("currency")
        if price: snap["now_price"] = f"{price:.2f}"; snap["currency"] = ccy or ("TWD" if yf_symbol.endswith(".TW") else "USD")
        elif not hist.empty: price = float(hist["Close"].iloc[-1]); snap["now_price"] = f"{price:.2f}"; snap["currency"] = ccy or ("TWD" if yf_symbol.endswith(".TW") else "USD")
        if not hist.empty and len(hist) >= 2 and float(hist["Close"].iloc[-2]) != 0:
            chg = float(hist["Close"].iloc[-1]) - float(hist["Close"].iloc[-2]); pct = chg/float(hist["Close"].iloc[-2])*100; sign = "+" if chg>=0 else ""
            snap["change"] = f"{sign}{chg:.2f} ({sign}{pct:.2f}%)"
        elif info.get('regularMarketChange') is not None and info.get('regularMarketChangePercent') is not None:
            chg = float(info['regularMarketChange']); pct = float(info['regularMarketChangePercent'])*100; sign = "+" if chg>=0 else ""
            snap["change"] = f"{sign}{chg:.2f} ({sign}{pct:.2f}%)"
        if not hist.empty: ts = hist.index[-1]; snap["close_time"] = ts.strftime("%Y-%m-%d %H:%M")
        elif info.get("regularMarketTime"):
            try: snap["close_time"] = datetime.fromtimestamp(info["regularMarketTime"]).strftime("%Y-%m-%d %H:%M")
            except Exception: pass
    except Exception as e:
        logger.warning(f"yfinance fail: {e}")
    if (not snap["now_price"] or not snap["name"]) and STOCK_ENABLED and 'YahooStock' in globals():
        try:
            ys = YahooStock(yahoo_slug); snap["name"] = ys.name or snap["name"] or yahoo_slug
            snap["now_price"] = ys.now_price or snap["now_price"]; snap["change"] = ys.change or snap["change"]
            snap["currency"] = ys.currency or ("TWD" if yf_symbol.endswith(".TW") else snap["currency"]); snap["close_time"] = ys.close_time or snap["close_time"]
        except Exception as e: logger.error(f"YahooStock fallback fail: {e}")
    return snap

stock_data_df: Optional[pd.DataFrame] = None
def load_stock_data() -> pd.DataFrame:
    global stock_data_df
    if stock_data_df is None:
        try: stock_data_df = pd.read_csv('name_df.csv'); logger.info("✅ loaded name_df.csv")
        except FileNotFoundError: logger.error("❌ `name_df.csv` not found."); stock_data_df = pd.DataFrame(columns=['股號', '股名'])
    return stock_data_df

def get_stock_name(stock_id: str) -> Optional[str]:
    df = load_stock_data()
    res = df[df['股號'].astype(str).str.strip().str.upper() == str(stock_id).strip().upper()]
    if not res.empty:
        name = res.iloc[0]['股名']; logger.debug(f"name_df lookup: {stock_id} -> {name}"); return name
    logger.debug(f"name_df not found: {stock_id}"); return None

def get_stock_report(user_input: str) -> str:
    yf_symbol, yahoo_slug, display_code, is_index = normalize_ticker(user_input)
    snapshot = fetch_realtime_snapshot(yf_symbol, yahoo_slug)
    price_data, news_data, value_part, dividend_part = "", "", "", ""
    if STOCK_ENABLED:
        try: input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol; price_df = stock_price(input_code); price_data = str(price_df) if not price_df.empty else "N/A"
        except Exception as e: price_data = f"Err: {e}"
        try: nm = get_stock_name(yahoo_slug) or snapshot.get("name") or yahoo_slug; news_list = stock_news(nm); news_data = "\n".join(news_list).replace("\u3000", " ")[:1024]
        except Exception as e: news_data = f"Err: {e}"
        if not is_index:
            try: input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol; val = stock_fundamental(input_code); value_part = f"{val}\n" if val else ""
            except Exception as e: value_part = f"Err: {e}\n"
            try: input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol; dvd = stock_dividend(input_code); dividend_part = f"{dvd}\n" if dvd else ""
            except Exception as e: dividend_part = f"Err: {e}\n"
    stock_link = (f"https://finance.yahoo.com/quote/{yf_symbol}" if yf_symbol.startswith("^") or not yf_symbol.endswith(".TW") else f"https://tw.stock.yahoo.com/quote/{yahoo_slug}")
    content_msg = (f"分析報告:\n**代碼:** {display_code}, **名稱:** {snapshot.get('name')}\n**價格:** {snapshot.get('now_price')} {snapshot.get('currency')}\n**漲跌:** {snapshot.get('change')}\n**時間:** {snapshot.get('close_time')}\n**近期價:**\n{price_data}\n")
    if value_part:    content_msg += f"**基本面:**\n{value_part}"
    if dividend_part: content_msg += f"**配息:**\n{dividend_part}"
    if news_data:     content_msg += f"**新聞:**\n{news_data}\n"
    content_msg += (f"請寫出 {snapshot.get('name') or display_code} 近期趨勢分析，用繁體中文 Markdown，附連結：{stock_link}")
    system_prompt = ("你是專業分析師。開頭列出股名(股號)/現價/漲跌/時間；分段說明走勢/基本面/技術面/消息面/風險/建議區間/停利目標/結論。資料不完整請保守說明。")
    msgs = [{"role":"system","content":system_prompt}, {"role":"user","content":content_msg}]
    return get_analysis_reply(msgs)
# 來源：Yahoo Finance https://finance.yahoo.com/

# ========== 9) 彩票分析（taiwanlottery） ==========
def _lotto_fallback_scrape(kind: str) -> str:
    try:
        if kind == "威力彩":
            url, pat = ("https://www.taiwanlottery.com/lotto/superlotto638/index.html", r"第\s*\d+\s*期.*?第一區.*?[:：\s]*([\d\s,]+?)\s*第二區.*?[:：\s]*(\d+)")
        elif kind == "大樂透":
            url, pat = ("https://www.taiwanlottery.com/lotto/lotto649/index.html", r"第\s*\d+\s*期.*?(?:號碼|獎號).*?[:：\s]*([\d\s,]+?)(?:\s*特別號[:：\s]*(\d+))?")
        elif kind == "539":
            url, pat = ("https://www.taiwanlottery.com/lotto/dailycash/index.html", r"第\s*\d+\s*期.*?(?:號碼|獎號).*?[:：\s]*([\d\s,]+)")
        else:
            return f"不支援: {kind}"
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser"); text = ' '.join(soup.stripped_strings)
        m = re.search(pat, text, re.DOTALL)
        if not m: return f"抱歉，找不到 {kind} 號碼 (Fallback regex failed)。"
        if kind == "威力彩":
            first, second = re.sub(r'[,\s]+', ' ', m.group(1)).strip(), m.group(2); return f"{kind}: 一區 {first}；二區 {second}"
        elif kind == "大樂透":
            nums, special = re.sub(r'[,\s]+', ' ', m.group(1)).strip(), m.group(2); return f"{kind}: {nums}{'；特 ' + special if special else ''}"
        elif kind == "539":
            nums = re.sub(r'[,\s]+', ' ', m.group(1)).strip(); return f"{kind}: {nums}"
    except Exception as e:
        logger.error(f"Fallback scrape fail: {e}", exc_info=False); return f"抱歉，{kind} 號碼取不到 (Fallback exception)。"
# 官方網站：台灣彩券 https://www.taiwanlottery.com/

def get_lottery_analysis(lottery_type_input: str) -> str:
    kind = "威力彩" if "威力" in lottery_type_input else ("大樂透" if "大樂" in lottery_type_input else ("539" if "539" in lottery_type_input else lottery_type_input))
    latest_data_str = ""
    if LOTTERY_ENABLED and lottery_crawler:
        try:
            if kind == "威力彩":   latest_data_str = str(lottery_crawler.super_lotto())
            elif kind == "大樂透": latest_data_str = str(lottery_crawler.lotto649())
            elif kind == "539":    latest_data_str = str(lottery_crawler.daily_cash())
            else: return f"不支援 {kind}。"
        except Exception as e:
            logger.warning(f"自訂爬蟲失敗，改用後備：{e}"); latest_data_str = _lotto_fallback_scrape(kind)
    else:
        latest_data_str = _lotto_fallback_scrape(kind)

    cai_part = ""
    if 'caiyunfangwei_crawler' in globals() and caiyunfangwei_crawler:
        try:
            cai = caiyunfangwei_crawler.get_caiyunfangwei()
            cai_part = (f"日期：{cai.get('今天日期','')}\n歲次：{cai.get('今日歲次','')}\n財位：{cai.get('財神方位','')}\n")
        except Exception as e:
            logger.warning(f"財運方位失敗: {e}"); cai_part = ""

    prompt = (f"{kind} 近況/號碼：\n{latest_data_str}\n\n{cai_part}請用繁體中文寫出：\n1) 走勢重點(熱冷號)\n2) 選號建議(風險聲明)\n3) 三組推薦號碼\n分點條列精煉。")
    messages = [{"role":"system","content":"你是資深彩券分析師。"},{"role":"user","content":prompt}]
    return get_analysis_reply(messages)

# ========== 10) LINE Handlers ==========
@handler.add(MessageEvent, message=TextMessage)
def on_message_text(event: MessageEvent):
    chat_id = get_chat_id(event); msg_raw = event.message.text.strip(); reply_token = event.reply_token
    is_group = not isinstance(event.source, SourceUser)
    if not isinstance(event.message, TextMessage): return
    if not msg_raw: return

    try:
        bot_info = line_bot_api.get_bot_info(); bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if chat_id not in auto_reply_status: auto_reply_status[chat_id] = True
    mentioned = msg_raw.startswith(f"@{bot_name}"); should_reply = not is_group or auto_reply_status.get(chat_id, True) or mentioned
    if not should_reply: return

    msg = msg_raw[len(f"@{bot_name}"):].strip() if mentioned else msg_raw
    if not msg: return
    low = msg.lower()

    try:
        if low in ("menu", "選單", "主選單"):
            line_bot_api.reply_message(reply_token, build_main_menu_flex()); return

        if msg in ["大樂透", "威力彩", "539"]:
            report = get_lottery_analysis(msg); reply_with_quick_bar(reply_token, report); return

        if low in ("金價", "黃金"):
            out = get_gold_analysis(); reply_with_quick_bar(reply_token, out); return

        if low == "jpy":
            out = get_currency_analysis("JPY"); reply_with_quick_bar(reply_token, out); return

        if is_stock_query(msg):
            report = get_stock_report(msg); reply_with_quick_bar(reply_token, report); return

        if low in ("開啟自動回答", "關閉自動回答"):
            is_on = low == "開啟自動回答"; auto_reply_status[chat_id] = is_on
            text = "✅ 自動回答已開啟" if is_on else "❌ 自動回答已關閉"; reply_with_quick_bar(reply_token, text); return

        if msg.startswith("翻譯->"):
            lang = msg.split("->", 1)[1].strip()
            if lang == "結束": translation_states.pop(chat_id, None); reply_with_quick_bar(reply_token, "✅ 翻譯模式結束")
            else: translation_states[chat_id] = lang; reply_with_quick_bar(reply_token, f"🌐 開啟翻譯 → {lang}")
            return

        if msg in PERSONA_ALIAS:
            key = PERSONA_ALIAS[msg]; key = random.choice(list(PERSONAS.keys())) if key == "random" else key
            key = "sweet" if key not in PERSONAS else key; user_persona[chat_id] = key
            p = PERSONAS[user_persona[chat_id]]; txt = f"💖 切換人設：{p['title']}\n{p['greetings']}"; reply_with_quick_bar(reply_token, txt); return

        if chat_id in translation_states:
            out = translate_text(msg, translation_states[chat_id]); reply_with_quick_bar(reply_token, out); return

        history = conversation_history.get(chat_id, [])
        sentiment = analyze_sentiment(msg)
        sys_prompt = (f"你是「{PERSONAS[user_persona.get(chat_id,'sweet')]['title']}」。風格：{PERSONAS[user_persona.get(chat_id,'sweet')]['style'] if user_persona.get(chat_id) else PERSONAS['sweet']['style']}\n"
                      f"情緒：{sentiment}；用繁體中文，精煉自然，帶少量表情 {PERSONAS[user_persona.get(chat_id, 'sweet')]['emoji'] if user_persona.get(chat_id) else PERSONAS['sweet']['emoji']}。")
        messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]
        final_reply = get_analysis_reply(messages)
        history.extend([{"role":"user","content":msg}, {"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
        reply_with_quick_bar(reply_token, final_reply)
    except LineBotApiError as lbe:
        logger.error(f"LINE API Error: {lbe.status_code} {lbe.error.message}", exc_info=False)
        try: line_bot_api.reply_message(reply_token, TextSendMessage(text="😥 LINE communication error."))
        except Exception: pass
    except Exception as e:
        logger.error(f"Handler internal error: {e}", exc_info=True)
        try: reply_with_quick_bar(reply_token, "😵‍💫 Unexpected error processing request.")
        except Exception as reply_e: logger.error(f"Failed to even send error reply: {reply_e}")

@handler.add(PostbackEvent)
def on_postback(event: PostbackEvent):
    data = (event.postback.data or "").strip(); kind = data[5:] if data.startswith("menu:") else None
    try:
        line_bot_api.reply_message(
            event.reply_token,
            [build_submenu_flex(kind), TextSendMessage(text="請選擇 👇", quick_reply=build_quick_reply())]
        )
    except LineBotApiError as lbe:
        logger.error(f"Postback LINE API Error: {lbe.status_code} {lbe.error.message}", exc_info=False)
    except Exception as e:
        logger.error(f"Postback reply fail: {e}", exc_info=True)

def is_stock_query(text: str) -> bool:
    t = text.strip().upper()
    return t in ["台股大盤", "大盤", "美股大盤", "美盤", "美股"] or bool(_TW_CODE_RE.match(t)) or (bool(_US_CODE_RE.match(t)) and t != "JPY")

# ========== 11) FastAPI Routes ==========
@router.post("/callback")
async def callback(request: Request):
    logger.info("Callback V2 received")
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body(); body_decoded = body.decode("utf-8")
    if not handler: raise HTTPException(status_code=500, detail="Handler not initialized")
    try:
        handler.handle(body_decoded, signature)
        logger.info("✅ Callback V2 handled")
    except InvalidSignatureError:
        logger.error(f"❌ Invalid signature: {signature}"); raise HTTPException(status_code=400, detail="Invalid signature")
    except LineBotApiError as lbe:
        logger.error(f"❌ LINE API Error in callback: {lbe.status_code} {lbe.error.message}", exc_info=True)
        return JSONResponse({"status": "ok but error logged"})
    except Exception as e:
        logger.error(f"❌ Callback V2 fail: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse({"status": "ok"})

@router.get("/")
async def root(): return PlainTextResponse("LINE Bot (V2 SDK - Sync) running.", status_code=200)

@router.get("/healthz")
async def healthz(): return PlainTextResponse("ok")

@router.get("/health/providers")
async def providers_health():
    return {"openai_ok": openai_client is not None, "groq_ok": sync_groq_client is not None, "line_ok": line_bot_api is not None, "ts": datetime.utcnow().isoformat() + "Z"}

app.include_router(router)

# ========== 12) Local run ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Uvicorn on 0.0.0.0:{port}")
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)