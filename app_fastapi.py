# app_fastapi.py (Version 2.0.4 - Syntax Fix)
# ========== 1) Imports ==========
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
import mplfinance as mpf

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

# ========== 2) Setup ==========
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(asctime)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("日誌系統初始化完成")

# --- 環境變數 ---
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
    logger.critical(error_message)
    raise RuntimeError(error_message)
else:
    logger.info("✅ 所有必要環境變數均已設定")


# --- API 用戶端初始化 (V2 SDK) ---
try:
    line_bot_api = LineBotApi(CHANNEL_TOKEN)
    handler = WebhookHandler(CHANNEL_SECRET)
    logger.info("✅ LINE Bot API (V2) 初始化成功")
except Exception as e:
    logger.critical(f"❌ LINE Bot API 初始化失敗: {e}", exc_info=True)
    line_bot_api = None
    handler = None
    raise RuntimeError(f"LINE Bot API 初始化失敗: {e}")


# --- AI Client 初始化 ---
async_groq_client = None
sync_groq_client = None
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
            logger.info(f"✅ OpenAI Client 初始化成功 (自訂 Base URL: {openai_base_url})")
        else:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            logger.info("✅ OpenAI Client 初始化成功 (官方 URL)")
    except Exception as e:
        logger.warning(f"⚠️ 初始化 OpenAI 失敗：{e}")
else:
    logger.info("ℹ️ 未設定 OPENAI_API_KEY，將僅使用 Groq")


# --- Groq 模型 ---
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")
logger.info(f"Groq 模型設定 - Primary: {GROQ_MODEL_PRIMARY}, Fallback: {GROQ_MODEL_FALLBACK}")

# --- 【靈活載入】自訂模組 ---
LOTTERY_ENABLED = True
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()
    logger.info("✅ 已載入自訂 TaiwanLotteryCrawler / CaiyunfangweiCrawler")
except ModuleNotFoundError:
    logger.error("❌ 找不到 'taiwanlottery' 模組。請確認 requirements.txt 並重新部署。將使用後備解析。")
    LOTTERY_ENABLED = False; lottery_crawler = None; caiyunfangwei_crawler = None
except Exception as e:
    logger.warning(f"⚠️ 無法載入自訂彩票模組：{e}，將使用後備解析。")
    LOTTERY_ENABLED = False; lottery_crawler = None; caiyunfangwei_crawler = None

STOCK_ENABLED = True
try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
    logger.info("✅ 已載入自訂股票模組 (my_commands.stock)")
except ModuleNotFoundError as e:
    if 'taiwanlottery' in str(e): logger.error("❌ 股票模組因找不到 'taiwanlottery' 而載入失敗。")
    else: logger.error(f"❌ 股票模組載入失敗 (ModuleNotFoundError): {e}")
    STOCK_ENABLED = False
except Exception as e:
    logger.warning(f"⚠️ 無法載入股票模組：{e}；將只顯示基本快照。")
    STOCK_ENABLED = False

if not STOCK_ENABLED:
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

PERSONAS = { "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji":"🌸💕😊"}, "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji":"😏🙄"}, "moe":   {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji":"✨🎀"}, "cool":  {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議", "greetings": "我在。說重點。", "emoji":"🧊⚡️"} }
LANGUAGE_MAP = {"英文": "English", "日文": "Japanese", "韓文": "Korean", "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"}
PERSONA_ALIAS = {"甜":"sweet", "鹹":"salty", "萌":"moe", "酷":"cool", "random":"random"}

# ========== 3) FastAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... (與 v2.0.3 相同) ...
    logger.info("應用程式啟動 (lifespan)...")
    if BASE_URL and CHANNEL_TOKEN != "dummy":
        try:
            async with httpx.AsyncClient() as c:
                headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
                payload = {"endpoint": f"{BASE_URL}/callback"}
                logger.info(f"準備更新 Webhook 至: {payload['endpoint']}")
                r = await c.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=payload, timeout=10.0)
                r.raise_for_status(); logger.info(f"✅ Webhook 更新成功: {r.status_code}")
        except Exception as e: logger.error(f"⚠️ Webhook 更新失敗: {e}", exc_info=True)
    else: logger.warning("⚠️ Webhook 未更新：未設定 BASE_URL 或 CHANNEL_ACCESS_TOKEN (Mock 模式)")
    logger.info("Lifespan 啟動程序完成，應用程式準備就緒."); yield; logger.info("應用程式關閉 (lifespan)...")

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="2.0.4-syntax-fix") # --- 繁體中文解：更新版本號 ---
router = APIRouter()

# ========== 4) Helpers (V2 SDK Style) ==========
# ... (get_chat_id, build_quick_reply, reply_with_quick_bar, build_main_menu_flex, build_submenu_flex 與 v2.0.3 相同) ...
def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom):  return event.source.room_id
    if isinstance(event.source, SourceUser): return event.source.user_id
    logger.warning(f"未知的 event source type: {type(event.source)}"); return "unknown_source"

def build_quick_reply() -> QuickReply:
    logger.debug("建立 QuickReply 按鈕"); return QuickReply(items=[ QuickReplyButton(action=MessageAction(label="主選單", text="選單")), QuickReplyButton(action=MessageAction(label="台股大盤", text="台股大盤")), QuickReplyButton(action=MessageAction(label="美股大盤", text="美股大盤")), QuickReplyButton(action=MessageAction(label="黃金價格", text="金價")), QuickReplyButton(action=MessageAction(label="查台積電", text="2330")), QuickReplyButton(action=MessageAction(label="查輝達", text="NVDA")), QuickReplyButton(action=MessageAction(label="查日圓", text="JPY")), QuickReplyButton(action=PostbackAction(label="💖 AI 人設", data="menu:persona")), QuickReplyButton(action=PostbackAction(label="🎰 彩票選單", data="menu:lottery")) ])

def reply_with_quick_bar(reply_token: str, text: str):
    if not line_bot_api: logger.error("LINE Bot API 未初始化，無法回覆"); print(f"[MOCK] Reply with Quick Bar: {text}"); return
    try: logger.debug(f"準備回覆 (含 QuickReply): {text[:50]}..."); line_bot_api.reply_message(reply_token, TextSendMessage(text=text, quick_reply=build_quick_reply())); logger.debug("回覆 (含 QuickReply) 成功")
    except LineBotApiError as lbe: logger.error(f"❌ 回覆 (含 QuickReply) 失敗: {lbe.status_code} {lbe.error.message}", exc_info=False)
    except Exception as e: logger.error(f"❌ 回覆 (含 QuickReply) 發生未知錯誤: {e}", exc_info=True)

def build_main_menu_flex() -> FlexSendMessage:
    logger.debug("建立主選單 FlexMessage"); bubble = BubbleContainer( direction="ltr", header=BoxComponent(layout="vertical", contents=[TextComponent(text="AI 助理主選單", weight="bold", size="lg")]), body=BoxComponent( layout="vertical", spacing="md", contents=[ TextComponent(text="請選擇功能分類：", size="sm"), SeparatorComponent(margin="md"), ButtonComponent(action=PostbackAction(label="💹 金融查詢", data="menu:finance"), style="primary", color="#5E86C1"), ButtonComponent(action=PostbackAction(label="🎰 彩票分析", data="menu:lottery"), style="primary", color="#5EC186"), ButtonComponent(action=PostbackAction(label="💖 AI 角色扮演", data="menu:persona"), style="secondary"), ButtonComponent(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate"), style="secondary"), ButtonComponent(action=PostbackAction(label="⚙️ 系統設定", data="menu:settings"), style="secondary"), ] ) ); return FlexSendMessage(alt_text="主選單", contents=bubble)

def build_submenu_flex(kind: str) -> FlexSendMessage:
    logger.debug(f"建立子選單 FlexMessage (kind={kind})"); title, buttons = "子選單", []
    if kind == "finance": title, buttons = "💹 金融查詢", [ButtonComponent(action=MessageAction(label="台股大盤", text="台股大盤")), ButtonComponent(action=MessageAction(label="美股大盤", text="美股大盤")), ButtonComponent(action=MessageAction(label="黃金價格", text="金價")), ButtonComponent(action=MessageAction(label="日圓匯率", text="JPY")), ButtonComponent(action=MessageAction(label="查 2330 台積電", text="2330")), ButtonComponent(action=MessageAction(label="查 NVDA 輝達", text="NVDA"))]
    elif kind == "lottery": title, buttons = "🎰 彩票分析", [ButtonComponent(action=MessageAction(label="大樂透", text="大樂透")), ButtonComponent(action=MessageAction(label="威力彩", text="威力彩")), ButtonComponent(action=MessageAction(label="今彩539", text="539"))]
    elif kind == "persona": title, buttons = "💖 AI 角色扮演", [ButtonComponent(action=MessageAction(label="甜美女友", text="甜")), ButtonComponent(action=MessageAction(label="傲嬌女友", text="鹹")), ButtonComponent(action=MessageAction(label="萌系女友", text="萌")), ButtonComponent(action=MessageAction(label="酷系御姐", text="酷")), ButtonComponent(action=MessageAction(label="隨機切換", text="random"))]
    elif kind == "translate": title, buttons = "🌐 翻譯工具", [ButtonComponent(action=MessageAction(label="翻成英文", text="翻譯->英文")), ButtonComponent(action=MessageAction(label="翻成日文", text="翻譯->日文")), ButtonComponent(action=MessageAction(label="翻成繁中", text="翻譯->繁體中文")), ButtonComponent(action=MessageAction(label="結束翻譯模式", text="翻譯->結束"))]
    elif kind == "settings": title, buttons = "⚙️ 系統設定", [ButtonComponent(action=MessageAction(label="開啟自動回答 (群組)", text="開啟自動回答")), ButtonComponent(action=MessageAction(label="關閉自動回答 (群組)", text="關閉自動回答"))]
    bubble = BubbleContainer( direction="ltr", header=BoxComponent(layout="vertical", contents=[TextComponent(text=title, weight="bold", size="lg")]), body=BoxComponent(layout="vertical", contents=buttons, spacing="sm") ); return FlexSendMessage(alt_text=title, contents=bubble)

# ========== 5) AI & 分析 ==========
# ... (get_analysis_reply, analyze_sentiment, translate_text 與 v2.0.3 相同) ...
def get_analysis_reply(messages: List[dict]) -> str:
    logger.debug(f"呼叫 get_analysis_reply (OpenAI優先), messages count: {len(messages)}")
    if openai_client:
        try:
            logger.debug("嘗試使用 OpenAI..."); resp = openai_client.chat.completions.create( model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=1500, )
            reply = resp.choices[0].message.content; logger.debug(f"OpenAI 成功回覆，長度: {len(reply)}"); return reply
        except Exception as e: logger.warning(f"⚠️ OpenAI 失敗：{e}")
    if not sync_groq_client: logger.error("❌ Groq Client 未初始化，無法回覆"); return "抱歉，AI 分析引擎目前無法連線。"
    try:
        logger.debug(f"嘗試使用 Groq 主模型: {GROQ_MODEL_PRIMARY}"); resp = sync_groq_client.chat.completions.create( model=GROQ_MODEL_PRIMARY, messages=messages, temperature=0.7, max_tokens=2000, )
        reply = resp.choices[0].message.content; logger.debug(f"Groq 主模型成功回覆，長度: {len(reply)}"); return reply
    except Exception as e:
        logger.warning(f"⚠️ Groq 主模型失敗：{e}")
        try:
            logger.debug(f"嘗試使用 Groq 備援模型: {GROQ_MODEL_FALLBACK}"); resp = sync_groq_client.chat.completions.create( model=GROQ_MODEL_FALLBACK, messages=messages, temperature=0.9, max_tokens=1500, )
            reply = resp.choices[0].message.content; logger.debug(f"Groq 備援模型成功回覆，長度: {len(reply)}"); return reply
        except Exception as ee: logger.error(f"❌ 所有 AI API 都失敗：{ee}", exc_info=True); return "抱歉，AI 分析師目前連線不穩定，請稍後再試。"

def analyze_sentiment(text: str) -> str:
    logger.debug(f"呼叫 analyze_sentiment for: {text[:30]}..."); msgs = [{"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},{"role":"user","content":text}]
    if not sync_groq_client: logger.error("❌ Groq Client 未初始化，無法分析情緒"); return "neutral"
    try:
        resp = sync_groq_client.chat.completions.create( model=GROQ_MODEL_FALLBACK, messages=msgs, max_tokens=10, temperature=0 ); result = (resp.choices[0].message.content or "neutral").strip().lower()
        logger.debug(f"Groq 同步情緒分析結果: {result}"); return result if result in ["positive", "neutral", "negative", "angry"] else "neutral"
    except Exception as e: logger.error(f"❌ Groq 同步情緒分析失敗: {e}", exc_info=True); return "neutral"

def translate_text(text: str, target_lang_display: str) -> str:
    logger.debug(f"呼叫 translate_text to {target_lang_display} for: {text[:30]}..."); target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text, without any introductory phrases or explanations."
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{text}"}}'
    if not sync_groq_client: logger.error("❌ Groq Client 未初始化，無法翻譯"); return "抱歉，翻譯引擎目前無法連線。"
    try:
        resp = sync_groq_client.chat.completions.create( model=GROQ_MODEL_FALLBACK, messages=[{"role":"system","content":sys},{"role":"user","content":usr}], max_tokens=len(text)*3 + 50, temperature=0.2 )
        translated_text = (resp.choices[0].message.content or "").strip(); logger.debug(f"Groq 同步翻譯結果: {translated_text[:50]}..."); return translated_text
    except Exception as e: logger.error(f"❌ Groq 同步翻譯失敗: {e}", exc_info=True); return "抱歉，翻譯功能暫時出錯了。"


# ========== 6) 金融工具 ==========
# ... (與 v2.0.3 相同) ...
def get_gold_analysis() -> str:
    logger.info("呼叫：get_gold_analysis()")
    try:
        r = requests.get(BOT_GOLD_URL, headers=DEFAULT_HEADERS, timeout=10); r.raise_for_status()
        data = _parse_bot_gold_text(r.text); logger.debug(f"金價原始資料: {data}")
        ts = data.get("listed_at") or "（頁面未標示）"; sell, buy = data["sell_twd_per_g"], data["buy_twd_per_g"]
        spread = sell - buy; bias = "盤整" if spread <= 30 else ("偏寬" if spread <= 60 else "價差偏大"); now = datetime.now().strftime("%Y-%m-%d %H:%M")
        report = (f"**金價快報（台灣銀行）**\n- 掛牌時間：{ts}\n- 本行賣出（1克）：**{sell:,.0f} 元**\n- 本行買進（1克）：**{buy:,.0f} 元**\n- 買賣價差：{spread:,.0f} 元（{bias}）\n\n資料來源：{BOT_GOLD_URL}\n（更新於 {now}）")
        logger.info("金價分析成功"); return report
    except Exception as e: logger.error(f"❌ 黃金價格流程失敗: {e}", exc_info=True); return "抱歉，目前無法從台灣銀行取得黃金牌價，稍後再試 🙏"

def get_currency_analysis(target_currency: str):
    logger.info(f"呼叫：get_currency_analysis(target_currency={target_currency})")
    try:
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"; res = requests.get(url, timeout=10); res.raise_for_status()
        data = res.json(); logger.debug(f"匯率 API 回應: {data}")
        if data.get("result") != "success": error_msg = f"抱歉，獲取匯率資料失敗：{data.get('error-type','未知錯誤')}"; logger.error(error_msg); return error_msg
        rate = data["rates"].get("TWD")
        if rate is None: logger.error("匯率 API 回應中無 TWD 資料"); return f"抱歉，API 無 TWD 匯率。"
        report = f"即時：1 {target_currency.upper()} ≈ {rate:.5f} 新台幣"; logger.info("匯率分析成功"); return report
    except Exception as e: logger.error(f"❌ 匯率分析錯誤: {e}", exc_info=True); return "抱歉，外匯資料暫時無法取得。"

_TW_CODE_RE = re.compile(r'^\d{4,6}[A-Za-z]?$')
_US_CODE_RE = re.compile(r'^[A-Za-z]{1,5}$')

def normalize_ticker(user_text: str) -> Tuple[str, str, str, bool]:
    t = user_text.strip().upper(); logger.debug(f"標準化股票代碼: {t}")
    if t in ["台股大盤", "大盤"]: return "^TWII", "^TWII", "^TWII", True
    if t in ["美股大盤", "美盤", "美股"]: return "^GSPC", "^GSPC", "^GSPC", True
    if _TW_CODE_RE.match(t): return f"{t}.TW", t, t, False
    if _US_CODE_RE.match(t) and t != "JPY": return t, t, t, False
    logger.warning(f"無法明確識別的股票/指數代碼: {t}"); return t, t, t, False

def fetch_realtime_snapshot(yf_symbol: str, yahoo_slug: str) -> dict:
    logger.debug(f"呼叫 fetch_realtime_snapshot (yf: {yf_symbol}, slug: {yahoo_slug})")
    snap: dict = {"name": None, "now_price": None, "change": None, "currency": None, "close_time": None}
    try:
        tk = yf.Ticker(yf_symbol); info = {}; hist = pd.DataFrame()
        try: info = tk.info or {}
        except Exception as info_e: logger.warning(f"yf tk.info 失敗 for {yf_symbol}: {info_e}")
        try: hist = tk.history(period="2d", interval="1d")
        except Exception as hist_e: logger.warning(f"yf tk.history 失敗 for {yf_symbol}: {hist_e}")
        name = info.get("shortName") or info.get("longName"); snap["name"] = name or yf_symbol
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"); ccy = info.get("currency")
        if price: snap["now_price"] = f"{price:.2f}"; snap["currency"] = ccy or ("TWD" if yf_symbol.endswith(".TW") else "USD")
        elif not hist.empty: price = float(hist["Close"].iloc[-1]); snap["now_price"] = f"{price:.2f}"; snap["currency"] = ccy or ("TWD" if yf_symbol.endswith(".TW") else "USD")
        if not hist.empty and len(hist) >= 2 and hist["Close"].iloc[-2] != 0: chg = float(hist["Close"].iloc[-1]) - float(hist["Close"].iloc[-2]); pct = chg / float(hist["Close"].iloc[-2]) * 100; sign = "+" if chg >= 0 else ""; snap["change"] = f"{sign}{chg:.2f} ({sign}{pct:.2f}%)"
        elif info.get('regularMarketChange') is not None and info.get('regularMarketChangePercent') is not None: chg = info['regularMarketChange']; pct = info['regularMarketChangePercent'] * 100; sign = "+" if chg >= 0 else ""; snap["change"] = f"{sign}{chg:.2f} ({sign}{pct:.2f}%)"
        if not hist.empty: ts = hist.index[-1]; snap["close_time"] = ts.strftime("%Y-%m-%d %H:%M")
        elif info.get("regularMarketTime"): try: snap["close_time"] = datetime.fromtimestamp(info["regularMarketTime"]).strftime("%Y-%m-%d %H:%M") except: pass
    except Exception as e: logger.warning(f"⚠️ yfinance 取得 {yf_symbol} 失敗：{e}")
    if (not snap["now_price"] or not snap["name"]) and STOCK_ENABLED and 'YahooStock' in globals():
        logger.debug(f"yfinance 失敗，嘗試使用 YahooStock 後備 for {yahoo_slug}")
        try: ys = YahooStock(yahoo_slug); snap["name"] = ys.name or snap["name"] or yahoo_slug; snap["now_price"] = ys.now_price or snap["now_price"]; snap["change"] = ys.change or snap["change"]; snap["currency"] = ys.currency or ("TWD" if yf_symbol.endswith(".TW") else snap["currency"]); snap["close_time"] = ys.close_time or snap["close_time"]; logger.debug("YahooStock 後備成功")
        except Exception as e: logger.error(f"❌ YahooStock 取得 {yahoo_slug} 失敗：{e}")
    logger.debug(f"Snapshot 結果: {snap}"); return snap

stock_data_df: Optional[pd.DataFrame] = None
def load_stock_data() -> pd.DataFrame:
    global stock_data_df
    if stock_data_df is None:
        try: stock_data_df = pd.read_csv('name_df.csv'); logger.info("✅ 成功載入 name_df.csv")
        except FileNotFoundError: logger.error("❌ `name_df.csv` not found."); stock_data_df = pd.DataFrame(columns=['股號', '股名'])
    return stock_data_df

def get_stock_name(stock_id_without_suffix: str) -> Optional[str]:
    df = load_stock_data(); res = df[df['股號'].astype(str).str.strip().str.upper() == str(stock_id_without_suffix).strip().upper()]
    if not res.empty: name = res.iloc[0]['股名']; logger.debug(f"從 name_df.csv 找到 {stock_id_without_suffix} -> {name}"); return name
    logger.debug(f"在 name_df.csv 中找不到 {stock_id_without_suffix}"); return None

def get_stock_report(user_input: str) -> str:
    logger.info(f"呼叫：get_stock_report(user_input={user_input})"); yf_symbol, yahoo_slug, display_code, is_index = normalize_ticker(user_input); snapshot = fetch_realtime_snapshot(yf_symbol, yahoo_slug)
    price_data, news_data, value_part, dividend_part = "", "", "", ""
    if STOCK_ENABLED:
        logger.debug("股票模組已啟用，嘗試獲取詳細資料...")
        try: input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol; logger.debug(f"呼叫 stock_price({input_code})"); price_df = stock_price(input_code); price_data = str(price_df) if not price_df.empty else "無法取得價格資料"
        except Exception as e: logger.warning(f"⚠️ stock_price 失敗：{e}"); price_data = f"錯誤: {e}"
        try: nm = get_stock_name(yahoo_slug) or snapshot.get("name") or yahoo_slug; logger.debug(f"呼叫 stock_news({nm})"); news_list = stock_news(nm); news_data = "\n".join(news_list).replace("\u3000", " ")[:1024]
        except Exception as e: logger.warning(f"⚠️ stock_news 失敗：{e}"); news_data = f"錯誤: {e}"
        if not is_index:
            try: input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol; logger.debug(f"呼叫 stock_fundamental({input_code})"); val = stock_fundamental(input_code); value_part = f"{val}\n" if val else ""
            except Exception as e: logger.warning(f"⚠️ stock_fundamental 失敗：{e}"); value_part = f"錯誤: {e}\n"
            try: input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol; logger.debug(f"呼叫 stock_dividend({input_code})"); dvd = stock_dividend(input_code); dividend_part = f"{dvd}\n" if dvd else ""
            except Exception as e: logger.warning(f"⚠️ stock_dividend 失敗：{e}"); dividend_part = f"錯誤: {e}\n"
    else: logger.warning("⚠️ 股票模組未啟用，僅顯示快照")
    stock_link = (f"https://finance.yahoo.com/quote/{yf_symbol}" if yf_symbol.startswith("^") or not yf_symbol.endswith(".TW") else f"https://tw.stock.yahoo.com/quote/{yahoo_slug}")
    content_msg = (f"你現在是一位專業的證券分析師, 依據以下資料寫一份分析報告：\n**股票代碼:** {display_code}, **股票名稱:** {snapshot.get('name')}\n**目前價格:** {snapshot.get('now_price')} {snapshot.get('currency')}\n**今日漲跌:** {snapshot.get('change')}\n**資料時間:** {snapshot.get('close_time')}\n**近期價格資訊:**\n{price_data}\n")
    if value_part:    content_msg += f"**基本面/營收資訊：**\n{value_part}"
    if dividend_part: content_msg += f"**配息資料：**\n{dividend_part}"
    if news_data:     content_msg += f"**近期新聞資訊：**\n{news_data}\n"
    content_msg += (f"請以嚴謹專業的角度寫出 {snapshot.get('name') or display_code} 近期趨勢，用繁體中文、Markdown 格式，最後**務必**附上這個連結：{stock_link}")
    system_prompt = ("你是專業的台股/美股分析師。請在開頭列出：股名(股號)、現價與漲跌幅、資料時間；接著分段說明：股價走勢、基本面、技術面、消息面、風險、建議區間與停利目標，最後給綜合結論。如果資料不完整或有錯誤，請保守說明。")
    msgs = [{"role":"system","content":system_prompt}, {"role":"user","content":content_msg}]
    logger.info("準備呼叫 AI 進行股票分析..."); analysis_result = get_analysis_reply(msgs); logger.info("股票分析完成"); return analysis_result


# ========== 7) 彩票分析 ==========
# ... (與 v2.0.3 相同) ...
def _lotto_fallback_scrape(kind: str) -> str:
    logger.warning(f"使用後備彩票爬蟲 for {kind}")
    try:
        if kind == "威力彩": url, pat = "https://www.taiwanlottery.com/lotto/superlotto638/index.html", r"第\s*\d+\s*期\s*開獎結果.*?第一區(?:中獎)?號碼(?:依大小順序排列)?[:：\s]*([\d\s,]+?)\s*第二區(?:中獎)?號碼[:：\s]*(\d+)"
        elif kind == "大樂透": url, pat = "https://www.taiwanlottery.com/lotto/lotto649/index.html", r"第\s*\d+\s*期\s*開獎結果.*?(?:中獎號碼|獎號)(?:依大小順序排列)?[:：\s]*([\d\s,]+?)(?:\s*特別號[:：\s]*(\d+))?"
        elif kind == "539": url, pat = "https://www.taiwanlottery.com/lotto/dailycash/index.html", r"第\s*\d+\s*期\s*開獎結果.*?(?:中獎號碼|獎號)(?:依大小順序排列)?[:：\s]*([\d\s,]+)"
        else: return f"不支援彩種：{kind}"
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser"); text = ' '.join(soup.stripped_strings)
        logger.debug(f"後備爬蟲取得文字 (前 200 字): {text[:200]}")
        m = re.search(pat, text, re.DOTALL)
        if not m: logger.error(f"後備爬蟲正則表達式匹配失敗 for {kind}"); return f"抱歉，暫時找不到 {kind} 最新號碼 (Fallback regex failed)。"
        if kind == "威力彩": first, second = re.sub(r'[,\s]+', ' ', m.group(1)).strip(), m.group(2); return f"{kind} 最新號碼：第一區 {first}；第二區 {second}"
        elif kind == "大樂透": nums, special = re.sub(r'[,\s]+', ' ', m.group(1)).strip(), m.group(2); return f"{kind} 最新號碼：{nums}{'；特別號 ' + special if special else ''}"
        elif kind == "539": nums = re.sub(r'[,\s]+', ' ', m.group(1)).strip(); return f"{kind} 最新號碼：{nums}"
    except Exception as e: logger.error(f"❌ 後備彩票爬取失敗：{e}", exc_info=True); return f"抱歉，{kind} 近期號碼暫時取不到 (Fallback exception)。"

def get_lottery_analysis(lottery_type_input: str) -> str:
    logger.info(f"呼叫：get_lottery_analysis(lottery_type_input={lottery_type_input})"); kind = "威力彩" if "威力" in lottery_type_input else ("大樂透" if "大樂" in lottery_type_input else ("539" if "539" in lottery_type_input else lottery_type_input)); latest_data_str = ""
    if LOTTERY_ENABLED and lottery_crawler:
        try:
            logger.debug(f"嘗試使用自訂爬蟲獲取 {kind} 資料...")
            if kind == "威力彩":   latest_data_str = str(lottery_crawler.super_lotto())
            elif kind == "大樂透": latest_data_str = str(lottery_crawler.lotto649())
            elif kind == "539":    latest_data_str = str(lottery_crawler.daily_cash())
            else: return f"不支援 {kind}。"
            logger.info("自訂爬蟲成功獲取資料")
        except Exception as e: logger.warning(f"⚠️ 自訂彩票爬蟲失敗，改用後備：{e}"); latest_data_str = _lotto_fallback_scrape(kind)
    else: logger.warning("自訂彩票模組未啟用或未載入，使用後備爬蟲"); latest_data_str = _lotto_fallback_scrape(kind)
    cai_part = "";
    if caiyunfangwei_crawler:
        try: logger.debug("嘗試獲取財運方位..."); cai = caiyunfangwei_crawler.get_caiyunfangwei(); cai_part = f"今天日期：{cai.get('今天日期','')}\n今日歲次：{cai.get('今日歲次','')}\n財神方位：{cai.get('財神方位','')}\n"; logger.info("財運方位獲取成功")
        except Exception as e: logger.warning(f"⚠️ 無法獲取財運方位: {e}"); cai_part = ""
    prompt = (f"你是一位資深彩券分析師。以下是 {kind} 近況/最新號碼資料：\n{latest_data_str}\n\n{cai_part}請用繁體中文寫出：\n1) 近期走勢重點（高機率區間/熱冷號）\n2) 選號建議與注意事項（理性與風險聲明）\n3) 提供三組推薦號碼（依彩種格式呈現）\n文字請精煉、分點條列。"); messages = [{"role":"system","content":"你是資深彩券分析師。"}, {"role":"user","content":prompt}]
    logger.info("準備呼叫 AI 進行彩票分析..."); analysis_result = get_analysis_reply(messages); logger.info("彩票分析完成"); return analysis_result

# ========== 8) 對話與翻譯 ==========
# ... (與 v2.0.3 相同) ...
def set_user_persona(chat_id: str, key: str):
    logger.debug(f"呼叫 set_user_persona for {chat_id[:10]}... with key={key}"); key = random.choice(list(PERSONAS.keys())) if key == "random" else key; key = "sweet" if key not in PERSONAS else key
    user_persona[chat_id] = key; logger.info(f"人設切換成功: {chat_id[:10]}... -> {key}"); return key

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    key = user_persona.get(chat_id, "sweet"); p = PERSONAS[key]
    prompt = (f"你是一位「{p['title']}」。風格：{p['style']}\n使用者情緒：{sentiment}；請調整語氣（開心→一起開心；難過/生氣→先共情、安撫再給建議；中性→自然聊天）。\n回覆使用繁體中文，精煉自然，帶少量表情 {p['emoji']}。")
    logger.debug(f"建構人設 Prompt (key={key}, sentiment={sentiment}): {prompt[:50]}..."); return prompt

# ========== 9) LINE Handlers (V2 SDK Style) ==========
@handler.add(MessageEvent, message=TextMessage)
def on_message_text(event: MessageEvent):
    chat_id = get_chat_id(event)
    if not isinstance(event.message, TextMessage): logger.warning(f"收到非文字訊息，忽略: {type(event.message)}"); return
    msg_raw = event.message.text.strip(); reply_token = event.reply_token; is_group = not isinstance(event.source, SourceUser)
    logger.info(f"處理文字訊息: '{msg_raw[:50]}...' from {chat_id[:10]}...")
    try: bot_info = line_bot_api.get_bot_info(); bot_name = bot_info.display_name; logger.debug(f"Bot name: {bot_name}")
    except Exception as e: logger.warning(f"⚠️ 獲取 Bot info 失敗: {e}"); bot_name = "AI 助手"
    if not msg_raw: logger.debug("空訊息，忽略"); return
    if chat_id not in auto_reply_status: auto_reply_status[chat_id] = True
    mentioned = msg_raw.startswith(f"@{bot_name}")
    should_reply_in_group = is_group and (auto_reply_status.get(chat_id, True) or mentioned)
    if is_group and not should_reply_in_group: logger.debug("群組中且未提及 Bot 且自動回覆關閉，忽略"); return
    msg = msg_raw[len(f"@{bot_name}"):].strip() if mentioned else msg_raw
    if not msg: logger.debug("移除 @ 後訊息為空，忽略"); return
    low = msg.lower()
    try:
        if low in ("menu", "選單", "主選單"): logger.info("分支：主選單"); return line_bot_api.reply_message(reply_token, build_main_menu_flex())
        if msg in ["大樂透", "威力彩", "539"]: logger.info(f"分支：彩票分析 ({msg})"); report = get_lottery_analysis(msg); return reply_with_quick_bar(reply_token, report)
        if low in ("金價", "黃金"): logger.info("分支：金價查詢"); out = get_gold_analysis(); return reply_with_quick_bar(reply_token, out)
        if low == "jpy": logger.info("分支：日圓匯率查詢"); out = get_currency_analysis("JPY"); return reply_with_quick_bar(reply_token, out)
        if is_stock_query(msg): logger.info(f"分支：股票查詢 ({msg})"); report = get_stock_report(msg); return reply_with_quick_bar(reply_token, report)
        if low in ("開啟自動回答", "關閉自動回答"): logger.info(f"分支：自動回覆設定 ({low})"); is_on = low == "開啟自動回答"; auto_reply_status[chat_id] = is_on; text = "✅ 已開啟自動回答 (群組訊息都會回)" if is_on else "❌ 已關閉自動回答 (群組需 @我 才回)"; return reply_with_quick_bar(reply_token, text)
        if msg.startswith("翻譯->"): lang = msg.split("->", 1)[1].strip(); logger.info(f"分支：翻譯模式切換 ({lang})"); (translation_states.pop(chat_id, None), reply_with_quick_bar(reply_token, "✅ 已結束翻譯模式")) if lang == "結束" else (translation_states.__setitem__(chat_id, lang), reply_with_quick_bar(reply_token, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")); return
        if msg in PERSONA_ALIAS: logger.info(f"分支：人設切換 ({msg})"); key_alias = msg; key = set_user_persona(chat_id, PERSONA_ALIAS[key_alias]); p = PERSONAS[user_persona[chat_id]]; txt = f"💖 已切換人設：{p['title']}\n\n{p['greetings']}"; return reply_with_quick_bar(reply_token, txt)
        if chat_id in translation_states: logger.info(f"分支：執行翻譯 (-> {translation_states[chat_id]})"); out = translate_text(msg, translation_states[chat_id]); return reply_with_quick_bar(reply_token, out)
        logger.info("分支：一般聊天 (Groq/OpenAI)"); history = conversation_history.get(chat_id, []); logger.debug("分析情緒..."); sentiment = analyze_sentiment(msg); logger.debug("建構 Prompt..."); sys_prompt = build_persona_prompt(chat_id, sentiment); messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]; logger.info("呼叫 AI 進行聊天回覆..."); final_reply = get_analysis_reply(messages); history.extend([{"role":"user","content":msg}, {"role":"assistant","content":final_reply}]); conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]; logger.debug("聊天歷史已更新"); return reply_with_quick_bar(reply_token, final_reply)
    # --- 繁體中文解：[修正] 修正 except 區塊的縮排和語法 ---
    except LineBotApiError as lbe:
        logger.error(f"❌ LINE API 錯誤: {lbe.status_code} {lbe.error.message}", exc_info=False)
        try:
            line_bot_api.reply_message(reply_token, TextSendMessage(text="抱歉，與 LINE 溝通時發生錯誤 😥"))
        except Exception as inner_e:
            logger.error(f"❌ 連錯誤訊息都無法回覆 (inner): {inner_e}")
            # 在這裡不需 pass，因為外層 except 會處理
    except Exception as e:
        logger.error(f"❌ on_message_text 內部錯誤: {e}", exc_info=True)
        try:
             reply_with_quick_bar(reply_token, "抱歉，處理您的請求時發生了未預期的錯誤 😵‍💫")
        except Exception as reply_e:
             logger.error(f"❌ 連錯誤訊息都無法回覆: {reply_e}")
             # 在這裡不需 pass


@handler.add(PostbackEvent)
def on_postback(event: PostbackEvent):
    # ... (與 v2.0.3 相同) ...
    logger.info(f"收到 V2 Postback Event from {get_chat_id(event)[:10]}..., data: {event.postback.data}"); data = (event.postback.data or "").strip(); kind = data[5:] if data.startswith("menu:") else None
    if kind: logger.info(f"匹配到 Postback 選單: {kind}")
    try: line_bot_api.reply_message( event.reply_token, [build_submenu_flex(kind), TextSendMessage(text="請選擇一項服務 👇", quick_reply=build_quick_reply())] ); logger.info("Postback 子選單回覆成功")
    except LineBotApiError as lbe: logger.error(f"❌ Postback 回覆 LINE API 錯誤: {lbe.status_code} {lbe.error.message}", exc_info=False)
    except Exception as e: logger.error(f"❌ Postback 回覆失敗: {e}", exc_info=True)
    else: logger.warning(f"⚠️ 未處理的 Postback data: {data}")

def is_stock_query(text: str) -> bool: t = text.strip().upper(); return t in ["台股大盤", "大盤", "美股大盤", "美盤", "美股"] or bool(_TW_CODE_RE.match(t)) or (bool(_US_CODE_RE.match(t)) and t not in ["JPY"])


# ========== 10) FastAPI Routes ==========
# ... (與 v2.0.3 相同) ...
@router.post("/callback")
async def callback(request: Request):
    logger.info("收到 /callback 請求 (V2)")
    if not handler: logger.critical("❌ WebhookHandler 未初始化，無法處理請求"); raise HTTPException(status_code=500, detail="WebhookHandler not initialized")
    signature = request.headers.get("X-Line-Signature", ""); body = await request.body(); body_decoded = body.decode("utf-8")
    logger.debug(f"Callback V2 - Signature: {signature[:10]}..., Body size: {len(body_decoded)}")
    try: handler.handle(body_decoded, signature); logger.info("✅ Callback V2 同步處理完成") # 直接同步調用
    except InvalidSignatureError: logger.error(f"❌ Invalid signature 驗證失敗 (Signature: {signature})，請檢查 CHANNEL_SECRET 是否正確。"); raise HTTPException(status_code=400, detail="Invalid signature")
    except LineBotApiError as lbe: logger.error(f"❌ Callback V2 處理期間 LINE API 錯誤: {lbe.status_code} {lbe.error.message}", exc_info=True); return JSONResponse({"status": "ok but error logged"})
    except Exception as e: logger.error(f"❌ Callback V2 處理失敗：{e}", exc_info=True); raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse({"status": "ok"})

@router.get("/")
async def root(): logger.debug("收到 / (root) 請求"); return PlainTextResponse("LINE Bot (V2 SDK - Sync Handler) is running.", status_code=200)
@router.get("/healthz")
async def healthz(): return PlainTextResponse("ok")
@router.get("/health/providers")
async def providers_health(): logger.info("收到 /health/providers 請求"); return {"openai_client_initialized": openai_client is not None, "groq_client_initialized": sync_groq_client is not None, "line_api_initialized": line_bot_api is not None, "ts": datetime.utcnow().isoformat() + "Z",}

app.include_router(router)

# ========== 11) Local run ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000)); logger.info(f"準備啟動 Uvicorn (app_fastapi:app) 於 0.0.0.0:{port}")
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)