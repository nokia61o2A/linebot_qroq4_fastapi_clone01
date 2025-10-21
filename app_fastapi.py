# app_fastapi.py (Version 2.0.8 - Uncompress load_stock_data)
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

required_vars = { "BASE_URL": BASE_URL, "CHANNEL_ACCESS_TOKEN": CHANNEL_TOKEN, "CHANNEL_SECRET": CHANNEL_SECRET, "GROQ_API_KEY": GROQ_API_KEY }
missing_vars = [name for name, value in required_vars.items() if not value]
if missing_vars: error_message = f"❌ 缺少必要環境變數: {', '.join(missing_vars)}"; logger.critical(error_message); raise RuntimeError(error_message)
else: logger.info("✅ 所有必要環境變數均已設定")

# --- API 用戶端初始化 (V2 SDK) ---
try: line_bot_api = LineBotApi(CHANNEL_TOKEN); handler = WebhookHandler(CHANNEL_SECRET); logger.info("✅ LINE Bot API (V2) 初始化成功")
except Exception as e: logger.critical(f"❌ LINE Bot API 初始化失敗: {e}", exc_info=True); line_bot_api = None; handler = None; raise RuntimeError(f"LINE Bot API 初始化失敗: {e}")

# --- AI Client 初始化 ---
async_groq_client, sync_groq_client = None, None
if GROQ_API_KEY:
    try: async_groq_client = AsyncGroq(api_key=GROQ_API_KEY); sync_groq_client = Groq(api_key=GROQ_API_KEY); logger.info("✅ Groq API Client 初始化成功 (Sync & Async)")
    except Exception as e: logger.error(f"❌ Groq API Client 初始化失敗: {e}")
else: logger.warning("⚠️ 未設定 GROQ_API_KEY")

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_base_url = os.getenv("OPENAI_API_BASE")
        if openai_base_url: openai_client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=openai_base_url); logger.info(f"✅ OpenAI Client (自訂 URL: {openai_base_url})")
        else: openai_client = openai.OpenAI(api_key=OPENAI_API_KEY); logger.info("✅ OpenAI Client (官方 URL)")
    except Exception as e: logger.warning(f"⚠️ 初始化 OpenAI 失敗：{e}")
else: logger.info("ℹ️ 未設定 OPENAI_API_KEY")

# --- Groq 模型 ---
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")
logger.info(f"Groq 模型: Primary={GROQ_MODEL_PRIMARY}, Fallback={GROQ_MODEL_FALLBACK}")

# --- 【靈活載入】自訂模組 ---
# ... (與 v2.0.7 相同) ...
LOTTERY_ENABLED = True
try: from TaiwanLottery import TaiwanLotteryCrawler; from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler; lottery_crawler = TaiwanLotteryCrawler(); caiyunfangwei_crawler = CaiyunfangweiCrawler(); logger.info("✅ 已載入彩票模組")
except ModuleNotFoundError: logger.error("❌ 找不到 'taiwanlottery' 模組。請檢查 requirements.txt。"); LOTTERY_ENABLED = False; lottery_crawler = None; caiyunfangwei_crawler = None
except Exception as e: logger.warning(f"⚠️ 無法載入彩票模組：{e}。"); LOTTERY_ENABLED = False; lottery_crawler = None; caiyunfangwei_crawler = None

STOCK_ENABLED = True
try: from my_commands.stock.stock_price import stock_price; from my_commands.stock.stock_news import stock_news; from my_commands.stock.stock_value import stock_fundamental; from my_commands.stock.stock_rate import stock_dividend; from my_commands.stock.YahooStock import YahooStock; logger.info("✅ 已載入股票模組")
except ModuleNotFoundError as e: logger.error(f"❌ 股票模組載入失敗 (ImportError): {e}"); STOCK_ENABLED = False
except Exception as e: logger.warning(f"⚠️ 無法載入股票模組：{e}"); STOCK_ENABLED = False

if not STOCK_ENABLED:
    def stock_price(id): logger.error("股票(備援): stock_price"); return pd.DataFrame()
    def stock_news(hint): logger.error("股票(備援): stock_news"); return ["股票模組未載入"]
    def stock_fundamental(id): logger.error("股票(備援): stock_fundamental"); return "股票模組未載入"
    def stock_dividend(id): logger.error("股票(備援): stock_dividend"); return "股票模組未載入"
    class YahooStock:
        def __init__(self, id): logger.error(f"股票(備援): YahooStock({id})"); self.name=id; self.now_price=None; self.change=None; self.currency=None; self.close_time=None

# --- 狀態字典與常數 ---
# ... (與 v2.0.7 相同) ...
conversation_history: Dict[str, List[dict]] = {}; MAX_HISTORY_LEN = 10; user_persona: Dict[str, str] = {}; translation_states: Dict[str, str] = {}; auto_reply_status: Dict[str, bool] = {}
PERSONAS = { "sweet": {"title": "甜美女友", "style": "溫柔體貼", "greetings": "親愛的～我在這🌸", "emoji":"🌸💕😊"}, "salty": {"title": "傲嬌女友", "style": "機智吐槽", "greetings": "你又來啦？說吧😏", "emoji":"😏🙄"}, "moe":   {"title": "萌系女友", "style": "動漫語氣", "greetings": "呀呼～(ﾉ>ω<)ﾉ", "emoji":"✨🎀"}, "cool":  {"title": "酷系御姐", "style": "冷靜精煉", "greetings": "我在。說重點。", "emoji":"🧊⚡️"} }
LANGUAGE_MAP = {"英文": "English", "日文": "Japanese", "韓文": "Korean", "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"}
PERSONA_ALIAS = {"甜":"sweet", "鹹":"salty", "萌":"moe", "酷":"cool", "random":"random"}

# ========== 3) FastAPI ==========
# ... (lifespan 與 v2.0.7 相同) ...
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
        except Exception as e: logger.error(f"⚠️ Webhook 更新失敗: {e}", exc_info=True)
    else: logger.warning("⚠️ Webhook 未更新：未設定 BASE_URL 或 CHANNEL_ACCESS_TOKEN (Mock 模式)")
    logger.info("Lifespan 啟動程序完成。"); yield; logger.info("應用程式關閉 (lifespan)...")

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="2.0.8-uncompress-load-stock") # --- 繁體中文解：更新版本號 ---
router = APIRouter()

# ========== 4) Helpers (V2 SDK Style) ==========
# ... (get_chat_id, build_quick_reply, reply_with_quick_bar, build_main_menu_flex, build_submenu_flex 與 v2.0.7 相同) ...
def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom):  return event.source.room_id
    if isinstance(event.source, SourceUser): return event.source.user_id
    logger.warning(f"未知的 event source type: {type(event.source)}"); return "unknown_source"

def build_quick_reply() -> QuickReply:
    logger.debug("建立 QuickReply"); return QuickReply(items=[ QuickReplyButton(action=MessageAction(label="主選單", text="選單")), QuickReplyButton(action=MessageAction(label="台股大盤", text="台股大盤")), QuickReplyButton(action=MessageAction(label="美股大盤", text="美股大盤")), QuickReplyButton(action=MessageAction(label="黃金價格", text="金價")), QuickReplyButton(action=MessageAction(label="查台積電", text="2330")), QuickReplyButton(action=MessageAction(label="查輝達", text="NVDA")), QuickReplyButton(action=MessageAction(label="查日圓", text="JPY")), QuickReplyButton(action=PostbackAction(label="💖 AI 人設", data="menu:persona")), QuickReplyButton(action=PostbackAction(label="🎰 彩票選單", data="menu:lottery")) ])

def reply_with_quick_bar(reply_token: str, text: str):
    if not line_bot_api: logger.error("LINE API 未初始化"); print(f"[MOCK] QR Reply: {text}"); return
    try: logger.debug(f"回覆 (QR): {text[:50]}..."); line_bot_api.reply_message(reply_token, TextSendMessage(text=text, quick_reply=build_quick_reply())); logger.debug("回覆 (QR) 成功")
    except LineBotApiError as lbe: logger.error(f"❌ 回覆 (QR) 失敗: {lbe.status_code} {lbe.error.message}", exc_info=False)
    except Exception as e: logger.error(f"❌ 回覆 (QR) 未知錯誤: {e}", exc_info=True)

def build_main_menu_flex() -> FlexSendMessage: logger.debug("建主選單 Flex"); bubble = BubbleContainer( direction="ltr", header=BoxComponent(layout="vertical", contents=[TextComponent(text="AI 助理選單", weight="bold", size="lg")]), body=BoxComponent( layout="vertical", spacing="md", contents=[ TextComponent(text="選擇功能：", size="sm"), SeparatorComponent(margin="md"), ButtonComponent(action=PostbackAction(label="💹 金融查詢", data="menu:finance"), style="primary", color="#5E86C1"), ButtonComponent(action=PostbackAction(label="🎰 彩票分析", data="menu:lottery"), style="primary", color="#5EC186"), ButtonComponent(action=PostbackAction(label="💖 AI 角色", data="menu:persona"), style="secondary"), ButtonComponent(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate"), style="secondary"), ButtonComponent(action=PostbackAction(label="⚙️ 系統設定", data="menu:settings"), style="secondary"), ] ) ); return FlexSendMessage(alt_text="主選單", contents=bubble)
def build_submenu_flex(kind: str) -> FlexSendMessage:
    logger.debug(f"建子選單 Flex ({kind})"); title, buttons = "子選單", []
    if kind == "finance": title, buttons = "💹 金融查詢", [ButtonComponent(action=MessageAction(label="台股", text="台股大盤")), ButtonComponent(action=MessageAction(label="美股", text="美股大盤")), ButtonComponent(action=MessageAction(label="黃金", text="金價")), ButtonComponent(action=MessageAction(label="日圓", text="JPY")), ButtonComponent(action=MessageAction(label="2330", text="2330")), ButtonComponent(action=MessageAction(label="NVDA", text="NVDA"))]
    elif kind == "lottery": title, buttons = "🎰 彩票分析", [ButtonComponent(action=MessageAction(label="大樂透", text="大樂透")), ButtonComponent(action=MessageAction(label="威力彩", text="威力彩")), ButtonComponent(action=MessageAction(label="今彩539", text="539"))]
    elif kind == "persona": title, buttons = "💖 AI 角色", [ButtonComponent(action=MessageAction(label="甜美女友", text="甜")), ButtonComponent(action=MessageAction(label="傲嬌女友", text="鹹")), ButtonComponent(action=MessageAction(label="萌系女友", text="萌")), ButtonComponent(action=MessageAction(label="酷系御姐", text="酷")), ButtonComponent(action=MessageAction(label="隨機", text="random"))]
    elif kind == "translate": title, buttons = "🌐 翻譯工具", [ButtonComponent(action=MessageAction(label="翻英文", text="翻譯->英文")), ButtonComponent(action=MessageAction(label="翻日文", text="翻譯->日文")), ButtonComponent(action=MessageAction(label="翻繁中", text="翻譯->繁體中文")), ButtonComponent(action=MessageAction(label="結束", text="翻譯->結束"))]
    elif kind == "settings": title, buttons = "⚙️ 系統設定", [ButtonComponent(action=MessageAction(label="開啟自動回答", text="開啟自動回答")), ButtonComponent(action=MessageAction(label="關閉自動回答", text="關閉自動回答"))]
    bubble = BubbleContainer( direction="ltr", header=BoxComponent(layout="vertical", contents=[TextComponent(text=title, weight="bold", size="lg")]), body=BoxComponent(layout="vertical", contents=buttons, spacing="sm") ); return FlexSendMessage(alt_text=title, contents=bubble)

# ========== 5) AI & 分析 ==========
# ... (與 v2.0.7 相同) ...
def get_analysis_reply(messages: List[dict]) -> str:
    logger.debug(f"呼叫 get_analysis_reply (OpenAI優先), messages count: {len(messages)}")
    if openai_client:
        try: logger.debug("嘗試 OpenAI..."); resp = openai_client.chat.completions.create( model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=1500, ); reply = resp.choices[0].message.content; logger.debug(f"OpenAI 成功, len: {len(reply)}"); return reply
        except Exception as e: logger.warning(f"⚠️ OpenAI 失敗：{e}")
    if not sync_groq_client: logger.error("❌ Groq Client 未初始化"); return "抱歉，AI 分析引擎無法連線。"
    try: logger.debug(f"嘗試 Groq Primary: {GROQ_MODEL_PRIMARY}"); resp = sync_groq_client.chat.completions.create( model=GROQ_MODEL_PRIMARY, messages=messages, temperature=0.7, max_tokens=2000, ); reply = resp.choices[0].message.content; logger.debug(f"Groq Primary 成功, len: {len(reply)}"); return reply
    except Exception as e:
        logger.warning(f"⚠️ Groq Primary 失敗：{e}")
        try: logger.debug(f"嘗試 Groq Fallback: {GROQ_MODEL_FALLBACK}"); resp = sync_groq_client.chat.completions.create( model=GROQ_MODEL_FALLBACK, messages=messages, temperature=0.9, max_tokens=1500, ); reply = resp.choices[0].message.content; logger.debug(f"Groq Fallback 成功, len: {len(reply)}"); return reply
        except Exception as ee: logger.error(f"❌ 所有 AI API 都失敗：{ee}", exc_info=True); return "抱歉，AI 分析師目前連線不穩定，請稍後再試。"

def analyze_sentiment(text: str) -> str:
    logger.debug(f"分析情緒: {text[:30]}..."); msgs = [{"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},{"role":"user","content":text}]
    if not sync_groq_client: logger.error("❌ Groq Client 未初始化"); return "neutral"
    try: resp = sync_groq_client.chat.completions.create( model=GROQ_MODEL_FALLBACK, messages=msgs, max_tokens=10, temperature=0 ); result = (resp.choices[0].message.content or "neutral").strip().lower(); logger.debug(f"情緒結果: {result}"); return result if result in ["positive", "neutral", "negative", "angry"] else "neutral"
    except Exception as e: logger.error(f"❌ Groq 情緒分析失敗: {e}", exc_info=False); return "neutral"

def translate_text(text: str, target_lang_display: str) -> str:
    logger.debug(f"翻譯 to {target_lang_display}: {text[:30]}..."); target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text, without intro."; usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{text}"}}'
    if not sync_groq_client: logger.error("❌ Groq Client 未初始化"); return "抱歉，翻譯引擎無法連線。"
    try: resp = sync_groq_client.chat.completions.create( model=GROQ_MODEL_FALLBACK, messages=[{"role":"system","content":sys},{"role":"user","content":usr}], max_tokens=len(text)*3 + 50, temperature=0.2 ); translated_text = (resp.choices[0].message.content or "").strip(); logger.debug(f"翻譯結果: {translated_text[:50]}..."); return translated_text
    except Exception as e: logger.error(f"❌ Groq 翻譯失敗: {e}", exc_info=False); return "抱歉，翻譯功能暫時出錯。"


# ========== 6) 金融工具 ==========
# ... (get_gold_analysis, get_currency_analysis 與 v2.0.7 相同) ...
def get_gold_analysis() -> str:
    logger.info("呼叫：get_gold_analysis()")
    try: r = requests.get(BOT_GOLD_URL, headers=DEFAULT_HEADERS, timeout=10); r.raise_for_status(); data = _parse_bot_gold_text(r.text); logger.debug(f"金價: {data}"); ts = data.get("listed_at") or "N/A"; sell, buy = data["sell_twd_per_g"], data["buy_twd_per_g"]; spread = sell - buy; bias = "盤整" if spread <= 30 else ("偏寬" if spread <= 60 else "價差大"); now = datetime.now().strftime("%H:%M"); report = (f"**金價({now})**\n賣: **{sell:,.0f}** | 買: **{buy:,.0f}** | 價差: {spread:,.0f} ({bias})\n掛牌: {ts}\n來源:台灣銀行"); logger.info("金價分析成功"); return report
    except Exception as e: logger.error(f"❌ 黃金分析失敗: {e}", exc_info=False); return "抱歉，目前無法取得黃金牌價 🙏"

def get_currency_analysis(target_currency: str):
    logger.info(f"呼叫：get_currency_analysis({target_currency})")
    try:
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"; res = requests.get(url, timeout=10); res.raise_for_status(); data = res.json(); logger.debug(f"匯率 API: {data}");
        if data.get("result") != "success": error_msg = f"匯率 API 錯誤: {data.get('error-type','未知')}"; logger.error(error_msg); return error_msg
        rate = data["rates"].get("TWD");
        if rate is None: logger.error("匯率 API 回應中無 TWD"); return f"抱歉，API 無 TWD 匯率。"
        report = f"即時：1 {target_currency.upper()} ≈ **{rate:.4f}** 新台幣"; logger.info("匯率分析成功"); return report
    except requests.exceptions.RequestException as req_e: logger.error(f"❌ 匯率 API 請求失敗: {req_e}", exc_info=False); return "抱歉，無法連線至匯率伺服器。"
    except Exception as e: logger.error(f"❌ 匯率分析未知錯誤: {e}", exc_info=True); return "抱歉，外匯資料暫無法取得。"

# --- 股票相關函數 ---
_TW_CODE_RE = re.compile(r'^\d{4,6}[A-Za-z]?$')
_US_CODE_RE = re.compile(r'^[A-Za-z]{1,5}$')
def normalize_ticker(t: str) -> Tuple[str, str, str, bool]: t = t.strip().upper(); logger.debug(f"正規化 ticker: {t}"); if t in ["台股大盤", "大盤"]: return "^TWII", "^TWII", "^TWII", True; if t in ["美股大盤", "美盤", "美股"]: return "^GSPC", "^GSPC", "^GSPC", True; if _TW_CODE_RE.match(t): return f"{t}.TW", t, t, False; if _US_CODE_RE.match(t) and t != "JPY": return t, t, t, False; logger.warning(f"無法識別 ticker: {t}"); return t, t, t, False
def fetch_realtime_snapshot(yf_symbol: str, yahoo_slug: str) -> dict:
    # ... (與 v2.0.7 相同) ...
    logger.debug(f"抓取快照 (yf: {yf_symbol}, slug: {yahoo_slug})")
    snap: dict = {"name": None, "now_price": None, "change": None, "currency": None, "close_time": None}
    try:
        tk = yf.Ticker(yf_symbol); info = {}; hist = pd.DataFrame()
        try: info = tk.info or {}
        except Exception as info_e: logger.warning(f"yf tk.info fail: {info_e}")
        try: hist = tk.history(period="2d", interval="1d")
        except Exception as hist_e: logger.warning(f"yf tk.history fail: {hist_e}")
        name = info.get("shortName") or info.get("longName"); snap["name"] = name or yf_symbol
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"); ccy = info.get("currency")
        if price: snap["now_price"] = f"{price:.2f}"; snap["currency"] = ccy or ("TWD" if yf_symbol.endswith(".TW") else "USD")
        elif not hist.empty: price = float(hist["Close"].iloc[-1]); snap["now_price"] = f"{price:.2f}"; snap["currency"] = ccy or ("TWD" if yf_symbol.endswith(".TW") else "USD")
        if not hist.empty and len(hist) >= 2 and hist["Close"].iloc[-2] != 0: chg = float(hist["Close"].iloc[-1]) - float(hist["Close"].iloc[-2]); pct = chg / float(hist["Close"].iloc[-2]) * 100; sign = "+" if chg >= 0 else ""; snap["change"] = f"{sign}{chg:.2f} ({sign}{pct:.2f}%)"
        elif info.get('regularMarketChange') is not None and info.get('regularMarketChangePercent') is not None: chg = info['regularMarketChange']; pct = info['regularMarketChangePercent'] * 100; sign = "+" if chg >= 0 else ""; snap["change"] = f"{sign}{chg:.2f} ({sign}{pct:.2f}%)"
        if not hist.empty: ts = hist.index[-1]; snap["close_time"] = ts.strftime("%Y-%m-%d %H:%M")
        elif info.get("regularMarketTime"):
            try: snap["close_time"] = datetime.fromtimestamp(info["regularMarketTime"]).strftime("%Y-%m-%d %H:%M")
            except Exception as ts_e: logger.warning(f"解析 timestamp {info.get('regularMarketTime')} 失敗: {ts_e}")
    except Exception as e: logger.warning(f"⚠️ yfinance fail: {e}")
    if (not snap["now_price"] or not snap["name"]) and STOCK_ENABLED and 'YahooStock' in globals():
        logger.debug(f"嘗試 YahooStock fallback for {yahoo_slug}")
        try: ys = YahooStock(yahoo_slug); snap["name"] = ys.name or snap["name"] or yahoo_slug; snap["now_price"] = ys.now_price or snap["now_price"]; snap["change"] = ys.change or snap["change"]; snap["currency"] = ys.currency or ("TWD" if yf_symbol.endswith(".TW") else snap["currency"]); snap["close_time"] = ys.close_time or snap["close_time"]; logger.debug("YahooStock fallback success")
        except Exception as e: logger.error(f"❌ YahooStock fallback fail: {e}")
    logger.debug(f"快照結果: {snap}"); return snap

stock_data_df: Optional[pd.DataFrame] = None
# --- 繁體中文解：[修正] 將 load_stock_data 恢復多行格式 ---
def load_stock_data() -> pd.DataFrame:
    global stock_data_df
    if stock_data_df is None:
        try:
            stock_data_df = pd.read_csv('name_df.csv')
            logger.info("✅ loaded name_df.csv")
        except FileNotFoundError:
            logger.error("❌ `name_df.csv` not found.")
            # --- 繁體中文解：[修正] 確保在 except 後仍然返回 DataFrame ---
            stock_data_df = pd.DataFrame(columns=['股號', '股名'])
    # --- 繁體中文解：[修正] 確保函數總是有返回值 ---
    return stock_data_df

# ... (get_stock_name, get_stock_report 與 v2.0.7 相同) ...
def get_stock_name(stock_id: str) -> Optional[str]: df = load_stock_data(); res = df[df['股號'].astype(str).str.strip().str.upper() == str(stock_id).strip().upper()]; if not res.empty: name = res.iloc[0]['股名']; logger.debug(f"name_df lookup: {stock_id} -> {name}"); return name; logger.debug(f"name_df not found: {stock_id}"); return None
def get_stock_report(user_input: str) -> str:
    logger.info(f"呼叫：get_stock_report({user_input})"); yf_symbol, yahoo_slug, display_code, is_index = normalize_ticker(user_input); snapshot = fetch_realtime_snapshot(yf_symbol, yahoo_slug)
    price_data, news_data, value_part, dividend_part = "", "", "", ""
    if STOCK_ENABLED:
        logger.debug("股票模組啟用，抓詳細資料...")
        try: input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol; logger.debug(f"call stock_price({input_code})"); price_df = stock_price(input_code); price_data = str(price_df) if not price_df.empty else "N/A"
        except Exception as e: logger.warning(f"⚠️ stock_price fail: {e}"); price_data = f"Err: {e}"
        try: nm = get_stock_name(yahoo_slug) or snapshot.get("name") or yahoo_slug; logger.debug(f"call stock_news({nm})"); news_list = stock_news(nm); news_data = "\n".join(news_list).replace("\u3000", " ")[:1024]
        except Exception as e: logger.warning(f"⚠️ stock_news fail: {e}"); news_data = f"Err: {e}"
        if not is_index:
            try: input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol; logger.debug(f"call stock_fundamental({input_code})"); val = stock_fundamental(input_code); value_part = f"{val}\n" if val else ""
            except Exception as e: logger.warning(f"⚠️ stock_fundamental fail: {e}"); value_part = f"Err: {e}\n"
            try: input_code = yahoo_slug if _TW_CODE_RE.match(yahoo_slug) else yf_symbol; logger.debug(f"call stock_dividend({input_code})"); dvd = stock_dividend(input_code); dividend_part = f"{dvd}\n" if dvd else ""
            except Exception as e: logger.warning(f"⚠️ stock_dividend fail: {e}"); dividend_part = f"Err: {e}\n"
    else: logger.warning("⚠️ 股票模組未啟用")
    stock_link = (f"https://finance.yahoo.com/quote/{yf_symbol}" if yf_symbol.startswith("^") or not yf_symbol.endswith(".TW") else f"https://tw.stock.yahoo.com/quote/{yahoo_slug}")
    content_msg = (f"分析報告:\n**代碼:** {display_code}, **名稱:** {snapshot.get('name')}\n**價格:** {snapshot.get('now_price')} {snapshot.get('currency')}\n**漲跌:** {snapshot.get('change')}\n**時間:** {snapshot.get('close_time')}\n**近期價:**\n{price_data}\n")
    if value_part:    content_msg += f"**基本面:**\n{value_part}"
    if dividend_part: content_msg += f"**配息:**\n{dividend_part}"
    if news_data:     content_msg += f"**新聞:**\n{news_data}\n"
    content_msg += (f"請寫出 {snapshot.get('name') or display_code} 近期趨勢分析，用繁體中文 Markdown，附連結：{stock_link}")
    system_prompt = ("你是專業分析師。開頭列出股名(股號)/現價/漲跌/時間；分段說明走勢/基本面/技術面/消息面/風險/建議區間/停利目標/結論。資料不完整請保守說明。")
    msgs = [{"role":"system","content":system_prompt}, {"role":"user","content":content_msg}]
    logger.info("呼叫 AI 股票分析..."); analysis_result = get_analysis_reply(msgs); logger.info("股票分析完成"); return analysis_result

# ========== 7) 彩票分析 ==========
# ... (與 v2.0.7 相同) ...
def _lotto_fallback_scrape(kind: str) -> str:
    logger.warning(f"使用後備彩票爬蟲 for {kind}")
    try:
        if kind == "威力彩": url, pat = "https://www.taiwanlottery.com/lotto/superlotto638/index.html", r"第\s*\d+\s*期.*?第一區.*?[:：\s]*([\d\s,]+?)\s*第二區.*?[:：\s]*(\d+)"
        elif kind == "大樂透": url, pat = "https://www.taiwanlottery.com/lotto/lotto649/index.html", r"第\s*\d+\s*期.*?(?:號碼|獎號).*?[:：\s]*([\d\s,]+?)(?:\s*特別號[:：\s]*(\d+))?"
        elif kind == "539": url, pat = "https://www.taiwanlottery.com/lotto/dailycash/index.html", r"第\s*\d+\s*期.*?(?:號碼|獎號).*?[:：\s]*([\d\s,]+)"
        else: return f"不支援: {kind}"
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=10); r.raise_for_status(); soup = BeautifulSoup(r.content, "html.parser"); text = ' '.join(soup.stripped_strings); logger.debug(f"Fallback text (200): {text[:200]}")
        m = re.search(pat, text, re.DOTALL);
        if not m: logger.error(f"Fallback regex fail for {kind}"); return f"抱歉，找不到 {kind} 號碼 (Fallback regex failed)。"
        if kind == "威力彩": first, second = re.sub(r'[,\s]+', ' ', m.group(1)).strip(), m.group(2); return f"{kind}: 一區 {first}；二區 {second}"
        elif kind == "大樂透": nums, special = re.sub(r'[,\s]+', ' ', m.group(1)).strip(), m.group(2); return f"{kind}: {nums}{'；特 ' + special if special else ''}"
        elif kind == "539": nums = re.sub(r'[,\s]+', ' ', m.group(1)).strip(); return f"{kind}: {nums}"
    except Exception as e: logger.error(f"❌ Fallback scrape fail: {e}", exc_info=False); return f"抱歉，{kind} 號碼取不到 (Fallback exception)。"

def get_lottery_analysis(lottery_type_input: str) -> str:
    logger.info(f"呼叫：get_lottery_analysis({lottery_type_input})"); kind = "威力彩" if "威力" in lottery_type_input else ("大樂透" if "大樂" in lottery_type_input else ("539" if "539" in lottery_type_input else lottery_type_input)); latest_data_str = ""
    if LOTTERY_ENABLED and lottery_crawler:
        try: logger.debug(f"嘗試自訂爬蟲 for {kind}...");
            if kind == "威力彩":   latest_data_str = str(lottery_crawler.super_lotto())
            elif kind == "大樂透": latest_data_str = str(lottery_crawler.lotto649())
            elif kind == "539":    latest_data_str = str(lottery_crawler.daily_cash())
            else: return f"不支援 {kind}。"
            logger.info("自訂爬蟲成功")
        except Exception as e: logger.warning(f"⚠️ 自訂爬蟲失敗，改用後備：{e}"); latest_data_str = _lotto_fallback_scrape(kind)
    else: logger.warning("彩票模組未啟用/載入，用後備"); latest_data_str = _lotto_fallback_scrape(kind)
    cai_part = "";
    if caiyunfangwei_crawler:
        try: logger.debug("嘗試財運方位..."); cai = caiyunfangwei_crawler.get_caiyunfangwei(); cai_part = f"日期：{cai.get('今天日期','')}\n歲次：{cai.get('今日歲次','')}\n財位：{cai.get('財神方位','')}\n"; logger.info("財運方位成功")
        except Exception as e: logger.warning(f"⚠️ 財運方位失敗: {e}"); cai_part = ""
    prompt = (f"{kind} 近況/號碼：\n{latest_data_str}\n\n{cai_part}請用繁體中文寫出：\n1) 走勢重點(熱冷號)\n2) 選號建議(風險聲明)\n3) 三組推薦號碼\n分點條列精煉。"); messages = [{"role":"system","content":"你是資深彩券分析師。"},{"role":"user","content":prompt}]
    logger.info("呼叫 AI 彩票分析..."); analysis_result = get_analysis_reply(messages); logger.info("彩票分析完成"); return analysis_result

# ========== 8) 對話與翻譯 ==========
# ... (與 v2.0.7 相同) ...
def set_user_persona(chat_id: str, key: str): logger.debug(f"Set persona: {chat_id[:10]} -> {key}"); key = random.choice(list(PERSONAS.keys())) if key == "random" else key; key = "sweet" if key not in PERSONAS else key; user_persona[chat_id] = key; logger.info(f"Persona set: {chat_id[:10]} -> {key}"); return key
def build_persona_prompt(chat_id: str, sentiment: str) -> str: key = user_persona.get(chat_id, "sweet"); p = PERSONAS[key]; prompt = (f"你是「{p['title']}」。風格：{p['style']}\n情緒：{sentiment}；調整語氣（開心→同樂；難過/生氣→共情安撫；中性→自然）。\n用繁體中文，精煉自然，帶少量表情 {p['emoji']}。"); logger.debug(f"Persona prompt (key={key}, sent={sentiment}): {prompt[:50]}..."); return prompt

# ========== 9) LINE Handlers (V2 SDK Style) ==========
# ... (與 v2.0.7 相同) ...
@handler.add(MessageEvent, message=TextMessage)
def on_message_text(event: MessageEvent):
    chat_id = get_chat_id(event); msg_raw = event.message.text.strip(); reply_token = event.reply_token; is_group = not isinstance(event.source, SourceUser)
    if not isinstance(event.message, TextMessage): logger.warning(f"Ignore non-text msg: {type(event.message)}"); return
    logger.info(f"Msg: '{msg_raw[:50]}...' from {chat_id[:10]}...")
    try: bot_info = line_bot_api.get_bot_info(); bot_name = bot_info.display_name; logger.debug(f"Bot name: {bot_name}")
    except Exception as e: logger.warning(f"⚠️ Get Bot info fail: {e}"); bot_name = "AI 助手"
    if not msg_raw: logger.debug("Empty msg"); return
    if chat_id not in auto_reply_status: auto_reply_status[chat_id] = True
    mentioned = msg_raw.startswith(f"@{bot_name}"); should_reply = not is_group or auto_reply_status.get(chat_id, True) or mentioned
    if not should_reply: logger.debug("Ignore msg in group"); return
    msg = msg_raw[len(f"@{bot_name}"):].strip() if mentioned else msg_raw
    if not msg: logger.debug("Empty after mention removal"); return
    low = msg.lower()
    try:
        if low in ("menu", "選單", "主選單"): logger.info("Route: Main menu"); return line_bot_api.reply_message(reply_token, build_main_menu_flex())
        if msg in ["大樂透", "威力彩", "539"]: logger.info(f"Route: Lottery ({msg})"); report = get_lottery_analysis(msg); return reply_with_quick_bar(reply_token, report)
        if low in ("金價", "黃金"): logger.info("Route: Gold"); out = get_gold_analysis(); return reply_with_quick_bar(reply_token, out)
        if low == "jpy": logger.info("Route: JPY"); out = get_currency_analysis("JPY"); return reply_with_quick_bar(reply_token, out)
        if is_stock_query(msg): logger.info(f"Route: Stock ({msg})"); report = get_stock_report(msg); return reply_with_quick_bar(reply_token, report)
        if low in ("開啟自動回答", "關閉自動回答"): logger.info(f"Route: Auto-reply ({low})"); is_on = low == "開啟自動回答"; auto_reply_status[chat_id] = is_on; text = "✅ 自動回答已開啟" if is_on else "❌ 自動回答已關閉"; return reply_with_quick_bar(reply_token, text)
        if msg.startswith("翻譯->"): lang = msg.split("->", 1)[1].strip(); logger.info(f"Route: Translate mode ({lang})"); (translation_states.pop(chat_id, None), reply_with_quick_bar(reply_token, "✅ 翻譯模式結束")) if lang == "結束" else (translation_states.__setitem__(chat_id, lang), reply_with_quick_bar(reply_token, f"🌐 開啟翻譯 → {lang}")); return
        if msg in PERSONA_ALIAS: logger.info(f"Route: Set Persona ({msg})"); key = set_user_persona(chat_id, PERSONA_ALIAS[msg]); p = PERSONAS[user_persona[chat_id]]; txt = f"💖 切換人設：{p['title']}\n{p['greetings']}"; return reply_with_quick_bar(reply_token, txt)
        if chat_id in translation_states: logger.info(f"Route: Translate content (-> {translation_states[chat_id]})"); out = translate_text(msg, translation_states[chat_id]); return reply_with_quick_bar(reply_token, out)
        logger.info("Route: General Chat"); history = conversation_history.get(chat_id, []); logger.debug("Analyze sentiment..."); sentiment = analyze_sentiment(msg); logger.debug("Build prompt..."); sys_prompt = build_persona_prompt(chat_id, sentiment); messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]; logger.info("Call AI chat..."); final_reply = get_analysis_reply(messages); history.extend([{"role":"user","content":msg}, {"role":"assistant","content":final_reply}]); conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]; logger.debug("History updated"); return reply_with_quick_bar(reply_token, final_reply)
    except LineBotApiError as lbe: logger.error(f"❌ LINE API Error: {lbe.status_code} {lbe.error.message}", exc_info=False); try: line_bot_api.reply_message(reply_token, TextSendMessage(text="😥 LINE communication error.")) except: pass
    except Exception as e: logger.error(f"❌ Handler internal error: {e}", exc_info=True); try: reply_with_quick_bar(reply_token, "😵‍💫 Unexpected error processing request.") except Exception as reply_e: logger.error(f"❌ Failed to even send error reply: {reply_e}")

@handler.add(PostbackEvent)
def on_postback(event: PostbackEvent):
    logger.info(f"Postback: data={event.postback.data} from {get_chat_id(event)[:10]}...")
    data = (event.postback.data or "").strip(); kind = data[5:] if data.startswith("menu:") else None
    if kind: logger.info(f"Postback menu: {kind}")
    try: line_bot_api.reply_message( event.reply_token, [build_submenu_flex(kind), TextSendMessage(text="請選擇 👇", quick_reply=build_quick_reply())] ); logger.info("Postback submenu reply OK")
    except LineBotApiError as lbe: logger.error(f"❌ Postback LINE API Error: {lbe.status_code} {lbe.error.message}", exc_info=False)
    except Exception as e: logger.error(f"❌ Postback reply fail: {e}", exc_info=True)
    else:
        if not kind: logger.warning(f"⚠️ Unhandled Postback data: {data}")

def is_stock_query(text: str) -> bool: t = text.strip().upper(); return t in ["台股大盤", "大盤", "美股大盤", "美盤", "美股"] or bool(_TW_CODE_RE.match(t)) or (bool(_US_CODE_RE.match(t)) and t not in ["JPY"])

# ========== 10) FastAPI Routes ==========
# ... (與 v2.0.7 相同) ...
@router.post("/callback")
async def callback(request: Request):
    logger.info("Callback V2 received"); signature = request.headers.get("X-Line-Signature", ""); body = await request.body(); body_decoded = body.decode("utf-8"); logger.debug(f"Sig: {signature[:10]}..., Body: {len(body_decoded)} bytes")
    if not handler: logger.critical("❌ Handler not init!"); raise HTTPException(status_code=500, detail="Handler not initialized")
    try: handler.handle(body_decoded, signature); logger.info("✅ Callback V2 handled")
    except InvalidSignatureError: logger.error(f"❌ Invalid signature: {signature}"); raise HTTPException(status_code=400, detail="Invalid signature")
    except LineBotApiError as lbe: logger.error(f"❌ LINE API Error in callback: {lbe.status_code} {lbe.error.message}", exc_info=True); return JSONResponse({"status": "ok but error logged"})
    except Exception as e: logger.error(f"❌ Callback V2 fail: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse({"status": "ok"})

@router.get("/")
async def root(): logger.debug("GET /"); return PlainTextResponse("LINE Bot (V2 SDK - Sync) running.", status_code=200)
@router.get("/healthz")
async def healthz(): return PlainTextResponse("ok")
@router.get("/health/providers")
async def providers_health(): logger.info("GET /health/providers"); return {"openai_ok": openai_client is not None, "groq_ok": sync_groq_client is not None, "line_ok": line_bot_api is not None, "ts": datetime.utcnow().isoformat() + "Z",}

app.include_router(router)

# ========== 11) Local run ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000)); logger.info(f"Starting Uvicorn on 0.0.0.0:{port}")
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)