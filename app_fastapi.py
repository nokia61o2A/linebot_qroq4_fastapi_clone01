# app_fastapi.py
# ========== 1) Imports ==========
import os
import re
import random
import logging
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

# --- FastAPI 與 LINE Bot SDK v2 ---
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

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
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# --- 環境變數 ---
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 可不設，會自動改用 Groq

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise RuntimeError("缺少必要環境變數：BASE_URL / CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET / GROQ_API_KEY")

# --- API 用戶端初始化 ---
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
sync_groq_client = Groq(api_key=GROQ_API_KEY)

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.warning(f"初始化 OpenAI 失敗：{e}")

# Groq 模型（改用未下架版本）
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

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
        logger.error(f"Webhook 更新失敗: {e}", exc_info=True)
    yield

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.2.0")
router = APIRouter()

# ========== 4) Helpers ==========
def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom):  return event.source.room_id
    return event.source.user_id

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

def reply_with_quick_bar(reply_token: str, text: str):
    line_bot_api.reply_message(
        reply_token,
        TextSendMessage(text=text, quick_reply=build_quick_reply())
    )

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
            )
            return resp.choices[0].message.content
        except Exception as ee:
            logger.error(f"所有 AI API 都失敗：{ee}")
            return "抱歉，AI 分析師目前連線不穩定，請稍後再試。"

async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
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
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text."
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{text}"}}'
    return await groq_chat_async([{"role":"system","content":sys},{"role":"user","content":usr}], 800, 0.2)

def set_user_persona(chat_id: str, key: str):
    if key == "random": key = random.choice(list(PERSONAS.keys()))
    if key not in PERSONAS: key = "sweet"
    user_persona[chat_id] = key
    return key

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    key = user_persona.get(chat_id, "sweet")
    p = PERSONAS[key]
    return (f"你是一位「{p['title']}」。風格：{p['style']}\n"
            f"使用者情緒：{sentiment}；請調整語氣（開心→一起開心；難過/生氣→先共情、安撫再給建議；中性→自然聊天）。\n"
            f"回覆使用繁體中文，精煉自然，帶少量表情 {p['emoji']}。")

# ========== 9) LINE Handlers ==========
@handler.add(MessageEvent, message=TextMessage)
def on_message_text(event: MessageEvent):
    try:
        asyncio.run(handle_message_async(event))
    except Exception as e:
        logger.error(f"Handle message failed: {e}", exc_info=True)

@handler.add(PostbackEvent)
def on_postback(event: PostbackEvent):
    data = (event.postback.data or "").strip()
    if data.startswith("menu:"):
        kind = data.split(":", 1)[-1]
        line_bot_api.reply_message(
            event.reply_token,
            [build_submenu_flex(kind), TextSendMessage(text="請選擇一項服務", quick_reply=build_quick_reply())]
        )

def is_stock_query(text: str) -> bool:
    t = text.strip().upper()
    if t in ["台股大盤", "大盤", "美股大盤", "美盤", "美股"]:
        return True
    if _TW_CODE_RE.match(t):  # 2330 / 00937B / 1101B ...
        return True
    if _US_CODE_RE.match(t) and t not in ["JPY"]:
        return True
    return False

async def handle_message_async(event: MessageEvent):
    chat_id = get_chat_id(event)
    msg_raw = event.message.text.strip()
    reply_token = event.reply_token
    is_group = not isinstance(event.source, SourceUser)

    try:
        bot_info = await run_in_threadpool(line_bot_api.get_bot_info)
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if not msg_raw:
        return

    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True

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
    except Exception as e:
        logger.error(f"AI 回覆失敗: {e}", exc_info=True)
        return reply_with_quick_bar(reply_token, "抱歉我剛剛走神了 😅 再說一次讓我補上！")

# ========== 10) FastAPI Routes ==========
@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        await run_in_threadpool(handler.handle, body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Callback 處理失敗：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse({"status": "ok"})

@router.get("/")
async def root():
    return PlainTextResponse("LINE Bot is running.", status_code=200)

@router.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

app.include_router(router)

# ========== 11) Local run ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)