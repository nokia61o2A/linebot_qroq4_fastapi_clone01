# ========== 1) Imports ==========
import os
import re
import random
import logging
import asyncio
from typing import Dict, List
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

# --- 數據處理與爬蟲 ---
import requests
from bs4 import BeautifulSoup
import httpx
import pandas as pd
import yfinance as yf

# --- FastAPI 與 LINE Bot SDK ---
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import LineBotApiError, InvalidSignatureError
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

# --- 【靈活載入】載入自訂的彩票與股票爬蟲模組 ---
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

# ========== 2) Setup ==========
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# --- 環境變數 ---
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise RuntimeError("缺少必要環境變數 (BASE_URL, CHANNEL_ACCESS_TOKEN, CHANNEL_SECRET, GROQ_API_KEY)")

# --- API 用戶端初始化 ---
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# AI 客戶端初始化
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
sync_groq_client = Groq(api_key=GROQ_API_KEY)
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)  # 你環境已裝 openai>=1.x
else:
    openai_client = None
    logger.warning("未設定 OPENAI_API_KEY，AI 分析功能將僅使用 Groq。")

# Groq 模型（可由環境變數覆寫）
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

if LOTTERY_ENABLED:
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()

# --- 狀態字典與常數 ---
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10  # 對話歷史截斷，避免提示詞過長
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
    "越南文": "Vietnamese", "繁體中文": "Traditional Chinese",
}

# ========== 3) FastAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時更新 LINE Webhook
    try:
        async with httpx.AsyncClient() as c:
            headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
            payload = {"endpoint": f"{BASE_URL}/callback"}
            r = await c.put("https://api.line.me/v2/bot/channel/webhook/endpoint",
                            headers=headers, json=payload, timeout=10.0)
            r.raise_for_status()
            logger.info(f"✅ Webhook 更新成功: {r.status_code}")
    except Exception as e:
        logger.error(f"Webhook 更新失敗: {e}", exc_info=True)
    yield

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.0.0")
router = APIRouter()

# ========== 4) Helpers ==========
def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom):  return event.source.room_id
    return event.source.user_id

# --- AI & 分析相關函式 ---
def get_analysis_reply(messages: List[Dict[str, str]]):
    """
    統一的 AI 回覆：優先 OpenAI，再回退 Groq 主模型 -> Groq 備用模型
    """
    # OpenAI（可選）
    if openai_client:
        try:
            # 你若無 gpt-4o 權限，改成 gpt-3.5-turbo 或你可用的型號
            resp = openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=messages,
                max_tokens=1800,
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as openai_err:
            logger.warning(f"OpenAI 失敗：{openai_err}")

    # Groq 主模型
    try:
        resp = sync_groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=messages,
            max_tokens=1800,
            temperature=0.8,
        )
        return resp.choices[0].message.content
    except Exception as e1:
        logger.warning(f"Groq 主模型失敗：{e1}")

    # Groq 備用模型
    try:
        resp = sync_groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK,
            messages=messages,
            max_tokens=1500,
            temperature=0.9,
        )
        return resp.choices[0].message.content
    except Exception as e2:
        logger.error(f"Groq 備用模型也失敗：{e2}")
        return "抱歉，AI 分析師目前連線不穩定，請稍後再試。"

async def groq_chat_async(messages: List[Dict[str, str]], max_tokens: int = 600, temperature: float = 0.7):
    """
    輕量任務用（如情感分析/翻譯），避免阻塞主流程
    """
    resp = await async_groq_client.chat.completions.create(
        model=GROQ_MODEL_FALLBACK,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

# --- 金融 & 彩票分析 ---

def get_gold_ai_analysis_report():
    """
    取台銀黃金（即時＋近30天）並產生分析
    """
    logger.info("開始獲取黃金數據並生成 AI 分析報告...")
    current_url = "https://rate.bot.com.tw/gold?Lang=zh-TW"
    history_url = "https://rate.bot.com.tw/gold/chart/year/TWD"
    headers = {'User-Agent': 'Mozilla/5.0'}

    # 即時
    try:
        r = requests.get(current_url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        table = soup.find("table", {"class": "table-striped"})
        rows = table.find("tbody").find_all("tr") if table else []
        sell_price = buy_price = None
        for row in rows:
            cells = row.find_all("td")
            if len(cells) > 1 and "黃金牌價" in cells[0].text:
                buy_price = cells[3].text.strip()
                sell_price = cells[4].text.strip()
                break
        if not sell_price:
            return "抱歉，目前無法從台灣銀行讀到黃金即時牌價。"
    except Exception as e:
        logger.error(f"黃金即時牌價失敗：{e}", exc_info=True)
        return "抱歉，目前無法獲取黃金即時牌價。"

    # 近30天
    try:
        df_list = pd.read_html(history_url)
        df = df_list[0]
        df = df[["日期", "本行賣出價格"]].copy()
        df.columns = ["Date", "Sell_Price"]
        df["Sell_Price"] = pd.to_numeric(df["Sell_Price"], errors="coerce")
        df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d")
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        recent = df[df.index >= (datetime.now() - timedelta(days=30))]
        if not recent.empty:
            max_p = recent["Sell_Price"].max()
            min_p = recent["Sell_Price"].min()
            avg_p = recent["Sell_Price"].mean()
            # 與30天前比較
            if len(df) >= 30:
                base = df["Sell_Price"].iloc[-30]
                now_p = float(sell_price.replace(",", ""))
                chg = now_p - base
                chg_pct = (chg / base * 100) if base else 0.0
                hist_summary = f"近30天高/低/均：{max_p:.2f}/{min_p:.2f}/{avg_p:.2f}；較30天前變動 {chg:.2f}（{chg_pct:.2f}%）。"
            else:
                hist_summary = f"近30天高/低/均：{max_p:.2f}/{min_p:.2f}/{avg_p:.2f}。"
        else:
            hist_summary = "近30天歷史數據不足。"
    except Exception as e:
        logger.error(f"黃金歷史數據失敗：{e}", exc_info=True)
        hist_summary = "無法取得近30天歷史摘要。"

    content = (
        "你是一位專業的黃金市場分析師，請用台灣繁體中文寫 200~300 字精簡報告。\n"
        f"即時數據：賣出價 {sell_price} 元/公克；買入價 {buy_price or 'N/A'} 元/公克。\n"
        f"{hist_summary}\n"
        "要求：1) 先點出最新價；2) 說明目前位階（高/低/盤整）；3) 簡述短期影響因素（美元、通膨、地緣政治、利率）；"
        "4) 給一般投資者一句實用建議；5) 語氣中立、清楚。"
    )
    msgs = [
        {"role": "system", "content": "你擅長從金融數據萃取重點與風險提示。"},
        {"role": "user", "content": content},
    ]
    return get_analysis_reply(msgs)

def get_currency_analysis(target_currency: str):
    """
    以 open.er-api 即時 JPY/TWD 匯率產生快訊
    """
    logger.info(f"開始執行 {target_currency} 匯率分析…")
    try:
        base_currency = "TWD"
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("result") != "success":
            return f"抱歉，獲取匯率資料失敗：{data.get('error-type', '未知錯誤')}"
        rate = data["rates"].get(base_currency)
        if rate is None:
            return f"抱歉，API 中找不到 {base_currency} 匯率。"

        content = (
            f"1 {target_currency.upper()} = {rate:.5f} TWD。\n"
            "請用繁中寫一段 80~120 字快訊：\n"
            "1) 直接報現價；2) 對旅遊/換匯族是相對划算或昂貴；3) 給一句實用提醒（如手續費、分批換等）。"
        )
        msgs = [
            {"role": "system", "content": "你是外匯市場小編，寫作簡短、清楚、務實。"},
            {"role": "user", "content": content},
        ]
        return get_analysis_reply(msgs)
    except Exception as e:
        logger.error(f"處理 {target_currency} 匯率時錯誤：{e}", exc_info=True)
        return "抱歉，外匯服務暫時無法使用。"

def get_lottery_analysis(lottery_type_input: str):
    """
    依據自訂爬蟲資料 + 財神方位提示，產出分析與建議號
    """
    logger.info(f"開始執行 {lottery_type_input} 彩票分析…")
    if not LOTTERY_ENABLED:
        return "抱歉，彩票分析功能目前設定不完整或模組未載入。"

    t = lottery_type_input.lower()
    if "威力" in t:
        last = lottery_crawler.super_lotto()
    elif "大樂" in t:
        last = lottery_crawler.lotto649()
    elif "539" in t:
        last = lottery_crawler.daily_cash()
    else:
        return f"抱歉，暫不支援 {lottery_type_input} 類型。"

    try:
        info = caiyunfangwei_crawler.get_caiyunfangwei()
        caiyun = (
            f"***財神方位提示***\n"
            f"國歷：{info.get('今天日期', '未知')}\n"
            f"農曆：{info.get('今日農曆', '未知')}\n"
            f"歲次：{info.get('今日歲次', '未知')}\n"
            f"財神方位：{info.get('財神方位', '未知')}\n"
        )
    except Exception as e:
        logger.error(f"財神方位取得失敗：{e}", exc_info=True)
        caiyun = "財神方位資訊暫時無法取得。"

    prompt = (
        f"你是專業彩券分析師，請用繁中撰寫 {lottery_type_input} 最新趨勢：\n"
        f"近期開獎資料：\n{last}\n\n"
        f"{caiyun}\n"
        "請完成：\n"
        "1) 熱門號/冷門號歸納；2) 依彩種規則給 3 組號（#1冷門組合、#2熱門組合、#3隨機），數字小到大；"
        "若有特別號/二區需另外列；3) 結尾附一則 20 字內勵志吉祥話；4) 數字一律不省略。"
    )
    msgs = [
        {"role": "system", "content": "你擅長根據歷史開獎歸納趨勢與風險提示。"},
        {"role": "user", "content": prompt},
    ]
    return get_analysis_reply(msgs)

# --- 股票分析 ---
def get_stock_name_from_yahoo(symbol: str) -> str:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        name = info.get("longName") or info.get("shortName")
        return name or symbol
    except Exception:
        return symbol

def remove_full_width_spaces(s):
    return s.replace("\u3000", " ") if isinstance(s, str) else s

def get_stock_analysis(stock_id_input: str):
    """
    支援：台股 4~6 碼（自動補 .TW）、美股代碼、大盤 ^TWII / ^GSPC、關鍵字「台股大盤 / 美股大盤」
    """
    logger.info(f"開始執行 {stock_id_input} 股票分析…")
    user_upper = stock_id_input.upper()

    # 大盤
    if user_upper in ["台股大盤", "大盤"]:
        symbol = "^TWII"; stock_name = "台灣加權指數"
    elif user_upper in ["美股大盤", "美盤", "美股"]:
        symbol = "^GSPC"; stock_name = "S&P 500 指數"
    # 台股代碼
    elif re.match(r"^\d{4,6}[A-Z]?$", user_upper):
        symbol = f"{user_upper}.TW"
        stock_name = get_stock_name_from_yahoo(symbol)
    # 其他（美股等）
    else:
        symbol = user_upper
        stock_name = get_stock_name_from_yahoo(symbol)

    try:
        # 即時 / 基本資料
        newprice = YahooStock(symbol)
        if not getattr(newprice, "name", None):
            newprice.name = stock_name

        # 若一般個股卻抓不到價格，直接返回提示
        if symbol not in ["^TWII", "^GSPC"] and not getattr(newprice, "currentPrice", None):
            return f"抱歉，無法獲取 {stock_name}（{stock_id_input}）的即時資料，請確認代碼是否正確。"

        price_data = stock_price(symbol)
        news_data = remove_full_width_spaces(str(stock_news(stock_name)))[:1024]

        content = (
            f"請以台灣繁中、Markdown，撰寫 {stock_name}（{symbol}）最新趨勢報告。\n"
            f"即時/快照：{vars(newprice)}\n"
            f"近30天價格：\n{price_data}\n"
        )

        if symbol not in ["^TWII", "^GSPC"]:
            # 基本面 / 配息
            value_data = stock_fundamental(symbol)
            dividend_data = stock_dividend(symbol)
            content += f"每季營收：\n{value_data if value_data is not None else '無資料'}\n"
            content += f"配息資料：\n{dividend_data if dividend_data is not None else '無資料'}\n"

        content += f"近期新聞（截斷 1KB 內）：\n{news_data or '無'}\n\n"
        content += (
            "請包含：\n"
            "- 現價與取得時間、走勢摘要\n"
            "- 基本面 / 技術面 / 消息面 / 籌碼面\n"
            "- 建議買進區間、停利點、建議買入張數（風險聲明）\n"
            "- 市場趨勢、配息觀點、最後附上正確連結\n"
        )
        link = f"https://finance.yahoo.com/quote/{symbol}"
        system = f"你是專業分析師。報告最後附：[股票資訊連結]({link})。"

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        return get_analysis_reply(msgs)
    except Exception as e:
        logger.error(f"股票分析流程失敗：{e}", exc_info=True)
        return f"抱歉，分析 {stock_id_input} 時發生錯誤，請稍後再試。"

# --- UI & 對話 Helpers ---
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
    return await groq_chat_async(
        [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        max_tokens=800, temperature=0.2
    )

def set_user_persona(chat_id: str, key: str):
    if key == "random": key = random.choice(list(PERSONAS.keys()))
    if key not in PERSONAS: key = "sweet"
    user_persona[chat_id] = key
    return key

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    key = user_persona.get(chat_id, "sweet")
    p = PERSONAS[key]
    return (
        f"你是一位「{p['title']}」。風格：{p['style']}。\n"
        f"使用者情緒：{sentiment}；請調整語氣（開心→一起開心；難過/生氣→先共情、安撫再建議；中性→自然聊天）。\n"
        f"回覆使用繁體中文，精煉自然，帶少量表情 {p['emoji']}。"
    )

def build_quick_reply() -> QuickReply:
    return QuickReply(items=[
        QuickReplyButton(action=MessageAction(label="主選單", text="選單")),
        QuickReplyButton(action=MessageAction(label="台股大盤", text="台股大盤")),
        QuickReplyButton(action=MessageAction(label="美股大盤", text="美股大盤")),
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
        header=BoxComponent(
            layout="vertical",
            contents=[TextComponent(text="AI 助理主選單", weight="bold", size="lg")]
        ),
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
    title = "子選單"; buttons: List[ButtonComponent] = []
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
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm"),
    )
    return FlexSendMessage(alt_text=title, contents=bubble)

# ========== 5) LINE Handlers ==========
@handler.add(MessageEvent, message=TextMessage)
def on_message_text(event: MessageEvent):
    """
    注意：這個回調在 Starlette 的 threadpool 中執行，該執行緒內沒有現成的 asyncio loop。
    因此用 asyncio.run(...) 開一個獨立事件圈，避免「no running event loop」。
    """
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
            [
                build_submenu_flex(kind),
                TextSendMessage(text="請選擇一項服務👇", quick_reply=build_quick_reply()),
            ]
        )

async def handle_message_async(event: MessageEvent):
    chat_id = get_chat_id(event)
    msg_raw = event.message.text.strip()
    reply_token = event.reply_token
    is_group = not isinstance(event.source, SourceUser)

    # 取得 Bot 顯示名（用於群組 @）
    try:
        bot_info = await run_in_threadpool(line_bot_api.get_bot_info)
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if not msg_raw:
        return
    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True

    # 群組下關閉自動回覆時，僅處理 @Bot 的訊息
    if is_group and not auto_reply_status.get(chat_id, True) and not msg_raw.startswith(f"@{bot_name}"):
        return

    msg = msg_raw[len(f"@{bot_name}"):].strip() if msg_raw.startswith(f"@{bot_name}") else msg_raw
    if not msg:
        return

    low = msg.lower()

    # --- 指令判斷（依優先順序） ---
    if low in ("menu", "選單", "主選單"):
        return line_bot_api.reply_message(reply_token, [
            build_main_menu_flex(),
            TextSendMessage(text="也可以直接點下方快速鍵唷👇", quick_reply=build_quick_reply())
        ])

    # 彩票
    if msg in ["大樂透", "威力彩", "539"]:
        if not LOTTERY_ENABLED:
            return reply_with_quick_bar(reply_token, "抱歉，彩票分析功能目前設定不完整。")
        try:
            report = await run_in_threadpool(get_lottery_analysis, msg)
            return reply_with_quick_bar(reply_token, report)
        except Exception as e:
            logger.error(f"彩票分析流程失敗：{e}", exc_info=True)
            return reply_with_quick_bar(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")

    # 股票 / 指數
    def is_stock_query(text: str) -> bool:
        t = text.upper()
        if t in ["台股大盤", "大盤", "美股大盤", "美盤", "美股"]:
            return True
        if re.match(r"^\d{4,6}[A-Z]?$", t):  # 台股代碼
            return True
        if re.match(r"^[A-Z]{1,5}$", t) and t not in ["JPY"]:  # 美股代碼（排除 JPY）
            return True
        return False

    if is_stock_query(msg):
        if not STOCK_ENABLED:
            return reply_with_quick_bar(reply_token, "抱歉，股票分析模組目前設定不完整或載入失敗。")
        try:
            report = await run_in_threadpool(get_stock_analysis, msg)
            return reply_with_quick_bar(reply_token, report)
        except Exception as e:
            logger.error(f"股票分析流程失敗：{e}", exc_info=True)
            return reply_with_quick_bar(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")

    # 黃金
    if low in ("金價", "黃金"):
        try:
            report = await run_in_threadpool(get_gold_ai_analysis_report)
            return reply_with_quick_bar(reply_token, report)
        except Exception as e:
            logger.error(f"黃金分析流程失敗：{e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "抱歉，金價分析服務暫時無法使用。")

    # 日圓
    if low == "jpy" or low == "日圓" or low == "日元":
        try:
            report = await run_in_threadpool(get_currency_analysis, "JPY")
            return reply_with_quick_bar(reply_token, report)
        except Exception as e:
            logger.error(f"日圓分析流程失敗：{e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "抱歉，日圓匯率分析服務暫時無法使用。")

    # 自動回覆開關（群組）
    if low in ("開啟自動回答", "關閉自動回答"):
        is_on = (low == "開啟自動回答")
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
    persona_keys = {"甜": "sweet", "鹹": "salty", "萌": "moe", "酷": "cool", "random": "random", "隨機": "random"}
    if low in persona_keys:
        key = set_user_persona(chat_id, persona_keys[low])
        p = PERSONAS[key]
        txt = f"💖 已切換人設：{p['title']}\n\n{p['greetings']}"
        return reply_with_quick_bar(reply_token, txt)

    # --- 一般聊天（最後預設） ---
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": msg}]
        final_reply = await groq_chat_async(messages)
        # 紀錄歷史
        history.extend([{"role": "user", "content": msg}, {"role": "assistant", "content": final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN * 2:]
        return reply_with_quick_bar(reply_token, final_reply)
    except Exception as e:
        logger.error(f"AI 回覆失敗：{e}", exc_info=True)
        return reply_with_quick_bar(reply_token, "抱歉我剛剛走神了 😅 再說一次讓我補上！")

# ========== 6) FastAPI Routes ==========
@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        # 在 threadpool 中執行官方 SDK 的同步 handle（避免阻塞）
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

# ========== 7) Local run ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Local server starting at http://0.0.0.0:{port}")
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)