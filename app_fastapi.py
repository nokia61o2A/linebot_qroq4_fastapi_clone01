# app_fastapi.py
# ========== 1) Imports ==========
import os
import re
import io
import random
import logging
import asyncio
from typing import Dict, List
from contextlib import asynccontextmanager
from datetime import datetime

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

# [最終修正] 1. 修正 AsyncWebhookHandler 的導入路徑
from linebot import AsyncLineBotApi
from linebot.webhook import AsyncWebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, AudioMessage,
    TextSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    QuickReply, QuickReplyButton, MessageAction,
    PostbackAction, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent,
    TextComponent, ButtonComponent, SeparatorComponent
)

# --- AI 相關 ---
from groq import AsyncGroq, Groq
import openai

# --- 自訂模組（有就載入，沒有就關閉功能） ---
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    LOTTERY_ENABLED = True
except Exception:
    logging.warning("無法載入彩票模組，彩票功能將停用。")
    LOTTERY_ENABLED = False

try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
    STOCK_ENABLED = True
except Exception as e:
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

if not CHANNEL_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("缺少必要環境變數：CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET")

# --- API 用戶端初始化 ---
line_bot_api = AsyncLineBotApi(CHANNEL_TOKEN)
handler = AsyncWebhookHandler(CHANNEL_SECRET)

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
sync_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

openai_client = None
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("未設定 OPENAI_API_KEY，語音轉文字與部分分析將優先使用 Groq。")

GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

if LOTTERY_ENABLED:
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()

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
LANGUAGE_MAP = { "英文": "English", "日文": "Japanese", "韓文": "Korean", "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"}

# ========== 3) FastAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    if BASE_URL:
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
    else:
        logger.warning("未設定 BASE_URL，跳過 Webhook 更新。")
    yield

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.1.0")
router = APIRouter()

# ========== 4) Helpers ==========
def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom):  return event.source.room_id
    return event.source.user_id

# --- AI & 分析相關函式 ---
def get_analysis_reply(messages):
    try:
        if openai_client:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, max_tokens=1500, temperature=0.7
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
    if not async_groq_client:
        return await run_in_threadpool(
            lambda: get_analysis_reply(messages)
        )
    resp = await async_groq_client.chat.completions.create(
        model=GROQ_MODEL_FALLBACK, messages=messages,
        max_tokens=max_tokens, temperature=temperature
    )
    return resp.choices[0].message.content.strip()

# ---------- 金價抓取（對應台銀新頁面文字） ----------
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

def format_gold_report(data: dict) -> str:
    ts = data.get("listed_at") or "（頁面未標示）"
    sell, buy = data["sell_twd_per_g"], data["buy_twd_per_g"]
    spread = sell - buy
    bias = "盤整" if spread <= 30 else ("偏寬" if spread <= 60 else "價差偏大")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"**金價快報（台灣銀行）**\n"
        f"- 資料時間：{ts}\n"
        f"- 本行賣出（1克）：**${sell:,.0f}** 元\n"
        f"- 本行買進（1克）：**${buy:,.0f}** 元\n"
        f"- 買賣價差：${spread:,.0f} 元（{bias}）\n"
        f"\n資料來源：{BOT_GOLD_URL}\n（更新於 {now}）"
    )

def get_gold_analysis():
    logger.info("開始執行黃金價格分析...")
    try:
        data = get_bot_gold_quote()
        return format_gold_report(data)
    except Exception as e:
        logger.error(f"金價流程失敗：{e}", exc_info=True)
        return "抱歉，目前無法從台灣銀行取得黃金牌價。稍後再試一次 🙏"

def get_currency_analysis(target_currency: str):
    logger.info(f"開始執行 {target_currency} 匯率分析...")
    try:
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("result") == "success":
            rate = data["rates"].get('TWD')
            if rate is None: return f"抱歉，API中找不到 TWD 的匯率資訊。"
            return f"最新：1 {target_currency.upper()} ≈ ${rate:.5f} 新台幣"
        else:
            return f"抱歉，獲取匯率資料失敗：{data.get('error-type', '未知錯誤')}"
    except Exception as e:
        logger.error(f"處理 {target_currency} API 資料時發生錯誤: {e}", exc_info=True)
        return f"抱歉，處理外匯資料時發生內部錯誤，請稍後再試。"

def get_lottery_analysis(lottery_type_input: str):
    logger.info(f"開始執行 {lottery_type_input} 彩票分析...")
    lottery_type = lottery_type_input.lower()
    if "威力" in lottery_type: last_lotto = lottery_crawler.super_lotto()
    elif "大樂" in lottery_type: last_lotto = lottery_crawler.lotto649()
    elif "539" in lottery_type: last_lotto = lottery_crawler.daily_cash()
    else: return f"抱歉，暫不支援 {lottery_type_input} 類型的分析。"
    try:
        caiyunfangwei_info = caiyunfangwei_crawler.get_caiyunfangwei()
        content_msg = (
            f'你現在是一位專業的樂透彩分析師, 使用{lottery_type_input}的資料來撰寫分析報告:\n'
            f'近幾期號碼資訊:\n{last_lotto}\n'
            f'顯示今天國歷/農歷日期：{caiyunfangwei_info.get("今天日期", "未知")}\n'
            f'今日歲次：{caiyunfangwei_info.get("今日歲次", "未知")}\n'
            f'財神方位：{caiyunfangwei_info.get("財神方位", "未知")}\n'
            '請寫詳細趨勢並給三組號（符合彩種格式）。使用繁體中文。'
        )
    except Exception:
        content_msg = (
            f'你現在是一位專業的樂透彩分析師, 使用{lottery_type_input}的資料來撰寫分析報告:\n'
            f'近幾期號碼資訊:\n{last_lotto}\n'
            '財神方位暫缺；仍請給趨勢與三組號。使用繁體中文。'
        )
    msg = [{"role": "system", "content": "你是資深彩券分析師。"}, {"role": "user", "content": content_msg}]
    return get_analysis_reply(msg)

stock_data_df = None
def load_stock_data():
    global stock_data_df
    if stock_data_df is None:
        try:
            stock_data_df = pd.read_csv('name_df.csv', dtype={'股號': str})
        except FileNotFoundError:
            logger.error("`name_df.csv` not found. Stock name lookup will be disabled.")
            stock_data_df = pd.DataFrame(columns=['股號', '股名'])
    return stock_data_df

def get_stock_name(stock_id):
    df = load_stock_data()
    result = df[df['股號'] == stock_id]
    return result.iloc[0]['股名'] if not result.empty else None

def remove_full_width_spaces(data):
    return data.replace('\u3000', ' ') if isinstance(data, str) else data

def get_stock_analysis(stock_id_input: str):
    logger.info(f"開始執行 {stock_id_input} 股票分析...")
    user_input_upper = stock_id_input.upper()
    if user_input_upper in ["台股大盤", "大盤"]: stock_id, stock_name = "^TWII", "台灣加權指數"
    elif user_input_upper in ["美股大盤", "美盤", "美股"]: stock_id, stock_name = "^GSPC", "S&P 500 指數"
    elif re.match(r'^\d{4,6}[A-Z]?$', user_input_upper):
        stock_id = f"{user_input_upper}.TW"
        stock_name = get_stock_name(stock_id_input) or stock_id_input
    else: stock_id, stock_name = user_input_upper, user_input_upper
    try:
        newprice_stock = YahooStock(stock_id)
        price_data = stock_price(stock_id)
        try: news_raw = str(stock_news(stock_name))
        except Exception: news_raw = "（新聞來源暫時無法取得）"
        news_data = remove_full_width_spaces(news_raw)[:1024]
        content_msg = (f'你現在是一位專業的證券分析師, 你會依據以下資料來進行分析並給出一份完整的分析報告:\n'
                       f'**股票代碼:** {stock_id}, **股票名稱:** {newprice_stock.name}\n'
                       f'**即時報價:** {vars(newprice_stock)}\n'
                       f'**近期價格資訊:**\n {price_data}\n')
        if stock_id not in ["^TWII", "^GSPC"]:
            try: stock_value_data = stock_fundamental(stock_id)
            except Exception: stock_value_data = None
            try: stock_vividend_data = stock_dividend(stock_id)
            except Exception: stock_vividend_data = None
            content_msg += f'**每季營收資訊：**\n {stock_value_data if stock_value_data is not None else "無法取得"}\n'
            content_msg += f'**配息資料：**\n {stock_vividend_data if stock_vividend_data is not None else "無法取得"}\n'
        content_msg += f'**近期新聞資訊:** \n {news_data}\n'
        content_msg += f'請給我 {stock_name} 近期的趨勢報告。請以詳細、嚴謹及專業的角度撰寫此報告，使用繁體中文。'
        stock_link = f"https://finance.yahoo.com/quote/{stock_id}"
        system_prompt = (
            "你現在是一位專業的證券分析師。請基於近期的股價走勢、基本面分析、新聞資訊等進行綜合分析。\n"
            "請至少包含：現價/漲跌、走勢、基本面、技術面、消息面、籌碼面、建議區間/停利、張數建議、趨勢、配息、綜合結論。\n"
            f"最後提供連結：[股票資訊連結]({stock_link})。\n"
            "回應請使用繁體中文並格式化為 Markdown。"
        )
        msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content_msg}]
        return get_analysis_reply(msg)
    except Exception as e:
        logger.error(f"股票分析流程失敗: {e}", exc_info=True)
        return f"抱歉，分析 {stock_id_input} 時發生錯誤，請確認股票代碼是否正確。"

# --- UI & 對話 Helpers ---
async def analyze_sentiment(text: str) -> str:
    msgs = [{"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
            {"role":"user","content":text}]
    out = await groq_chat_async(msgs, max_tokens=10, temperature=0)
    return (out or "neutral").strip().lower()

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

async def reply_with_quick_bar(reply_token: str, text: str):
    await line_bot_api.reply_message(
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
    title, buttons = "子選單", []
    if kind == "finance":
        title, buttons = "💹 金融查詢", [
            ButtonComponent(action=MessageAction(label="台股大盤", text="台股大盤")), ButtonComponent(action=MessageAction(label="美股大盤", text="美股大盤")),
            ButtonComponent(action=MessageAction(label="黃金價格", text="金價")), ButtonComponent(action=MessageAction(label="日圓匯率", text="JPY")),
            ButtonComponent(action=MessageAction(label="查 2330 台積電", text="2330")), ButtonComponent(action=MessageAction(label="查 NVDA 輝達", text="NVDA")),
        ]
    elif kind == "lottery":
        title, buttons = "🎰 彩票分析", [
            ButtonComponent(action=MessageAction(label="大樂透", text="大樂透")), ButtonComponent(action=MessageAction(label="威力彩", text="威力彩")),
            ButtonComponent(action=MessageAction(label="今彩539", text="539")),
        ]
    elif kind == "persona":
        title, buttons = "💖 AI 角色扮演", [
            ButtonComponent(action=MessageAction(label="甜美女友", text="甜")), ButtonComponent(action=MessageAction(label="傲嬌女友", text="鹹")),
            ButtonComponent(action=MessageAction(label="萌系女友", text="萌")), ButtonComponent(action=MessageAction(label="酷系御姐", text="酷")),
            ButtonComponent(action=MessageAction(label="隨機切換", text="random")),
        ]
    elif kind == "translate":
        title, buttons = "🌐 翻譯工具", [
            ButtonComponent(action=MessageAction(label="翻成英文", text="翻譯->英文")), ButtonComponent(action=MessageAction(label="翻成日文", text="翻譯->日文")),
            ButtonComponent(action=MessageAction(label="翻成繁中", text="翻譯->繁體中文")), ButtonComponent(action=MessageAction(label="結束翻譯模式", text="翻譯->結束")),
        ]
    elif kind == "settings":
        title, buttons = "⚙️ 系統設定", [
            ButtonComponent(action=MessageAction(label="開啟自動回答 (群組)", text="開啟自動回答")),
            ButtonComponent(action=MessageAction(label="關閉自動回答 (群組)", text="關閉自動回答")),
        ]
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[TextComponent(text=title, weight="bold", size="lg")]),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm")
    )
    return FlexSendMessage(alt_text=title, contents=bubble)

# ========== 5) 語音（錄音）→ 轉文字 → 回覆 ==========
async def _save_line_content_to_bytes_async(message_id: str) -> bytes:
    """非同步下載 LINE 音訊內容為 bytes。"""
    message_content = await line_bot_api.get_message_content(message_id)
    return await message_content.read()

def _transcribe_with_openai(audio_bytes: bytes, filename: str = "audio.m4a") -> str | None:
    if not openai_client: return None
    try:
        f = io.BytesIO(audio_bytes)
        f.name = filename
        resp = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
        return resp.text.strip() if resp.text else None
    except Exception as e:
        logger.warning(f"OpenAI 轉錄失敗：{e}")
        return None

def _transcribe_with_groq(audio_bytes: bytes, filename: str = "audio.m4a") -> str | None:
    if not sync_groq_client: return None
    try:
        f = io.BytesIO(audio_bytes)
        f.name = filename
        resp = sync_groq_client.audio.transcriptions.create(file=f, model="whisper-large-v3")
        return resp.text.strip() if resp.text else None
    except Exception as e:
        logger.warning(f"Groq 轉錄失敗：{e}")
        return None

# ========== 6) LINE Handlers ==========
@handler.add(MessageEvent, message=TextMessage)
async def on_message_text(event: MessageEvent):
    try:
        await handle_message_async(event)
    except Exception as e:
        logger.error(f"Handle message failed: {e}", exc_info=True)

@handler.add(MessageEvent, message=AudioMessage)
async def on_message_audio(event: MessageEvent):
    try:
        await handle_audio_async(event)
    except Exception as e:
        logger.error(f"Handle audio failed: {e}", exc_info=True)

@handler.add(PostbackEvent)
async def on_postback(event: PostbackEvent):
    data = (event.postback.data or "").strip()
    if data.startswith("menu:"):
        kind = data.split(":", 1)[-1]
        await line_bot_api.reply_message(
            event.reply_token,
            [build_submenu_flex(kind), TextSendMessage(text="請選擇一項服務", quick_reply=build_quick_reply())]
        )

async def handle_audio_async(event: MessageEvent):
    chat_id = get_chat_id(event)
    reply_token = event.reply_token
    try:
        audio_bytes = await _save_line_content_to_bytes_async(event.message.id)
        text = await run_in_threadpool(_transcribe_with_openai, audio_bytes)
        if not text:
            text = await run_in_threadpool(_transcribe_with_groq, audio_bytes)
        if not text:
            raise RuntimeError("語音轉文字失敗 (OpenAI 和 Groq 皆失敗)")
    except Exception as e:
        logger.error(f"語音轉文字失敗：{e}", exc_info=True)
        await reply_with_quick_bar(reply_token, "抱歉我剛剛沒聽清楚 🙈 能再說一次或改用文字嗎？")
        return

    try:
        sentiment = await analyze_sentiment(text)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role":"system","content":sys_prompt},
                    {"role":"user","content":f"(以下是使用者語音轉文字)\n{text}"}]
        final_reply = await groq_chat_async(messages)
        await reply_with_quick_bar(reply_token, f"🎧 我聽到了：\n{text}\n\n—\n{final_reply}")
    except Exception as e:
        logger.error(f"語音回覆失敗：{e}", exc_info=True)
        await reply_with_quick_bar(reply_token, "我在～只是有點恍神😅 你再說一次，我會好好聽。")

async def handle_message_async(event: MessageEvent):
    chat_id, msg_raw = get_chat_id(event), event.message.text.strip()
    reply_token, is_group = event.reply_token, not isinstance(event.source, SourceUser)

    try:
        bot_info = await line_bot_api.get_bot_info()
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if not msg_raw: return
    if chat_id not in auto_reply_status: auto_reply_status[chat_id] = True

    if is_group and not auto_reply_status.get(chat_id, True) and not msg_raw.startswith(f"@{bot_name}"):
        return
    msg = msg_raw[len(f"@{bot_name}"):].strip() if msg_raw.startswith(f"@{bot_name}") else msg_raw
    if not msg: return

    low = msg.lower()
    
    def is_stock_query(text: str) -> bool:
        text_upper = text.upper()
        if text_upper in ["台股大盤", "大盤", "美股大盤", "美盤", "美股"]: return True
        if re.match(r'^\d{4,6}[A-Z]?$', text_upper): return True
        if re.match(r'^[A-Z]{1,5}$', text_upper) and text_upper not in ["JPY"]: return True
        return False

    if low in ("menu", "選單", "主選單"):
        await line_bot_api.reply_message(reply_token, build_main_menu_flex())
        return

    LOTTERY_KEYWORDS = ["大樂透", "威力彩", "539"]
    if msg in LOTTERY_KEYWORDS:
        if not LOTTERY_ENABLED:
            await reply_with_quick_bar(reply_token, "抱歉，彩票分析功能目前設定不完整。")
            return
        try:
            analysis_report = await run_in_threadpool(get_lottery_analysis, msg)
            await reply_with_quick_bar(reply_token, analysis_report)
        except Exception as e:
            logger.error(f"彩票分析流程失敗: {e}", exc_info=True)
            await reply_with_quick_bar(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")
        return

    if is_stock_query(msg):
        if not STOCK_ENABLED:
            await reply_with_quick_bar(reply_token, "抱歉，股票分析模組目前設定不完整或載入失敗。")
            return
        try:
            analysis_report = await run_in_threadpool(get_stock_analysis, msg)
            await reply_with_quick_bar(reply_token, analysis_report)
        except Exception as e:
            logger.error(f"股票分析流程失敗: {e}", exc_info=True)
            await reply_with_quick_bar(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")
        return

    if low in ("金價", "黃金"):
        try:
            analysis_report = await run_in_threadpool(get_gold_analysis)
            await reply_with_quick_bar(reply_token, analysis_report)
        except Exception as e:
            logger.error(f"黃金分析流程失敗: {e}", exc_info=True)
            await reply_with_quick_bar(reply_token, "抱歉，金價分析服務暫時無法使用。")
        return

    if low == "jpy":
        try:
            analysis_report = await run_in_threadpool(get_currency_analysis, "JPY")
            await reply_with_quick_bar(reply_token, analysis_report)
        except Exception as e:
            logger.error(f"日圓分析流程失敗: {e}", exc_info=True)
            await reply_with_quick_bar(reply_token, "抱歉，日圓匯率分析服務暫時無法使用。")
        return

    if low in ("開啟自動回答", "關閉自動回答"):
        is_on = low == "開啟自動回答"
        auto_reply_status[chat_id] = is_on
        text = "✅ 已開啟自動回答" if is_on else "❌ 已關閉自動回答（群組需 @我 才回）"
        await reply_with_quick_bar(reply_token, text)
        return

    if low.startswith("翻譯->"):
        lang = msg.split("->", 1)[1].strip()
        if lang == "結束":
            translation_states.pop(chat_id, None)
            await reply_with_quick_bar(reply_token, "✅ 已結束翻譯模式")
        else:
            translation_states[chat_id] = lang
            await reply_with_quick_bar(reply_token, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")
        return

    persona_keys = {"甜":"sweet", "鹹":"salty", "萌":"moe", "酷":"cool", "random":"random"}
    if low in persona_keys:
        key = set_user_persona(chat_id, persona_keys[low])
        p = PERSONAS[key]
        txt = f"💖 已切換人設：{p['title']}\n\n{p['greetings']}"
        await reply_with_quick_bar(reply_token, txt)
        return

    if chat_id in translation_states:
        try:
            out = await translate_text(msg, translation_states[chat_id])
            await reply_with_quick_bar(reply_token, f"🌐 ({translation_states[chat_id]})\n{out}")
        except Exception as e:
            logger.error(f"翻譯失敗: {e}", exc_info=True)
            await reply_with_quick_bar(reply_token, "翻譯暫時失效，等我回神再來一次 🙏")
        return

    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]
        final_reply = await groq_chat_async(messages)
        history.extend([{"role":"user","content":msg}, {"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
        await reply_with_quick_bar(reply_token, final_reply)
    except Exception as e:
        logger.error(f"AI 回覆失敗: {e}", exc_info=True)
        await reply_with_quick_bar(reply_token, "抱歉我剛剛走神了 😅 再說一次讓我補上！")

# ========== 7) FastAPI Routes ==========
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
    return PlainTextResponse("LINE Bot is running.", status_code=200)

@router.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

app.include_router(router)

# ========== 8) Local run ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)