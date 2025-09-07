# app_fastapi.py
# ========== 1) Imports ==========
import os
import re
import random
import logging
import asyncio
from typing import Dict, List
from contextlib import asynccontextmanager

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

# --- 自訂功能 ---
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    LOTTERY_ENABLED = True
except Exception:
    LOTTERY_ENABLED = False

try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
    STOCK_ENABLED = True
except Exception as e:
    print(f"[WARN] 股票模組載入失敗：{e}")
    STOCK_ENABLED = False

# ========== 2) Setup ==========
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# --- 環境變數 ---
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise RuntimeError("缺少必要環境變數：BASE_URL / CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET / GROQ_API_KEY")

# --- API 用戶端初始化 ---
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

async_groq = AsyncGroq(api_key=GROQ_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Groq 模型（避免使用已下架名稱）
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-70b-specdec")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

if LOTTERY_ENABLED:
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()

# --- 狀態與常數 ---
conversation_history: Dict[str, List[dict]] = {}
translation_states: Dict[str, str] = {}  # chat_id -> 目標語言顯示文字（如「英文」）
auto_reply_status: Dict[str, bool] = {}

LANGUAGE_MAP = {
    "英文": "English", "日文": "Japanese", "韓文": "Korean",
    "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"
}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji":"🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji":"😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji":"✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議", "greetings": "我在。說重點。", "emoji":"🧊⚡️"}
}
MAX_HISTORY_LEN = 10

# ========== 3) FastAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with httpx.AsyncClient() as c:
            headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
            payload = {"endpoint": f"{BASE_URL}/callback"}
            r = await c.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=payload, timeout=10.0)
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
    line_bot_api.reply_message(reply_token, TextSendMessage(text=text, quick_reply=build_quick_reply()))

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

# --- AI helpers ---
async def groq_chat_async(messages, model=None, max_tokens=800, temperature=0.7):
    model = model or GROQ_MODEL_FALLBACK
    try:
        resp = await async_groq.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq async 主要模型失敗：{e}")
        # fallback
        resp = await async_groq.chat.completions.create(
            model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return resp.choices[0].message.content.strip()

def get_analysis_reply(messages):
    # 一律走 Groq（你目前 OpenAI 金鑰報 401）
    try:
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY, messages=messages, max_tokens=2000, temperature=0.7
        )
        return r.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq 主要模型失敗：{e}")
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=1500, temperature=0.9
        )
        return r.choices[0].message.content

# --- 翻譯 ---
async def translate_text(text: str, target_lang_display: str) -> str:
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text."
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{text}"}}'
    return await groq_chat_async([{"role":"system","content":sys},{"role":"user","content":usr}], max_tokens=800, temperature=0.2)

# --- 股票工具 ---
_stock_name_cache_df = None
def load_stock_data():
    global _stock_name_cache_df
    if _stock_name_cache_df is None:
        try:
            _stock_name_cache_df = pd.read_csv("name_df.csv")
        except Exception:
            _stock_name_cache_df = pd.DataFrame(columns=["股號", "股名"])
    return _stock_name_cache_df

def get_stock_name(stock_id):
    df = load_stock_data()
    r = df[df["股號"] == stock_id]
    return r.iloc[0]["股名"] if not r.empty else None

def remove_full_width_spaces(s: str) -> str:
    return s.replace("\u3000", " ") if isinstance(s, str) else s

def normalize_stock_input(text: str):
    """
    回傳 (norm_code, display_name)
    - 台股數字/數字+字母 → 加 .TW（2330 → 2330.TW、00937B → 00937B.TW）
    - 大盤關鍵詞 → ^TWII / ^GSPC
    - 其他 → 原樣大寫
    """
    t = text.strip()
    up = t.upper()
    if up in ("台股大盤", "大盤"): return "^TWII", "台灣加權指數"
    if up in ("美股大盤", "美盤", "美股"): return "^GSPC", "S&P 500 指數"
    if re.fullmatch(r"\d{4,6}[A-Z]?", up):  # 台股
        name = get_stock_name(t) or t
        return f"{up}.TW", name
    return up, up  # 美股/指數

def fetch_yahoo_html_price(symbol: str) -> dict | None:
    """同 YahooStock 的 HTML 備援，供主流程直接用。"""
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        name = soup.select_one("h1.D\\(ib\\)")
        price = soup.select_one('fin-streamer[data-field="regularMarketPrice"]')
        chg_pct = soup.select_one('fin-streamer[data-field="regularMarketChangePercent"]')
        when = soup.find("div", string=lambda t: t and isinstance(t, str) and ("GMT" in t or "台北" in t))
        if price and price.text:
            return {
                "name": name.get_text(strip=True) if name else symbol,
                "now_price": price.text,
                "change": chg_pct.text if chg_pct else None,
                "time": when.text.strip() if when else None
            }
    except Exception:
        pass
    return None

def fetch_twse_stock(stock_no: str) -> dict | None:
    """TWSE 近一年日成交資訊，取最近一筆收盤。"""
    try:
        url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=&stockNo={stock_no}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        if j.get("stat") != "OK": return None
        rows = j.get("data") or []
        if not rows: return None
        d = rows[-1]
        price = d[6].replace(",", "")
        chg = d[7]
        return {"name": j.get("title", stock_no), "now_price": price, "change": chg, "time": d[0]}
    except Exception:
        return None

def get_stock_analysis(user_input: str):
    logger.info(f"開始執行 {user_input} 股票分析…")
    stock_id, stock_name = normalize_stock_input(user_input)

    # 實時快照（YahooStock 內建 API→HTML→TWSE 備援）
    newprice = YahooStock(stock_id)
    snap = {
        "name": newprice.name or stock_name,
        "now_price": newprice.now_price,
        "change": newprice.change,
        "time": newprice.close_time
    }

    # 歷史價格
    price_data = None
    try:
        price_data = stock_price(stock_id)
    except Exception as e:
        logger.warning(f"stock_price 失敗：{e}")
        price_data = "（歷史價格抓取失敗）"

    # 新聞
    try:
        news_data = remove_full_width_spaces(str(stock_news(stock_name)))[:1024]
    except Exception as e:
        logger.warning(f"stock_news 失敗：{e}")
        news_data = "（新聞抓取失敗）"

    # 基本面/配息（大盤略過）
    fundamental = dividend = None
    if stock_id not in ("^TWII", "^GSPC"):
        try:
            fundamental = stock_fundamental(stock_id)
        except Exception as e:
            logger.warning(f"fundamental 失敗：{e}")
        try:
            dividend = stock_dividend(stock_id)
        except Exception as e:
            logger.warning(f"dividend 失敗：{e}")

    # 如果即時價仍是 None，再做一次 HTML/TWSE 額外補救（多一層）
    if snap["now_price"] is None:
        alt = fetch_yahoo_html_price(stock_id)
        if alt and alt.get("now_price"):
            snap["name"] = alt["name"]
            snap["now_price"] = alt["now_price"]
            snap["change"] = alt.get("change")
            snap["time"] = alt.get("time")
        elif stock_id.endswith(".TW"):
            twse = fetch_twse_stock(stock_id.replace(".TW", ""))
            if twse:
                snap["name"] = twse["name"]
                snap["now_price"] = twse["now_price"]
                snap["change"] = twse.get("change")
                snap["time"] = twse.get("time")

    stock_link = f"https://finance.yahoo.com/quote/{stock_id}"
    content_msg = (
        f"你現在是一位專業的證券分析師，請依據以下資料撰寫一份完整的報告：\n"
        f"**股票代碼:** {stock_id}，**股票名稱:** {snap['name']}\n"
        f"**即時報價:** 現價={snap['now_price']}, 變動={snap['change']}, 時間={snap['time']}\n"
        f"**近期價格資訊：**\n{price_data}\n"
    )
    if stock_id not in ("^TWII", "^GSPC"):
        content_msg += f"**每季營收資訊：**\n{fundamental if fundamental is not None else '無法取得'}\n"
        content_msg += f"**配息資料：**\n{dividend if dividend is not None else '無法取得'}\n"

    content_msg += f"**近期新聞資訊：**\n{news_data}\n"
    system_prompt = (
        "你是專業的證券分析師。請綜合基本面、技術面、消息面、籌碼面，"
        "以繁體中文、Markdown 格式輸出，並包含：股名(股號)、現價與取得時間、"
        "股價走勢、基本/技術/消息/籌碼面、建議買進區間、停利點%、建議買入張數、"
        "市場趨勢、配息分析與綜合結論；最後附上正確連結："
        f"[股票資訊連結]({stock_link})。"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content_msg}]
    return get_analysis_reply(messages)

# ========== 5) LINE Handlers ==========
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

async def handle_message_async(event: MessageEvent):
    chat_id = (event.source.group_id if isinstance(event.source, SourceGroup)
               else event.source.room_id if isinstance(event.source, SourceRoom)
               else event.source.user_id)
    msg_raw = event.message.text.strip()
    reply_token = event.reply_token
    is_group = not isinstance(event.source, SourceUser)

    if not msg_raw:
        return

    # 群組自動回覆開關
    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True
    if is_group and not auto_reply_status.get(chat_id, True):
        # 只回應 @BotName
        try:
            bot_name = (await run_in_threadpool(line_bot_api.get_bot_info)).display_name
        except Exception:
            bot_name = "AI 助手"
        if not msg_raw.startswith(f"@{bot_name}"):
            return
        msg = msg_raw[len(f"@{bot_name}"):].strip()
    else:
        msg = msg_raw

    low = msg.lower()

    # --- 選單 ---
    if low in ("menu", "選單", "主選單"):
        return line_bot_api.reply_message(reply_token, build_main_menu_flex())

    # --- 彩票 ---
    if msg in ("大樂透", "威力彩", "539"):
        if not LOTTERY_ENABLED:
            return reply_with_quick_bar(reply_token, "抱歉，彩票分析功能未啟用。")
        try:
            # 省略：你的彩票分析函式，可復用原本版本
            return reply_with_quick_bar(reply_token, "彩票分析：此處可接入你的 crawler 與 LLM。")
        except Exception as e:
            logger.error(f"彩票分析流程失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")

    # --- 金價 / 匯率 ---
    if low in ("金價", "黃金"):
        try:
            url = "https://rate.bot.com.tw/gold?Lang=zh-TW"
            r = await run_in_threadpool(requests.get, url)
            soup = BeautifulSoup(r.text, "html.parser")
            price = soup.select_one("table.table-striped tbody tr td:nth-of-type(5)")
            text = f"台銀黃金牌價（1g 賣出）：{price.text.strip()} 元" if price else "暫無法取得"
            return reply_with_quick_bar(reply_token, text)
        except Exception:
            return reply_with_quick_bar(reply_token, "抱歉，金價服務暫時無法使用。")

    if low == "jpy":
        try:
            r = await run_in_threadpool(requests.get, "https://open.er-api.com/v6/latest/JPY",)
            rate = r.json().get("rates", {}).get("TWD")
            if not rate:
                return reply_with_quick_bar(reply_token, "取不到匯率")
            txt = f"最新：1 日圓 ≈ {rate:.5f} 新台幣"
            return reply_with_quick_bar(reply_token, txt)
        except Exception:
            return reply_with_quick_bar(reply_token, "抱歉，匯率服務暫時無法使用。")

    # --- 自動回覆開關 ---
    if low in ("開啟自動回答", "關閉自動回答"):
        auto_reply_status[chat_id] = low == "開啟自動回答"
        return reply_with_quick_bar(reply_token, "✅ 已開啟自動回答" if auto_reply_status[chat_id] else "❌ 已關閉自動回答（群組需 @我 才回）")

    # --- 翻譯模式切換 ---
    if low.startswith("翻譯->"):
        lang = msg.split("->", 1)[1].strip()
        if lang == "結束":
            translation_states.pop(chat_id, None)
            return reply_with_quick_bar(reply_token, "✅ 已結束翻譯模式")
        translation_states[chat_id] = lang
        return reply_with_quick_bar(reply_token, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")

    # --- 股票查詢（支援 00937B / 2881A） ---
    def is_stock_query(text: str) -> bool:
        up = text.upper()
        if up in ("台股大盤", "大盤", "美股大盤", "美盤", "美股"): return True
        if re.fullmatch(r"\d{4,6}[A-Z]?", up): return True  # 台股（含字尾）
        if re.fullmatch(r"^[A-Z\.^]{1,12}$", up) and up not in ("JPY",): return True
        return False

    if is_stock_query(msg):
        if not STOCK_ENABLED:
            return reply_with_quick_bar(reply_token, "抱歉，股票分析模組未啟用。")
        try:
            analysis = await run_in_threadpool(get_stock_analysis, msg)
            return reply_with_quick_bar(reply_token, analysis)
        except Exception as e:
            logger.error(f"股票分析流程失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")

    # --- 翻譯模式優先 ---
    if chat_id in translation_states:
        try:
            out = await translate_text(msg, translation_states[chat_id])
            return reply_with_quick_bar(reply_token, f"{out}")
        except Exception as e:
            logger.error(f"翻譯失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "翻譯暫時失效，等我回神再來一次 🙏")

    # --- 一般對話 ---
    try:
        sys = "你是一位自然、精煉、友善的繁體中文聊天助手。"
        messages = [{"role": "system", "content": sys}, {"role": "user", "content": msg}]
        out = await groq_chat_async(messages)
        return reply_with_quick_bar(reply_token, out)
    except Exception as e:
        logger.error(f"AI 回覆失敗: {e}", exc_info=True)
        return reply_with_quick_bar(reply_token, "抱歉我剛剛走神了 😅 再說一次讓我補上！")

# ========== 6) FastAPI Routes ==========
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

# ========== 7) Local run ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)