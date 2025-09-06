# app_fastapi.py
# ========== 1) Imports ==========
import os
import re
import random
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import json
import time
import requests
import httpx
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

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

from groq import AsyncGroq, Groq
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # 沒安裝也能跑

# （可選）彩票模組；沒有就自動關閉此功能
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    LOTTERY_ENABLED = True
except Exception:
    LOTTERY_ENABLED = False

# ========== 2) Setup ==========
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise RuntimeError("缺少必要環境變數：BASE_URL / CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET / GROQ_API_KEY")

line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# Groq 最新可用模型（Primary → Backup1 → Backup2）
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.3-70b-versatile")
GROQ_MODEL_BACKUP1 = os.getenv("GROQ_MODEL_BACKUP1", "llama-3.3-8b-instant")
GROQ_MODEL_BACKUP2 = os.getenv("GROQ_MODEL_BACKUP2", "deepseek-r1-distill-llama-70b")

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
sync_groq_client = Groq(api_key=GROQ_API_KEY)

openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None
if not openai_client:
    logger.warning("未設定/未啟用 OPENAI_API_KEY，AI 生成將使用 Groq。")

if LOTTERY_ENABLED:
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()

# 對話狀態
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}
auto_reply_status: Dict[str, bool] = {}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji":"🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji":"😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji":"✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議", "greetings": "我在。說重點。", "emoji":"🧊⚡️"},
}
LANGUAGE_MAP = {
    "英文":"English","日文":"Japanese","韓文":"Korean","越南文":"Vietnamese","繁體中文":"Traditional Chinese"
}

# ========== 3) FastAPI lifespan ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
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

def make_quick_reply() -> QuickReply:
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
    line_bot_api.reply_message(reply_token, TextSendMessage(text=text, quick_reply=make_quick_reply()))

def build_main_menu_flex() -> FlexSendMessage:
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[TextComponent(text="AI 助理主選單", weight="bold", size="lg")]),
        body=BoxComponent(layout="vertical", spacing="md", contents=[
            TextComponent(text="請選擇功能分類：", size="sm"),
            SeparatorComponent(margin="md"),
            ButtonComponent(action=PostbackAction(label="💹 金融查詢", data="menu:finance"), style="primary", color="#5E86C1"),
            ButtonComponent(action=PostbackAction(label="🎰 彩票分析", data="menu:lottery"), style="primary", color="#5EC186"),
            ButtonComponent(action=PostbackAction(label="💖 AI 角色扮演", data="menu:persona"), style="secondary"),
            ButtonComponent(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate"), style="secondary"),
            ButtonComponent(action=PostbackAction(label="⚙️ 系統設定", data="menu:settings"), style="secondary"),
        ])
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
            ButtonComponent(action=MessageAction(label="開啟自動回答(群組)", text="開啟自動回答")),
            ButtonComponent(action=MessageAction(label="關閉自動回答(群組)", text="關閉自動回答")),
        ]
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[TextComponent(text=title, weight="bold", size="lg")]),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm")
    )
    return FlexSendMessage(alt_text=title, contents=bubble)

# ========== 5) AI helpers ==========
def ai_complete(messages: List[Dict[str, str]], max_tokens: int = 1800, temperature: float = 0.7) -> str:
    # 先 OpenAI（若可），失敗則 Groq 主→備1→備2
    if openai_client:
        try:
            r = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return r.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI 失敗：{e}")

    for model in (GROQ_MODEL_PRIMARY, GROQ_MODEL_BACKUP1, GROQ_MODEL_BACKUP2):
        try:
            r = sync_groq_client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature
            )
            return r.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq 模型失敗（{model}）：{e}")
    return "抱歉，AI 產生內容暫時不可用。"

async def groq_small_async(messages: List[Dict[str, str]], max_tokens=300, temperature=0.2) -> str:
    # 輕量任務（情緒、翻譯等）
    for model in (GROQ_MODEL_BACKUP1, GROQ_MODEL_PRIMARY):
        try:
            r = await async_groq_client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Groq async 失敗（{model}）：{e}")
    return "neutral"

# ========== 6) 金價 / 匯率 ==========
def get_gold_ai_analysis_report() -> str:
    logger.info("開始：黃金報告")
    current_price_url = "https://rate.bot.com.tw/gold?Lang=zh-TW"
    history_chart_url = "https://rate.bot.com.tw/gold/chart/year/TWD"

    headers = {"User-Agent": "Mozilla/5.0"}
    current_gold = {}
    try:
        resp = requests.get(current_price_url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"class": "table-striped"})
        rows = table.find("tbody").find_all("tr") if table else []
        for r in rows:
            tds = r.find_all("td")
            if len(tds) > 4 and "黃金牌價" in tds[0].get_text():
                current_gold["sell_price"] = tds[4].get_text(strip=True)
                current_gold["buy_price"]  = tds[3].get_text(strip=True)
                break
        if not current_gold:
            raise ValueError("找不到黃金牌價欄位")
        current_gold["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"黃金即時價抓取失敗：{e}", exc_info=True)
        return "抱歉，目前無法取得即時黃金牌價。"

    hist_summary = "歷史數據不足"
    try:
        df = pd.read_html(history_chart_url)[0]
        df = df[["日期", "本行賣出價格"]].copy()
        df["日期"] = pd.to_datetime(df["日期"], format="%Y/%m/%d")
        df["本行賣出價格"] = pd.to_numeric(df["本行賣出價格"], errors="coerce")
        df = df.dropna().set_index("日期").sort_index()
        last30 = df[df.index >= (datetime.now() - timedelta(days=30))]
        if not last30.empty:
            mx, mn, avg = last30["本行賣出價格"].max(), last30["本行賣出價格"].min(), last30["本行賣出價格"].mean()
            try:
                cur = float(current_gold["sell_price"].replace(",", ""))
                ago = df["本行賣出價格"].iloc[-30] if len(df) >= 30 else last30["本行賣出價格"].iloc[0]
                delta = cur - ago
                pct = (delta / ago * 100) if ago else 0
                hist_summary = f"近30天高 {mx:.2f}、低 {mn:.2f}、均 {avg:.2f}；較30天前變化 {delta:.2f}（{pct:.2f}%）"
            except Exception:
                hist_summary = f"近30天高 {mx:.2f}、低 {mn:.2f}、均 {avg:.2f}"
    except Exception as e:
        logger.warning(f"黃金歷史數據處理失敗：{e}")

    content = (
        f"**最新牌價**：賣出 {current_gold['sell_price']} 元 / 買入 {current_gold['buy_price']} 元（{current_gold['update_time']}）\n"
        f"**近30天摘要**：{hist_summary}"
    )
    msgs = [
        {"role":"system","content":"你是專業黃金市場分析師，請用繁體中文、約 250 字，給出簡潔客觀的行情解讀與一個給一般人的建議。"},
        {"role":"user","content":content},
    ]
    return ai_complete(msgs, max_tokens=500, temperature=0.6)

def get_currency_analysis(cur: str) -> str:
    logger.info(f"開始：{cur} 匯率")
    try:
        url = f"https://open.er-api.com/v6/latest/{cur.upper()}"
        data = requests.get(url, timeout=10).json()
        if data.get("result") != "success":
            return f"抱歉，無法取得 {cur.upper()} 匯率資料。"
        twd = data["rates"].get("TWD")
        if not twd:
            return f"抱歉，API 中沒有 TWD 對 {cur.upper()}。"
        content = f"1 {cur.upper()} ≈ {twd:.5f} TWD"
        msgs = [
            {"role":"system","content":"你是外匯分析師，請以繁體中文寫 120~180 字快訊，包含目前匯率、旅遊換匯概念與一句實用建議。"},
            {"role":"user","content":content},
        ]
        return ai_complete(msgs, max_tokens=300, temperature=0.5)
    except Exception as e:
        logger.error(f"匯率分析錯誤：{e}", exc_info=True)
        return "抱歉，匯率服務暫時無法使用。"

# ========== 7) 股票：Quote API + yfinance ==========
YF_QUOTE_API = "https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols}"
REQ_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

def _normalize_symbol(s: str) -> str:
    t = s.strip().upper()
    if t in ["台股大盤", "大盤"]: return "^TWII"
    if t in ["美股大盤", "美盤", "美股"]: return "^GSPC"
    if re.fullmatch(r"\d{4,6}[A-Z]?", t): return f"{t}.TW"  # 台股自動補 .TW
    return t

def yf_quote(symbol: str) -> Optional[dict]:
    url = YF_QUOTE_API.format(symbols=symbol)
    r = requests.get(url, headers=REQ_HEADERS, timeout=8)
    r.raise_for_status()
    res = r.json()
    arr = (res or {}).get("quoteResponse", {}).get("result", [])
    return arr[0] if arr else None

def get_stock_snapshot(user_input: str) -> Tuple[str, str, Optional[float], Optional[str], Optional[str]]:
    """
    回傳： (norm_symbol, name, price, change_str, currency)
    """
    norm = _normalize_symbol(user_input)
    q = yf_quote(norm)
    if not q:
        return (norm, user_input, None, None, None)
    name = q.get("longName") or q.get("shortName") or norm
    price = q.get("regularMarketPrice")
    chg = q.get("regularMarketChange"); chgp = q.get("regularMarketChangePercent")
    change_str = f"{chg:+.2f} ({chgp:+.2f}%)" if (chg is not None and chgp is not None) else None
    currency = q.get("currency")
    return (norm, name, price, change_str, currency)

def get_stock_history_text(norm_symbol: str) -> str:
    try:
        # 指數/股票皆可，抓近 1 個月日 K
        hist = yf.Ticker(norm_symbol).history(period="1mo", interval="1d")
        if hist is None or hist.empty:
            return "（近1個月歷史價格不可用）"
        # 簡要摘要
        close = hist["Close"]
        last = float(close.iloc[-1])
        first = float(close.iloc[0])
        delta = last - first
        pct = (delta/first*100) if first else 0
        return f"近1月收盤：起點 {first:.2f} → 目前 {last:.2f}，變化 {delta:.2f}（{pct:.2f}%）。"
    except Exception as e:
        logger.warning(f"抓歷史價失敗：{norm_symbol} - {e}")
        return "（歷史價格抓取失敗）"

def get_stock_news_text(norm_symbol: str, fallback_name: str) -> str:
    try:
        tk = yf.Ticker(norm_symbol)
        items = tk.news or []
        if not items:
            return "（最近沒有可用新聞）"
        out = []
        for n in items[:5]:
            title = n.get("title")
            publisher = n.get("publisher")
            ts = n.get("providerPublishTime")
            when = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d") if ts else ""
            if title:
                out.append(f"- {when} {publisher or ''}：{title}")
        return "\n".join(out) if out else "（最近沒有可用新聞）"
    except Exception as e:
        logger.warning(f"抓新聞失敗：{norm_symbol} - {e}")
        return f"（{fallback_name} 新聞抓取失敗）"

def build_stock_report(user_input: str) -> str:
    logger.info(f"開始執行 {user_input} 股票分析…")
    norm, name, price, change_str, ccy = get_stock_snapshot(user_input)
    if price is None and norm not in ["^TWII", "^GSPC"]:
        return f"抱歉，無法獲取 {user_input} 的即時資料，請確認代碼是否正確。"

    hist_txt = get_stock_history_text(norm)
    news_txt = get_stock_news_text(norm, name)

    stock_link = f"https://finance.yahoo.com/quote/{norm.replace('^', '%5E')}"
    snapshot_line = f"{name}（{norm}） 現價 {price:.2f} {ccy or ''}，變動 {change_str}" if price is not None else f"{name}（{norm}）"
    user_content = (
        f"{snapshot_line}\n"
        f"{hist_txt}\n\n"
        f"最新新聞：\n{news_txt}\n"
        f"連結：{stock_link}"
    )

    sys = (
        "你是一位專業的證券分析師，請用繁體中文、Markdown 結構化輸出。"
        "請包含：\n"
        "- **股名(股號)**、即時現價/漲跌（含幣別）、價格資料時間（若可）\n"
        "- 技術面（近一段時間趨勢與關鍵價位）\n"
        "- 基本面（若為指數可略過基本面）\n"
        "- 消息面重點\n"
        "- 策略建議：建議買進區間、停利/停損參考與倉位建議（張數或比例）\n"
        "- 風險提示\n"
        f"- 最後附上有效連結：{stock_link}\n"
    )
    msgs = [{"role":"system","content":sys},{"role":"user","content":user_content}]
    return ai_complete(msgs, max_tokens=1400, temperature=0.7)

# ========== 8) 角色 / 翻譯 / 情緒 ==========
def set_user_persona(chat_id: str, key: str) -> str:
    if key == "random": key = random.choice(list(PERSONAS.keys()))
    if key not in PERSONAS: key = "sweet"
    user_persona[chat_id] = key
    return key

async def analyze_sentiment(text: str) -> str:
    msgs = [
        {"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
        {"role":"user","content":text}
    ]
    out = await groq_small_async(msgs, max_tokens=5, temperature=0.0)
    return (out or "neutral").strip().lower()

async def translate_text(text: str, target_lang_display: str) -> str:
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    msgs = [
        {"role":"system","content":"You are a precise translation engine. Output ONLY the translated text."},
        {"role":"user","content":json.dumps({
            "source_language":"auto", "target_language":target, "text_to_translate":text
        }, ensure_ascii=False)}
    ]
    return await groq_small_async(msgs, max_tokens=800, temperature=0.2)

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    k = user_persona.get(chat_id, "sweet"); p = PERSONAS[k]
    return (f"你是一位「{p['title']}」。風格：{p['style']}。\n"
            f"使用者情緒：{sentiment}。請先共情再給建議；回覆使用繁體中文，精煉自然，搭配少量表情 {p['emoji']}。")

# ========== 9) LINE Handlers ==========
@handler.add(PostbackEvent)
def on_postback(event: PostbackEvent):
    data = (event.postback.data or "").strip()
    if not data.startswith("menu:"):
        return
    kind = data.split(":",1)[1]
    msgs = [build_submenu_flex(kind), TextSendMessage(text="👇 選一個開始吧", quick_reply=make_quick_reply())]
    line_bot_api.reply_message(event.reply_token, msgs)

@handler.add(MessageEvent, message=TextMessage)
def on_message_text(event: MessageEvent):
    # 這裡在 Webhook 的 worker thread，沒有 running loop；用 asyncio.run 執行 async 邏輯
    try:
        asyncio.run(handle_message_async(event))
    except Exception as e:
        logger.error(f"處理訊息失敗：{e}", exc_info=True)

async def handle_message_async(event: MessageEvent):
    chat_id = get_chat_id(event)
    msg_raw = (event.message.text or "").strip()
    reply_token = event.reply_token
    is_group = not isinstance(event.source, SourceUser)

    try:
        bot_name = (await run_in_threadpool(line_bot_api.get_bot_info)).display_name
    except Exception:
        bot_name = "AI 助手"

    if not msg_raw:
        return

    # 預設群組自動回覆 ON
    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True

    # 若關閉自動回覆，群組需 @我
    if is_group and not auto_reply_status.get(chat_id, True) and not msg_raw.startswith(f"@{bot_name}"):
        return
    msg = msg_raw[len(f"@{bot_name}"):].strip() if msg_raw.startswith(f"@{bot_name}") else msg_raw
    low = msg.lower()

    # --- 指令 ---
    if low in ("menu","選單","主選單"):
        line_bot_api.reply_message(reply_token, build_main_menu_flex()); return

    if low in ("開啟自動回答","關閉自動回答"):
        is_on = (low == "開啟自動回答")
        auto_reply_status[chat_id] = is_on
        reply_with_quick_bar(reply_token, "✅ 已開啟自動回答" if is_on else "❌ 已關閉自動回答（群組需 @我 才回）")
        return

    if low in ("金價","黃金"):
        txt = await run_in_threadpool(get_gold_ai_analysis_report)
        reply_with_quick_bar(reply_token, txt); return

    if low == "jpy":
        txt = await run_in_threadpool(get_currency_analysis, "JPY")
        reply_with_quick_bar(reply_token, txt); return

    # 彩票（如果模組可用）
    if LOTTERY_ENABLED and msg in ("大樂透","威力彩","539"):
        try:
            cai = caiyunfangwei_crawler.get_caiyunfangwei()
            cai_msg = (f"***財神方位***\n國歷：{cai.get('今天日期','')}\n"
                       f"農曆：{cai.get('今日農曆','')}\n歲次：{cai.get('今日歲次','')}\n"
                       f"方位：{cai.get('財神方位','')}")
        except Exception:
            cai_msg = "（財神方位暫時無法取得）"
        try:
            if "威力" in msg: last = lottery_crawler.super_lotto()
            elif "大樂" in msg: last = lottery_crawler.lotto649()
            else: last = lottery_crawler.daily_cash()
        except Exception as e:
            reply_with_quick_bar(reply_token, f"抱歉，彩票資料無法取得：{e}"); return

        prompt = (
            f"近期待號：\n{last}\n\n{cai_msg}\n\n"
            "請以繁體中文給一份短評：熱門/冷門號段、3組號碼建議（若有特別號/二區請分開），最後附一句 20 字內吉祥話。"
        )
        out = ai_complete(
            [{"role":"system","content":"你是彩券分析師。"},{"role":"user","content":prompt}],
            max_tokens=600, temperature=0.9
        )
        reply_with_quick_bar(reply_token, out); return

    # 人設 / 翻譯
    persona_keys = {"甜":"sweet","鹹":"salty","萌":"moe","酷":"cool","random":"random","隨機":"random"}
    if low in persona_keys:
        key = set_user_persona(chat_id, persona_keys[low]); p = PERSONAS[key]
        reply_with_quick_bar(reply_token, f"💖 已切換人設：{p['title']}\n\n{p['greetings']}"); return

    if low.startswith("翻譯->"):
        lang = msg.split("->",1)[1].strip()
        if lang == "結束":
            translation_states.pop(chat_id, None)
            reply_with_quick_bar(reply_token, "✅ 已結束翻譯模式"); return
        translation_states[chat_id] = lang
        reply_with_quick_bar(reply_token, f"🌐 已開啟翻譯 → {lang}，直接輸入要翻的內容。"); return

    if chat_id in translation_states:
        tgt = translation_states[chat_id]
        try:
            out = await translate_text(msg, tgt)
        except Exception:
            out = "翻譯暫時失效，等我回神再來一次 🙏"
        reply_with_quick_bar(reply_token, f"🌐 ({tgt})\n{out}"); return

    # --- 股票關鍵字判斷 ---
    def is_stock_query(txt: str) -> bool:
        t = txt.strip().upper()
        if t in ["台股大盤","大盤","美股大盤","美盤","美股"]:
            return True
        if re.fullmatch(r"\d{4,6}[A-Z]?", t):  # 台股
            return True
        if re.fullmatch(r"[A-Z]{1,5}", t):    # 美股/ETF
            return True
        return False

    if is_stock_query(msg):
        out = await run_in_threadpool(build_stock_report, msg)
        reply_with_quick_bar(reply_token, out); return

    # --- 一般對話 ---
    try:
        history = conversation_history.get(chat_id, [])
        senti = await analyze_sentiment(msg)
        sys = build_persona_prompt(chat_id, senti)
        messages = [{"role":"system","content":sys}] + history + [{"role":"user","content":msg}]
        # 用 Groq 輕量回即可
        reply = await groq_small_async(messages, max_tokens=600, temperature=0.7)
        history.extend([{"role":"user","content":msg},{"role":"assistant","content":reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
        reply_with_quick_bar(reply_token, reply)
    except Exception as e:
        logger.error(f"一般對話失敗：{e}", exc_info=True)
        reply_with_quick_bar(reply_token, "抱歉我剛剛走神了 😅 再說一次讓我補上！")

# ========== 10) Routes ==========
@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        # handler 是同步的，放到 threadpool 跑
        await run_in_threadpool(handler.handle, body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Callback 處理失敗：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse({"status":"ok"})

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