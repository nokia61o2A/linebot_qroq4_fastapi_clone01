# app_fastapi.py
# =============================================================================
# LINE Bot + FastAPI (金價 / 股票 / 彩票 / 翻譯 / TTS / 單聊 Loading 動畫)
# -----------------------------------------------------------------------------
# ✅ 修正版要點（含本次「@ai 行為」修正 + QuickReplyButton 自動應答OFF）
# 1) 指令匹配不到時，最終一律回到「一般 LLM 對話（代入人設）」。
# 2) 加回 _TW_CODE_RE / _US_CODE_RE，避免 NameError 導致整段對話中斷。
# 3) 彩票：「優先」呼叫你獨立模組 my_commands/lottery_gpt.py（大樂透/威力彩/今彩539），
#    其他彩種「後備」走 TaiwanLotteryCrawler，皆含錯誤保底。
# 4) 翻譯模式新增「中英雙向」；TTS 在雙向模式會依輸出語種自動選 en/zh-TW。
# 5) 任何錯誤都以文字回覆保底，避免 LINE 空訊息。
# 6) ✅ 自動應答模式：
#    - 私聊：預設自動應答 ON，照舊回覆所有訊息。
#    - 群組 / 聊天室：預設自動應答 OFF，不會主動回覆。
#      * OFF 時：僅在「@ai 指令」或「@機器人名 + 指令」才會處理該次指令，但 **不改變** 自動應答狀態。
#      * OFF 時：若「只有 @ai」或「只有 @機器人名（無其它文字）」→ 將自動應答切到 ON，並回覆 "I'm back!"。
#      * ON 時：群組 / 聊天室回復所有訊息（原本的所有分析功能）。
#    - 使用「開啟自動回答／關閉自動回答」可手動切換。
#    - 當自動應答 OFF 時，QuickReply 整排按鈕會隱藏；ON 時才會顯示。
#    - 自動應答 OFF 時關閉訊息會回「我先退下了」。
# 7) ✅ QuickReplyButton 新增「自動應答OFF」按鈕：
#    - 只在自動應答為 ON 時顯示。
#    - 點擊後會送出文字「關閉自動回答」，由 on_message 既有邏輯處理。
# 8) ✅ 本次重點修正：
#    - 「@ai 有帶訊息/指令」：會把 @ 前綴去掉後的內容當指令處理，但 **不改變** 自動應答 ON/OFF。
#    - 「只有 @ai 或只有 @機器人名」：才會把自動應答模式切成 ON，並回覆 "I'm back!"。
# =============================================================================

import os
import re
import io
import json
import time
import random
import logging
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime

import requests
import httpx
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, AudioSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    QuickReply, QuickReplyButton, MessageAction,
    PostbackAction, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent,
    TextComponent, ButtonComponent
)

from gtts import gTTS
import cloudinary
import cloudinary.uploader
import uvicorn

# ==== 外部彩票資料來源（後備用：全彩種）====
try:
    from TaiwanLottery import TaiwanLotteryCrawler  # 支援 9 彩種，作為備援
    _LT_CRAWLER_OK = True
    logging.info("✅ TaiwanLotteryCrawler 模組載入成功")
except Exception as e:
    _LT_CRAWLER_OK = False
    logging.warning(f"⚠️ TaiwanLotteryCrawler 載入失敗：{e}")

# ==== 你的獨立分析模組（優先用：大樂透/威力彩/今彩539）====
try:
    from my_commands.lottery_gpt import lottery_gpt as ext_lottery_gpt
    _EXT_LOTTERY_OK = True
    logging.info("✅ my_commands.lottery_gpt 模組載入成功")
except Exception as e:
    _EXT_LOTTERY_OK = False
    logging.warning(f"⚠️ 外掛 lottery_gpt 模組載入失敗：{e}")

# ========= Logging =========
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(asctime)s:%(message)s"
)
log = logging.getLogger("app")

# ========= ENV =========
BASE_URL = os.getenv("BASE_URL")  # e.g. https://your-domain/callback
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")  # e.g. 官方或自建代理

# ✅ Bot 名稱關鍵字（用於群組 / 聊天室偵測 @ 提及）
#    - BOT_NAME：可設定完整顯示名稱或暱稱，例如 "AI醬"、"金價小幫手"。
#    - BOT_NAME_KEYWORDS：若要額外自訂多個關鍵字（逗號分隔），例如 "ai,AI,小幫手"。
BOT_NAME = os.getenv("BOT_NAME", "").strip()
BOT_NAME_KEYWORDS = [
    kw.strip().lower()
    for kw in os.getenv("BOT_NAME_KEYWORDS", "ai,ＡＩ,ai醬,ai bot").split(",")
    if kw.strip()
]

if not BASE_URL or not CHANNEL_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("請設定環境變數：BASE_URL、CHANNEL_ACCESS_TOKEN、CHANNEL_SECRET")

# ========= LINE SDK =========
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# ========= Cloudinary（可選，用於語音上傳）=========
CLOUD_OK = False
try:
    if os.getenv("CLOUDINARY_URL"):
        cloudinary.config(cloudinary_url=os.getenv("CLOUDINARY_URL"))
    else:
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            secure=True
        )
    if cloudinary.config().cloud_name:
        CLOUD_OK = True
        log.info("✅ Cloudinary 配置成功")
except Exception as e:
    log.warning(f"⚠️ Cloudinary 初始化失敗：{e}")

# ========= AI Clients（OpenAI/Groq，雙引擎）=========
openai_client = None
if OPENAI_API_KEY:
    try:
        import openai as openai_lib
        if OPENAI_API_BASE:
            openai_client = openai_lib.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
            log.info(f"✅ OpenAI Client (base={OPENAI_API_BASE})")
        else:
            openai_client = openai_lib.OpenAI(api_key=OPENAI_API_KEY)
            log.info("✅ OpenAI Client (official)")
    except Exception as e:
        log.warning(f"OpenAI 初始化失敗：{e}")

from groq import Groq
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        log.info("✅ Groq Client 初始化成功")
    except Exception as e:
        log.warning(f"Groq 初始化失敗：{e}")

# 強制採用 Groq 穩定模型
GROQ_MODEL_PRIMARY = "llama-3.1-8b-instant"

# ========= 全域狀態 =========
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125 Safari/537.36"
}
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"

conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY = 10
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}  # chat_id -> 目標語言顯示字串（英文/日文/繁體中文/中英雙向）
auto_reply_status: Dict[str, bool] = {}  # 自動應答 ON/OFF（key = chat_id）
tts_enabled: Dict[str, bool] = {}
tts_lang: Dict[str, str] = {}  # gTTS 用語言碼（e.g. zh-TW）

# ========= 人設 =========
PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼", "greet": "我在這🌸", "emoji": "🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽", "greet": "你又來啦？😏", "emoji": "😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣", "greet": "呀呼～(ﾉ>ω<)ﾉ", "emoji": "✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉", "greet": "我在。說重點。", "emoji": "🧊⚡️"},
}
PERSONA_ALIAS = {"甜": "sweet", "鹹": "salty", "萌": "moe", "酷": "cool", "random": "random"}

# ========= 股票代碼 Regex（修正缺失）=========
# 台股代碼：4~5 位數字，可選擇結尾 1 個大寫英文字母
_TW_CODE_RE = re.compile(r'^\d{4,5}[A-Z]?$')
# 美股：1~5 英文字母（NVDA、AAPL、TSLA…）
_US_CODE_RE = re.compile(r'^[A-Z]{1,5}$')

# ========= App Lifespan：啟動時更新 Webhook =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 應用啟動")
    try:
        async with httpx.AsyncClient() as c:
            headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
            payload = {"endpoint": f"{BASE_URL}/callback"}
            r = await c.put(
                "https://api.line.me/v2/bot/channel/webhook/endpoint",
                headers=headers, json=payload, timeout=10
            )
            r.raise_for_status()
            log.info("✅ Webhook 更新成功")
    except Exception as e:
        log.warning(f"⚠️ Webhook 更新失敗：{e}")
    yield
    log.info("👋 應用關閉")

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="5.1.0")
router = APIRouter()

# ========= Loading 動畫（僅單人聊天有效；失敗不影響流程）=========
def send_loading_animation(user_id: str, seconds: int = 5):
    """
    觸發 LINE 官方 Loading 動畫（單人 1:1 有效；群組/聊天室無效）
    文件：/v2/bot/chat/loading/start
    """
    try:
        url = "https://api.line.me/v2/bot/chat/loading/start"
        headers = {
            "Authorization": f"Bearer {CHANNEL_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {"chatId": user_id, "loadingSeconds": max(1, min(15, int(seconds)))}
        resp = requests.post(url, headers=headers, json=payload, timeout=5)
        resp.raise_for_status()
        log.info(f"✅ Loading 動畫觸發成功 chatId={user_id}")
    except Exception as e:
        # 這裡只警告，不中斷後續流程
        log.warning(f"⚠️ Loading 動畫觸發失敗：{e}")

# ========= QuickReply（依 TTS 與翻譯模式動態顯示）=========
def quick_bar(chat_id: Optional[str] = None) -> Optional[QuickReply]:
    """
    ✅ 依「自動應答模式」決定是否顯示 QuickReply：
    - 私聊：預設 auto_reply_status[chat_id] = True → 一律顯示。
    - 群組 / 聊天室：
        * 自動應答 ON：顯示完整 QuickReply。
        * 自動應答 OFF：整排 QuickReply 隱藏（回傳 None）。
    並加入：
    - 「自動應答OFF」按鈕：
        * 只在 auto_reply_status 為 True 時顯示。
        * 點擊後送出文字「關閉自動回答」，由 on_message 中既有邏輯關閉自動應答。
    """
    if chat_id is not None:
        # 群組 / 聊天室若 auto_reply_status=False → 不顯示 QuickReply
        if not auto_reply_status.get(chat_id, True):
            return None

    items: List[QuickReplyButton] = [
        QuickReplyButton(action=MessageAction(label="主選單", text="選單")),
        QuickReplyButton(action=MessageAction(label="台股大盤", text="台股大盤")),
        QuickReplyButton(action=MessageAction(label="美股大盤", text="美股大盤")),
        QuickReplyButton(action=MessageAction(label="黃金價格", text="金價")),
        QuickReplyButton(action=MessageAction(label="日圓匯率", text="JPY")),
        QuickReplyButton(action=MessageAction(label="查 2330", text="2330")),
        QuickReplyButton(action=MessageAction(label="查 NVDA", text="NVDA")),
        QuickReplyButton(action=PostbackAction(label="💖 AI 人設", data="menu:persona")),
        QuickReplyButton(action=PostbackAction(label="🎰 彩票選單", data="menu:lottery")),
    ]

    # 僅顯示 TTS「其中之一」按鈕
    if chat_id and tts_enabled.get(chat_id, False):
        items.insert(7, QuickReplyButton(action=MessageAction(label="語音 關", text="TTS OFF")))
    else:
        items.insert(7, QuickReplyButton(action=MessageAction(label="語音 開✅", text="TTS ON")))

    # ✅ 自動應答 OFF 快速按鈕（只有在目前自動應答為 ON 時顯示）
    if chat_id and auto_reply_status.get(chat_id, True):
        items.append(
            QuickReplyButton(
                action=MessageAction(label="自動應答OFF", text="關閉自動回答")
            )
        )

    # 翻譯模式：最後一鍵換成「結束翻譯」
    if chat_id and chat_id in translation_states:
        items.append(QuickReplyButton(action=MessageAction(label="結束翻譯", text="翻譯->結束")))
    else:
        items.append(QuickReplyButton(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate")))

    return QuickReply(items=items)

# ========= sender.name（翻譯模式顯示「翻譯模式（中↔英）」）=========
def display_sender_name(chat_id: str) -> Tuple[str, Optional[str]]:
    if chat_id in translation_states:
        target = translation_states.get(chat_id) or ""
        mapping = {"英文": "中→英", "日文": "中→日", "繁體中文": "→ 繁中", "中英雙向": "中↔英"}
        arrow = mapping.get(target, f"→ {target}") if target else ""
        name = f"翻譯模式（{arrow}）" if arrow else "翻譯模式"
        return name, None
    return "AI 助理", None

# ========= Flex 提示卡（無分隔線、字型 md）=========
def minimal_flex_hint(
    alt_text: str = "提示",
    hint_text: str = "要聽語音請按上方播放鈕👆",
    chat_id: Optional[str] = None
) -> FlexSendMessage:
    safe_alt = (alt_text or hint_text or "提示").strip() or "提示"
    bubble = BubbleContainer(
        direction="ltr",
        body=BoxComponent(
            layout="vertical",
            spacing="sm",
            contents=[
                TextComponent(text=hint_text, size="md", color="#888888", wrap=True)
            ]
        )
    )
    return FlexSendMessage(alt_text=safe_alt, contents=bubble, quick_reply=quick_bar(chat_id))

# ========= 統一回覆：Text → Audio →（可選）Flex =========
def reply_text_audio_flex(
    reply_token: str,
    chat_id: str,
    text: str,
    audio_url: Optional[str],
    duration_ms: int,
    hint_text: str = "（👆要聽語音請按上方播放鈕）"
):
    """
    - 只有 audio_url 存在時才附 Flex 提示卡（TTS OFF 不出現）
    - 所有訊息 sender.name 隨翻譯模式顯示「翻譯模式（中↔英）」等
    - QuickReply 是否顯示，依 quick_bar(chat_id)（會看 auto_reply_status）
    """
    sender_name, sender_icon = display_sender_name(chat_id)
    msgs = []

    # 1) Text
    qr = quick_bar(chat_id)
    text_msg = TextSendMessage(text=text, quick_reply=qr)
    text_msg.sender = {"name": sender_name}
    if sender_icon:
        text_msg.sender["iconUrl"] = sender_icon
    msgs.append(text_msg)

    # 2) Audio（可選）
    if audio_url:
        audio_msg = AudioSendMessage(original_content_url=audio_url, duration=duration_ms)
        audio_msg.sender = {"name": sender_name}
        if sender_icon:
            audio_msg.sender["iconUrl"] = sender_icon
        msgs.append(audio_msg)

        # 3) 提示 Flex（僅在有音訊時加入）
        flex_msg = minimal_flex_hint(
            alt_text=(text[:60] + "…") if text else "提示",
            hint_text=hint_text,
            chat_id=chat_id
        )
        flex_msg.sender = {"name": sender_name}
        if sender_icon:
            flex_msg.sender["iconUrl"] = sender_icon
        msgs.append(flex_msg)

    # 一次回覆多則訊息
    line_bot_api.reply_message(reply_token, msgs)

# ========= AI / 翻譯 =========
def ai_chat(messages: List[dict]) -> str:
    """
    先嘗試 OpenAI；失敗再走 Groq；最後回穩定錯誤訊息（避免空回覆）
    """
    # OpenAI
    if openai_client:
        try:
            r = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1600
            )
            return r.choices[0].message.content
        except Exception as e:
            log.warning(f"OpenAI 失敗：{e}")

    # Groq
    if groq_client:
        try:
            r = groq_client.chat.completions.create(
                model=GROQ_MODEL_PRIMARY,
                messages=messages,
                temperature=0.7,
                max_tokens=1800
            )
            return r.choices[0].message.content
        except Exception as e:
            log.warning(f"Groq 失敗：{e}")

    return "AI 引擎連線不穩定，請稍後再試。"

def translate_text(content: str, target_lang_display: str) -> str:
    """
    使用 Groq 進行單向翻譯（中→目標語言）
    """
    if not groq_client:
        return "抱歉，翻譯引擎暫不可用。"
    try:
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=[
                {"role": "system", "content": "You are a precise translator. Output ONLY the translated text. Do NOT converse. Do NOT explain."},
                {"role": "user", "content": f"Translate to {target_lang_display}:\n{content}"}
            ],
            temperature=0.2,
            max_tokens=len(content) * 2 + 60
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning(f"翻譯失敗：{e}")
        return "抱歉，翻譯失敗。"

# ========= 股票 =========
# 台股代碼規則：4~6 位數字，可能含一位英文字母後綴（如 1101B → 1101B.TW）
_TW_CODE_FULL_RE = re.compile(r"^(?:[0-9]{4,6})(?:[A-Za-z])?$")
# 美股代碼規則：1~5 個英文字（排除 JPY 關鍵字以免和匯率指令衝突）
_US_CODE_FULL_RE = re.compile(r"^[A-Za-z]{1,5}$")

def normalize_ticker(raw: str) -> Tuple[str, str]:
    """
    將使用者輸入的股票代碼/別名正規化為 Yahoo Finance 可用的 symbol
    回傳 (yahoo_symbol, display_symbol)
    - 台股：純數字或數字+字母 → 一律加 .TW（例如 '2002' → '2002.TW'）
    - 大盤別名：台股大盤/大盤 → ^TWII，美股大盤/美盤/美股 → ^GSPC
    - 其他純英文字：視為美股（不自動加市場尾碼）
    """
    t = (raw or "").strip()
    u = t.upper()

    # 大盤別名
    if u in ("台股大盤", "大盤"):
        return "^TWII", "^TWII"
    if u in ("美股大盤", "美盤", "美股"):
        return "^GSPC", "^GSPC"

    # 台股：數字或數字+字母 → 強制 .TW
    if _TW_CODE_FULL_RE.match(u):
        return f"{u}.TW", u

    # 英文代碼（美股）
    if _US_CODE_FULL_RE.match(u):
        return u, u

    # 其他情況：原樣回傳（讓上層保守處理）
    return u, u

def yahoo_snapshot(symbol: str) -> dict:
    """
    以 yfinance 取得基本即時/近日快照資訊
    - 會盡力從 info 或 history 補齊價格
    - 輸出鍵：name/now_price/change/currency/close_time
    """
    out = {"name": symbol, "now_price": None, "change": None, "currency": "", "close_time": ""}
    try:
        tk = yf.Ticker(symbol)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            pass

        hist = pd.DataFrame()
        try:
            hist = tk.history(period="2d", interval="1d")
        except Exception:
            pass

        out["name"] = info.get("shortName") or info.get("longName") or symbol
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None and not hist.empty:
            price = float(hist["Close"].iloc[-1])
        if price is not None:
            out["now_price"] = f"{price:.2f}"
            # 台股預設 TWD
            if symbol.endswith(".TW"):
                out["currency"] = "TWD"
            else:
                out["currency"] = info.get("currency") or "USD"

        if not hist.empty and len(hist) >= 2:
            prev = float(hist["Close"].iloc[-2]) if float(hist["Close"].iloc[-2]) != 0 else None
            last = float(hist["Close"].iloc[-1])
            if prev:
                chg = last - prev
                pct = chg / prev * 100
                sign = "+" if chg >= 0 else ""
                out["change"] = f"{sign}{chg:.2f} ({sign}{pct:.2f}%)"
        if not hist.empty:
            out["close_time"] = hist.index[-1].strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        log.warning(f"yfinance 快照失敗：{e}")
    return out

def stock_report(q: str) -> str:
    """
    產生條列式股票分析。對台股輸入（例如 '2002'）會自動轉 '2002.TW'
    並附正確的 Yahoo 連結（台股走 https://tw.stock.yahoo.com/quote/代碼）
    """
    code, disp = normalize_ticker(q)
    snap = yahoo_snapshot(code)

    # Yahoo 連結（指向對應語系站點）
    if code.startswith("^"):  # 指數
        link = f"https://finance.yahoo.com/quote/{code}"
    elif code.endswith(".TW"):  # 台股
        link = f"https://tw.stock.yahoo.com/quote/{disp}"
    else:  # 其他（多半是美股）
        link = f"https://finance.yahoo.com/quote/{code}"

    sys = (
        "你是專業證券分析師，請用繁體中文分段條列："
        "1) 近期走勢 2) 技術面 3) 基本面 4) 消息 5) 風險 6) 建議與合理區間 7) 結論。"
        "若資料不足，請保守陳述；勿杜撰精確數字。"
    )
    user = (
        f"代碼：{disp}\n"
        f"名稱：{snap.get('name')}\n"
        f"價格：{snap.get('now_price')} {snap.get('currency')}\n"
        f"漲跌：{snap.get('change')}\n"
        f"時間：{snap.get('close_time')}\n"
        f"參考連結：{link}"
    )
    return ai_chat([{"role": "system", "content": sys}, {"role": "user", "content": user}])

# ========= 金價（台灣銀行）=========
def _extract_numbers_from_text(text: str) -> dict:
    out = {}
    m_sell = re.search(r"(?:賣出|賣價|賣出價)[^\d]{0,8}([\d,]+(?:\.\d+)?)", text)
    if m_sell:
        out["sell_twd_per_g"] = float(m_sell.group(1).replace(",", ""))
    m_buy = re.search(r"(?:買進|買價|買入價)[^\d]{0,8}([\d,]+(?:\.\d+)?)", text)
    if m_buy:
        out["buy_twd_per_g"] = float(m_buy.group(1).replace(",", ""))
    m_time = re.search(r"(?:掛牌時間|最後更新)[：:\s]*([0-9\/\-\s:]{8,})", text)
    if m_time:
        out["listed_at"] = m_time.group(1).strip()
    return out

def _parse_gold_html(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    out = {}
    try:
        big_text = " ".join(soup.stripped_strings)
        got = _extract_numbers_from_text(big_text)
        out.update(got)
    except Exception:
        pass

    try:
        for sc in soup.find_all("script"):
            s = sc.string or ""
            if not s:
                continue
            if "sell" in s.lower() and "buy" in s.lower():
                nums = _extract_numbers_from_text(s)
                for k, v in nums.items():
                    out.setdefault(k, v)
    except Exception:
        pass
    return out

def get_bot_gold() -> Tuple[str, Optional[float], Optional[float], Optional[str]]:
    urls = [
        "https://rate.bot.com.tw/gold?Lang=zh-TW",
        "https://rate.bot.com.tw/gold",
    ]
    data = {}
    html_any = ""

    for u in urls:
        try:
            r = requests.get(u, headers=DEFAULT_HEADERS, timeout=12)
            r.raise_for_status()
            html_any = r.text
            d = _parse_gold_html(r.text)
            for k, v in d.items():
                data.setdefault(k, v)
            if data.get("sell_twd_per_g") and data.get("buy_twd_per_g"):
                break
        except Exception as e:
            log.warning(f"抓取 {u} 失敗：{e}")

    if not (data.get("sell_twd_per_g") and data.get("buy_twd_per_g")) and html_any:
        more = _extract_numbers_from_text(" ".join(BeautifulSoup(html_any, "html.parser").stripped_strings))
        for k, v in more.items():
            data.setdefault(k, v)

    sell = data.get("sell_twd_per_g")
    buy = data.get("buy_twd_per_g")
    ts = data.get("listed_at")

    if sell is None or buy is None:
        msg = "抱歉，目前無法取得台銀黃金牌價。"
        return msg, sell, buy, ts

    spread = sell - buy if (sell is not None and buy is not None) else None
    bias = ""
    if spread is not None:
        bias = "（價差小）" if spread <= 30 else ("（偏寬）" if spread <= 60 else "（價差大）")
    now = datetime.now().strftime("%H:%M")
    msg = (
        f"**台銀黃金**（{now}）\n"
        f"賣：**{sell:,.0f}** 元/g\n"
        f"買：**{buy:,.0f}** 元/g\n"
        f"{'價差：' + format(spread, ',.0f') + bias if spread is not None else ''}\n"
        f"掛牌：{ts or '—'}\n"
        f"來源：台灣銀行（{BOT_GOLD_URL}）"
    )
    return msg, sell, buy, ts

# ========= 匯率（JPY→TWD）=========
def jpy_twd() -> str:
    try:
        res = requests.get("https://open.er-api.com/v6/latest/JPY", timeout=10)
        res.raise_for_status()
        js = res.json()
        if js.get("result") != "success":
            return "外匯 API 異常。"
        rate = js["rates"].get("TWD")
        if not rate:
            return "API 無 TWD 匯率。"
        return f"即時：1 JPY ≈ **{rate:.4f}** TWD"
    except Exception as e:
        log.error(f"匯率失敗：{e}")
        return "外匯資料暫時無法取得。"

# ========= 彩票（備援：全彩種，使用 TaiwanLotteryCrawler）=========
# 你的 my_commands.lottery_gpt 會優先處理「大樂透／威力彩／今彩539」，
# 其餘彩種或外掛失敗時走這支後備函式。
def lottery_report_all(kind: str) -> str:
    try:
        if not _LT_CRAWLER_OK:
            return (
                f"**{kind} 分析報告**\n\n"
                "資料來源暫不可用，請稍後再試。\n\n"
                "[官方歷史開獎查詢](https://www.taiwanlottery.com.tw/)"
            )
        # 對應表：彩種 -> TaiwanLotteryCrawler 方法名、主號數量、最大號
        mapping = {
            "大樂透":   ("lotto649",    6, 49, "特別號"),
            "威力彩":   ("super_lotto", 6, 39, "第二區"),
            "今彩539":  ("daily_cash",  5, 39, None),
            "雙贏彩":   ("lotto1224",   6, 49, None),
            "3星彩":    ("lotto3d",     3, 10, None),
            "4星彩":    ("lotto4d",     4, 10, None),
            "38樂合彩": ("lotto38m6",   6, 38, None),
            "39樂合彩": ("lotto39m5",   5, 39, None),
            "49樂合彩": ("lotto49m6",   6, 49, None),
        }
        if kind not in mapping:
            return (
                f"**{kind} 分析報告**\n\n"
                "目前支援：大樂透／威力彩／今彩539／雙贏彩／3星彩／4星彩／38/39/49樂合彩。\n\n"
                "[官方歷史開獎查詢](https://www.taiwanlottery.com.tw/)"
            )
        func_name, num_main, max_num, special_label = mapping[kind]
        crawler = TaiwanLotteryCrawler()
        func = getattr(crawler, func_name)
        result = func()
        latest = result[0] if isinstance(result, list) and result else None

        if latest:
            draw_date = getattr(latest, "draw_date", None)
            if draw_date:
                draw_date = draw_date.strftime("%Y/%m/%d")
            else:
                draw_date = "—"
            numbers = getattr(latest, "numbers", None) or getattr(latest, "number", None)
            if isinstance(numbers, (list, tuple)):
                numbers_str = ", ".join(f"{n:02d}" for n in numbers)
            else:
                numbers_str = str(numbers)
            special_str = ""
            if special_label:
                special_val = getattr(latest, "special", None)
                if special_val is not None:
                    try:
                        special_str = f"（{special_label}：{int(special_val):02d}）"
                    except Exception:
                        special_str = f"（{special_label}：{special_val}）"
        else:
            # 抓不到 → 隨機保底
            draw_date = datetime.now().strftime("%Y/%m/%d")
            numbers = sorted(random.sample(range(1, max_num + 1), num_main))
            numbers_str = ", ".join(f"{n:02d}" for n in numbers)
            special_str = ""
            if special_label:
                special_rand = random.randint(1, max_num if special_label else max_num)
                special_str = f"（{special_label}：{special_rand:02d}）"

        # 建議（隨機保底）
        suggest = sorted(random.sample(range(1, max_num + 1), num_main))
        suggest_str = ", ".join(f"{n:02d}" for n in suggest)
        suggest_special_str = ""
        if special_label:
            special_sug = random.randint(1, max_num)
            suggest_special_str = f"（{special_label}：{special_sug:02d}）"

        analysis = f"{kind}：近期開獎號碼動態多變，建議理性娛樂。"
        return (
            f"**{kind} 分析報告**\n\n"
            f"📅 最新開獎（{draw_date}）：{numbers_str} {special_str}\n\n"
            f"🎯 下期建議：{suggest_str} {suggest_special_str}\n\n"
            f"💡 分析：{analysis}\n\n"
            f"[官方歷史開獎查詢](https://www.taiwanlottery.com.tw/)"
        )
    except Exception as e:
        log.error(f"{kind} 擷取失敗：{e}", exc_info=True)
        # 錯誤保底
        num_main = 6
        max_num = 49
        rnd = sorted(random.sample(range(1, max_num + 1), num_main))
        rnd_str = ", ".join(f"{n:02d}" for n in rnd)
        return (
            f"**{kind} 分析報告**\n\n"
            f"📅 最新開獎：資料取得失敗（顯示隨機）\n\n"
            f"🎯 下期建議：{rnd_str}\n\n"
            f"💡 分析：資料來源異常，請稍後再試。\n\n"
            f"[官方歷史開獎查詢](https://www.taiwanlottery.com.tw/)"
        )

# ========= Flex 主選單與子選單 =========
def flex_main(chat_id: Optional[str] = None) -> FlexSendMessage:
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[
            TextComponent(text="AI 助理主選單", weight="bold", size="lg")
        ]),
        body=BoxComponent(
            layout="vertical", spacing="md",
            contents=[
                TextComponent(text="請選擇功能：", size="md"),
                ButtonComponent(action=PostbackAction(label="💹 金融查詢", data="menu:finance"),
                                style="primary", color="#5E86C1"),
                ButtonComponent(action=PostbackAction(label="🎰 彩票分析", data="menu:lottery"),
                                style="primary", color="#5EC186"),
                ButtonComponent(action=PostbackAction(label="💖 AI 角色", data="menu:persona"),
                                style="secondary"),
                ButtonComponent(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate"),
                                style="secondary"),
                ButtonComponent(action=PostbackAction(label="⚙️ 系統設定", data="menu:settings"),
                                style="secondary"),
            ]
        )
    )
    return FlexSendMessage(alt_text="主選單", contents=bubble, quick_reply=quick_bar(chat_id))

def flex_submenu(kind: str, chat_id: Optional[str] = None) -> FlexSendMessage:
    title, buttons = "子選單", []
    if kind == "finance":
        title = "💹 金融查詢"
        buttons = [
            ButtonComponent(action=MessageAction(label="台股大盤", text="台股大盤")),
            ButtonComponent(action=MessageAction(label="美股大盤", text="美股大盤")),
            ButtonComponent(action=MessageAction(label="黃金價格", text="金價")),
            ButtonComponent(action=MessageAction(label="日圓匯率", text="JPY")),
            ButtonComponent(action=MessageAction(label="查 2330", text="2330")),
            ButtonComponent(action=MessageAction(label="查 NVDA", text="NVDA")),
        ]
    elif kind == "lottery":
        title = "🎰 彩票分析"
        buttons = [
            ButtonComponent(action=MessageAction(label="大樂透", text="大樂透")),
            ButtonComponent(action=MessageAction(label="威力彩", text="威力彩")),
            ButtonComponent(action=MessageAction(label="今彩539", text="今彩539")),
            ButtonComponent(action=MessageAction(label="雙贏彩", text="雙贏彩")),
            ButtonComponent(action=MessageAction(label="3星彩", text="3星彩")),
            ButtonComponent(action=MessageAction(label="4星彩", text="4星彩")),
            ButtonComponent(action=MessageAction(label="38樂合彩", text="38樂合彩")),
            ButtonComponent(action=MessageAction(label="39樂合彩", text="39樂合彩")),
            ButtonComponent(action=MessageAction(label="49樂合彩", text="49樂合彩")),
        ]
    elif kind == "persona":
        title = "💖 AI 角色"
        buttons = [
            ButtonComponent(action=MessageAction(label="甜美女友", text="甜")),
            ButtonComponent(action=MessageAction(label="傲嬌女友", text="鹹")),
            ButtonComponent(action=MessageAction(label="萌系女友", text="萌")),
            ButtonComponent(action=MessageAction(label="酷系御姐", text="酷")),
            ButtonComponent(action=MessageAction(label="隨機", text="random")),
        ]
    elif kind == "translate":
        title = "🌐 翻譯工具"
        buttons = [
            ButtonComponent(action=MessageAction(label="翻英文", text="翻譯->英文")),
            ButtonComponent(action=MessageAction(label="翻日文", text="翻譯->日文")),
            ButtonComponent(action=MessageAction(label="翻繁中", text="翻譯->繁體中文")),
            ButtonComponent(action=MessageAction(label="中↔英", text="翻譯->中英雙向")),
            ButtonComponent(action=MessageAction(label="結束翻譯", text="翻譯->結束")),
        ]
    elif kind == "settings":
        title = "⚙️ 系統設定"
        buttons = [
            ButtonComponent(action=MessageAction(label="開啟自動回答", text="開啟自動回答")),
            ButtonComponent(action=MessageAction(label="關閉自動回答", text="關閉自動回答")),
        ]
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[
            TextComponent(text=title, weight="bold", size="lg")
        ]),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm")
    )
    return FlexSendMessage(alt_text=title, contents=bubble, quick_reply=quick_bar(chat_id))

# ========== ✅ TTS 功能 ==========
def ensure_defaults(chat_id: str, is_private: bool):
    """
    ✅ 依 chat 類型初始化預設值：
    - 私聊 (SourceUser)：auto_reply_status = True（預設會自動回覆所有訊息）
    - 群組 / 聊天室：auto_reply_status = False（預設不主動回覆，等待被 @ 才開啟）
    其餘 TTS / 人設維持原本預設。
    """
    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True if is_private else False
    if chat_id not in tts_enabled:
        tts_enabled[chat_id] = False
    if chat_id not in tts_lang:
        tts_lang[chat_id] = "zh-TW"
    if chat_id not in user_persona:
        user_persona[chat_id] = "sweet"

def tts_make_url(text: str, lang_code: str) -> Tuple[Optional[str], int]:
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        data = buf.getvalue()

        if not CLOUD_OK:
            return None, 0

        res = cloudinary.uploader.upload(
            data, resource_type="video",
            folder="line-bot-tts",
            public_id=f"tts_{int(time.time()*1000)}",
            overwrite=True
        )

        url = res.get("secure_url")
        dur = max(1000, int(len(data)/32))
        return url, dur
    except Exception as e:
        log.error(f"TTS 失敗: {e}")
        return None, 0


# ========== ✅ 中英雙向翻譯（此版本會中英雙語一起回）=========
def translate_bilingual(content: str) -> str:
    """
    ✅ 最終採用版本：
    - 讓 model 同時輸出中英對照（方便看原文 + 翻譯）
    - 強制翻譯模式，不進行對話
    """
    if not groq_client:
        return content
    try:
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=[
                {"role": "system", "content": (
                    "You are a strict bilingual translator. "
                    "You must translate the user's input directly. "
                    "If the input is Chinese, translate it to English. "
                    "If the input is English, translate it to Chinese. "
                    "Output BOTH the original text and the translation. "
                    "Do NOT converse, do NOT answer questions, and do NOT explain. "
                    "Just provide the translation."
                )},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
            max_tokens=400
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning(f"中英雙向翻譯失敗: {e}")
        return content

# ========== ✅ 解析 @ai / @機器人名 ==========
def parse_bot_mention(text: str) -> Tuple[bool, bool, str]:
    """
    ✅ 專門處理「@ai 訊息/指令」與「只打 @ai」的行為：

    回傳:
    - mentioned: 是否有明確提到 bot（@ai 或 @機器人名 或關鍵字）
    - bare_mention: 是否為「只有提到 bot，無其他內容」
        * 例如："@ai"、"ai"、"@AI醬" → bare_mention=True
        * 例如："@ai 金價"、"ai 幫我查 2330" → bare_mention=False
    - cleaned_text: 去掉 @ 部分後的剩餘指令文字（已去除常見前導標點）
        * 若 bare_mention=True 則 cleaned_text=""
    """
    if not text:
        return False, False, ""

    raw = text.strip()
    if not raw:
        return False, False, ""

    low = raw.lower()

    # 建立候選名稱清單（優先長的，避免 "ai" 吃掉 "ai bot"）
    candidates: List[str] = []

    # 1) BOT_NAME（若有設定）
    if BOT_NAME:
        candidates.append(BOT_NAME.lower())

    # 2) 其它環境變數關鍵字
    for kw in BOT_NAME_KEYWORDS:
        if kw and kw not in candidates:
            candidates.append(kw)

    # 3) 保底關鍵字 "ai" / "ＡＩ"
    if "ai" not in candidates:
        candidates.append("ai")
    if "ＡＩ".lower() not in candidates:
        candidates.append("ＡＩ".lower())

    # 依長度排序，避免短字先吃掉
    candidates.sort(key=len, reverse=True)

    best_prefix_len = -1
    best_payload = None

    for cand in candidates:
        # cand 已經是小寫
        for with_at in (True, False):
            prefix = f"@{cand}" if with_at else cand
            if low.startswith(prefix):
                l = len(prefix)
                if l > best_prefix_len:
                    best_prefix_len = l
                    # 從原字串切掉對應長度（保留原大小寫與中英文標點）
                    payload_raw = raw[len(raw) - len(low) + l:] if len(raw) != len(low) else raw[l:]
                    # 去掉前後空白與常見標點
                    payload = payload_raw.lstrip().lstrip(" ,，、:：;；").rstrip()
                    best_payload = payload

    if best_prefix_len == -1:
        # 沒偵測到任何 @ai / BOT 名稱
        return False, False, text

    if not best_payload:
        # 只有 @ai 或 @BOT_NAME，沒有其它文字
        return True, True, ""

    # 有提到 bot 且後面還有指令文字
    return True, False, best_payload

# ========== ✅ LINE Message Event ==========
@handler.add(MessageEvent, message=TextMessage)
def on_message(event: MessageEvent):
    # 依來源判斷 chat_id 與聊天型態
    if isinstance(event.source, SourceGroup):
        chat_id = event.source.group_id
        is_private = False
    elif isinstance(event.source, SourceRoom):
        chat_id = event.source.room_id
        is_private = False
    else:
        chat_id = event.source.user_id
        is_private = True

    # 初始化預設值
    ensure_defaults(chat_id, is_private)

    original_text = (event.message.text or "")
    text = original_text.strip()
    low = text.lower()

    mentioned = False
    bare_mention = False
    cmd_text = text

    # ✅ 群組 / 聊天室才需要特別解析 @ai 行為
    if not is_private:
        mentioned, bare_mention, cleaned = parse_bot_mention(text)
        if mentioned:
            cmd_text = cleaned or ""  # 有指令就用指令，沒有就空字串
        else:
            cmd_text = text  # 沒有 @ 到，就維持原文字
    else:
        # 私聊不用解析 @ai，直接用原文字
        cmd_text = text

    # ======= ✅ 自動應答模式：是否要處理這一則訊息？ =======
    if is_private:
        # 私聊：永遠處理
        should_handle = True
    else:
        ar = auto_reply_status.get(chat_id, False)
        if ar:
            # 群組 / 聊天室，自動應答 ON：處理所有訊息（cmd_text 可能已去掉 @ai）
            should_handle = True
        else:
            # 自動應答 OFF
            if mentioned:
                if bare_mention:
                    # ✅ 只有 "@ai" 或 "@機器人名"：
                    #    → 把自動應答切到 ON，並回覆 "I'm back!"，然後結束這次處理。
                    auto_reply_status[chat_id] = True
                    reply_text_audio_flex(
                        event.reply_token,
                        chat_id,
                        "I'm back!",
                        None,
                        0
                    )
                    return
                else:
                    # ✅ "@ai 訊息/指令"：
                    #    → 處理這一次指令，但 **不改變** 自動應答 ON/OFF 狀態
                    should_handle = True
                # 此時 cmd_text 已經是去掉 @ 前綴的內容
            else:
                # 沒有提到 bot，且目前自動應答 OFF → 不處理
                should_handle = False

    if not should_handle:
        return

    # 從這裡開始，一律改用「cmd_text」當作指令內容
    text = cmd_text or ""
    low = text.lower()

    # 單聊 → Loading 動畫
    if is_private:
        send_loading_animation(chat_id, seconds=3)

    try:
        # ======= ✅ 主選單 =======
        if low in ("menu", "主選單", "選單"):
            line_bot_api.reply_message(event.reply_token, flex_main(chat_id))
            return

        # ======= ✅ TTS =======
        if low in ("tts on", "tts on✅"):
            tts_enabled[chat_id] = True
            reply_text_audio_flex(event.reply_token, chat_id, "TTS 已開啟 ✅", None, 0)
            return
        if low in ("tts off", "tts off❌"):
            tts_enabled[chat_id] = False
            reply_text_audio_flex(event.reply_token, chat_id, "TTS 已關閉 ❎", None, 0)
            return

        # ======= ✅ 金價 =======
        if low in ("金價", "黃金", "黃金價格"):
            msg, _, _, _ = get_bot_gold()
            audio, dur = (tts_make_url(msg, tts_lang[chat_id]) if tts_enabled[chat_id] else (None, 0))
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # ======= ✅ 匯率 =======
        if low == "jpy":
            msg = jpy_twd()
            audio, dur = (tts_make_url(msg, tts_lang[chat_id]) if tts_enabled[chat_id] else (None, 0))
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # ======= ✅ 股票 =======
        if low in ("台股大盤", "大盤", "美股大盤", "美盤", "美股") or \
           _TW_CODE_RE.match(text.upper()) or \
           (_US_CODE_RE.match(text.upper()) and text.upper() != "JPY"):

            msg = stock_report(text)
            audio, dur = (tts_make_url(msg, tts_lang[chat_id]) if tts_enabled[chat_id] else (None, 0))
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # ======= ✅ 彩票 =======
        lottery_names = ("大樂透","威力彩","今彩539","539","雙贏彩","3星彩","4星彩","38樂合彩","39樂合彩","49樂合彩")
        if text in lottery_names:
            kind = "今彩539" if text=="539" else text

            if _EXT_LOTTERY_OK and kind in ("大樂透","威力彩","今彩539"):
                try:
                    msg = ext_lottery_gpt(kind)
                except Exception:
                    msg = lottery_report_all(kind)
            else:
                msg = lottery_report_all(kind)

            audio, dur = (tts_make_url(msg,tts_lang[chat_id]) if tts_enabled[chat_id] else (None,0))
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # ======= ✅ 自動回覆 / 自動應答開關 =======
        if text in ("開啟自動回答","關閉自動回答"):
            if text == "開啟自動回答":
                auto_reply_status[chat_id] = True
                reply_text_audio_flex(
                    event.reply_token,
                    chat_id,
                    "自動應答已開啟 ✅ 之後我會在這個聊天室主動回覆大家的訊息。",
                    None,
                    0
                )
            else:
                auto_reply_status[chat_id] = False
                reply_text_audio_flex(
                    event.reply_token,
                    chat_id,
                    "自動應答已關閉，我先退下了 🙏 有需要再 @ 我把我叫出來。",
                    None,
                    0
                )
            return

        # ======= ✅ 人設切換 =======
        if text in PERSONA_ALIAS:
            role = PERSONA_ALIAS[text]
            if role=="random":
                role = random.choice(list(PERSONAS.keys()))
            user_persona[chat_id] = role
            p = PERSONAS[role]
            reply_text_audio_flex(event.reply_token, chat_id, f"角色切換：{p['title']} {p['greet']}", None, 0)
            return

        # ======= ✅ 翻譯模式 =======
        if text.startswith("翻譯->"):
            lang = text.split("->")[1]
            if lang in ("結束","結束翻譯"):
                translation_states.pop(chat_id,None)
                reply_text_audio_flex(event.reply_token, chat_id,"已退出翻譯模式 ✅",None,0)
            else:
                translation_states[chat_id] = lang
                mode = "中↔英" if lang=="中英雙向" else f"→ {lang}"
                reply_text_audio_flex(event.reply_token, chat_id,f"翻譯模式 {mode}",None,0)
            return

        # ======= ✅ 處於翻譯模式時處理訊息 =======
        if chat_id in translation_states:
            mode = translation_states[chat_id]
            out = translate_bilingual(text) if mode=="中英雙向" else translate_text(text, mode)

            audio, dur = (tts_make_url(out,tts_lang[chat_id]) if tts_enabled[chat_id] else (None,0))
            reply_text_audio_flex(event.reply_token, chat_id, out, audio, dur)
            return

        # ======= ✅ 匹配不到任何指令 → 一般聊天（人設） =======
        persona = PERSONAS[user_persona[chat_id]]
        sys_prompt = f"你是 {persona['title']}。風格：{persona['style']}。回覆請使用繁體中文 {persona['emoji']}。"

        hist = conversation_history.get(chat_id,[])
        msgs = [{"role":"system","content":sys_prompt}] + hist + [{"role":"user","content":text}]
        out = ai_chat(msgs)

        hist += [{"role":"user","content":text},{"role":"assistant","content":out}]
        conversation_history[chat_id] = hist[-20:]

        audio, dur = (tts_make_url(out,tts_lang[chat_id]) if tts_enabled[chat_id] else (None,0))
        reply_text_audio_flex(event.reply_token, chat_id, out, audio, dur)

    except Exception as e:
        log.error(f"處理訊息錯誤: {e}")
        reply_text_audio_flex(event.reply_token, chat_id, "系統錯誤，請再試一次 🙏", None, 0)


# ========== ✅ Postback ==========
@handler.add(PostbackEvent)
def on_postback(event: PostbackEvent):
    """
    Postback 事件也要套用自動應答邏輯：
    - 私聊：照常。
    - 群組 / 聊天室：若自動應答 OFF，則不處理 Postback（避免亂入）。
    """
    if isinstance(event.source, SourceGroup):
        chat_id = event.source.group_id
        is_private = False
    elif isinstance(event.source, SourceRoom):
        chat_id = event.source.room_id
        is_private = False
    else:
        chat_id = event.source.user_id
        is_private = True

    ensure_defaults(chat_id, is_private)

    if not is_private and not auto_reply_status.get(chat_id, False):
        # 群組 / 聊天室且自動應答 OFF → 不處理 Postback
        return

    sub = (event.postback.data or "").replace("menu:","")
    line_bot_api.reply_message(
        event.reply_token,
        [
            flex_submenu(sub or "finance", chat_id),
            TextSendMessage(text="請選擇 👇", quick_reply=quick_bar(chat_id))
        ]
    )


# ========== ✅ FastAPI Routes ==========
@router.post("/callback")
async def callback(request: Request):
    sig = request.headers.get("X-Line-Signature","")
    body = (await request.body()).decode("utf-8")

    try:
        handler.handle(body, sig)
        return JSONResponse({"ok":True})
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Bad signature")
    except Exception as e:
        log.error(f"/callback 錯誤: {e}")
        raise HTTPException(status_code=500, detail="Server error")

@router.get("/")
async def index():
    return PlainTextResponse("LINE Bot Ready")

@router.get("/healthz")
async def health():
    return PlainTextResponse("ok")


# ========== ✅ Main ==========
app.include_router(router)

if __name__ == "__main__":
    port = int(os.getenv("PORT","8000"))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, reload=True)
