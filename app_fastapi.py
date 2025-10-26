# app_fastapi.py
# =============================================================================
# LINE Bot + FastAPI (金價/股票/彩票/翻譯/TTS)
# - 文字在前、音訊在中；僅「有音訊時」才會在最後附 Flex 提示卡
# - 進入翻譯模式：以 sender.name 顯示「翻譯模式（中→英）」等格式
# - 翻譯模式下，QuickReply 最右鍵由「🌐 翻譯工具」改為「結束翻譯」
# =============================================================================
# 參考（Messaging API Overview）：https://developers.line.biz/en/docs/messaging-api/overview/
# 參考（Webhook 設定）：https://developers.line.biz/en/docs/messaging-api/building-bot/#setting-webhook-url
# 參考（Icon/Nickname Switch）：https://developers.line.biz/en/docs/messaging-api/icon-nickname-switch/
# 參考（Flex 規格 / altText 必填）：https://developers.line.biz/en/docs/messaging-api/using-flex-messages/
# 參考（Quick Reply 規格）：https://developers.line.biz/en/docs/messaging-api/using-quick-reply/

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

# ========= Logging =========
# 參考（Python logging）：https://docs.python.org/3/library/logging.html
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:%(name)s:%(asctime)s:%(message)s")
log = logging.getLogger("app")

# ========= ENV =========
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")

if not BASE_URL or not CHANNEL_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError(
        "請設定環境變數：BASE_URL、CHANNEL_ACCESS_TOKEN、CHANNEL_SECRET"
    )
# 參考（LINE Console）：https://developers.line.biz/console/

# ========= LINE =========
# 參考（LINE Python SDK）：https://github.com/line/line-bot-sdk-python
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# ========= Cloudinary（可選）語音上傳 =========
# 參考（Cloudinary Upload API）：https://cloudinary.com/documentation/image_upload_api_reference
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

# ========= AI Clients（OpenAI/Groq，可選） =========
# 參考（OpenAI Chat Completions）：https://platform.openai.com/docs/api-reference/chat
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

# 參考（Groq API）：https://console.groq.com/docs
from groq import Groq
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        log.info("✅ Groq Client 初始化成功")
    except Exception as e:
        log.warning(f"Groq 初始化失敗：{e}")
GROQ_MODEL_PRIMARY = "llama-3.1-8b-instant"  # 避免 404

# ========= 狀態 =========
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"}
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"  # 台銀金價頁

conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY = 10
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}  # chat_id -> 目標語言顯示字串（英文/日文/繁體中文...）
auto_reply_status: Dict[str, bool] = {}
tts_enabled: Dict[str, bool] = {}
tts_lang: Dict[str, str] = {}  # gTTS 用語言碼

PERSONAS = {
    "sweet": {"title":"甜美女友","style":"溫柔體貼","greet":"我在這🌸","emoji":"🌸💕😊"},
    "salty": {"title":"傲嬌女友","style":"機智吐槽","greet":"你又來啦？😏","emoji":"😏🙄"},
    "moe"  : {"title":"萌系女友","style":"動漫語氣","greet":"呀呼～(ﾉ>ω<)ﾉ","emoji":"✨🎀"},
    "cool" : {"title":"酷系御姐","style":"冷靜精煉","greet":"我在。說重點。","emoji":"🧊⚡️"},
}
PERSONA_ALIAS = {"甜":"sweet","鹹":"salty","萌":"moe","酷":"cool","random":"random"}

# ========= App Lifespan =========
# 參考（FastAPI lifespan）：https://fastapi.tiangolo.com/advanced/events/
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 應用啟動")
    try:
        async with httpx.AsyncClient() as c:
            headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
            payload = {"endpoint": f"{BASE_URL}/callback"}
            r = await c.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=payload, timeout=10)
            r.raise_for_status()
            log.info("✅ Webhook 更新成功")
    except Exception as e:
        log.warning(f"⚠️ Webhook 更新失敗：{e}")
    yield
    log.info("👋 應用關閉")

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="3.9.0")
router = APIRouter()

# ========= QuickReply（依 TTS 與翻譯模式動態顯示） =========
# 只顯示必要的 TTS 切換按鈕：
# - TTS ON 中：顯示「語音 關」（點了會傳 TTS OFF）
# - TTS OFF 中：顯示「語音 開✅」（點了會傳 TTS ON）
# 翻譯模式中：最後一鍵改為「結束翻譯」，否則為「🌐 翻譯工具」
def quick_bar(chat_id: Optional[str] = None) -> QuickReply:
    # 基本功能鍵（與你原本一致）
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

    # 依目前 TTS 狀態只放「其一」按鈕
    # 說明：quick bar 是在每次回覆時重建，因此切換 TTS 後，下一則回覆就會反映最新狀態
    if chat_id and tts_enabled.get(chat_id, False):
        # 目前是開啟狀態 → 顯示「關」
        items.insert(7, QuickReplyButton(action=MessageAction(label="語音 關", text="TTS OFF")))
    else:
        # 目前是關閉狀態 → 顯示「開✅」
        items.insert(7, QuickReplyButton(action=MessageAction(label="語音 開✅", text="TTS ON")))

    # 翻譯模式：最後一鍵換成「結束翻譯」
    if chat_id and chat_id in translation_states:
        items.append(QuickReplyButton(action=MessageAction(label="結束翻譯", text="翻譯->結束")))
    else:
        items.append(QuickReplyButton(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate")))

    return QuickReply(items=items)

# ========= sender 名稱（翻譯模式顯示「翻譯模式（中→英）」） =========
# 參考（Icon/Nickname Switch）：https://developers.line.biz/en/docs/messaging-api/icon-nickname-switch/
def display_sender_name(chat_id: str) -> Tuple[str, Optional[str]]:
    if chat_id in translation_states:
        target = translation_states.get(chat_id) or ""
        mapping = {"英文": "中→英", "日文": "中→日", "繁體中文": "→ 繁中"}
        arrow = mapping.get(target, f"→ {target}") if target else ""
        name = f"翻譯模式（{arrow}）" if arrow else "翻譯模式"
        return name, None
    return "AI 助理", None

# ========= Flex（無分隔線） =========
# 參考（Flex/altText）：https://developers.line.biz/en/docs/messaging-api/using-flex-messages/
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
                TextComponent(text=hint_text, size="sm", color="#888888", wrap=True)
            ]
        )
    )
    return FlexSendMessage(alt_text=safe_alt, contents=bubble, quick_reply=quick_bar(chat_id))

# ========= 統一回覆：Text → Audio → Flex（僅在有音訊時才附 Flex） =========
# 參考（訊息共同欄位/quickReply）：https://developers.line.biz/en/reference/messaging-api/#common-properties
def reply_text_audio_flex(
    reply_token: str,
    chat_id: str,
    text: str,
    audio_url: Optional[str],
    duration_ms: int,
    hint_text: str = "（👆要聽語音請按上方播放鈕）"
):
    sender_name, sender_icon = display_sender_name(chat_id)

    msgs = []
    # 1) Text
    text_msg = TextSendMessage(text=text, quick_reply=quick_bar(chat_id))
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

        # 3) 只有有音訊時才送 Flex 提示
        flex_msg = minimal_flex_hint(
            alt_text=(text[:60] + "…") if text else "提示",
            hint_text=hint_text,
            chat_id=chat_id
        )
        flex_msg.sender = {"name": sender_name}
        if sender_icon:
            flex_msg.sender["iconUrl"] = sender_icon
        msgs.append(flex_msg)

    line_bot_api.reply_message(reply_token, msgs)

# ========= AI / 翻譯 =========
# 參考（OpenAI Chat Completions）：https://platform.openai.com/docs/api-reference/chat
# 參考（Groq Chat）：https://console.groq.com/docs
def ai_chat(messages: List[dict]) -> str:
    if openai_client:
        try:
            r = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7, max_tokens=1600
            )
            return r.choices[0].message.content
        except Exception as e:
            log.warning(f"OpenAI 失敗：{e}")

    if not groq_client:
        return "目前 AI 引擎不可用。"

    try:
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY, messages=messages, temperature=0.7, max_tokens=1800
        )
        return r.choices[0].message.content
    except Exception as e:
        log.warning(f"Groq 失敗：{e}")
        return "AI 引擎連線不穩定，請稍後再試。"

def translate_text(content: str, target_lang_display: str) -> str:
    if not groq_client:
        return "抱歉，翻譯引擎暫不可用。"
    try:
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=[
                {"role":"system","content":"You are a precise translator. Output ONLY the translated text in the requested language."},
                {"role":"user","content":f"Translate to {target_lang_display}:\n{content}"}
            ],
            temperature=0.2, max_tokens=len(content)*2+60
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning(f"翻譯失敗：{e}")
        return "抱歉，翻譯失敗。"

# ========= 股票 =========
# 參考（yfinance）：https://pypi.org/project/yfinance/
_TW_CODE_RE = re.compile(r'^\d{4,6}[A-Za-z]?$')
_US_CODE_RE = re.compile(r'^[A-Za-z]{1,5}$')

def normalize_ticker(t: str) -> Tuple[str, str]:
    t = t.strip().upper()
    if t in ("台股大盤","大盤"): return "^TWII", "^TWII"
    if t in ("美股大盤","美盤","美股"): return "^GSPC", "^GSPC"
    if _TW_CODE_RE.match(t): return f"{t}.TW", t
    return t, t

def yahoo_snapshot(symbol: str) -> dict:
    out = {"name": symbol, "now_price": None, "change": None, "currency": "", "close_time": ""}
    try:
        tk = yf.Ticker(symbol)
        info = {}
        try: info = tk.info or {}
        except Exception: pass
        hist = pd.DataFrame()
        try: hist = tk.history(period="2d", interval="1d")
        except Exception: pass

        out["name"] = info.get("shortName") or info.get("longName") or symbol
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price and not hist.empty:
            price = float(hist["Close"].iloc[-1])
        if price is not None:
            out["now_price"] = f"{price:.2f}"
            out["currency"] = info.get("currency") or ("TWD" if symbol.endswith(".TW") else "USD")
        if not hist.empty and len(hist) >= 2 and float(hist["Close"].iloc[-2]) != 0:
            chg = float(hist["Close"].iloc[-1]) - float(hist["Close"].iloc[-2])
            pct = chg / float(hist["Close"].iloc[-2]) * 100
            sign = "+" if chg >= 0 else ""
            out["change"] = f"{sign}{chg:.2f} ({sign}{pct:.2f}%)"
        if not hist.empty:
            out["close_time"] = hist.index[-1].strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        log.warning(f"yfinance 快照失敗：{e}")
    return out

def stock_report(q: str) -> str:
    code, disp = normalize_ticker(q)
    snap = yahoo_snapshot(code)
    link = f"https://finance.yahoo.com/quote/{code}" if (code.startswith("^") or not code.endswith(".TW")) else f"https://tw.stock.yahoo.com/quote/{disp}"
    sys = "你是專業分析師。分段條列：走勢/技術/基本/消息/風險/建議與區間/結論。缺資料則保守陳述。"
    user = (
        f"分析代碼：{disp}\n"
        f"名稱：{snap.get('name')}\n"
        f"價格：{snap.get('now_price')} {snap.get('currency')}\n"
        f"漲跌：{snap.get('change')}\n"
        f"時間：{snap.get('close_time')}\n"
        f"請用繁體中文分析近期走勢並附連結：{link}"
    )
    return ai_chat([{"role":"system","content":sys},{"role":"user","content":user}])

# ========= 金價（台灣銀行） =========
# 來源（台銀金價）：https://rate.bot.com.tw/gold?Lang=zh-TW
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

# ========= 匯率 =========
# 來源（ExchangeRate-API）：https://www.exchangerate-api.com/docs/free
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

# ========= 彩票（簡化：資料→AI說明） =========
# 來源（台灣彩券）：https://www.taiwanlottery.com.tw/
def lottery_text(kind: str) -> str:
    try:
        if kind == "威力彩":
            url = "https://www.taiwanlottery.com/lotto/superlotto638/index.html"
        elif kind == "大樂透":
            url = "https://www.taiwanlottery.com/lotto/lotto649/index.html"
        else:
            url = "https://www.taiwanlottery.com/lotto/dailycash/index.html"
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=12)
        r.raise_for_status()
        txt = " ".join(BeautifulSoup(r.text, "html.parser").stripped_strings)
        nums = re.findall(r"\b\d{1,2}\b", txt)
        brief = "、".join(nums[:12]) if nums else "（官網資料結構變更，僅能部分解析）"
        prompt = (
            f"{kind} 近期資料（僅做參考）：{brief}\n"
            f"請以繁體中文條列：1) 近期走勢與熱冷號 2) 合理選號建議（含風險聲明）3) 推薦 3 組號碼"
        )
        return ai_chat([{"role":"system","content":"你是資深彩券分析師。"},{"role":"user","content":prompt}])
    except Exception as e:
        log.error(f"彩票抓取失敗：{e}")
        return f"{kind} 官網讀取失敗。"

# ========= TTS =========
# 參考（gTTS）：https://pypi.org/project/gTTS/
def ensure_defaults(chat_id: str):
    if chat_id not in auto_reply_status: auto_reply_status[chat_id] = True
    if chat_id not in tts_enabled:       tts_enabled[chat_id] = False
    if chat_id not in tts_lang:          tts_lang[chat_id] = "zh-TW"
    if chat_id not in user_persona:      user_persona[chat_id] = "sweet"

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
            public_id=f"say_{int(time.time()*1000)}",
            overwrite=True
        )
        url = res.get("secure_url")
        dur = max(1000, int(len(data)/32))  # 粗估時長，避免 0ms
        return url, dur if url else (None, 0)
    except Exception as e:
        log.error(f"TTS 生成/上傳失敗：{e}")
        return None, 0

# ========= Handlers =========
@handler.add(MessageEvent, message=TextMessage)
def on_message(event: MessageEvent):
    chat_id = (
        event.source.group_id if isinstance(event.source, SourceGroup) else
        event.source.room_id  if isinstance(event.source, SourceRoom)  else
        event.source.user_id
    )
    ensure_defaults(chat_id)

    text = (event.message.text or "").strip()
    if not text:
        return

    # 群組尊重自動回覆開關
    should = isinstance(event.source, SourceUser) or auto_reply_status.get(chat_id, True)
    if not should:
        return

    low = text.lower()

    try:
        # 主選單
        if low in ("menu","選單","主選單"):
            line_bot_api.reply_message(event.reply_token, flex_main(chat_id))
            return

        # TTS 切換
        if low in ("tts on","tts on✅"):
            tts_enabled[chat_id] = True
            reply_text_audio_flex(event.reply_token, chat_id, "已開啟語音播報 ✅", None, 0)
            return
        if low in ("tts off","tts off❌","tts off✖"):
            tts_enabled[chat_id] = False
            reply_text_audio_flex(event.reply_token, chat_id, "已關閉語音播報", None, 0)
            return

        # 金價
        if low in ("金價","黃金","黃金價格"):
            msg, sell, buy, ts = get_bot_gold()
            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                audio, dur = tts_make_url(msg, tts_lang[chat_id])
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # 匯率
        if low == "jpy":
            msg = jpy_twd()
            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                audio, dur = tts_make_url(msg, tts_lang[chat_id])
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # 股票
        if low in ("台股大盤","大盤","美股大盤","美盤","美股") or _TW_CODE_RE.match(text.upper()) or (_US_CODE_RE.match(text.upper()) and text.upper() != "JPY"):
            msg = stock_report(text)
            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                audio, dur = tts_make_url(msg, tts_lang[chat_id])
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # 彩票
        if text in ("大樂透","威力彩","今彩539","539"):
            kind = "威力彩" if "威力" in text else ("大樂透" if "樂" in text else "今彩539")
            msg = lottery_text(kind)
            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                audio, dur = tts_make_url(msg, tts_lang[chat_id])
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # 自動回覆開關
        if text in ("開啟自動回答","關閉自動回答"):
            auto_reply_status[chat_id] = (text == "開啟自動回答")
            reply_text_audio_flex(event.reply_token, chat_id, f"自動回答：{'開啟' if auto_reply_status[chat_id] else '關閉'}", None, 0)
            return

        # 人設
        if text in PERSONA_ALIAS:
            key = PERSONA_ALIAS[text]
            if key == "random":
                key = random.choice(list(PERSONAS.keys()))
            user_persona[chat_id] = key
            p = PERSONAS[key]
            reply_text_audio_flex(event.reply_token, chat_id, f"💖 角色切換：{p['title']}\n{p['greet']}", None, 0)
            return

        # 翻譯模式開關
        if text.startswith("翻譯->"):
            lang = text.split("->",1)[1]
            if lang in ("結束","結束翻譯"):
                translation_states.pop(chat_id, None)
                reply_text_audio_flex(event.reply_token, chat_id, "✅ 已結束翻譯模式", None, 0)
            else:
                translation_states[chat_id] = lang
                reply_text_audio_flex(event.reply_token, chat_id, f"🈯 已開啟翻譯模式（→ {lang}）", None, 0)
            return

        # 翻譯內容
        if chat_id in translation_states:
            out = translate_text(text, translation_states[chat_id])
            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                audio, dur = tts_make_url(out, tts_lang[chat_id])
            reply_text_audio_flex(event.reply_token, chat_id, out, audio, dur)
            return

        # 一般聊天（帶人設）
        key = user_persona.get(chat_id, "sweet")
        p = PERSONAS[key]
        sys = f"你是「{p['title']}」。風格：{p['style']}。用繁體中文，自然精煉，適量表情 {p['emoji']}。"
        hist = conversation_history.get(chat_id, [])
        msgs = [{"role":"system","content":sys}] + hist + [{"role":"user","content":text}]
        out = ai_chat(msgs)
        hist.extend([{"role":"user","content":text},{"role":"assistant","content":out}])
        conversation_history[chat_id] = hist[-MAX_HISTORY*2:]

        audio, dur = (None, 0)
        if tts_enabled[chat_id]:
            audio, dur = tts_make_url(out, tts_lang[chat_id])
        reply_text_audio_flex(event.reply_token, chat_id, out, audio, dur)

    except LineBotApiError as e:
        log.error(f"LINE 回覆失敗：{e}")
    except Exception as e:
        log.error(f"處理訊息錯誤：{e}", exc_info=True)
        try:
            reply_text_audio_flex(event.reply_token, chat_id, "😵‍💫 發生錯誤，請稍後再試。", None, 0)
        except Exception:
            pass

@handler.add(PostbackEvent)
def on_postback(event: PostbackEvent):
    data = (event.postback.data or "")
    sub = data[5:] if data.startswith("menu:") else ""
    chat_id = (
        event.source.group_id if isinstance(event.source, SourceGroup) else
        event.source.room_id  if isinstance(event.source, SourceRoom)  else
        event.source.user_id
    )
    try:
        line_bot_api.reply_message(
            event.reply_token,
            [flex_submenu(sub or "finance", chat_id), TextSendMessage(text="請選擇 👇", quick_reply=quick_bar(chat_id))]
        )
    except Exception as e:
        log.error(f"Postback 失敗：{e}")

# ========= Menu Flex =========
def flex_main(chat_id: Optional[str] = None) -> FlexSendMessage:
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[TextComponent(text="AI 助理主選單", weight="bold", size="lg")]),
        body=BoxComponent(
            layout="vertical", spacing="md",
            contents=[
                TextComponent(text="請選擇功能：", size="sm"),
                ButtonComponent(action=PostbackAction(label="💹 金融查詢", data="menu:finance"), style="primary", color="#5E86C1"),
                ButtonComponent(action=PostbackAction(label="🎰 彩票分析", data="menu:lottery"), style="primary", color="#5EC186"),
                ButtonComponent(action=PostbackAction(label="💖 AI 角色", data="menu:persona"), style="secondary"),
                ButtonComponent(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate"), style="secondary"),
                ButtonComponent(action=PostbackAction(label="⚙️ 系統設定", data="menu:settings"), style="secondary"),
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
            ButtonComponent(action=MessageAction(label="今彩539", text="539")),
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
        header=BoxComponent(layout="vertical", contents=[TextComponent(text=title, weight="bold", size="lg")]),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm")
    )
    return FlexSendMessage(alt_text=title, contents=bubble, quick_reply=quick_bar(chat_id))

# ========= Routes =========
@router.post("/callback")
async def callback(request: Request):
    sig = request.headers.get("X-Line-Signature","")
    body = (await request.body()).decode("utf-8")
    try:
        handler.handle(body, sig)
        return JSONResponse({"status":"ok"})
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        log.error(f"/callback 失敗：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="internal error")

@router.get("/")
async def root():
    return PlainTextResponse("LINE Bot running.", status_code=200)

@router.get("/healthz")
async def health():
    return PlainTextResponse("ok")

app.include_router(router)

# ========= Local run =========
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)