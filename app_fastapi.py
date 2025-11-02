# app_fastapi.py
# =============================================================================
# LINE Bot + FastAPI (金價 / 股票 / 彩票(含所有彩種) / 翻譯 / TTS / 單聊 Loading 動畫)
# -----------------------------------------------------------------------------
# 功能重點：
# - 支援所有遊戲彩種：大樂透 / 威力彩 / 今彩539 / 雙贏彩 / 3星彩 / 4星彩 / 38樂合彩 / 39樂合彩 / 49樂合彩（來源：TaiwanLotteryCrawler）  [oai_citation:0‡GitHub](https://github.com/stu01509/TaiwanLotteryCrawler?utm_source=chatgpt.com)
# - 同時保留你原有的 my_commands/lottery_gpt.py 模組做部分彩種分析
# - 其餘功能維持你原本架構
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

# === 導入 TaiwanLotteryCrawler 庫 ===
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    _LT_CRAWLER_OK = True
    logging.info("✅ TaiwanLotteryCrawler 模組載入成功")
except Exception as e:
    _LT_CRAWLER_OK = False
    logging.warning(f"⚠️ TaiwanLotteryCrawler 載入失敗：{e}")

# === 導入你原有的分析模組 my_commands/lottery_gpt.py ===
try:
    from my_commands.lottery_gpt import lottery_gpt as ext_lottery_gpt
    _EXT_LOTTERY_OK = True
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
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")  # e.g. https://api.openai.com/v1 或自建代理

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

# 強制採用當前可用的 Groq 模型（避免 404 / decommission）
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
translation_states: Dict[str, str] = {}
auto_reply_status: Dict[str, bool] = {}
tts_enabled: Dict[str, bool] = {}
tts_lang: Dict[str, str] = {}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼", "greet": "我在這🌸", "emoji": "🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽", "greet": "你又來啦？😏", "emoji": "😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣", "greet": "呀呼～(ﾉ>ω<)ﾉ", "emoji": "✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉", "greet": "我在。說重點。", "emoji": "🧊⚡️"},
}
PERSONA_ALIAS = {"甜": "sweet", "鹹": "salty", "萌": "moe", "酷": "cool", "random": "random"}

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

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="4.0.0")
router = APIRouter()

# ========= Loading 動畫（僅單人聊天有效）=========
def send_loading_animation(user_id: str, seconds: int = 5):
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
        log.warning(f"⚠️ Loading 動畫觸發失敗：{e}")

# ========= QuickReply（依 TTS 與翻譯模式動態顯示）=========
def quick_bar(chat_id: Optional[str] = None) -> QuickReply:
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

    if chat_id and tts_enabled.get(chat_id, False):
        items.insert(7, QuickReplyButton(action=MessageAction(label="語音 關", text="TTS OFF")))
    else:
        items.insert(7, QuickReplyButton(action=MessageAction(label="語音 開✅", text="TTS ON")))

    if chat_id and chat_id in translation_states:
        items.append(QuickReplyButton(action=MessageAction(label="結束翻譯", text="翻譯->結束")))
    else:
        items.append(QuickReplyButton(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate")))

    return QuickReply(items=items)

# ========= sender.name（翻譯模式顯示「翻譯模式（中→英/中↔英）」）=========
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
    sender_name, sender_icon = display_sender_name(chat_id)
    msgs = []
    text_msg = TextSendMessage(text=text, quick_reply=quick_bar(chat_id))
    text_msg.sender = {"name": sender_name}
    if sender_icon:
        text_msg.sender["iconUrl"] = sender_icon
    msgs.append(text_msg)

    if audio_url:
        audio_msg = AudioSendMessage(original_content_url=audio_url, duration=duration_ms)
        audio_msg.sender = {"name": sender_name}
        if sender_icon:
            audio_msg.sender["iconUrl"] = sender_icon
        msgs.append(audio_msg)

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
def ai_chat(messages: List[dict]) -> str:
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
    if not groq_client:
        return "抱歉，翻譯引擎暫不可用。"
    try:
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=[
                {"role": "system", "content": "You are a precise translator. Output ONLY the translated text."},
                {"role": "user", "content": f"Translate to {target_lang_display}:\n{content}"}
            ],
            temperature=0.2,
            max_tokens=len(content) * 2 + 60
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning(f"翻譯失敗：{e}")
        return "抱歉，翻譯失敗。"

def translate_bilingual(content: str) -> str:
    if not groq_client:
        return "抱歉，翻譯引擎暫不可用。"
    try:
        sys_prompt = (
            "You are a bilingual translator for Traditional Chinese and English.\n"
            "Rules:\n"
            "1) Detect the main language of the input.\n"
            "2) If input is mainly Traditional Chinese, translate to natural English.\n"
            "3) If input is mainly English, translate to natural Traditional Chinese.\n"
            "4) Keep formatting; preserve numbers, symbols, inline code, and code blocks.\n"
            "5) Output ONLY the translation text."
        )
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": content}
            ],
            temperature=0.2,
            max_tokens=len(content) * 2 + 120
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning(f"雙向翻譯失敗：{e}")
        return "抱歉，雙向翻譯失敗。"

# ========= 股票 =========
_TW_CODE_RE = re.compile(r'^\d{4,6}[A-Za-z]?$')
_US_CODE_RE = re.compile(r'^[A-Za-z]{1,5}$')

def normalize_ticker(t: str) -> Tuple[str, str]:
    t = t.strip().upper()
    if t in ("台股大盤", "大盤"):
        return "^TWII", "^TWII"
    if t in ("美股大盤", "美盤", "美股"):
        return "^GSPC", "^GSPC"
    if _TW_CODE_RE.match(t):
        return f"{t}.TW", t
    return t, t

def yahoo_snapshot(symbol: str) -> dict:
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
    link = (
        f"https://finance.yahoo.com/quote/{code}"
        if (code.startswith("^") or not code.endswith(".TW"))
        else f"https://tw.stock.yahoo.com/quote/{disp}"
    )
    sys_prompt = "你是專業分析師。分段條列：走勢/技術/基本/消息/風險/建議與區間/結論。缺資料則保守陳述。"
    user_prompt = (
        f"分析代碼：{disp}\n"
        f"名稱：{snap.get('name')}\n"
        f"價格：{snap.get('now_price')} {snap.get('currency')}\n"
        f"漲跌：{snap.get('change')}\n"
        f"時間：{snap.get('close_time')}\n"
        f"請用繁體中文分析近期走勢並附連結：{link}"
    )
    return ai_chat([{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}])

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

# ========= 彩票（全彩種支援）=========
def lottery_report_all(kind: str) -> str:
    """
    支援所有彩種：
    - 大樂透：lotto649()
    - 威力彩：super_lotto()
    - 今彩539：daily_cash()
    - 雙贏彩：lotto1224()
    - 3星彩：lotto3d()
    - 4星彩：lotto4d()
    - 38樂合彩：lotto38m6()
    - 39樂合彩：lotto39m5()
    - 49樂合彩：lotto49m6()
    使用 TaiwanLotteryCrawler 擷取資料；若失敗則回 fallback 隨機建議。  [oai_citation:1‡GitHub](https://github.com/stu01509/TaiwanLotteryCrawler?utm_source=chatgpt.com)
    """
    if not _LT_CRAWLER_OK:
        return f"📌 {kind} 分析報告：\n資料來源暫不可用，請稍後再試。"

    lottery = TaiwanLotteryCrawler()
    try:
        kind_map = {
            "大樂透": ("lotto649", 6, 49),
            "威力彩": ("super_lotto", 6, 39),
            "今彩539": ("daily_cash", 5, 39),
            "雙贏彩": ("lotto1224", 6, 49),
            "3星彩": ("lotto3d", 3, 10),
            "4星彩": ("lotto4d", 4, 10),
            "38樂合彩": ("lotto38m6", 6, 38),
            "39樂合彩": ("lotto39m5", 5, 39),
            "49樂合彩": ("lotto49m6", 6, 49),
        }
        if kind not in kind_map:
            return f"📌 {kind} 分析報告：\n目前未支援此彩種，請輸入以上支援名稱。"

        func_name, num_main, max_num = kind_map[kind]
        func = getattr(lottery, func_name)
        result = func()
        latest = result[0] if isinstance(result, list) and result else None
        if not latest:
            raise RuntimeError("未取得開獎資料")

        draw_date = getattr(latest, "draw_date", None)
        numbers = getattr(latest, "numbers", None) or getattr(latest, "number", None)

        if draw_date:
            draw_date = draw_date.strftime("%Y/%m/%d")
        else:
            draw_date = "—"

        if isinstance(numbers, (list, tuple)):
            numbers_str = ", ".join(f"{n:02d}" for n in numbers)
        else:
            numbers_str = str(numbers)

        suggest = sorted(random.sample(range(1, max_num+1), num_main))
        suggest_str = ", ".join(f"{n:02d}" for n in suggest)

        analysis = f"{kind}：近期開獎號碼動態且猜測難度高，建議理性娛樂。"

        return (
            f"**{kind} 分析報告**\n\n"
            f"📅 最新開獎（{draw_date}）：{numbers_str}\n\n"
            f"🎯 下期建議：{suggest_str}\n\n"
            f"💡 分析：{analysis}\n\n"
            f"[官方歷史開獎查詢](https://www.taiwanlottery.com.tw/)"
        )

    except Exception as e:
        log.error(f"{kind} 擷取失敗：{e}")
        rnd = sorted(random.sample(range(1, max_num+1), num_main))
        rnd_str = ", ".join(f"{n:02d}" for n in rnd)
        return (
            f"**{kind} 分析報告**\n\n"
            f"📅 最新開獎：資料取得失敗（顯示隨機）\n\n"
            f"🎯 下期建議：{rnd_str}\n\n"
            f"💡 分析：資料來源暫時異常，請稍後再試。\n\n"
            f"[官方歷史開獎查詢](https://www.taiwanlottery.com.tw/)"
        )

# ========= 路由／事件處理：MessageEvent =========
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

    should = isinstance(event.source, SourceUser) or auto_reply_status.get(chat_id, True)
    if not should:
        return

    if isinstance(event.source, SourceUser):
        send_loading_animation(chat_id, seconds=4)

    low = text.lower()

    try:
        # 主選單
        if low in ("menu", "選單", "主選單"):
            line_bot_api.reply_message(event.reply_token, flex_main(chat_id))
            return

        # TTS 切換
        if low in ("tts on", "tts on✅"):
            tts_enabled[chat_id] = True
            reply_text_audio_flex(event.reply_token, chat_id, "已開啟語音播報 ✅", None, 0)
            return
        if low in ("tts off", "tts off❌", "tts off✖"):
            tts_enabled[chat_id] = False
            reply_text_audio_flex(event.reply_token, chat_id, "已關閉語音播報", None, 0)
            return

        # 金價查詢
        if low in ("金價", "黃金", "黃金價格"):
            msg, sell, buy, ts = get_bot_gold()
            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                audio, dur = tts_make_url(msg, tts_lang[chat_id])
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # 匯率 JPY
        if low == "jpy":
            msg = jpy_twd()
            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                audio, dur = tts_make_url(msg, tts_lang[chat_id])
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # 股票查詢
        if low in ("台股大盤", "大盤", "美股大盤", "美盤", "美股") \
           or _TW_CODE_RE.match(text.upper()) \
           or (_US_CODE_RE.match(text.upper()) and text.upper() != "JPY"):
            msg = stock_report(text)
            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                audio, dur = tts_make_url(msg, tts_lang[chat_id])
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # 彩票觸發（支援所有彩種）
        lottery_names = ("大樂透", "威力彩", "今彩539", "539", "雙贏彩", "3星彩", "4星彩", "38樂合彩", "39樂合彩", "49樂合彩")
        if text in lottery_names:
            mapping = {
                "539": "今彩539"
            }
            kind = mapping.get(text, text)
            # 若 ext_lottery_gpt 支援該彩種且你希望優先使用：
            if _EXT_LOTTERY_OK and kind in ("大樂透", "威力彩", "今彩539"):
                try:
                    msg = ext_lottery_gpt(kind)
                except Exception as e:
                    log.warning(f"外掛分析模組失敗：{e}")
                    msg = lottery_report_all(kind)
            else:
                msg = lottery_report_all(kind)

            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                audio, dur = tts_make_url(msg, tts_lang[chat_id])
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # 自動回覆開關
        if text in ("開啟自動回答", "關閉自動回答"):
            auto_reply_status[chat_id] = (text == "開啟自動回答")
            reply_text_audio_flex(
                event.reply_token, chat_id,
                f"自動回答：{'開啟' if auto_reply_status[chat_id] else '關閉'}",
                None, 0
            )
            return

        # 人設切換
        if text in PERSONA_ALIAS:
            key = PERSONA_ALIAS[text]
            if key == "random":
                key = random.choice(list(PERSONAS.keys()))
            user_persona[chat_id] = key
            p = PERSONAS[key]
            reply_text_audio_flex(
                event.reply_token, chat_id,
                f"💖 角色切換：{p['title']}\n{p['greet']}",
                None, 0
            )
            return

        # 翻譯模式切換
        if text.startswith("翻譯->"):
            lang = text.split("->", 1)[1]
            if lang in ("結束", "結束翻譯"):
                translation_states.pop(chat_id, None)
                reply_text_audio_flex(event.reply_token, chat_id, "✅ 已結束翻譯模式", None, 0)
            else:
                if lang in ("英文", "日文", "繁體中文", "中英雙向"):
                    translation_states[chat_id] = lang
                    label = "中↔英" if lang == "中英雙向" else f"→ {lang}"
                    reply_text_audio_flex(event.reply_token, chat_id, f"🈯 已開啟翻譯模式（{label}）", None, 0)
                else:
                    reply_text_audio_flex(event.reply_token, chat_id, "未支援的翻譯目標。", None, 0)
            return

        # 翻譯模式內容
        if chat_id in translation_states:
            mode = translation_states[chat_id]
            if mode == "中英雙向":
                out = translate_bilingual(text)
            else:
                out = translate_text(text, mode)
            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                lang_code = tts_lang[chat_id]
                if mode == "中英雙向":
                    ascii_ratio = sum(1 for ch in out if ord(ch) < 128) / max(1, len(out))
                    lang_code = "en" if ascii_ratio > 0.6 else "zh-TW"
                audio, dur = tts_make_url(out, lang_code)
            reply_text_audio_flex(event.reply_token, chat_id, out, audio, dur)
            return

        # 一般聊天（帶人設）
        key = user_persona.get(chat_id, "sweet")
        p = PERSONAS[key]
        sys_prompt = f"你是「{p['title']}」。風格：{p['style']}。用繁體中文，自然精煉，適量表情 {p['emoji']}。"
        hist = conversation_history.get(chat_id, [])
        msgs = [{"role": "system", "content": sys_prompt}] + hist + [{"role": "user", "content": text}]
        out = ai_chat(msgs)
        hist.extend([{"role": "user", "content": text}, {"role": "assistant", "content": out}])
        conversation_history[chat_id] = hist[-MAX_HISTORY * 2:]

        audio, dur = (None, 0)
        if tts_enabled[chat_id]:
            audio, dur = tts_make_url(out, tts_lang[chat_id])
        reply_text_audio_flex(event.reply_token, chat_id, out, audio, dur)

    except LineBotApiError as e:
        log.error(f"LINE 回覆失敗：{e}")
        try:
            reply_text_audio_flex(event.reply_token, chat_id, "⚠️ LINE 回覆失敗，請稍後再試。", None, 0)
        except Exception:
            pass
    except Exception as e:
        log.error(f"處理訊息錯誤：{e}", exc_info=True)
        try:
            reply_text_audio_flex(event.reply_token, chat_id, "😵‍💫 發生錯誤，請稍後再試。", None, 0)
        except Exception:
            pass

# ========= 事件處理：PostbackEvent =========
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

@router.post("/callback")
async def callback(request: Request):
    sig = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode("utf-8")
    try:
        handler.handle(body, sig)
        return JSONResponse({"status": "ok"})
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
    return PlainTextResponse("ok", status_code=200)

app.include_router(router)

# ========= Local run =========
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)