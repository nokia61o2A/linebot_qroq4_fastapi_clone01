# app_fastapi.py
# Version: 2.2.1 (Always-Visible Quick Reply)
# 變更重點：
# - 保證每次回覆 Quick Reply 永遠顯示：最後物件若為 Flex/Audio 仍補一個空白 Text(帶 QR)
# - 仍保留：翻譯模式 Sender 顯示「翻譯模式(中->英)」、TTS(Cloudinary 上傳)、台銀金價實抓、
#          TaiwanLottery + 官網 fallback、股票/外匯/聊天等

import os, re, io, sys, random, logging, asyncio
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
MC_DIR = os.path.join(BASE_DIR, "my_commands")
if MC_DIR not in sys.path: sys.path.append(MC_DIR)

import requests, httpx
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent, AudioMessageContent, PostbackEvent
from linebot.v3.webhook import WebhookParser
from linebot.v3.messaging import (
    Configuration, ApiClient, AsyncMessagingApi, ReplyMessageRequest,
    TextMessage, AudioMessage, FlexMessage, FlexBubble, FlexBox,
    FlexText, FlexButton, QuickReply, QuickReplyItem, MessageAction, PostbackAction,
    Sender,
)

# Cloudinary
CLOUDINARY_AVAILABLE = False
CLOUDINARY_CONFIGURED = False
cloudinary = None
cloudinary_uploader = None
if 'CLOUDINARY_URL' in os.environ:
    try:
        import cloudinary, cloudinary.uploader
        CLOUDINARY_AVAILABLE = True
        cloudinary = cloudinary
        cloudinary_uploader = cloudinary.uploader
    except ImportError:
        CLOUDINARY_AVAILABLE = False

# gTTS
GTTS_AVAILABLE = False
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

from groq import AsyncGroq, Groq
import openai

# Lottery
LOTTERY_OK = False
LOTTERY_IMPORT_ERR = ""
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    lottery_crawler = TaiwanLotteryCrawler()
    LOTTERY_OK = True
except Exception as e:
    LOTTERY_OK = False
    LOTTERY_IMPORT_ERR = f"{e.__class__.__name__}: {e}"
    lottery_crawler = None

# Stock modules
try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
    STOCK_OK = True
except Exception as e:
    logging.warning(f"股票模組載入失敗：{e}")
    def stock_price(s): return pd.DataFrame()
    def stock_news(s): return ["（股票新聞模組未載入）"]
    def stock_fundamental(s): return "（股票基本面模組未載入）"
    def stock_dividend(s): return "（股票股利模組未載入）"
    class YahooStock:
        def __init__(self, s):
            self.name = s; self.now_price=None; self.change=None; self.currency=None; self.close_time=None
    STOCK_OK = False

logger = logging.getLogger("uvicorn.error"); logger.setLevel(logging.INFO)
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto").lower()
TTS_SEND_ALWAYS = os.getenv("TTS_SEND_ALWAYS", "true").lower() == "true"
logger.info(f"ENV: BASE_URL={bool(BASE_URL)}, TTS_PROVIDER={TTS_PROVIDER}, TTS_DEFAULT={TTS_SEND_ALWAYS}, Cloudinary={bool(CLOUDINARY_URL)}")

if not CHANNEL_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("缺少必要環境變數：CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET")

if CLOUDINARY_URL and CLOUDINARY_AVAILABLE:
    try:
        import re as _re
        cloudinary.config(
            cloud_name=_re.search(r"@(.+)", CLOUDINARY_URL).group(1),
            api_key=_re.search(r"//(\d+):", CLOUDINARY_URL).group(1),
            api_secret=_re.search(r":([A-Za-z0-9_-]+)@", CLOUDINARY_URL).group(1),
        )
        CLOUDINARY_CONFIGURED = True
        logger.info("Cloudinary 配置成功")
    except Exception as e:
        logger.error(f"Cloudinary 設定失敗: {e}")
        CLOUDINARY_CONFIGURED = False

configuration = Configuration(access_token=CHANNEL_TOKEN)
api_client = ApiClient(configuration=configuration)
line_bot_api = AsyncMessagingApi(api_client=api_client)
parser = WebhookParser(CHANNEL_SECRET)

sync_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI 客戶端初始化成功")
    except Exception as e:
        logger.warning(f"初始化 OpenAI 失敗：{e}")

GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
translation_states: Dict[str, str] = {}
translation_states_ttl: Dict[str, datetime] = {}
TRANSLATE_TTL_SECONDS = int(os.getenv("TRANSLATE_TTL_SECONDS", "7200"))
user_persona: Dict[str, str] = {}
tts_switch_per_chat: Dict[str, bool] = {}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji": "🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji": "😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji": "✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議", "greetings": "我在。說重點。", "emoji": "🧊⚡️"},
}
LANGUAGE_MAP = {
    "英文": "English","日文": "Japanese","韓文": "Korean","越南文":"Vietnamese",
    "繁體中文":"Traditional Chinese","中文":"Traditional Chinese",
    "english":"English","japanese":"Japanese","korean":"Korean","vietnamese":"Vietnamese"
}
PERSONA_ALIAS = {"甜":"sweet","鹹":"salty","萌":"moe","酷":"cool","random":"random"}

TRANSLATE_CMD = re.compile(
    r"^(?:翻譯|翻成)\s*(?:->|→|>)?\s*(英文|English|日文|Japanese|韓文|Korean|越南文|Vietnamese|繁體中文|中文)\s*$",
    re.IGNORECASE
)
INLINE_TRANSLATE = re.compile(r"^(en|eng|英文|ja|jp|日文|zh|繁中|中文)\s*[:：>]\s*(.+)$", re.IGNORECASE)

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"

def _now() -> datetime: return datetime.utcnow()

def _tstate_set(chat_id: str, lang_display: str):
    translation_states[chat_id] = lang_display
    translation_states_ttl[chat_id] = _now() + timedelta(seconds=TRANSLATE_TTL_SECONDS)

def _tstate_get(chat_id: str) -> Optional[str]:
    exp = translation_states_ttl.get(chat_id)
    if exp and _now() > exp:
        _tstate_clear(chat_id); return None
    return translation_states.get(chat_id)

def _tstate_clear(chat_id: str):
    translation_states.pop(chat_id, None)
    translation_states_ttl.pop(chat_id, None)

def get_chat_id(event: MessageEvent) -> str:
    src = event.source
    uid = getattr(src, "userId", None) or getattr(src, "user_id", None)
    gid = getattr(src, "groupId", None) or getattr(src, "group_id", None)
    rid = getattr(src, "roomId", None) or getattr(src, "room_id", None)
    if gid: return f"group:{gid}"
    if rid: return f"room:{rid}"
    if uid: return f"user:{uid}"
    return f"unknown:{abs(hash(str(src)))%10_000_000}"

_LANG_PAIR_LABEL = {
    "英文": "中->英", "English": "中->英",
    "日文": "中->日", "Japanese": "中->日",
    "韓文": "中->韓", "Korean": "中->韓",
    "越南文": "中->越", "Vietnamese": "中->越",
    "繁體中文": "任->中", "中文": "任->中",
}
def _build_translation_sender(chat_id: str) -> Optional[Sender]:
    tgt = translation_states.get(chat_id)
    if not tgt: return None
    label = _LANG_PAIR_LABEL.get(tgt, f"中->{tgt}")
    return Sender(name=f"翻譯模式({label})")

def build_quick_reply(chat_id: Optional[str]=None) -> QuickReply:
    tts_on = tts_switch_per_chat.get(chat_id, TTS_SEND_ALWAYS)
    on_label  = "TTS ON✅" if tts_on else "TTS ON"
    off_label = "TTS OFF" if tts_on else "TTS OFF✅"
    return QuickReply(items=[
        QuickReplyItem(action=MessageAction(label="主選單", text="選單")),
        QuickReplyItem(action=MessageAction(label="台股大盤", text="大盤")),
        QuickReplyItem(action=MessageAction(label="美股大盤", text="美盤")),
        QuickReplyItem(action=MessageAction(label="黃金價格", text="金價")),
        QuickReplyItem(action=MessageAction(label="查 2330", text="2330")),
        QuickReplyItem(action=MessageAction(label="查 NVDA", text="NVDA")),
        QuickReplyItem(action=MessageAction(label="日圓匯率", text="JPY")),
        QuickReplyItem(action=MessageAction(label=on_label, text="TTS ON")),
        QuickReplyItem(action=MessageAction(label=off_label, text="TTS OFF")),
        QuickReplyItem(action=PostbackAction(label="💖 AI 人設", data="menu:persona")),
        QuickReplyItem(action=PostbackAction(label="🎰 彩票選單", data="menu:lottery")),
        QuickReplyItem(action=MessageAction(label="結束翻譯", text="翻譯->結束")),
    ])

def build_main_menu() -> FlexMessage:
    buttons = [
        FlexButton(action=PostbackAction(label="💹 金融查詢", data="menu:finance"), style="primary"),
        FlexButton(action=PostbackAction(label="🎰 彩票分析", data="menu:lottery"), style="primary"),
        FlexButton(action=PostbackAction(label="💖 AI 角色扮演", data="menu:persona"), style="secondary"),
        FlexButton(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate"), style="secondary"),
    ]
    bubble = FlexBubble(
        header=FlexBox(layout="vertical", contents=[FlexText(text="AI 助理主選單", weight="bold", size="lg")]),
        body=FlexBox(layout="vertical", spacing="md", contents=buttons),
    )
    return FlexMessage(alt_text="主選單", contents=bubble)

def build_submenu(kind: str) -> FlexMessage:
    menus = {
        "finance": ("💹 金融查詢", [
            ("台股大盤", "大盤"), ("美股大盤", "美盤"), ("黃金價格", "金價"),
            ("日圓匯率", "JPY"), ("查 2330 台積", "2330"), ("查 NVDA 輝達", "NVDA"),
        ]),
        "lottery": ("🎰 彩票分析", [("大樂透","大樂透"),("威力彩","威力彩"),("今彩539","今彩539")]),
        "persona": ("💖 AI 角色扮演", [("甜美女友","甜"),("傲嬌女友","鹹"),("萌系女友","萌"),("酷系御姐","酷"),("隨機切換","random")]),
        "translate": ("🌐 翻譯工具", [("翻成英文","翻譯->英文"),("翻成日文","翻譯->日文"),("翻成繁中","翻譯->繁體中文"),("結束翻譯模式","翻譯->結束")]),
    }
    title, items = menus.get(kind, ("無效選單", []))
    rows=[]
    for i in range(0, len(items), 2):
        pair = items[i:i+2]
        row = [FlexButton(action=MessageAction(label=lbl, text=txt), style="primary") for (lbl,txt) in pair]
        rows.append(FlexBox(layout="horizontal", spacing="sm", contents=row))
    bubble = FlexBubble(
        header=FlexBox(layout="vertical", contents=[FlexText(text=title, weight="bold", size="lg")]),
        body=FlexBox(layout="vertical", spacing="md", contents=rows or [FlexText(text="（尚無項目）")]),
    )
    return FlexMessage(alt_text=title, contents=bubble)

# ---------- TTS / STT ----------
async def _stt_openai(audio_bytes: bytes, filename="audio.m4a") -> Optional[str]:
    if not openai_client: return None
    try:
        f = io.BytesIO(audio_bytes); f.name = filename
        resp = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
        return (resp.text or "").strip() or None
    except Exception as e:
        logger.warning(f"OpenAI STT 失敗：{e}")
        return None

async def speech_to_text_async(audio_bytes: bytes) -> Optional[str]:
    return await _stt_openai(audio_bytes)

def _tts_openai(text: str) -> Optional[bytes]:
    if not openai_client: return None
    try:
        clean = re.sub(r"[*_`~#]", "", text) or "內容為空"
        resp = openai_client.audio.speech.create(model="tts-1", voice="nova", input=clean)
        return resp.read()
    except Exception as e:
        logger.error(f"OpenAI TTS 失敗: {e}")
        return None

def _tts_gtts(text: str) -> Optional[bytes]:
    if not GTTS_AVAILABLE: return None
    try:
        clean = re.sub(r"[*_`~#]", "", text).strip() or "嗨，我在這裡。"
        tts = gTTS(text=clean, lang="zh-TW", tld="com.tw", slow=False)
        buf = io.BytesIO(); tts.write_to_fp(buf); buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"gTTS 失敗: {e}")
        return None

async def text_to_speech_async(text: str) -> Optional[bytes]:
    clean_text = re.sub(r"[*_`~#]", "", text).strip() or "內容為空"
    if TTS_PROVIDER == "openai":
        return (await run_in_threadpool(_tts_openai, clean_text)) or (await run_in_threadpool(_tts_gtts, clean_text))
    elif TTS_PROVIDER == "gtts":
        return await run_in_threadpool(_tts_gtts, clean_text)
    else:
        return (await run_in_threadpool(_tts_openai, clean_text)) or (await run_in_threadpool(_tts_gtts, clean_text))

# ---------- Quick Reply 保證顯示 ----------
def _ensure_qr_visible(messages: List, chat_id: Optional[str], sender: Optional[Sender]):
    """
    保證最後一個訊息也帶 Quick Reply；若客戶端對 Audio/Flex 不顯示 QR，就補一個空白 Text。
    """
    if not messages:
        return

    qr = build_quick_reply(chat_id)

    # 盡量把 QR 設在最後一個物件上
    last = messages[-1]
    try:
        # v3 SDK 的物件都接受 quick_reply
        if hasattr(last, "quick_reply") and getattr(last, "quick_reply", None) is None:
            last.quick_reply = qr  # type: ignore
            return
    except Exception:
        pass

    # 為安全起見，再補一個極短文字（空白）帶 QR，確保看得到
    messages.append(TextMessage(text=" ", quick_reply=qr, sender=sender))

# ---------- 統一回覆 ----------
async def reply_text_with_tts_and_extras(reply_token: str, text: str, extras: Optional[List]=None, chat_id: Optional[str]=None):
    if not text: text = "（無內容）"
    sender = _build_translation_sender(chat_id) if chat_id else None

    messages: List = [TextMessage(text=text, quick_reply=build_quick_reply(chat_id), sender=sender)]
    if extras:
        # 讓 Flex 也帶上 QR（但仍會再保險補一個空白 Text）
        for m in extras:
            try:
                m.quick_reply = build_quick_reply(chat_id)  # type: ignore
            except Exception:
                pass
        messages.extend(extras)

    tts_enabled = tts_switch_per_chat.get(chat_id, TTS_SEND_ALWAYS)
    if tts_enabled and CLOUDINARY_CONFIGURED:
        try:
            audio_bytes = await text_to_speech_async(text)
            if audio_bytes:
                upload_result = await run_in_threadpool(
                    lambda: cloudinary_uploader.upload(
                        io.BytesIO(audio_bytes),
                        resource_type="video", folder="line-bot-tts", format="mp3"
                    )
                )
                url = upload_result.get("secure_url")
                if url:
                    est = max(3000, min(30000, len(text) * 60))
                    # 讓 Audio 本身也帶 QR
                    messages.append(AudioMessage(original_content_url=url, duration=est, quick_reply=build_quick_reply(chat_id)))
        except Exception as e:
            logger.error(f"TTS/Cloudinary 失敗：{e}")

    # 最後保險：一定讓“最後一個”有 QR
    _ensure_qr_visible(messages, chat_id, sender)

    try:
        return line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=messages))
    except Exception as line_e:
        logger.error(f"LINE 回覆失敗：{line_e}")
        try:
            simple_msg = TextMessage(text=text[:100]+"..." if len(text)>100 else text,
                                     quick_reply=build_quick_reply(chat_id), sender=sender)
            line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=[simple_msg]))
        except Exception as backup_e:
            logger.error(f"LINE 備用回覆也失敗：{backup_e}")
            raise line_e

# ---------- AI / 翻譯 ----------
def get_analysis_reply(messages: List[dict]) -> str:
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=1500
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"OpenAI 失敗：{e}")
    if not sync_groq_client:
        return "抱歉，AI 服務目前無法連線。"
    try:
        resp = sync_groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY, messages=messages, temperature=0.7, max_tokens=2000
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"Groq Primary 失敗：{e}")
        try:
            resp = sync_groq_client.chat.completions.create(
                model=GROQ_MODEL_FALLBACK, messages=messages, temperature=0.9, max_tokens=1500
            )
            return resp.choices[0].message.content or ""
        except Exception as e2:
            logger.error(f"Groq Fallback 失敗：{e2}")
            return "AI 分析服務暫時不可用，請稍後再試。"

async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
    if not async_groq_client:
        return await run_in_threadpool(lambda: get_analysis_reply(messages))
    try:
        resp = await async_groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"Groq 異步失敗：{e}")
        return await run_in_threadpool(lambda: get_analysis_reply(messages))

async def analyze_sentiment(text: str) -> str:
    msgs = [{"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
            {"role":"user","content":text}]
    try:
        out = await groq_chat_async(msgs, max_tokens=10, temperature=0)
        s = (out or "neutral").strip().lower()
        return s if s in {"positive","neutral","negative","angry"} else "neutral"
    except Exception:
        return "neutral"

async def translate_text(text: str, target_lang_display: str) -> str:
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys_prompt = "You are a precise translation engine. Output ONLY the translated text with no extra words."
    clean = re.sub(r"[\u200B-\u200D\uFEFF]", "", text).strip()
    if not clean: return "無內容可翻譯"
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{clean}"}}'
    try:
        result = await groq_chat_async([{"role":"system","content":sys_prompt},{"role":"user","content":usr}], 800, 0.2)
        return result if result.strip() else f"翻譯失敗：{text[:20]}..."
    except Exception as e:
        logger.error(f"翻譯失敗：{e}")
        return f"翻譯服務暫時不可用：{text[:20]}..."

# ---------- 金價（台銀，實抓） ----------
def get_bot_gold_quote() -> dict:
    try:
        r = requests.get(BOT_GOLD_URL, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        m_time = re.search(r"掛牌時間[:：]\s*([0-9]{4}/[0-9]{2}/[0-9]{2}\s+[0-9]{2}:[0-9]{2})", text)
        m_sell = re.search(r"本行賣出\s*([0-9,]+(?:\.[0-9]+)?)", text)
        m_buy  = re.search(r"本行買進\s*([0-9,]+(?:\.[0-9]+)?)", text)
        if not (m_sell and m_buy):
            raise RuntimeError("找不到『本行賣出/本行買進』欄位")
        listed_at = m_time.group(1) if m_time else "未知"
        sell = float(m_sell.group(1).replace(",", ""))
        buy  = float(m_buy.group(1).replace(",", ""))
        return {"listed_at": listed_at, "sell_twd_per_g": sell, "buy_twd_per_g": buy}
    except Exception as e:
        logger.error(f"金價獲取失敗：{e}")
        return {"listed_at": "錯誤", "sell_twd_per_g": 0, "buy_twd_per_g": 0}

# ---------- 外匯（yfinance） ----------
FX_CODES = {"USD","TWD","JPY","EUR","GBP","CNY","HKD","AUD","CAD","CHF","SGD","KRW","NZD","THB","MYR","IDR","PHP","INR","ZAR"}
FX_ALIAS = {"日圓":"JPY","日元":"JPY","美元":"USD","台幣":"TWD","新台幣":"TWD","人民幣":"CNY","港幣":"HKD","韓元":"KRW","歐元":"EUR","英鎊":"GBP"}

def _is_fx_query(text: str) -> bool:
    t = text.strip().upper()
    if t in FX_CODES or t in set(FX_ALIAS.values()): return True
    return bool(re.match(r"^[A-Za-z]{3}([/\s-]?[A-Za-z]{3})?$", t))

def _normalize_fx_token(tok: str) -> str:
    return FX_ALIAS.get(tok.strip().upper(), tok.strip().upper())

def parse_fx_pair(user_text: str) -> Tuple[str,str,str]:
    raw = user_text.strip()
    m = re.findall(r"[A-Za-z\u4e00-\u9fa5]{2,5}", raw)
    toks = [_normalize_fx_token(x) for x in m]
    toks = [x for x in toks if x in FX_CODES]
    if not toks:
        t = _normalize_fx_token(raw)
        if len(t) == 3 and t in FX_CODES: base, quote = t, "TWD"
        else: base, quote = "USD", "JPY"
    elif len(toks) == 1:
        base, quote = toks[0], "TWD"
    else:
        base, quote = toks[0], toks[1]
    symbol = f"{base}{quote}=X"
    link = f"https://finance.yahoo.com/quote/{symbol}/"
    return base, quote, link

def fetch_fx_quote_yf(symbol: str):
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(period="5d", interval="1d")
        if df is None or df.empty: return None, None, None, None
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df)>=2 else None
        last_price = float(last_row["Close"])
        change_pct = None if prev_row is None else (last_price/float(prev_row["Close"]) - 1.0)*100.0
        ts = last_row.name
        ts_iso = ts.tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M %Z") if hasattr(ts, "tz_convert") else str(ts)
        return last_price, change_pct, ts_iso, df
    except Exception as e:
        logger.error(f"fetch_fx_quote_yf error for {symbol}: {e}")
        return None, None, None, None

def render_fx_report(base, quote, link, last, chg, ts, df) -> str:
    trend = ""
    if df is not None and not df.empty:
        diff = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[0])
        trend = "上升" if diff>0 else ("下跌" if diff<0 else "持平")
    lines = [f"#### 外匯報告（Yahoo Finance）\n- 幣別對：**{base}/{quote}**\n- 連結：{link}"]
    if last is not None: lines.append(f"- 目前匯率：**{last:.6f}**（{base}/{quote}）")
    if chg is not None:  lines.append(f"- 日變動：**{chg:+.2f}%**")
    if ts:               lines.append(f"- 資料時間：{ts}")
    if trend:            lines.append(f"- 近 5 日趨勢：{trend}")
    lines.append(f"\n[Yahoo Finance Quote]({link})")
    return "\n".join(lines)

# ---------- 股票 ----------
TW_TICKER_RE = re.compile(r"^\d{4,6}[A-Za-z]?$")
US_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}$")
def _is_stock_query(text: str) -> bool:
    t = text.strip()
    if t in ("大盤","台股大盤","台灣大盤","美盤","美股大盤","美股"): return True
    if TW_TICKER_RE.match(t): return True
    if US_TICKER_RE.match(t): return True
    return False

def _normalize_ticker_and_name(user_text: str) -> Tuple[str,str,str]:
    raw = user_text.strip()
    if raw in ("大盤","台股大盤","台灣大盤"): return "^TWII","台灣大盤","https://tw.finance.yahoo.com/quote/%5ETWII/"
    if raw in ("美盤","美股大盤","美股"):     return "^GSPC","美國大盤","https://tw.finance.yahoo.com/quote/%5EGSPC/"
    ticker = raw.upper()
    link = f"https://tw.stock.yahoo.com/quote/{ticker}" if TW_TICKER_RE.match(ticker) else f"https://tw.finance.yahoo.com/quote/{ticker}"
    return ticker, ticker, link

def build_stock_prompt_block(stock_id: str, stock_name_hint: str) -> Tuple[str, dict]:
    try:
        ys = YahooStock(stock_id)
        price_df = stock_price(stock_id)
        news = stock_news(stock_name_hint)
        news = [n.replace('\u3000',' ') for n in news]
        news_text = "\n".join(news)[:1024]
        fund_text = div_text = ""
        if stock_id not in ["^TWII","^GSPC"]:
            try: fund_text = str(stock_fundamental(stock_id)) or "（無法取得）"
            except Exception as e: fund_text = f"（基本面錯誤：{e}）"
            try: div_text = str(stock_dividend(stock_id)) or "（無法取得）"
            except Exception as e: div_text = f"（配息錯誤：{e}）"
        blk = [
            f"**股票代碼:** {stock_id}, **股票名稱:** {ys.name}",
            f"**即時資訊(vars):** {vars(ys)}",
            f"近期價格資訊:\n{price_df if not price_df.empty else '(價格資料缺)'}",
        ]
        if stock_id not in ["^TWII","^GSPC"]:
            blk += [f"每季營收資訊:\n{fund_text}", f"配息資料:\n{div_text}"]
        blk.append(f"近期新聞資訊:\n{news_text}")
        return "\n".join(str(x) for x in blk), {}
    except Exception as e:
        logger.error(f"股票資料建構失敗：{e}")
        return f"股票資料獲取錯誤：{e}", {}

def render_stock_report(stock_id: str, stock_link: str, content_block: str) -> str:
    sys_prompt = ("你現在是一位專業的證券分析師。請基於近期走勢、基本面、新聞與籌碼概念進行綜合分析，"
                  "條列清楚、數字精確、可讀性高。\n"
                  "- 股名(股號)/現價(與漲跌幅)/資料時間\n"
                  "- 走勢 / 基本面 / 技術面 / 消息面 / 籌碼面\n"
                  "- 建議買進區間 / 停利點 / 建議部位\n"
                  f"最後附上正確連結：[股票資訊連結]({stock_link})。\n"
                  "使用台灣繁體中文，回覆精簡有力。")
    try:
        result = get_analysis_reply([{"role":"system","content":sys_prompt},{"role":"user","content":content_block}])
        return result
    except Exception as e:
        logger.error(f"股票分析失敗：{e}")
        return f"（分析模型不可用）原始資料：\n{content_block[:500]}...\n\n連結：{stock_link}"

# ---------- 彩票 ----------
def _lotto_fallback_scrape(kind: str) -> str:
    try:
        if kind == "威力彩":
            url, pat = ("https://www.taiwanlottery.com/lotto/superlotto638/index.html",
                        r"第\s*\d+\s*期.*?第一區.*?[:：\s]*([\d\s,]+?)\s*第二區.*?[:：\s]*(\d+)")
        elif kind == "大樂透":
            url, pat = ("https://www.taiwanlottery.com/lotto/lotto649/index.html",
                        r"第\s*\d+\s*期.*?(?:號碼|獎號).*?[:：\s]*([\d\s,]+?)(?:\s*特別號[:：\s]*(\d+))?")
        elif kind in ("今彩539","539"):
            url, pat = ("https://www.taiwanlottery.com/lotto/dailycash/index.html",
                        r"第\s*\d+\s*期.*?(?:號碼|獎號).*?[:：\s]*([\d\s,]+)")
        else:
            return f"不支援: {kind}"
        r = requests.get(url, headers=_HEADERS, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser"); text = ' '.join(soup.stripped_strings)
        m = re.search(pat, text, re.DOTALL)
        if not m: return f"{kind}：官網解析失敗（版面可能更動）"
        if kind == "威力彩":
            first, second = re.sub(r'[,\s]+',' ', m.group(1)).strip(), m.group(2)
            return f"{kind}：第一區 {first}；第二區 {second}"
        elif kind == "大樂透":
            nums, special = re.sub(r'[,\s]+',' ', m.group(1)).strip(), m.group(2)
            return f"{kind}：{nums}{'；特別號 '+special if special else ''}"
        else:
            nums = re.sub(r'[,\s]+',' ', m.group(1)).strip()
            return f"{kind}：{nums}"
    except Exception as e:
        logger.error(f"Fallback scrape fail: {e}")
        return f"{kind}：無法取得最新號碼（fallback 例外）"

def get_lottery_analysis(lottery_type_input: str) -> str:
    kind = "威力彩" if "威力" in lottery_type_input else ("大樂透" if "大樂" in lottery_type_input else ("今彩539" if "539" in lottery_type_input or "今彩" in lottery_type_input else lottery_type_input))
    latest_data_str = ""
    if LOTTERY_OK and lottery_crawler:
        try:
            if kind == "威力彩":   latest_data_str = str(lottery_crawler.super_lotto())
            elif kind == "大樂透": latest_data_str = str(lottery_crawler.lotto649())
            elif kind == "今彩539":latest_data_str = str(lottery_crawler.daily_cash())
            else: return f"不支援 {kind}"
        except Exception as e:
            logger.warning(f"TaiwanLottery 套件失敗：{e}，改用官網 fallback")
            latest_data_str = _lotto_fallback_scrape(kind)
    else:
        latest_data_str = _lotto_fallback_scrape(kind)

    prompt = (f"{kind} 最新資料：\n{latest_data_str}\n\n"
              "請用繁體中文條列：\n"
              "1) 近期走勢重點（熱號/冷號）\n"
              "2) 選號思路與風險聲明（理性投注）\n"
              "3) 推薦三組號碼（僅供娛樂，不保證中獎）")
    messages = [{"role":"system","content":"你是資深彩券分析師。"},{"role":"user","content":prompt}]
    return get_analysis_reply(messages)

# ---------- 事件處理 ----------
async def handle_text_message(event: MessageEvent):
    chat_id = get_chat_id(event)
    msg_raw = (event.message.text or "").strip()
    reply_tok = event.reply_token
    if not msg_raw: return

    if msg_raw.upper() == "TTS ON":
        tts_switch_per_chat[chat_id] = True
        await reply_text_with_tts_and_extras(reply_tok, "🔊 已開啟語音播報", chat_id=chat_id); return
    if msg_raw.upper() == "TTS OFF":
        tts_switch_per_chat[chat_id] = False
        await reply_text_with_tts_and_extras(reply_tok, "🔇 已關閉語音播報", chat_id=chat_id); return

    m = TRANSLATE_CMD.match(msg_raw)
    if m:
        lang_token = m.group(1)
        rev = {"english":"英文","japanese":"日文","korean":"韓文","vietnamese":"越南文","繁體中文":"繁體中文","中文":"繁體中文"}
        lang_display = rev.get(lang_token.lower(), lang_token)
        _tstate_set(chat_id, lang_display)
        await reply_text_with_tts_and_extras(reply_tok, f"🌐 已開啟翻譯 → {lang_display}，請直接輸入要翻的內容。", chat_id=chat_id)
        return

    if msg_raw.startswith("翻譯->"):
        lang = msg_raw.split("->",1)[1].strip()
        if lang == "結束":
            _tstate_clear(chat_id)
            await reply_text_with_tts_and_extras(reply_tok, "✅ 已結束翻譯模式", chat_id=chat_id)
        else:
            _tstate_set(chat_id, lang)
            await reply_text_with_tts_and_extras(reply_tok, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。", chat_id=chat_id)
        return

    im = INLINE_TRANSLATE.match(msg_raw)
    if im:
        lang_key, text_to_translate = im.group(1).lower(), im.group(2)
        mapping = {"en":"英文","eng":"英文","英文":"英文","ja":"日文","jp":"日文","日文":"日文","zh":"繁體中文","繁中":"繁體中文","中文":"繁體中文"}
        lang_display = mapping.get(lang_key, "英文")
        out = await translate_text(text_to_translate, lang_display)
        await reply_text_with_tts_and_extras(reply_tok, out, chat_id=chat_id)
        return

    current_lang = _tstate_get(chat_id)
    if current_lang:
        out = await translate_text(msg_raw, current_lang)
        await reply_text_with_tts_and_extras(reply_tok, out, chat_id=chat_id)
        return

    low = msg_raw.lower()
    if low in ("menu","選單","主選單"):
        await reply_text_with_tts_and_extras(reply_tok, "👇 功能選單", chat_id=chat_id, extras=[build_main_menu()])
        return

    if msg_raw in PERSONA_ALIAS:
        key = PERSONA_ALIAS[msg_raw]
        key = random.choice(list(PERSONAS.keys())) if key=="random" else key
        if key not in PERSONAS: key = "sweet"
        user_persona[chat_id] = key
        p = PERSONAS[key]
        await reply_text_with_tts_and_extras(reply_tok, f"已切換為「{p['title']}」模式～{p['emoji']}", chat_id=chat_id)
        return

    if msg_raw in ("金價","黃金"):
        d = get_bot_gold_quote()
        ts, sell, buy = d.get("listed_at") or "（未標示）", d["sell_twd_per_g"], d["buy_twd_per_g"]
        spread = sell - buy
        txt = (f"**金價（台灣銀行）**\n"
               f"- 掛牌時間：{ts}\n"
               f"- 賣出(1g)：{sell:,.0f} 元\n"
               f"- 買進(1g)：{buy:,.0f} 元\n"
               f"- 價差：{spread:,.0f} 元\n"
               f"來源：{BOT_GOLD_URL}")
        await reply_text_with_tts_and_extras(reply_tok, txt, chat_id=chat_id)
        return

    if msg_raw in ("大樂透","威力彩","今彩539","539"):
        kind = "今彩539" if msg_raw in ("今彩539","539") else msg_raw
        report = get_lottery_analysis(kind)
        await reply_text_with_tts_and_extras(reply_tok, report, chat_id=chat_id)
        return

    if _is_fx_query(msg_raw):
        base, quote, link = parse_fx_pair(msg_raw)
        last, chg, ts, df = fetch_fx_quote_yf(f"{base}{quote}=X")
        report = render_fx_report(base, quote, link, last, chg, ts, df)
        await reply_text_with_tts_and_extras(reply_tok, report, chat_id=chat_id)
        return

    if _is_stock_query(msg_raw):
        ticker, name_hint, link = _normalize_ticker_and_name(msg_raw)
        content_block, _ = await run_in_threadpool(build_stock_prompt_block, ticker, name_hint)
        report = await run_in_threadpool(render_stock_report, ticker, link, content_block)
        await reply_text_with_tts_and_extras(reply_tok, report, chat_id=chat_id)
        return

    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg_raw)
        key = user_persona.get(chat_id, "sweet"); p = PERSONAS[key]
        sys_prompt = (f"你是一位「{p['title']}」。風格：{p['style']}。\n"
                      f"使用者情緒：{sentiment}。\n"
                      f"回覆請精煉自然，使用繁體中文，帶少量表情 {p['emoji']}。")
        messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg_raw}]
        final_reply = await groq_chat_async(messages)
        history.extend([{"role":"user","content":msg_raw},{"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
        await reply_text_with_tts_and_extras(reply_tok, final_reply, chat_id=chat_id)
    except Exception as e:
        logger.error(f"一般聊天失敗：{e}")
        await reply_text_with_tts_and_extras(reply_tok, "抱歉我剛剛走神了 😅 再說一次讓我補上！", chat_id=chat_id)

async def handle_audio_message(event: MessageEvent):
    chat_id = get_chat_id(event)
    reply_tok = event.reply_token
    try:
        content_stream = await line_bot_api.get_message_content(event.message.id)
        audio_in = await content_stream.read()
        text = await speech_to_text_async(audio_in)
        if not text:
            await reply_text_with_tts_and_extras(reply_tok, "🎧 語音收到！目前語音轉文字失敗，請稍後再試。", chat_id=chat_id)
            return
        msgs: List = [TextMessage(text=f"🎧 我聽到了：\n{text}", quick_reply=build_quick_reply(chat_id), sender=_build_translation_sender(chat_id))]
        tts_enabled = tts_switch_per_chat.get(chat_id, TTS_SEND_ALWAYS)
        if tts_enabled and CLOUDINARY_CONFIGURED:
            try:
                echo_bytes = await text_to_speech_async(f"你說了：{text}")
                if echo_bytes:
                    upload_result = await run_in_threadpool(
                        lambda: cloudinary_uploader.upload(
                            io.BytesIO(echo_bytes), resource_type="video", folder="line-bot-tts", format="mp3"
                        )
                    )
                    url = upload_result.get("secure_url")
                    if url:
                        est = max(3000, min(30000, len(text) * 60))
                        msgs.append(AudioMessage(original_content_url=url, duration=est, quick_reply=build_quick_reply(chat_id)))
            except Exception as e:
                logger.warning(f"語音回音失敗：{e}")
        # 保證最後有 QR
        _ensure_qr_visible(msgs, chat_id, _build_translation_sender(chat_id))
        line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_tok, messages=msgs))
    except Exception as e:
        logger.error(f"語音處理失敗: {e}", exc_info=True)
        await reply_text_with_tts_and_extras(reply_tok, "抱歉，語音處理失敗，請稍後再試。", chat_id=chat_id)

async def handle_postback(event: PostbackEvent):
    chat_id = get_chat_id(event)
    data = event.postback.data or ""
    if data.startswith("menu:"):
        kind = data.split(":",1)[-1]
        await reply_text_with_tts_and_extras(event.reply_token, "👇 子選單", chat_id=chat_id, extras=[build_submenu(kind)])
    else:
        await reply_text_with_tts_and_extras(event.reply_token, "收到你的選擇，正在處理中...", chat_id=chat_id)

async def handle_events(events):
    for event in events:
        try:
            if isinstance(event, MessageEvent):
                if isinstance(event.message, TextMessageContent):
                    await handle_text_message(event)
                elif isinstance(event.message, AudioMessageContent):
                    await handle_audio_message(event)
            elif isinstance(event, PostbackEvent):
                await handle_postback(event)
        except Exception as e:
            logger.error(f"事件處理失敗：{e}", exc_info=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 LINE Bot 啟動中...")
    if BASE_URL:
        async with httpx.AsyncClient() as c:
            for endpoint in ("https://api-data.line.me/v2/bot/channel/webhook/endpoint",
                             "https://api.line.me/v2/bot/channel/webhook/endpoint"):
                try:
                    headers={"Authorization":f"Bearer {CHANNEL_TOKEN}","Content-Type":"application/json"}
                    payload={"endpoint":f"{BASE_URL}/callback"}
                    r = await c.put(endpoint, headers=headers, json=payload, timeout=10.0)
                    r.raise_for_status(); logger.info(f"✅ Webhook 更新成功: {r.status_code}")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Webhook 更新失敗：{e}")
    logger.info("✅ 應用程式啟動完成")
    yield
    logger.info("👋 應用程式關閉")

app = FastAPI(lifespan=lifespan, title="AI醬 LINE Bot", version="2.2.1", description="彩票/股票/外匯/翻譯/TTS")
router = APIRouter()

@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        events = parser.parse(body.decode("utf-8"), signature)
        await handle_events(events)
        return JSONResponse({"status":"ok"})
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Callback 失敗：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")

@router.get("/")
async def root():
    return PlainTextResponse(
        "🤖 AI醬 LINE Bot v2.2.1 運行中！\n"
        "功能：彩票分析 💰 | 股票查詢 📈 | 外匯匯率 💱 | 即時翻譯 🌐 | 語音互動 🎤\n"
        "健康檢查：/healthz"
    )

@router.get("/healthz")
async def healthz():
    status = {
        "status": "ok",
        "version": "2.2.1",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "line_bot": "active",
            "lottery_module": LOTTERY_OK,
            "stock_module": STOCK_OK,
            "cloudinary": CLOUDINARY_CONFIGURED,
            "tts_default": TTS_SEND_ALWAYS,
            "tts_provider": TTS_PROVIDER,
            "gtts_available": GTTS_AVAILABLE,
            "openai_available": openai_client is not None,
            "groq_available": sync_groq_client is not None,
        }
    }
    return JSONResponse(status)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🎬 開發伺服器啟動：0.0.0.0:{port}，TTS_DEFAULT={TTS_SEND_ALWAYS}, PROVIDER={TTS_PROVIDER}")
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info")