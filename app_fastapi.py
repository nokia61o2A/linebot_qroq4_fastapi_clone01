# app_fastapi.py  v1.5.12
# 變更重點：
# - 嚴查 Quick Reply 顯示問題：多訊息回覆中，將 TextMessage（帶 Quick Reply）置於最後，確保 QR 顯示在底部（LINE 行為：QR 附加最後訊息最可靠）
# - 調整 reply_text_with_tts_and_extras：extras (Flex) + Audio + Text with QR
# - 新增詳細 log：回覆順序、QR 位置、訊息類型
# - 確認：單/多訊息均有 QR；Audio/Flex 不影響 QR 顯示
# - 其他功能完整：TTS 備用 gTTS、stub 模組正常

import os, re, io, sys, random, logging, asyncio
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

# ── 先定義 logger，避免 NameError ────────────────────────────────────────
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

logger.info("=== 🚀 AI醬 LINE Bot v1.5.12 啟動 (Quick Reply 最後置版) ===")

# ── 專案路徑 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
MC_DIR = os.path.join(BASE_DIR, "my_commands")
if MC_DIR not in sys.path:
    sys.path.append(MC_DIR)

# ── HTTP / 解析 ───────────────────────────────────────────────────────────────
import requests, httpx
from bs4 import BeautifulSoup

# ── 資料處理 / 金融 ───────────────────────────────────────────────────────────
import pandas as pd
import yfinance as yf

# ── FastAPI / LINE SDK v3 ────────────────────────────────────────────────────
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
    BotInfoResponse,
)

# ── Cloudinary（可選） ───────────────────────────────────────────────────────
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
        logger.info("✅ Cloudinary 模組載入成功")
    except ImportError as e:
        logger.warning(f"⚠️ Cloudinary 模組載入失敗：{e}")

# ── TTS/STT（可選） ─────────────────────────────────────────────────────────
GTTS_AVAILABLE = False
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
    logger.info("✅ gTTS 模組載入成功")
except ImportError as e:
    logger.warning(f"⚠️ gTTS 模組載入失敗：{e}")

# ── LLM ──────────────────────────────────────────────────────────────────────
from groq import AsyncGroq, Groq
import openai

# ── 你的彩票分析程式庫（唯一入口） ───────────────────────────────────────────
LOTTERY_OK = False
LOTTERY_IMPORT_ERR = ""
try:
    from my_commands.lottery_gpt import lottery_gpt as run_lottery_analysis
    LOTTERY_OK = True
    logger.info("✅ 彩票模組載入成功")
except Exception as e:
    LOTTERY_OK = False
    LOTTERY_IMPORT_ERR = f"{e.__class__.__name__}: {e}"
    run_lottery_analysis = None
    logger.error(f"❌ 彩票模組載入失敗：{LOTTERY_IMPORT_ERR}")

# ── 股票模組（若失敗則降級為安全 stub） ───────────────────────────────────────
try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
    STOCK_OK = True
    logger.info("✅ 股票模組載入成功")
except Exception as e:
    logger.warning(f"⚠️ 股票模組載入失敗：{e}，啟用備用版本")
    STOCK_OK = False
    def stock_price(s): return pd.DataFrame()
    def stock_news(s): return "（股票新聞模組未載入）"
    def stock_fundamental(s): return "（股票基本面模組未載入）"
    def stock_dividend(s): return "（股票股利模組未載入）"
    class YahooStock:
        def __init__(self, s): self.name = "（YahooStock 未載入）"

# ── 環境變數 ──────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto").lower()
TTS_SEND_ALWAYS = os.getenv("TTS_SEND_ALWAYS", "true").lower() == "true"

if not CHANNEL_TOKEN or not CHANNEL_SECRET:
    logger.error("❌ 缺少必要環境變數：CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET")
    raise RuntimeError("缺少必要環境變數")

logger.info(f"環境變數檢查：TTS={TTS_SEND_ALWAYS}, Provider={TTS_PROVIDER}")
logger.info(f"模組狀態：彩票={LOTTERY_OK}, 股票={STOCK_OK}, Cloudinary={CLOUDINARY_AVAILABLE}, gTTS={GTTS_AVAILABLE}")

# ── Cloudinary 配置 ──────────────────────────────────────────────────────────
if CLOUDINARY_URL and CLOUDINARY_AVAILABLE:
    try:
        cloudinary.config(
            cloud_name=re.search(r"@(.+)", CLOUDINARY_URL).group(1),
            api_key=re.search(r"//(\d+):", CLOUDINARY_URL).group(1),
            api_secret=re.search(r":([A-Za-z0-9_-]+)@", CLOUDINARY_URL).group(1),
        )
        logger.info("✅ Cloudinary 配置成功")
        CLOUDINARY_CONFIGURED = True
    except Exception as e:
        logger.error(f"❌ Cloudinary 設定失敗: {e}")
        CLOUDINARY_CONFIGURED = False
else:
    logger.info("ℹ️ Cloudinary 未配置或不可用")
    CLOUDINARY_CONFIGURED = False

# ── LINE / LLM Client ────────────────────────────────────────────────────────
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
        logger.info("✅ OpenAI 客戶端初始化成功")
    except Exception as e:
        logger.warning(f"⚠️ 初始化 OpenAI 失敗：{e}")

GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

logger.info(f"LLM 設定：Groq Primary={GROQ_MODEL_PRIMARY}, Fallback={GROQ_MODEL_FALLBACK}, OpenAI={'可用' if openai_client else '不可用'}")

# ── 會話狀態 / 翻譯 / 人設 ───────────────────────────────────────────────────
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
translation_states: Dict[str, str] = {}
translation_states_ttl: Dict[str, datetime] = {}
TRANSLATE_TTL_SECONDS = int(os.getenv("TRANSLATE_TTL_SECONDS", "7200"))
auto_reply_status: Dict[str, bool] = {}
user_persona: Dict[str, str] = {}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji": "🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji": "😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji": "✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議", "greetings": "我在。說重點。", "emoji": "🧊⚡️"},
}
LANGUAGE_MAP = {
    "英文": "English","日文": "Japanese","韓文": "Korean","越南文":"Vietnamese",
    "繁體中文":"Traditional Chinese","中文":"Traditional Chinese",
    "en":"English","ja":"Japanese","jp":"Japanese","ko":"Korean","vi":"Vietnamese","zh":"Traditional Chinese"
}
PERSONA_ALIAS = {"甜":"sweet","鹹":"salty","萌":"moe","酷":"cool","random":"random"}

TRANSLATE_CMD = re.compile(
    r"^(?:翻譯|翻成)\s*(?:->|→|>)?\s*(英文|English|日文|Japanese|韓文|Korean|越南文|Vietnamese|繁體中文|中文)\s*$",
    re.IGNORECASE
)
INLINE_TRANSLATE = re.compile(r"^(en|eng|英文|ja|jp|日文|zh|繁中|中文)\s*[:：>]\s*(.+)$", re.IGNORECASE)

# ── 小工具 ───────────────────────────────────────────────────────────────────
def _now() -> datetime: return datetime.utcnow()
 
def get_chat_id(event: MessageEvent) -> str:
    """穩健取得 chat id（支援駝峰屬性與 to_dict()）"""
    source = event.source
    stype = getattr(source, "type", None) or getattr(source, "_type", None)
    uid = getattr(source, "userId", None) or getattr(source, "user_id", None)
    gid = getattr(source, "groupId", None) or getattr(source, "group_id", None)
    rid = getattr(source, "roomId", None) or getattr(source, "room_id", None)
    try:
        if hasattr(source, "to_dict"):
            d = source.to_dict() or {}
            stype = stype or d.get("type")
            uid = uid or d.get("userId") or d.get("user_id")
            gid = gid or d.get("groupId") or d.get("group_id")
            rid = rid or d.get("roomId") or d.get("room_id")
    except Exception:
        pass
    if gid: return f"group:{gid}"
    if rid: return f"room:{rid}"
    if uid: return f"user:{uid}"
    return f"{stype or 'unknown'}:{abs(hash(str(source))) % 10_000_000}"

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

# ── Quick Reply（每則回覆都會帶） ────────────────────────────────────────────
def build_quick_reply() -> QuickReply:
    qr = QuickReply(items=[
        QuickReplyItem(action=MessageAction(label="主選單", text="選單")),
        QuickReplyItem(action=MessageAction(label="台股大盤", text="大盤")),
        QuickReplyItem(action=MessageAction(label="美股大盤", text="美盤")),
        QuickReplyItem(action=MessageAction(label="黃金價格", text="金價")),
        QuickReplyItem(action=MessageAction(label="查 2330", text="2330")),
        QuickReplyItem(action=MessageAction(label="查 NVDA", text="NVDA")),
        QuickReplyItem(action=MessageAction(label="日圓匯率", text="JPY")),
        QuickReplyItem(action=MessageAction(label="大樂透", text="大樂透")),
        QuickReplyItem(action=MessageAction(label="威力彩", text="威力彩")),
        QuickReplyItem(action=MessageAction(label="今彩539", text="今彩539")),
        QuickReplyItem(action=PostbackAction(label="💖 AI 人設", data="menu:persona")),
        QuickReplyItem(action=PostbackAction(label="🎰 彩票選單", data="menu:lottery")),
        QuickReplyItem(action=MessageAction(label="結束翻譯", text="翻譯->結束")),
    ])
    logger.debug("Quick Reply 建構完成（13 項）")
    return qr

def build_main_menu() -> FlexMessage:
    items = [
        ("💹 金融查詢", PostbackAction(label="💹 金融查詢", data="menu:finance")),
        ("🎰 彩票分析", PostbackAction(label="🎰 彩票分析", data="menu:lottery")),
        ("💖 AI 角色扮演", PostbackAction(label="💖 AI 角色扮演", data="menu:persona")),
        ("🌐 翻譯工具", PostbackAction(label="🌐 翻譯工具", data="menu:translate")),
    ]
    buttons = [FlexButton(action=i[1], style="primary" if idx < 2 else "secondary") for idx, i in enumerate(items)]
    bubble = FlexBubble(
        header=FlexBox(layout="vertical", contents=[FlexText(text="AI 助理主選單", weight="bold", size="lg")]),
        body=FlexBox(layout="vertical", spacing="md", contents=buttons),
    )
    return FlexMessage(alt_text="主選單", contents=bubble)

def build_submenu(kind: str) -> FlexMessage:
    menus = {
        "finance": ("💹 金融查詢", [
            ("台股大盤", MessageAction(label="台股大盤", text="大盤")),
            ("美股大盤", MessageAction(label="美股大盤", text="美盤")),
            ("黃金價格", MessageAction(label="黃金價格", text="金價")),
            ("日圓匯率", MessageAction(label="日圓匯率", text="JPY")),
            ("查 2330 台積電", MessageAction(label="查 2330 台積電", text="2330")),
            ("查 NVDA 輝達", MessageAction(label="查 NVDA 輝達", text="NVDA")),
        ]),
        "lottery": ("🎰 彩票分析", [
            ("大樂透", MessageAction(label="大樂透", text="大樂透")),
            ("威力彩", MessageAction(label="威力彩", text="威力彩")),
            ("今彩539", MessageAction(label="今彩539", text="今彩539")),
        ]),
        "persona": ("💖 AI 角色扮演", [
            ("甜美女友", MessageAction(label="甜美女友", text="甜")),
            ("傲嬌女友", MessageAction(label="傲嬌女友", text="鹹")),
            ("萌系女友", MessageAction(label="萌系女友", text="萌")),
            ("酷系御姐", MessageAction(label="酷系御姐", text="酷")),
            ("隨機切換", MessageAction(label="隨機切換", text="random")),
        ]),
        "translate": ("🌐 翻譯工具", [
            ("翻成英文", MessageAction(label="翻成英文", text="翻譯->英文")),
            ("翻成日文", MessageAction(label="翻成日文", text="翻譯->日文")),
            ("翻成繁中", MessageAction(label="翻成繁中", text="翻譯->繁體中文")),
            ("結束翻譯模式", MessageAction(label="結束翻譯模式", text="翻譯->結束")),
        ]),
    }
    title, items = menus.get(kind, ("無效選單", []))
    rows, row = [], []
    for _, action in items:
        row.append(FlexButton(action=action, style="primary"))
        if len(row) == 2:
            rows.append(FlexBox(layout="horizontal", spacing="sm", contents=row)); row = []
    if row: rows.append(FlexBox(layout="horizontal", spacing="sm", contents=row))
    bubble = FlexBubble(
        header=FlexBox(layout="vertical", contents=[FlexText(text=title, weight="bold", size="lg")]),
        body=FlexBox(layout="vertical", spacing="md", contents=rows or [FlexText(text="（尚無項目）")]),
    )
    return FlexMessage(alt_text=title, contents=bubble)

# ── STT/TTS 與統一回覆 ────────────────────────────────────────────────────────
async def _stt_openai(audio_bytes: bytes, filename="audio.m4a") -> Optional[str]:
    if not openai_client: return None
    try:
        f = io.BytesIO(audio_bytes); f.name = filename
        resp = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
        return (resp.text or "").strip() or None
    except Exception as e:
        logger.warning(f"OpenAI STT 失敗：{e}")
        return None

async def _stt_groq(audio_bytes: bytes, filename="audio.m4a") -> Optional[str]:
    if not GROQ_API_KEY: return None
    try:
        f = io.BytesIO(audio_bytes); f.name = filename
        resp = Groq(api_key=GROQ_API_KEY).audio.transcriptions.create(file=f, model="whisper-large-v3")
        return (resp.text or "").strip() or None
    except Exception as e:
        logger.warning(f"Groq STT 失敗：{e}")
        return None

async def speech_to_text_async(audio_bytes: bytes) -> Optional[str]:
    """語音轉文字（非同步）"""
    return await _stt_openai(audio_bytes) or await _stt_groq(audio_bytes)

def _tts_openai(text: str) -> Optional[bytes]:
    """OpenAI TTS（同步）"""
    if not openai_client: return None
    try:
        clean = re.sub(r"[*_`~#]", "", text)
        if not clean: clean = "內容為空"
        resp = openai_client.audio.speech.create(model="tts-1", voice="nova", input=clean)
        return resp.read()
    except Exception as e:
        logger.error(f"OpenAI TTS 失敗: {e}")
        return None

def _tts_gtts(text: str) -> Optional[bytes]:
    """gTTS（同步）"""
    if not GTTS_AVAILABLE: return None
    try:
        clean = re.sub(r"[*_`~#]", "", text).strip()
        if not clean: clean = "嗨，我在這裡。"
        tts = gTTS(text=clean, lang="zh-TW", tld="com.tw", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"gTTS 失敗: {e}")
        return None

async def text_to_speech_async(text: str) -> Optional[bytes]:
    """文字轉語音（非同步）"""
    clean_text = re.sub(r"[*_`~#]", "", text).strip()
    if not clean_text:
        clean_text = "內容為空"
    
    if TTS_PROVIDER == "openai":
        # 強制 OpenAI，失敗後用 gTTS 備用
        try:
            result = await run_in_threadpool(_tts_openai, clean_text)
            if result: return result
        except Exception as e:
            logger.warning(f"OpenAI TTS 失敗，使用 gTTS 備用：{e}")
        
        return await run_in_threadpool(_tts_gtts, clean_text)
    
    elif TTS_PROVIDER == "gtts":
        # 強制 gTTS
        return await run_in_threadpool(_tts_gtts, clean_text)
    
    else:  # "auto"
        # 自動選擇：OpenAI → gTTS
        try:
            result = await run_in_threadpool(_tts_openai, clean_text)
            if result: 
                logger.debug("使用 OpenAI TTS")
                return result
        except Exception as e:
            logger.warning(f"OpenAI TTS 失敗，改用 gTTS：{e}")
        
        return await run_in_threadpool(_tts_gtts, clean_text)

async def reply_text_with_tts_and_extras(reply_token: str, text: str, extras: Optional[List]=None):
    """
    所有文字回覆統一走這裡，強制 Quick Reply 附加在最後一個 TextMessage 上（確保 QR 在底部顯示）。
    修正版：多訊息順序 = extras (Flex) + Audio + Text with QR（QR 置最後，LINE 顯示底部）
    """
    if not text: 
        text = "（無內容）"
    
    logger.debug(f"準備回覆：{text[:50]}...，TTS={TTS_SEND_ALWAYS}, Cloudinary={CLOUDINARY_CONFIGURED}")
    
    # 建立基本文字訊息，並強制附加 Quick Reply（置於最後）
    text_message = TextMessage(text=text, quick_reply=build_quick_reply())
    
    # 建構訊息順序：extras 先、Audio 中、Text with QR 最後（確保 QR 底部）
    messages = []
    if extras: 
        messages.extend(extras)
        logger.debug(f"附加 extras：{len(extras)} 項 (e.g., Flex)")
    
    audio_msg = None
    if TTS_SEND_ALWAYS and CLOUDINARY_CONFIGURED and cloudinary_uploader:
        try:
            logger.debug("開始 TTS 處理...")
            audio_bytes = await text_to_speech_async(text)
            
            if audio_bytes and len(audio_bytes) > 0:
                logger.debug(f"TTS 音檔生成成功，大小：{len(audio_bytes)} bytes")
                
                # 上傳到 Cloudinary
                upload_result = await run_in_threadpool(
                    lambda: cloudinary_uploader.upload(
                        io.BytesIO(audio_bytes),
                        resource_type="video", 
                        folder="line-bot-tts", 
                        format="mp3"
                    )
                )
                
                url = upload_result.get("secure_url")
                if url:
                    est_duration = max(3000, min(30000, len(text) * 60))
                    audio_msg = AudioMessage(original_content_url=url, duration=est_duration)
                    messages.append(audio_msg)
                    logger.debug("附加 Audio 到中間位置")
                    logger.info(f"TTS 上傳成功：{url}，持續時間：{est_duration}ms")
                else:
                    logger.warning("Cloudinary 上傳成功但無 secure_url")
            else:
                logger.warning("TTS 生成空音檔")
        except Exception as tts_e:
            logger.error(f"TTS 處理失敗：{tts_e}")
    
    # Text with QR 總在最後（確保 QR 顯示底部）
    messages.append(text_message)
    
    # LINE API 調用（同步）
    try:
        line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=messages))
        has_qr = any(hasattr(m, 'quick_reply') and m.quick_reply for m in messages)
        msg_types = [m.__class__.__name__ for m in messages]
        logger.info(f"LINE 回覆成功：{reply_token[:20]}...，訊息數：{len(messages)}，類型：{msg_types}，Quick Reply 附加（最後）：{has_qr}")
    except Exception as line_e:
        logger.error(f"LINE 回覆失敗：{line_e}")
        # 備用方案：只發送文字 + Quick Reply（置最後）
        try:
            simple_msg = TextMessage(text=text[:100] + "..." if len(text) > 100 else text, quick_reply=build_quick_reply())
            line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=[simple_msg]))
            logger.info("LINE 備用回覆成功（僅文字 + Quick Reply 最後）")
        except Exception as backup_e:
            logger.error(f"LINE 備用回覆也失敗：{backup_e}")

async def reply_menu_with_hint(reply_token: str, flex: FlexMessage, hint: str="👇 功能選單"):
    """選單回覆：Flex 先 + 文字 with QR 最後，確保 Quick Reply 底部顯示"""
    # 強制附加 Quick Reply 在文字訊息上（置最後）
    text_msg = TextMessage(text=hint, quick_reply=build_quick_reply())
    messages = [flex, text_msg]  # Flex 先，QR Text 最後
    
    try:
        line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=messages))
        has_qr = any(hasattr(m, 'quick_reply') and m.quick_reply for m in messages)
        msg_types = [m.__class__.__name__ for m in messages]
        logger.info(f"LINE 選單回覆成功：{reply_token[:20]}...，訊息數：{len(messages)}，類型：{msg_types}，Quick Reply 附加（最後）：{has_qr}")
    except Exception as e:
        logger.error(f"LINE 選單回覆失敗：{e}")
        # 備用方案：僅文字 + Quick Reply 最後
        try:
            simple_msg = TextMessage(text=hint, quick_reply=build_quick_reply())
            line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=[simple_msg]))
            logger.info("LINE 選單備用回覆成功（僅文字 + Quick Reply 最後）")
        except Exception as backup_e:
            logger.error(f"LINE 選單備用回覆也失敗：{backup_e}")

# ── 一般聊天/翻譯 LLM ────────────────────────────────────────────────────────
def get_analysis_reply(messages: List[dict]) -> str:
    """同步 LLM 分析（備用方法）"""
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=1500
            )
            content = resp.choices[0].message.content or ""
            logger.debug(f"OpenAI 回應長度：{len(content)}")
            return content
        except Exception as e:
            logger.warning(f"OpenAI 失敗：{e}")
    
    if not sync_groq_client:
        return "抱歉，AI 服務目前無法連線。"
    
    try:
        resp = sync_groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY, messages=messages, temperature=0.7, max_tokens=2000
        )
        content = resp.choices[0].message.content or ""
        logger.debug(f"Groq Primary 回應長度：{len(content)}")
        return content
    except Exception as e:
        logger.warning(f"Groq Primary 失敗：{e}")
        try:
            resp = sync_groq_client.chat.completions.create(
                model=GROQ_MODEL_FALLBACK, messages=messages, temperature=0.9, max_tokens=1500
            )
            content = resp.choices[0].message.content or ""
            logger.debug(f"Groq Fallback 回應長度：{len(content)}")
            return content
        except Exception as e2:
            logger.error(f"Groq 備用模型失敗：{e2}")
            return "AI 分析服務暫時不可用，請稍後再試。"

async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
    """非同步 Groq 聊天（主要方法）"""
    if not async_groq_client:
        return await run_in_threadpool(lambda: get_analysis_reply(messages))
    
    try:
        resp = await async_groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        content = resp.choices[0].message.content.strip() or ""
        logger.debug(f"Groq Async 回應長度：{len(content)}")
        return content
    except Exception as e:
        logger.error(f"Groq 異步調用失敗：{e}")
        return await run_in_threadpool(lambda: get_analysis_reply(messages))

async def analyze_sentiment(text: str) -> str:
    """情緒分析"""
    msgs = [{"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
            {"role":"user","content":text}]
    try:
        out = await groq_chat_async(msgs, max_tokens=10, temperature=0)
        sentiment = (out or "neutral").strip().lower()
        logger.debug(f"情緒分析結果：{sentiment} for '{text[:30]}...'")
        return sentiment
    except Exception as e:
        logger.warning(f"情緒分析失敗：{e}")
        return "neutral"

async def translate_text(text: str, target_lang_display: str) -> str:
    """翻譯文字"""
    target = LANGUAGE_MAP.get(target_lang_display.lower(), target_lang_display)
    sys_prompt = "You are a precise translation engine. Output ONLY the translated text with no extra words."
    clean = re.sub(r"[\u200B-\u200D\uFEFF]", "", text).strip()
    if not clean:
        return "無內容可翻譯"
    
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{clean}"}}'
    try:
        result = await groq_chat_async([{"role":"system","content":sys_prompt},{"role":"user","content":usr}], 800, 0.2)
        logger.debug(f"翻譯結果：{target} <- {result[:50]}...")
        return result if result.strip() else f"翻譯失敗：{text[:20]}..."
    except Exception as e:
        logger.error(f"翻譯失敗：{e}")
        return f"翻譯服務暫時不可用：{text[:20]}..."

def set_user_persona(chat_id: str, key: str):
    """設定使用者人設"""
    key_mapped = PERSONA_ALIAS.get(key, key)
    if key_mapped == "random": 
        key_mapped = random.choice(list(PERSONAS.keys()))
    if key_mapped not in PERSONAS: 
        key_mapped = "sweet"
    user_persona[chat_id] = key_mapped
    logger.debug(f"人設切換：{chat_id[:20]}... -> {PERSONAS[key_mapped]['title']}")
    return key_mapped

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    """建構人設 Prompt"""
    key = user_persona.get(chat_id, "sweet")
    p = PERSONAS[key]
    return (f"你是一位「{p['title']}」。風格：{p['style']}。\n"
            f"使用者情緒：{sentiment}。\n"
            f"回覆請精煉自然，使用繁體中文，帶少量表情 {p['emoji']}。")

# ── 金價 / 外匯 / 股票 ──────────────────────────────────────────────────────
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def get_bot_gold_quote() -> dict:
    """獲取台灣銀行金價"""
    try:
        r = requests.get(BOT_GOLD_URL, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        
        m_time = re.search(r"掛牌時間[:：]\s*([0-9]{4}/[0-9]{2}/[0-9]{2}\s+[0-9]{2}:[0-9]{2})", text)
        listed_at = m_time.group(1) if m_time else "未知"
        
        m_sell = re.search(r"本行賣出\s*([0-9,]+(?:\.[0-9]+)?)", text)
        m_buy = re.search(r"本行買進\s*([0-9,]+(?:\.[0-9]+)?)", text)
        
        if not (m_sell and m_buy): 
            raise RuntimeError("找不到『本行賣出/本行買進』欄位")
        
        sell = float(m_sell.group(1).replace(",", ""))
        buy = float(m_buy.group(1).replace(",", ""))
        
        logger.debug(f"金價資料：賣出={sell}, 買進={buy}")
        return {"listed_at": listed_at, "sell_twd_per_g": sell, "buy_twd_per_g": buy}
    except Exception as e:
        logger.error(f"金價獲取失敗：{e}")
        return {"listed_at": "錯誤", "sell_twd_per_g": 0, "buy_twd_per_g": 0}

FX_CODES = {"USD","TWD","JPY","EUR","GBP","CNY","HKD","AUD","CAD","CHF","SGD","KRW","NZD","THB","MYR","IDR","PHP","INR","ZAR"}
FX_ALIAS = {"日圓":"JPY","日元":"JPY","美元":"USD","台幣":"TWD","新台幣":"TWD","人民幣":"CNY","港幣":"HKD","韓元":"KRW","歐元":"EUR","英鎊":"GBP"}

def _is_fx_query(text: str) -> bool:
    """判斷是否為外匯查詢"""
    t = text.strip().upper()
    if t in FX_CODES or t in set(FX_ALIAS.values()): 
        return True
    return bool(re.match(r"^[A-Za-z]{3}[\s/\-_]?([A-Za-z]{3})?$", t))

def _normalize_fx_token(tok: str) -> str: 
    """標準化外匯代碼"""
    return FX_ALIAS.get(tok.strip().upper(), tok.strip().upper())

def parse_fx_pair(user_text: str) -> Tuple[str,str,str]:
    """解析外匯貨幣對"""
    raw = user_text.strip()
    m = re.findall(r"[A-Za-z\u4e00-\u9fa5]{2,5}", raw)
    toks = [_normalize_fx_token(x) for x in m]
    toks = [x for x in toks if x in FX_CODES]
    
    if not toks:
        t = _normalize_fx_token(raw)
        if len(t) == 3 and t in FX_CODES: 
            base, quote = t, "TWD"
        else: 
            base, quote = "USD", "JPY"
    elif len(toks) == 1: 
        base, quote = toks[0], "TWD"
    else: 
        base, quote = toks[0], toks[1]
    
    symbol = f"{base}{quote}=X"
    link = f"https://finance.yahoo.com/quote/{symbol}/"
    return base, quote, link

def fetch_fx_quote_yf(symbol: str):
    """從 Yahoo Finance 獲取外匯報價"""
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(period="5d", interval="1d")
        if df is None or df.empty: 
            return None, None, None, None
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df)>=2 else None
        last_price = float(last_row["Close"])
        change_pct = None if prev_row is None else (last_price/float(prev_row["Close"]) - 1.0)*100.0
        
        ts = last_row.name
        ts_iso = ts.tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M %Z") if hasattr(ts, "tz_convert") else str(ts)
        
        logger.debug(f"外匯 {symbol}：價格={last_price}, 變動={change_pct}")
        return last_price, change_pct, ts_iso, df
    except Exception as e:
        logger.error(f"fetch_fx_quote_yf error for {symbol}: {e}")
        return None, None, None, None

def render_fx_report(base, quote, link, last, chg, ts, df) -> str:
    """渲染外匯報表"""
    trend = ""
    if df is not None and not df.empty:
        diff = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[0])
        trend = "上升" if diff>0 else ("下跌" if diff<0 else "持平")
    
    lines = [f"#### 外匯報告（查匯優先）\n- 幣別對：**{base}/{quote}**\n- 來源：Yahoo Finance\n- 連結：{link}"]
    if last is not None: 
        lines.append(f"- 目前匯率：**{last:.6f}**（{base}/{quote}）")
    if chg is not None: 
        lines.append(f"- 日變動：**{chg:+.2f}%**")
    if ts: 
        lines.append(f"- 資料時間：{ts}")
    if trend: 
        lines.append(f"- 近 5 日趨勢：{trend}")
    lines.append(f"\n[外匯連結（Yahoo）]({link})")
    return "\n".join(lines)

TW_TICKER_RE = re.compile(r"^\d{4,6}[A-Za-z]?$")
US_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}$")

def _is_stock_query(text: str) -> bool:
    """判斷是否為股票查詢"""
    t = text.strip()
    if t in ("大盤","台股大盤","台灣大盤","美盤","美股大盤","美股"): 
        return True
    if TW_TICKER_RE.match(t): 
        return True
    if US_TICKER_RE.match(t) and t.upper() in {"NVDA","AAPL","TSLA","MSFT"}: 
        return True
    return False

def _normalize_ticker_and_name(user_text: str) -> Tuple[str,str,str]:
    """標準化股票代碼和名稱"""
    raw = user_text.strip()
    if raw in ("大盤","台股大盤","台灣大盤"): 
        return "^TWII","台灣大盤","https://tw.finance.yahoo.com/quote/%5ETWII/"
    if raw in ("美盤","美股大盤","美股"):     
        return "^GSPC","美國大盤","https://tw.finance.yahoo.com/quote/%5EGSPC/"
    
    ticker = raw.upper()
    link = f"https://tw.stock.yahoo.com/quote/{ticker}" if TW_TICKER_RE.match(ticker) else f"https://tw.finance.yahoo.com/quote/{ticker}"
    return ticker, ticker, link

def _safe_to_str(x) -> str:
    """安全轉換為字串"""
    try: 
        return str(x)
    except Exception: 
        return repr(x)

def _remove_full_width_spaces(data):
    """移除全形空格"""
    if isinstance(data, list): 
        return [_remove_full_width_spaces(i) for i in data]
    if isinstance(data, str):  
        return data.replace('\u3000',' ')
    return data

def _truncate_text(data, max_length=1024):
    """截斷文字"""
    if isinstance(data, list): 
        return [_truncate_text(i, max_length) for i in data]
    if isinstance(data, str):  
        return data[:max_length]
    return data

def build_stock_prompt_block(stock_id: str, stock_name_hint: str) -> Tuple[str, dict]:
    """建構股票分析 Prompt"""
    try:
        ys = YahooStock(stock_id)
        price_df = stock_price(stock_id)
        news = _remove_full_width_spaces(stock_news(stock_name_hint))
        news = _truncate_text(news, 1024)
        
        fund_text = div_text = None
        if stock_id not in ["^TWII","^GSPC"]:
            try:    
                fund_text = _safe_to_str(stock_fundamental(stock_id)) or "（無法取得）"
            except Exception as e: 
                fund_text = f"（基本面錯誤：{e}）"
            try:    
                div_text = _safe_to_str(stock_dividend(stock_id)) or "（無法取得）"
            except Exception as e: 
                div_text = f"（配息錯誤：{e}）"
        
        blk = [
            f"**股票代碼:** {stock_id}, **股票名稱:** {ys.name}",
            f"**即時資訊(vars):** {vars(ys)}",
            f"近期價格資訊:\n{price_df}"
        ]
        if stock_id not in ["^TWII","^GSPC"]:
            blk += [f"每季營收資訊:\n{fund_text}", f"配息資料:\n{div_text}"]
        blk.append(f"近期新聞資訊:\n{news}")
        
        result = "\n".join(_safe_to_str(x) for x in blk)
        logger.debug(f"股票 Prompt 建構完成，長度：{len(result)}")
        return result, {}
    except Exception as e:
        logger.error(f"股票資料建構失敗：{e}")
        return f"股票資料獲取錯誤：{e}", {}

def render_stock_report(stock_id: str, stock_link: str, content_block: str) -> str:
    """渲染股票分析報告"""
    sys_prompt = ("你現在是一位專業的證券分析師。請基於近期走勢、基本面、新聞與籌碼概念進行綜合分析，"
                  "條列清楚、數字精確、可讀性高。\n"
                  "- 股名(股號)/現價(與漲跌幅)/資料時間\n"
                  "- 走勢\n"
                  "- 基本面\n"
                  "- 技術面\n"
                  "- 消息面\n"
                  "- 籌碼面\n"
                  "- 建議買進區間\n"
                  "- 停利點\n"
                  "- 建議部位\n"
                  "- 總結\n"
                  f"最後附上正確連結：[股票資訊連結]({stock_link})。\n"
                  "使用台灣繁體中文，回覆精簡有力。")
    
    try:
        result = get_analysis_reply([{"role":"system","content":sys_prompt},{"role":"user","content":content_block}])
        logger.debug(f"股票分析完成，長度：{len(result)}")
        return result
    except Exception as e:
        logger.error(f"股票分析失敗：{e}")
        return f"（分析模型不可用）原始資料：\n{content_block[:500]}...\n\n連結：{stock_link}"

# ── 事件處理 ─────────────────────────────────────────────────────────────────
async def handle_text_message(event: MessageEvent):
    """處理文字訊息（所有分支均走統一回覆，確保 Quick Reply 底部顯示）"""
    chat_id = get_chat_id(event)
    msg_raw = (event.message.text or "").strip()
    reply_tok = event.reply_token
    
    logger.info(f"收到文字訊息：{msg_raw[:50]}... (chat_id: {chat_id[:20]}...)")
    
    if not msg_raw: 
        logger.debug("空訊息，忽略")
        return
    
    try:
        # 同步方法，無 await
        bot_info = line_bot_api.get_bot_info()
        bot_name = bot_info.display_name
        logger.debug(f"Bot 名稱：{bot_name}")
    except Exception as e:
        logger.warning(f"獲取 Bot 資訊失敗：{e}")
        bot_name = "AI 助手"

    msg = msg_raw
    if msg_raw.startswith(f"@{bot_name}"):
        msg = re.sub(f'^@{re.escape(bot_name)}\\s*','', msg_raw).strip()
        logger.debug(f"提及 Bot，清理後訊息：{msg[:30]}...")
    
    if not msg: 
        logger.debug("清理後訊息為空，忽略")
        return

    # ── 翻譯模式命令 ──────────────────────────────────────────────────────────
    m = TRANSLATE_CMD.match(msg)
    if m:
        lang_token = m.group(1)
        rev = {"english":"英文","japanese":"日文","korean":"韓文","vietnamese":"越南文","繁體中文":"繁體中文","中文":"繁體中文"}
        lang_display = rev.get(lang_token.lower(), lang_token)
        _tstate_set(chat_id, lang_display)
        await reply_text_with_tts_and_extras(reply_tok, f"🌐 已開啟翻譯 → {lang_display}，請直接輸入要翻的內容。")
        logger.info(f"開啟翻譯模式：{lang_display}")
        return
    
    if msg.startswith("翻譯->"):
        lang = msg.split("->",1)[1].strip()
        if lang=="結束":
            _tstate_clear(chat_id)
            await reply_text_with_tts_and_extras(reply_tok, "✅ 已結束翻譯模式")
            logger.info(f"結束翻譯模式：{chat_id}")
        else:
            _tstate_set(chat_id, lang)
            await reply_text_with_tts_and_extras(reply_tok, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")
            logger.info(f"開啟翻譯模式：{lang}")
        return
    
    # ── 內聯翻譯 ─────────────────────────────────────────────────────────────
    im = INLINE_TRANSLATE.match(msg)
    if im:
        lang_key, text_to_translate = im.group(1).lower(), im.group(2)
        lang_display = {"en":"英文","eng":"英文","英文":"英文","ja":"日文","jp":"日文","日文":"日文","zh":"繁體中文","繁中":"繁體中文","中文":"繁體中文"}.get(lang_key,"英文")
        logger.info(f"內聯翻譯：{lang_display} <- {text_to_translate[:30]}...")
        out = await translate_text(text_to_translate, lang_display)
        await reply_text_with_tts_and_extras(reply_tok, out)
        return

    # ── 翻譯模式中 ────────────────────────────────────────────────────────────
    current_lang = _tstate_get(chat_id)
    if current_lang:
        logger.debug(f"翻譯模式中：{current_lang}")
        out = await translate_text(msg, current_lang)
        await reply_text_with_tts_and_extras(reply_tok, out)
        return

    # ── 主選單 / 子選單 / 人設 ─────────────────────────────────────────────────
    low = msg.lower()
    if low in ("menu","選單","主選單"):
        logger.info("顯示主選單")
        await reply_menu_with_hint(reply_tok, build_main_menu())
        return
    
    if msg in PERSONA_ALIAS:
        key = set_user_persona(chat_id, msg)
        p = PERSONAS[key]
        await reply_text_with_tts_and_extras(reply_tok, f"已切換為「{p['title']}」模式～{p['emoji']}")
        return

    # ── 金價 ───────────────────────────────────────────────────────────────────
    if msg in ("金價","黃金"):
        logger.info("查詢金價")
        try:
            d = get_bot_gold_quote()
            ts, sell, buy = d.get("listed_at") or "（未標示）", d["sell_twd_per_g"], d["buy_twd_per_g"]
            spread = sell - buy
            txt = (f"**金價（台灣銀行）**\n"
                   f"- 掛牌時間：{ts}\n"
                   f"- 賣出(1g)：{sell:,.0f} 元\n"
                   f"- 買進(1g)：{buy:,.0f} 元\n"
                   f"- 價差：{spread:,.0f} 元\n"
                   f"來源：{BOT_GOLD_URL}")
            await reply_text_with_tts_and_extras(reply_tok, txt)
        except Exception as e:
            logger.error(f"金價查詢失敗：{e}")
            await reply_text_with_tts_and_extras(reply_tok, "抱歉，目前無法取得金價，請稍後再試。")
        return

    # ── 彩票分析 ──────────────────────────────────────────────────────────────
    if msg in ("大樂透","威力彩","539","今彩539","雙贏彩","3星彩","三星彩","4星彩","38樂合彩","39樂合彩","49樂合彩","運彩"):
        logger.info(f"收到彩票查詢：{msg}，模組狀態：LOTTERY_OK={LOTTERY_OK}")
        if LOTTERY_OK and callable(run_lottery_analysis):
            try:
                report = await run_in_threadpool(run_lottery_analysis, msg)
                await reply_text_with_tts_and_extras(reply_tok, report)
            except Exception as e:
                logger.error(f"彩票分析失敗: {e}", exc_info=True)
                await reply_text_with_tts_and_extras(reply_tok, f"抱歉，分析 {msg} 時發生錯誤：{e}")
        else:
            await reply_text_with_tts_and_extras(
                reply_tok,
                f"彩票分析模組未載入（匯入失敗）。詳情：{LOTTERY_IMPORT_ERR}\n"
                "請確認 my_commands/lottery_gpt.py、TaiwanLottery.py、CaiyunfangweiCrawler.py 都存在，且使用相對匯入。"
            )
        return

    # ── 外匯 ───────────────────────────────────────────────────────────────────
    if _is_fx_query(msg):
        logger.info(f"外匯查詢：{msg}")
        try:
            base, quote, link = parse_fx_pair(msg)
            last, chg, ts, df = fetch_fx_quote_yf(f"{base}{quote}=X")
            report = render_fx_report(base, quote, link, last, chg, ts, df)
            await reply_text_with_tts_and_extras(reply_tok, report)
        except Exception as e:
            logger.error(f"外匯查詢失敗：{e}")
            await reply_text_with_tts_and_extras(reply_tok, f"抱歉，取得 {msg} 匯率時發生錯誤：{e}")
        return

    # ── 股票 ───────────────────────────────────────────────────────────────────
    if _is_stock_query(msg):
        logger.info(f"股票查詢：{msg}")
        try:
            ticker, name_hint, link = _normalize_ticker_and_name(msg)
            content_block, _ = await run_in_threadpool(build_stock_prompt_block, ticker, name_hint)
            report = await run_in_threadpool(render_stock_report, ticker, link, content_block)
            await reply_text_with_tts_and_extras(reply_tok, report)
        except Exception as e:
            logger.error(f"股票查詢失敗：{e}")
            await reply_text_with_tts_and_extras(reply_tok, f"抱歉，取得 {msg} 分析時發生錯誤：{e}\n請稍後再試或換個代碼。")
        return

    # ── 一般聊天 ──────────────────────────────────────────────────────────────
    logger.info(f"一般聊天：{msg[:30]}...")
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        
        messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]
        final_reply = await groq_chat_async(messages)
        
        history.extend([{"role":"user","content":msg},{"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
        
        await reply_text_with_tts_and_extras(reply_tok, final_reply)
    except Exception as e:
        logger.error(f"一般聊天失敗：{e}")
        await reply_text_with_tts_and_extras(reply_tok, "抱歉我剛剛走神了 😅 再說一次讓我補上！")

async def handle_audio_message(event: MessageEvent):
    """處理語音訊息（統一走帶 Quick Reply 的回覆，底部顯示）"""
    reply_tok = event.reply_token
    logger.info(f"收到語音訊息：{event.message.id}")
    
    try:
        content_stream = await line_bot_api.get_message_content(event.message.id)
        audio_in = await content_stream.read()
        text = await speech_to_text_async(audio_in)
        if not text:
            await reply_text_with_tts_and_extras(reply_tok, "🎧 語音收到！目前語音轉文字失敗，請稍後再試。")
            return
        
        # 統一使用 reply_text_with_tts_and_extras 確保 Quick Reply 底部
        await reply_text_with_tts_and_extras(reply_tok, f"🎧 我聽到了：\n{text}")
        
    except Exception as e:
        logger.error(f"語音處理失敗: {e}", exc_info=True)
        await reply_text_with_tts_and_extras(reply_tok, "抱歉，語音處理失敗，請稍後再試。")

async def handle_postback(event: PostbackEvent):
    """處理 Postback 事件（走選單回覆，確保 Quick Reply 底部）"""
    data = event.postback.data or ""
    if data.startswith("menu:"):
        kind = data.split(":",1)[-1]
        await reply_menu_with_hint(event.reply_token, build_submenu(kind), hint="👇 子選單")

async def handle_events(events):
    """處理事件列表"""
    for event in events:
        if isinstance(event, MessageEvent):
            if isinstance(event.message, TextMessageContent):
                await handle_text_message(event)
            elif isinstance(event.message, AudioMessageContent):
                await handle_audio_message(event)
        elif isinstance(event, PostbackEvent):
            await handle_postback(event)

# ── FastAPI ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if BASE_URL:
        async with httpx.AsyncClient() as c:
            for endpoint in ("https://api-data.line.me/v2/bot/channel/webhook/endpoint",
                             "https://api.line.me/v2/bot/channel/webhook/endpoint"):
                try:
                    headers={"Authorization":f"Bearer {CHANNEL_TOKEN}","Content-Type":"application/json"}
                    payload={"endpoint":f"{BASE_URL}/callback"}
                    r = await c.put(endpoint, headers=headers, json=payload, timeout=10.0)
                    r.raise_for_status()
                    logger.info(f"Webhook 更新成功: {endpoint} {r.status_code}")
                    break
                except Exception as e:
                    logger.warning(f"Webhook 更新失敗：{e}")
    yield

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.5.12")
router = APIRouter()

@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        events = parser.parse(body.decode("utf-8"), signature)
        await handle_events(events)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Callback 失敗：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse({"status":"ok"})

@router.get("/")
async def root(): return PlainTextResponse("LINE Bot is running.", status_code=200)

@router.get("/healthz")
async def healthz(): return PlainTextResponse("ok", status_code=200)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)