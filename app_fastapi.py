# app_fastapi.py v1.4.9 (整合彩票分析 & 強化 Quick Reply)
# 修正摘要：
# - [FIX] 將 lottery_gpt.py 的邏輯直接整合進來，解決模組匯入失敗問題。
# - [FIX] 即使找不到 TaiwanLottery.py 等自訂爬蟲，程式也不會崩潰，而是回傳錯誤提示。
# - [CHG] 強化 reply_menu_with_hint 函式，確保 Flex 選單出現時，下方的 Quick Reply 按鈕列也會穩定顯示。
# - [INFO] 完整註解，方便您理解與後續維護。

import os
import re
import io
import sys
import random
import logging
import pkg_resources
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

# --- 讓本機與雲端都能找得到 my_commands 與專案根目錄 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
MC_DIR = os.path.join(BASE_DIR, "my_commands")
if MC_DIR not in sys.path:
    sys.path.append(MC_DIR)

# --- HTTP / 解析 ---
import requests
import httpx
from bs4 import BeautifulSoup

# --- 資料處理 / 金融 ---
import pandas as pd
import yfinance as yf

# --- FastAPI / LINE SDK v3 ---
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    AudioMessageContent,
    PostbackEvent,
)
from linebot.v3.webhook import WebhookParser
from linebot.v3.messaging import (
    Configuration, ApiClient, AsyncMessagingApi, ReplyMessageRequest,
    TextMessage, AudioMessage, FlexMessage, FlexBubble, FlexBox,
    FlexText, FlexButton, QuickReply, QuickReplyItem, MessageAction, PostbackAction,
    BotInfoResponse,
)

# --- Cloudinary（可選） ---
import cloudinary
import cloudinary.uploader

# --- 語音 TTS/STT（可選） ---
from gtts import gTTS

# --- LLM ---
from groq import AsyncGroq, Groq
import openai

# ====== 股票分析模組（沿用） ======
try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
    STOCK_OK = True
except ImportError as e:
    logging.warning(f"無法載入股票模組，功能將受限：{e}")
    STOCK_OK = False


# ====== [FIX] 彩票分析模組：建立安全的預備方案 ======
# 如果找不到您的自訂爬蟲檔案，會使用下面的 Dummy Class，避免程式崩潰
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    logging.info("成功載入 TaiwanLotteryCrawler。")
except ImportError:
    class TaiwanLotteryCrawler:
        def _not_found(self): return {"error": "找不到 'TaiwanLottery.py' 檔案，請檢查 my_commands 資料夾。"}
        def super_lotto(self): return self._not_found()
        def lotto649(self): return self._not_found()
        def daily_cash(self): return self._not_found()
        def lotto1224(self): return self._not_found()
        def lotto3d(self): return self._not_found()
        def lotto4d(self): return self._not_found()
        def lotto38m6(self): return self._not_found()
        def lotto39m5(self): return self._not_found()
        def lotto49m6(self): return self._not_found()
    logging.warning("無法從 my_commands 載入 'TaiwanLotteryCrawler'，已使用預備方案。")

try:
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    logging.info("成功載入 CaiyunfangweiCrawler。")
except ImportError:
    class CaiyunfangweiCrawler:
        def get_caiyunfangwei(self):
            return {"error": "找不到 'CaiyunfangweiCrawler.py' 檔案，請檢查 my_commands 資料夾。"}
    logging.warning("無法從 my_commands 載入 'CaiyunfangweiCrawler'，已使用預備方案。")


# ====== 基本設定 ======
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto").lower()
TTS_SEND_ALWAYS = os.getenv("TTS_SEND_ALWAYS", "true").lower() == "true"

if not CHANNEL_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("缺少必要環境變數：CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET")

# Cloudinary
if CLOUDINARY_URL:
    try:
        cloudinary.config(
            cloud_name=re.search(r"@(.+)", CLOUDINARY_URL).group(1),
            api_key=re.search(r"//(\d+):", CLOUDINARY_URL).group(1),
            api_secret=re.search(r":([A-Za-z0-9_-]+)@", CLOUDINARY_URL).group(1),
        )
        logger.info("Cloudinary OK")
    except Exception as e:
        logger.error(f"Cloudinary 設定失敗: {e}")
        CLOUDINARY_URL = None

# LINE / LLM 客戶端
configuration = Configuration(access_token=CHANNEL_TOKEN)
async_api_client = ApiClient(configuration=configuration)
line_bot_api = AsyncMessagingApi(api_client=async_api_client)
parser = WebhookParser(CHANNEL_SECRET)

sync_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.warning(f"初始化 OpenAI 失敗：{e}")

GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")


# ====== 狀態管理 ======
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
translation_states: Dict[str, str] = {}
translation_states_ttl: Dict[str, datetime] = {}
TRANSLATE_TTL_SECONDS = int(os.getenv("TRANSLATE_TTL_SECONDS", "7200"))
auto_reply_status: Dict[str, bool] = {}
user_persona: Dict[str, str] = {}

PERSONAS = {
    "sweet": {"title":"甜美女友","style":"溫柔體貼","greetings":"親愛的～我在這裡聽你說 🌸","emoji":"🌸💕😊"},
    "salty": {"title":"傲嬌女友","style":"機智吐槽","greetings":"你又來啦？說吧，哪裡卡住了。😏","emoji":"😏🙄"},
    "moe":   {"title":"萌系女友","style":"動漫語氣","greetings":"呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ","emoji":"✨🎀"},
    "cool":  {"title":"酷系御姐","style":"冷靜精煉","greetings":"我在。說重點。","emoji":"🧊⚡️"},
}
LANGUAGE_MAP = {"英文":"English","日文":"Japanese","韓文":"Korean","繁體中文":"Traditional Chinese","中文":"Traditional Chinese", "en":"English","ja":"Japanese","ko":"Korean","zh":"Traditional Chinese"}
PERSONA_ALIAS = {"甜":"sweet","鹹":"salty","萌":"moe","酷":"cool","random":"random"}
TRANSLATE_CMD = re.compile(r"^(?:翻譯|翻成)\s*(?:->|→|>)?\s*(英文|日文|韓文|繁體中文|中文)\s*$", re.IGNORECASE)
INLINE_TRANSLATE = re.compile(r"^(en|ja|jp|ko|zh|英文|日文|韓文|中文)\s*[:：>]\s*(.+)$", re.IGNORECASE)


# ====== 核心小工具 ======
def _now() -> datetime: return datetime.utcnow()

def get_chat_id(event: MessageEvent) -> str:
    source = event.source
    stype = getattr(source, "type", "")
    uid = getattr(source, "user_id", None)
    gid = getattr(source, "group_id", None)
    rid = getattr(source, "room_id", None)
    if gid: return f"group:{gid}"
    if rid: return f"room:{rid}"
    if uid: return f"user:{uid}"
    return f"{stype or 'unknown'}:{abs(hash(str(source))) % 10_000_000}"

def _tstate_set(chat_id: str, lang_display: str):
    translation_states[chat_id] = lang_display
    translation_states_ttl[chat_id] = _now() + timedelta(seconds=TRANSLATE_TTL_SECONDS)

def _tstate_get(chat_id: str) -> Optional[str]:
    if translation_states_ttl.get(chat_id, _now()) < _now():
        _tstate_clear(chat_id)
        return None
    return translation_states.get(chat_id)

def _tstate_clear(chat_id: str):
    translation_states.pop(chat_id, None)
    translation_states_ttl.pop(chat_id, None)


# ====== UI 元件 ======
def build_quick_reply() -> QuickReply:
    # 確保每次呼叫都建立新的物件，避免狀態汙染
    return QuickReply(items=[
        QuickReplyItem(action=MessageAction(label="主選單", text="選單")),
        QuickReplyItem(action=MessageAction(label="大盤", text="大盤")),
        QuickReplyItem(action=MessageAction(label="金價", text="金價")),
        QuickReplyItem(action=MessageAction(label="查 2330", text="2330")),
        QuickReplyItem(action=MessageAction(label="查 NVDA", text="NVDA")),
        QuickReplyItem(action=MessageAction(label="日圓", text="JPY")),
        QuickReplyItem(action=MessageAction(label="大樂透", text="大樂透")),
        QuickReplyItem(action=PostbackAction(label="💖 AI 人設", data="menu:persona")),
        QuickReplyItem(action=PostbackAction(label="🎰 彩票", data="menu:lottery")),
        QuickReplyItem(action=MessageAction(label="結束翻譯", text="翻譯->結束")),
    ])

def build_main_menu() -> FlexMessage:
    # ... (此函式內容不變，可沿用您 v1.4.8 的版本)
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
    # ... (此函式內容不變，可沿用您 v1.4.8 的版本)
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
            ("今彩539", MessageAction(label="今彩539", text="539")),
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
            rows.append(FlexBox(layout="horizontal", spacing="sm", contents=row))
            row = []
    if row:
        rows.append(FlexBox(layout="horizontal", spacing="sm", contents=row))
    bubble = FlexBubble(
        header=FlexBox(layout="vertical", contents=[FlexText(text=title, weight="bold", size="lg")]),
        body=FlexBox(layout="vertical", spacing="md", contents=rows or [FlexText(text="（尚無項目）")]),
    )
    return FlexMessage(alt_text=title, contents=bubble)

# ====== [FIX] 整合後的彩票分析邏輯 ======
def get_lottery_reply_from_groq(messages):
    if not sync_groq_client: return "Groq API 金鑰未設定。"
    try:
        response = sync_groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", messages=messages, max_tokens=2000, temperature=1.2
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API 在彩票分析中失敗: {e}")
        return f"GROQ API 發生錯誤: {e}"

# 初始化爬蟲 (若找不到檔案會使用 Dummy Class)
lottery_crawler = TaiwanLotteryCrawler()
caiyunfangwei_crawler = CaiyunfangweiCrawler()

def get_lottery_analysis(lottery_type: str) -> str:
    lottery_map = {
        "威力": lottery_crawler.super_lotto, "大樂": lottery_crawler.lotto649,
        "539": lottery_crawler.daily_cash, "雙贏": lottery_crawler.lotto1224,
        "3星": lottery_crawler.lotto3d, "三星": lottery_crawler.lotto3d, "4星": lottery_crawler.lotto4d,
        "38樂": lottery_crawler.lotto38m6, "39樂": lottery_crawler.lotto39m5, "49樂": lottery_crawler.lotto49m6
    }
    last_lotto = "未知的彩券類型"
    for key, func in lottery_map.items():
        if key in lottery_type:
            last_lotto = func()
            break
    
    if isinstance(last_lotto, dict) and 'error' in last_lotto:
        return f"彩票資料獲取失敗：{last_lotto['error']}"

    content_msg = f'你現在是一位專業的樂透彩分析師, 使用{lottery_type}的資料來撰寫分析報告:\n'
    content_msg += f'近幾期號碼資訊:\n{last_lotto}\n'
    try:
        cai_info = caiyunfangwei_crawler.get_caiyunfangwei()
        if 'error' not in cai_info:
            content_msg += f'國歷/農曆：{cai_info.get("今天日期", "未知")}\n'
            content_msg += f'今日歲次：{cai_info.get("今日歲次", "未知")}\n'
            content_msg += f'財神方位：{cai_info.get("財神方位", "未知")}\n'
    except Exception: pass
    
    content_msg += '請分析冷熱門號碼、奇偶分佈等趨勢。\n'
    content_msg += '提供三組推薦號碼(符合該彩種格式，由小到大排序)。威力彩需含第二區。\n'
    content_msg += '最後附上一句20字內勵志的發財吉祥話。\n'
    content_msg += '請使用台灣繁體中文回覆。'

    msg_list = [{"role": "system", "content": f"你是專業的{lottery_type}分析師。"}, {"role": "user", "content": content_msg}]
    return get_lottery_reply_from_groq(msg_list)


# ====== 語音處理, LLM, 金融等其他函式 (沿用 v1.4.8) ======
# ... (此處省略其他未變更的輔助函式，如 speech_to_text_async, get_bot_gold_quote 等，請保留您原有的程式碼)
# 为了简洁，此处不再重复贴出，请确保您的档案中保留这些函式
async def speech_to_text_async(audio_bytes: bytes) -> Optional[str]:
    # ... (No changes)
    pass
async def text_to_speech_async(text: str) -> Optional[str]:
    # ... (No changes)
    pass
def get_analysis_reply(messages: List[dict]) -> str:
    # ... (No changes)
    pass
def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    # ... (No changes)
    pass
# ... etc ...

# ====== [CHG] 統一的回覆出口 ======
async def reply_text_message(reply_token: str, text: str, extras: Optional[List] = None):
    if not text: text = "（無內容）"
    messages = [TextMessage(text=text, quick_reply=build_quick_reply())]
    if extras: messages.extend(extras)
    
    if TTS_SEND_ALWAYS and CLOUDINARY_URL:
        # ... (TTS logic from your v1.4.8 is fine)
        pass
        
    try:
        await line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=messages)
        )
    except Exception as e:
        logger.error(f"Reply message failed: {e}", exc_info=True)


def reply_with_menu(reply_token: str, flex_message: FlexMessage, hint_text: str = "👇 請點選下方功能選單"):
    """
    [FIX] 確保 QuickReply 持續顯示的關鍵函式。
    此函式「總是」先傳送一則帶有 QuickReply 的文字訊息，再附上 FlexMessage，
    確保使用者介面上永遠都看得到快速按鈕列。
    """
    try:
        await line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[
                    TextMessage(text=hint_text, quick_reply=build_quick_reply()),
                    flex_message
                ]
            )
        )
    except Exception as e:
        logger.error(f"Reply with menu failed: {e}", exc_info=True)


# ====== 事件處理主迴圈 ======
async def handle_text_message(event: MessageEvent):
    chat_id = get_chat_id(event)
    msg_raw = (event.message.text or "").strip()
    reply_tok = event.reply_token
    if not msg_raw: return

    # ... (bot name & auto reply logic is fine, can be kept)
    
    msg = msg_raw # Simplified for clarity, keep your @bot logic
    
    # 優先處理翻譯模式
    # ... (Your translation logic from v1.4.8 is fine)
    
    # 彩票關鍵字列表
    lottery_keywords = ["大樂透", "威力彩", "539", "今彩539", "雙贏彩", "3星彩", "三星彩", "4星彩", "38樂合彩", "39樂合彩", "49樂合彩"]
    if msg in lottery_keywords:
        try:
            report = await run_in_threadpool(get_lottery_analysis, msg)
            await reply_text_message(reply_tok, report)
        except Exception as e:
            logger.error(f"彩票分析流程失敗: {e}", exc_info=True)
            await reply_text_message(reply_tok, f"抱歉，分析 {msg} 時發生錯誤。")
        return

    # 主選單
    if msg.lower() in ("menu", "選單", "主選單"):
        await reply_with_menu(reply_tok, build_main_menu())
        return

    # 其他指令 (金價、股票...)
    # ...
    
    # 一般聊天
    # ...
    
async def handle_postback(event: PostbackEvent):
    reply_tok = event.reply_token
    data = event.postback.data or ""
    if data.startswith("menu:"):
        kind = data.split(":", 1)[-1]
        await reply_with_menu(reply_tok, build_submenu(kind), hint_text="👇 子選單")

async def handle_events(events):
    for event in events:
        if isinstance(event, MessageEvent):
            if isinstance(event.message, TextMessageContent):
                await handle_text_message(event)
            # ... handle audio etc.
        elif isinstance(event, PostbackEvent):
            await handle_postback(event)

# ====== FastAPI 路由設定 ======
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... (lifespan logic is fine, can be kept)
    yield

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.4.9")
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
    return JSONResponse({"status": "ok"})

# ... (root and healthz routes are fine)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)