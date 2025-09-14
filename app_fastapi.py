# app_fastapi.py
# ========== 1) Imports ==========
import os
import re
import io
import random
import logging
import asyncio
from typing import Dict, List, Optional
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

# --- 雲端儲存 (Cloudinary) ---
import cloudinary
import cloudinary.uploader

# --- gTTS（免費 TTS） ---
from gtts import gTTS

# --- LINE Bot SDK v3 Imports ---
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    AudioMessageContent,
    PostbackEvent,
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    AsyncMessagingApi,
    ReplyMessageRequest,
    TextMessage,
    AudioMessage,
    FlexMessage,
    FlexBubble,
    FlexBox,
    FlexText,
    FlexButton,
    QuickReply,
    QuickReplyItem,
    MessageAction,
    PostbackAction,
    BotInfoResponse,
)

# --- AI 相關 ---
from groq import AsyncGroq, Groq
import openai

# --- 自訂模組（錯誤處理） ---
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
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto").lower()  # auto / openai / gtts

if not CHANNEL_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("缺少必要環境變數：CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET")

# --- Cloudinary 設定 ---
if CLOUDINARY_URL:
    try:
        cloudinary.config(
            cloud_name=re.search(r"@(.+)", CLOUDINARY_URL).group(1),
            api_key=re.search(r"//(\d+):", CLOUDINARY_URL).group(1),
            api_secret=re.search(r":([A-Za-z0-9_-]+)@", CLOUDINARY_URL).group(1),
        )
        logger.info("✅ Cloudinary 設定成功！")
    except Exception as e:
        logger.error(f"Cloudinary 設定失敗: {e}")
        CLOUDINARY_URL = None
else:
    logger.warning("未設定 CLOUDINARY_URL，TTS 語音訊息將無法傳送。")

# --- API 用戶端初始化 ---
configuration = Configuration(access_token=CHANNEL_TOKEN)
async_api_client = ApiClient(configuration=configuration)
line_bot_api = AsyncMessagingApi(api_client=async_api_client)
handler = WebhookHandler(CHANNEL_SECRET)

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
sync_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

openai_client = None
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("未設定 OPENAI_API_KEY，語音轉文字與 OpenAI TTS 將停用（將以 gTTS 為主）。")

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
    "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji": "🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji": "😏🙄"},
    "moe": {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji": "✨🎀"},
    "cool": {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議", "greetings": "我在。說重點。", "emoji": "🧊⚡️"},
}
LANGUAGE_MAP = {
    "英文": "English",
    "日文": "Japanese",
    "韓文": "Korean",
    "越南文": "Vietnamese",
    "繁體中文": "Traditional Chinese",
}

# ========== 3) FastAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    嘗試兩個官方端點：api-data 與 api（避免區域性限制）
    """
    if BASE_URL:
        async with httpx.AsyncClient() as c:
            for endpoint in (
                "https://api-data.line.me/v2/bot/channel/webhook/endpoint",
                "https://api.line.me/v2/bot/channel/webhook/endpoint",
            ):
                try:
                    headers = {
                        "Authorization": f"Bearer {CHANNEL_TOKEN}",
                        "Content-Type": "application/json",
                    }
                    payload = {"endpoint": f"{BASE_URL}/callback"}
                    r = await c.put(endpoint, headers=headers, json=payload, timeout=10.0)
                    r.raise_for_status()
                    logger.info(f"✅ Webhook 更新成功: {endpoint} {r.status_code}")
                    break
                except Exception as e:
                    logger.warning(f"Webhook 更新失敗（嘗試 {endpoint}）: {e}")
    else:
        logger.warning("未設定 BASE_URL，跳過 Webhook 更新。")
    yield

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.2.0")
router = APIRouter()

# ========== 4) Helpers ==========
def get_chat_id(event: MessageEvent) -> str:
    """
    透過判斷 source.type 取得 chat ID（不依賴型別檢查，較穩定）
    """
    source = event.source
    if getattr(source, "type", "") == "group":
        return source.group_id
    if getattr(source, "type", "") == "room":
        return source.room_id
    return source.user_id

# ---------- 金價抓取（強化版，對應台銀文字內容） ----------
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

    # 掛牌時間（例如：2025/09/05 19:30）
    m_time = re.search(r"掛牌時間[:：]\s*([0-9]{4}/[0-9]{2}/[0-9]{2}\s+[0-9]{2}:[0-9]{2})", text)
    listed_at = m_time.group(1) if m_time else None

    # 本行賣出 / 本行買進（允許千分位與小數）
    m_sell = re.search(r"本行賣出\s*([0-9,]+(?:\.[0-9]+)?)", text)
    m_buy = re.search(r"本行買進\s*([0-9,]+(?:\.[0-9]+)?)", text)
    if not (m_sell and m_buy):
        raise RuntimeError("找不到『本行賣出/本行買進』欄位")

    sell = float(m_sell.group(1).replace(",", ""))
    buy = float(m_buy.group(1).replace(",", ""))

    return {
        "listed_at": listed_at,
        "sell_twd_per_g": sell,  # 本行賣出（TWD/克）
        "buy_twd_per_g": buy,    # 本行買進（TWD/克）
        "source": BOT_GOLD_URL,
    }

def get_bot_gold_quote() -> dict:
    r = requests.get(BOT_GOLD_URL, headers=DEFAULT_HEADERS, timeout=10)
    r.raise_for_status()
    return parse_bot_gold_text(r.text)

def format_gold_report(data: dict) -> str:
    ts = data.get("listed_at") or "（頁面未標示）"
    sell = data["sell_twd_per_g"]
    buy = data["buy_twd_per_g"]
    spread = sell - buy
    bias = "盤整" if spread <= 30 else ("偏寬" if spread <= 60 else "價差偏大")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return (
        f"**金價快報（台灣銀行）**\n"
        f"- 資料時間：{ts}\n"
        f"- 本行賣出（1克）：**{sell:,.0f} 元**\n"
        f"- 本行買進（1克）：**{buy:,.0f} 元**\n"
        f"- 買賣價差：{spread:,.0f} 元（{bias}）\n"
        f"\n資料來源：{BOT_GOLD_URL}\n（更新於 {now}）"
    )

def get_gold_analysis() -> str:
    try:
        data = get_bot_gold_quote()
        return format_gold_report(data)
    except Exception as e:
        logger.error(f"金價流程失敗：{e}", exc_info=True)
        return "抱歉，目前無法從台灣銀行取得黃金牌價。稍後再試一次 🙏"

# ---------- 匯率 ----------
def get_currency_analysis(target_currency: str) -> str:
    try:
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("result") == "success":
            rate = data["rates"].get("TWD")
            if rate is None:
                return "抱歉，API中找不到 TWD 的匯率資訊。"
            return f"最新：1 {target_currency.upper()} ≈ {rate:.5f} 新台幣"
        else:
            return f"抱歉，獲取匯率資料失敗：{data.get('error-type', '未知錯誤')}"
    except Exception as e:
        logger.error(f"處理 {target_currency} API 資料時發生錯誤: {e}", exc_info=True)
        return "抱歉，處理外匯資料時發生內部錯誤，請稍後再試。"

# ---------- 標準聊天/分析 ----------
def get_analysis_reply(messages: List[dict]) -> str:
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
        return await run_in_threadpool(lambda: get_analysis_reply(messages))
    resp = await async_groq_client.chat.completions.create(
        model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    return resp.choices[0].message.content.strip()

# ---------- 彩票 (已更新) ----------
def get_lottery_analysis(lottery_type_input: str) -> str:
    if not LOTTERY_ENABLED:
        return "彩票模組未啟用。"
    
    lt = lottery_type_input.lower()
    if "威力" in lt:
        data = lottery_crawler.super_lotto()
        lottery_name = "威力彩"
    elif "大樂" in lt:
        data = lottery_crawler.lotto649()
        lottery_name = "大樂透"
    elif "539" in lt:
        data = lottery_crawler.daily_cash()
        lottery_name = "今彩539"
    else:
        return f"不支援 {lottery_type_input}。"

    # 獲取財神方位資訊，如果失敗則優雅地跳過
    extra_info = ""
    try:
        info = caiyunfangwei_crawler.get_caiyunfangwei()
        extra_info = (
            f'***財神方位提示***\n'
            f'國歷：{info.get("今天日期", "未知")}\n'
            f'農曆：{info.get("今日歲次", "未知")}\n'
            f'今日財神方位：**{info.get("財神方位", "未知")}**\n\n'
        )
    except Exception as e:
        logger.warning(f"無法獲取財神方位資訊: {e}")
        extra_info = "財神方位資訊暫時無法獲取。\n\n"

    # 建立更詳細的分析指令 (Prompt)
    prompt = (
        f"你是一位專業的樂透彩分析師，請基於以下「{lottery_name}」的最近幾期開獎號碼資料，撰寫一份詳細的趨勢分析報告，並遵循以下指示：\n\n"
        f"1.  **開頭資訊**：請先顯示我提供的「財神方位提示」。\n"
        f"2.  **數據來源**：清楚列出最近幾期的開獎號碼。\n"
        f"   - 資料:\n{data}\n\n"
        f"3.  **趨勢分析**：\n"
        f"   - 分析並列出「最熱門的號碼」(Hot Numbers) 和「最冷門的號碼」(Cold Numbers)。\n"
        f"   - 根據號碼分佈（例如大小、奇偶比例）提供簡要的趨勢觀察。\n\n"
        f"4.  **推薦號碼**：\n"
        f"   - 根據你的專業分析，提供三組推薦號碼。\n"
        f"   - 號碼組合必須符合「{lottery_name}」的遊戲規則（例如：大樂透為6個號碼，威力彩為6+1個號碼）。\n"
        f"   - 每組號碼請由小到大排序。\n\n"
        f"5.  **結語**：最後，請附上一句20字以內、具有勵志感的發財吉祥話。\n\n"
        f"請務必使用台灣用語的繁體中文回覆。"
    )

    # 呼叫 AI 模型進行分析
    return get_analysis_reply(
        [{"role": "system", "content": f"你是一位專業且詳細的「{lottery_name}」彩券分析師。"}, {"role": "user", "content": prompt}]
    )

# ---------- 股票（簡版） ----------
def get_stock_analysis(stock_id_input: str) -> str:
    if not STOCK_ENABLED:
        return "股票模組未啟用。"
    try:
        stock = yf.Ticker(f"{stock_id_input}.TW" if stock_id_input.isdigit() else stock_id_input)
        info = stock.info
        name = info.get("longName", stock_id_input)
        price = info.get("currentPrice", "N/A")
        prev_close = info.get("previousClose", "N/A")
        return f"**{name} ({stock_id_input})**\n- 即時股價: {price}\n- 昨日收盤: {prev_close}"
    except Exception as e:
        logger.error(f"股票查詢失敗：{e}", exc_info=True)
        return f"查詢 {stock_id_input} 失敗：{e}"

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
    return await groq_chat_async([{"role": "system", "content": sys}, {"role": "user", "content": usr}], 800, 0.2)

def set_user_persona(chat_id: str, key: str):
    key_mapped = {
        "甜": "sweet",
        "鹹": "salty",
        "萌": "moe",
        "酷": "cool",
        "random": "random",
    }.get(key, key)

    if key_mapped == "random":
        key_mapped = random.choice(list(PERSONAS.keys()))
    if key_mapped not in PERSONAS:
        key_mapped = "sweet"

    user_persona[chat_id] = key_mapped
    return key_mapped

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    key = user_persona.get(chat_id, "sweet")
    p = PERSONAS[key]
    return (
        f"你是一位「{p['title']}」。風格：{p['style']}\n"
        f"使用者情緒：{sentiment}；請調整語氣（開心→一起開心；難過/生氣→先共情、安撫再給建議；中性→自然聊天）。\n"
        f"回覆使用繁體中文，精煉自然，帶少量表情 {p['emoji']}。"
    )

# --- Quick Reply / Flex ---
def build_quick_reply() -> QuickReply:
    actions = [
        MessageAction(label="主選單", text="選單"),
        MessageAction(label="台股大盤", text="^TWII"),
        MessageAction(label="查台積電", text="2330"),
        PostbackAction(label="💖 AI 人設", data="menu:persona"),
    ]
    return QuickReply(items=[QuickReplyItem(action=a) for a in actions])

def build_flex_menu(title, items_data, alt_text):
    buttons = []
    for label, action_obj, _ in items_data:
        buttons.append(FlexButton(action=action_obj))
    bubble = FlexBubble(
        header=FlexBox(layout="vertical", contents=[FlexText(text=title, weight="bold", size="lg")]),
        body=FlexBox(layout="vertical", spacing="md", contents=buttons),
    )
    return FlexMessage(alt_text=alt_text, contents=bubble)

def build_main_menu():
    items = [
        ("💹 金融查詢", PostbackAction(label="💹 金融查詢", data="menu:finance"), "finance"),
        ("🎰 彩票分析", PostbackAction(label="🎰 彩票分析", data="menu:lottery"), "lottery"),
        ("💖 AI 角色扮演", PostbackAction(label="💖 AI 角色扮演", data="menu:persona"), "persona"),
        ("🌐 翻譯工具", PostbackAction(label="🌐 翻譯工具", data="menu:translate"), "translate"),
    ]
    return build_flex_menu("AI 助理主選單", items, "主選單")

def build_submenu(kind: str):
    menus = {
        "finance": (
            "💹 金融查詢",
            [
                ("台股大盤", MessageAction(label="台股大盤", text="^TWII"), ""),
                ("美股 S&P500", MessageAction(label="美股 S&P500", text="^GSPC"), ""),
                ("黃金價格", MessageAction(label="黃金價格", text="金價"), ""),
                ("日圓匯率", MessageAction(label="日圓匯率", text="JPY"), ""),
            ],
        ),
        "lottery": (
            "🎰 彩票分析",
            [
                ("大樂透", MessageAction(label="大樂透", text="大樂透"), ""),
                ("威力彩", MessageAction(label="威力彩", text="威力彩"), ""),
                ("今彩539", MessageAction(label="今彩539", text="539"), ""),
            ],
        ),
        "persona": (
            "💖 AI 角色扮演",
            [
                ("甜美女友", MessageAction(label="甜美女友", text="甜"), ""),
                ("傲嬌女友", MessageAction(label="傲嬌女友", text="鹹"), ""),
                ("萌系女友", MessageAction(label="萌系女友", text="萌"), ""),
                ("酷系御姐", MessageAction(label="酷系御姐", text="酷"), ""),
            ],
        ),
        "translate": (
            "🌐 翻譯工具",
            [
                ("翻成英文", MessageAction(label="翻成英文", text="翻譯->英文"), ""),
                ("翻成日文", MessageAction(label="翻成日文", text="翻譯->日文"), ""),
                ("結束翻譯", MessageAction(label="結束翻譯", text="翻譯->結束"), ""),
            ],
        ),
    }
    title, items = menus.get(kind, ("無效選單", []))
    return build_flex_menu(title, items, title)

# ========== 5) 上傳與 TTS ==========
def _upload_audio_sync(audio_bytes: bytes) -> Optional[dict]:
    if not CLOUDINARY_URL:
        return None
    try:
        response = cloudinary.uploader.upload(
            io.BytesIO(audio_bytes),
            resource_type="video",  # 用 video 才能穩定播放 MP3
            folder="line-bot-tts",
            format="mp3",
        )
        return response
    except Exception as e:
        logger.error(f"Cloudinary 上傳失敗: {e}")
        return None

async def upload_audio_to_cloudinary(audio_bytes: bytes) -> Optional[str]:
    response = await run_in_threadpool(_upload_audio_sync, audio_bytes)
    return response.get("secure_url") if response else None

def _create_tts_with_openai_sync(text: str) -> Optional[bytes]:
    if not openai_client:
        return None
    try:
        clean = re.sub(r"[*_`~#]", "", text)
        resp = openai_client.audio.speech.create(model="tts-1", voice="nova", input=clean)
        return resp.read()
    except Exception as e:
        logger.error(f"OpenAI TTS 生成失敗: {e}", exc_info=True)
        return None

def _create_tts_with_gtts_sync(text: str) -> Optional[bytes]:
    try:
        clean = re.sub(r"[*_`~#]", "", text).strip() or "嗨，我在這裡。"
        tts = gTTS(text=clean, lang="zh-TW", tld="com.tw", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"gTTS 生成失敗: {e}", exc_info=True)
        return None

async def text_to_speech_async(text: str) -> Optional[bytes]:
    """
    依照 TTS_PROVIDER 決定用哪個 TTS。
    - auto: 先 openai 後 gtts
    - openai: 只用 openai
    - gtts: 只用 gtts
    """
    provider = TTS_PROVIDER

    async def try_openai():
        return await run_in_threadpool(_create_tts_with_openai_sync, text)

    async def try_gtts():
        return await run_in_threadpool(_create_tts_with_gtts_sync, text)

    if provider == "openai":
        return await try_openai()

    if provider == "gtts":
        return await try_gtts()

    # auto：先 OpenAI（若有）再 gTTS
    if openai_client:
        b = await try_openai()
        if b:
            return b
    return await try_gtts()

# ---------- STT（語音轉文字） ----------
def _transcribe_with_openai_sync(audio_bytes: bytes, filename: str = "audio.m4a") -> Optional[str]:
    if not openai_client:
        return None
    try:
        f = io.BytesIO(audio_bytes)
        f.name = filename
        resp = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
        return (resp.text or "").strip() or None
    except Exception as e:
        logger.warning(f"OpenAI STT 失敗：{e}")
        return None

def _transcribe_with_groq_sync(audio_bytes: bytes, filename: str = "audio.m4a") -> Optional[str]:
    if not sync_groq_client:
        return None
    try:
        f = io.BytesIO(audio_bytes)
        f.name = filename
        # Groq Python SDK 的音訊端點（兼容 whisper-large-v3）
        resp = sync_groq_client.audio.transcriptions.create(file=f, model="whisper-large-v3")
        return (resp.text or "").strip() or None
    except Exception as e:
        logger.warning(f"Groq STT 失敗：{e}")
        return None

async def speech_to_text_async(audio_bytes: bytes) -> Optional[str]:
    # 先 OpenAI 再 Groq（若都有 Key）
    text = await run_in_threadpool(_transcribe_with_openai_sync, audio_bytes)
    if text:
        return text
    text = await run_in_threadpool(_transcribe_with_groq_sync, audio_bytes)
    return text

# ========== 6) LINE Event Handlers ==========
@handler.add(MessageEvent, message=TextMessageContent)
async def handle_text_message(event: MessageEvent):
    chat_id, msg, reply_token = get_chat_id(event), event.message.text.strip(), event.reply_token
    try:
        bot_info: BotInfoResponse = await line_bot_api.get_bot_info()
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    # 群組/聊天室提及判定（簡易版）
    if hasattr(event.source, "group_id") or hasattr(event.source, "room_id"):
        if not msg.startswith(f"@{bot_name}"):
            return
        msg = re.sub(f'^@{re.escape(bot_name)}\\s*', "", msg)

    if not msg:
        return

    final_reply_text, low = "", msg.lower()
    try:
        if low in ("menu", "選單"):
            await line_bot_api.reply_message(
                ReplyMessageRequest(reply_token=reply_token, messages=[build_main_menu()])
            )
            return

        elif low in ("大樂透", "威力彩", "539"):
            final_reply_text = get_lottery_analysis(low)

        elif low in ("金價", "黃金"):
            final_reply_text = get_gold_analysis()

        elif low.upper() in ("JPY", "USD", "EUR"):
            final_reply_text = get_currency_analysis(low)

        elif re.fullmatch(r"\^?[A-Z0-9.]{2,10}", msg) or msg.isdigit():
            final_reply_text = get_stock_analysis(msg.upper())

        elif low in ("甜", "鹹", "萌", "酷", "random"):
            key = set_user_persona(chat_id, low)
            p = PERSONAS[key]
            final_reply_text = f"💖 已切換人設：{p['title']}\n{p['greetings']}"

        elif low.startswith("翻譯->"):
            lang = low.split("->", 1)[1].strip()
            if lang == "結束":
                translation_states.pop(chat_id, None)
                final_reply_text = "✅ 已結束翻譯模式"
            else:
                translation_states[chat_id] = lang
                final_reply_text = f"🌐 已開啟翻譯 → {lang}"

        elif chat_id in translation_states:
            final_reply_text = await translate_text(msg, translation_states[chat_id])

        else:
            sentiment = await analyze_sentiment(msg)
            sys_prompt = build_persona_prompt(chat_id, sentiment)
            history = conversation_history.setdefault(chat_id, [])
            messages = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": msg}]
            final_reply_text = await groq_chat_async(messages)
            history.extend([{"role": "user", "content": msg}, {"role": "assistant", "content": final_reply_text}])
            conversation_history[chat_id] = history[-MAX_HISTORY_LEN * 2 :]
    except Exception as e:
        logger.error(f"指令 '{msg}' 處理失敗: {e}", exc_info=True)
        final_reply_text = "抱歉，處理時發生錯誤 😵"

    # --- 最終回覆（文字 + 可選語音） ---
    messages_to_send = [TextMessage(text=final_reply_text, quick_reply=build_quick_reply())]

    if final_reply_text and CLOUDINARY_URL:
        audio_bytes = await text_to_speech_async(final_reply_text)
        if audio_bytes:
            public_audio_url = await upload_audio_to_cloudinary(audio_bytes)
            if public_audio_url:
                # duration 估值（粗估：字數 * 60ms，再 clamp）
                est_dur = max(3000, min(30000, len(final_reply_text) * 60))
                messages_to_send.append(
                    AudioMessage(original_content_url=public_audio_url, duration=est_dur)
                )
                logger.info("✅ 成功上傳 TTS 語音並加入回覆佇列。")

    await line_bot_api.reply_message(
        ReplyMessageRequest(reply_token=reply_token, messages=messages_to_send)
    )

@handler.add(MessageEvent, message=AudioMessageContent)
async def handle_audio_message(event: MessageEvent):
    reply_token = event.reply_token
    try:
        content_stream = await line_bot_api.get_message_content(event.message.id)
        audio_in = await content_stream.read()

        text = await speech_to_text_async(audio_in)
        if not text:
            raise RuntimeError("語音轉文字失敗")

        sentiment = await analyze_sentiment(text)
        sys_prompt = build_persona_prompt(get_chat_id(event), sentiment)
        final_reply_text = await groq_chat_async(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}]
        )

        messages_to_send = [
            TextMessage(text=f"🎧 我聽到了：\n{text}\n\n—\n{final_reply_text}", quick_reply=build_quick_reply())
        ]

        if final_reply_text and CLOUDINARY_URL:
            audio_out = await text_to_speech_async(final_reply_text)
            if audio_out:
                public_audio_url = await upload_audio_to_cloudinary(audio_out)
                if public_audio_url:
                    est_dur = max(3000, min(30000, len(final_reply_text) * 60))
                    messages_to_send.append(
                        AudioMessage(original_content_url=public_audio_url, duration=est_dur)
                    )
                    logger.info("✅ 成功上傳 TTS 語音並加入回覆佇列。")

        await line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=messages_to_send)
        )

    except Exception as e:
        logger.error(f"處理語音訊息失敗: {e}", exc_info=True)
        await line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text="抱歉，我沒聽清楚，可以再說一次嗎？")])
        )

@handler.add(PostbackEvent)
async def handle_postback(event: PostbackEvent):
    data = event.postback.data
    if data.startswith("menu:"):
        kind = data.split(":", 1)[-1]
        await line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=event.reply_token, messages=[build_submenu(kind)])
        )

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
    return PlainTextResponse("LINE Bot is running.")

app.include_router(router)

# ========== 8) Local run ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)