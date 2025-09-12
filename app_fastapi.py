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

# --- 雲端儲存 (Cloudinary) ---
import cloudinary
import cloudinary.uploader

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
    FlexSeparator,
    QuickReply,
    QuickReplyItem,
    MessageAction,
    PostbackAction,
    BotInfoResponse,  # 正確的 Bot Info 回應類別
    SourceUser,
    SourceGroup,
    SourceRoom,
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

if not CHANNEL_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("缺少必要環境變數：CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET")

# --- Cloudinary 設定 ---
if CLOUDINARY_URL:
    try:
        cloudinary.config(cloud_name = re.search(r"@(.+)", CLOUDINARY_URL).group(1),
                          api_key = re.search(r"//(\d+):", CLOUDINARY_URL).group(1),
                          api_secret = re.search(r":([A-Za-z0-9_-]+)@", CLOUDINARY_URL).group(1))
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
    logger.warning("未設定 OPENAI_API_KEY，語音轉文字與 TTS 功能將停用。")

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
                r = await c.put("https://api-data.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=payload, timeout=10.0)
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
    source = event.source
    if isinstance(source, SourceGroup): return source.group_id
    if isinstance(source, SourceRoom): return source.room_id
    return source.user_id

# --- 上傳與 TTS 輔助函式 ---
def _upload_audio_sync(audio_bytes: bytes) -> dict | None:
    if not CLOUDINARY_URL: return None
    try:
        response = cloudinary.uploader.upload(
            io.BytesIO(audio_bytes),
            resource_type="video",
            folder="line-bot-tts",
            format="mp3"
        )
        return response
    except Exception as e:
        logger.error(f"Cloudinary 上傳失敗: {e}")
        return None

async def upload_audio_to_cloudinary(audio_bytes: bytes) -> str | None:
    response = await run_in_threadpool(_upload_audio_sync, audio_bytes)
    return response.get("secure_url") if response else None

def _create_tts_with_openai_sync(text: str) -> bytes | None:
    if not openai_client: return None
    try:
        text_for_speech = re.sub(r'[*_`~#]', '', text)
        response = openai_client.audio.speech.create(model="tts-1", voice="nova", input=text_for_speech)
        return response.read()
    except Exception as e:
        logger.error(f"OpenAI TTS 生成失敗: {e}", exc_info=True)
        return None

async def text_to_speech_async(text: str) -> bytes | None:
    return await run_in_threadpool(_create_tts_with_openai_sync, text)

# --- 其他輔助函式 ---
def get_analysis_reply(messages):
    try:
        if openai_client:
            resp = openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=1500, temperature=0.7)
            return resp.choices[0].message.content
        raise Exception("OpenAI client not initialized.")
    except Exception as openai_err:
        logger.warning(f"OpenAI API 失敗: {openai_err}")
        try:
            if not sync_groq_client: raise Exception("Groq client not initialized.")
            resp = sync_groq_client.chat.completions.create(model=GROQ_MODEL_PRIMARY, messages=messages, max_tokens=2000, temperature=0.7)
            return resp.choices[0].message.content
        except Exception as groq_err:
            logger.warning(f"Groq 主要模型失敗: {groq_err}")
            try:
                if not sync_groq_client: raise Exception("Groq client not initialized.")
                resp = sync_groq_client.chat.completions.create(model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=1500, temperature=0.9)
                return resp.choices[0].message.content
            except Exception as fallback_err:
                logger.error(f"所有 AI API 都失敗: {fallback_err}"); return "（分析模組暫時連線不穩定）"
async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
    if not async_groq_client: return await run_in_threadpool(lambda: get_analysis_reply(messages))
    resp = await async_groq_client.chat.completions.create(model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature)
    return resp.choices[0].message.content.strip()

def get_gold_analysis():
    # ... (此處省略部分輔助函式，維持與你原碼一致) ...
    r = requests.get("https://rate.bot.com.tw/gold?Lang=zh-TW", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    m_time = re.search(r"掛牌時間[:：]\s*(\S+\s+\S+)", text)
    m_sell = re.search(r"本行賣出\s*([\d,]+)", text)
    m_buy = re.search(r"本行買進\s*([\d,]+)", text)
    if not (m_time and m_sell and m_buy): raise RuntimeError("無法解析台銀金價頁面")
    ts, sell, buy = m_time.group(1), int(m_sell.group(1).replace(",", "")), int(m_buy.group(1).replace(",", ""))
    return f"**金價快報（台灣銀行）**\n- 資料時間：{ts}\n- 賣出價：**${sell:,}**\n- 買進價：**${buy:,}**"

def get_currency_analysis(target: str):
    r = requests.get(f"https://open.er-api.com/v6/latest/{target.upper()}", timeout=10)
    r.raise_for_status(); data = r.json()
    if data.get("result") != "success": return f"獲取匯率資料失敗：{data.get('error-type')}"
    return f"最新匯率：1 {target.upper()} ≈ ${data['rates'].get('TWD', 0):.5f} 新台幣"

def get_lottery_analysis(lotto_type: str):
    if not LOTTERY_ENABLED: return "彩票模組未啟用。"
    # ... (此處省略部分輔助函式，維持與你原碼一致) ...
    if "威力" in lotto_type: data = lottery_crawler.super_lotto()
    elif "大樂" in lotto_type: data = lottery_crawler.lotto649()
    elif "539" in lotto_type: data = lottery_crawler.daily_cash()
    else: return f"不支援 {lotto_type}。"
    prompt = f'你是一位專業的樂透彩分析師，請基於以下近幾期號碼資料，撰寫詳細趨勢分析並給出三組推薦號碼（符合彩種格式）。\n\n資料:\n{data}\n\n請用繁體中文回覆。'
    return get_analysis_reply([{"role": "system", "content": "你是彩券分析師。"}, {"role": "user", "content": prompt}])

def get_stock_analysis(stock_id: str):
    if not STOCK_ENABLED: return "股票模組未啟用。"
    # ... (此處省略部分輔助函式，維持與你原碼一致) ...
    try:
        stock = yf.Ticker(f"{stock_id}.TW" if stock_id.isdigit() else stock_id)
        info = stock.info
        name = info.get('longName', stock_id)
        price = info.get('currentPrice', 'N/A')
        prev_close = info.get('previousClose', 'N/A')
        return f"**{name} ({stock_id})**\n- 即時股價: ${price}\n- 昨日收盤: ${prev_close}"
    except Exception as e:
        return f"查詢 {stock_id} 失敗: {e}"

async def analyze_sentiment(text: str):
    return (await groq_chat_async([{"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."}, {"role":"user","content":text}], 10, 0) or "neutral").lower()

async def translate_text(text: str, lang: str):
    target = LANGUAGE_MAP.get(lang, lang)
    return await groq_chat_async([{"role":"system","content":"You are a precise translation engine. Output ONLY the translated text."}, {"role":"user","content":f'{{"text":"{text}","target_language":"{target}"}}'}], 800, 0.2)

def set_user_persona(chat_id, key):
    chosen_key = random.choice(list(PERSONAS.keys())) if key == "random" else key
    user_persona[chat_id] = chosen_key
    return chosen_key

def build_persona_prompt(chat_id, sentiment):
    key = user_persona.get(chat_id, "sweet")
    p = PERSONAS.get(key, PERSONAS["sweet"])
    return f"你是一位「{p['title']}」。風格：{p['style']}。使用者情緒：{sentiment}。回覆請簡短、自然，並帶少量表情符號 {p['emoji']}。"

# --- UI Builders ---
def build_quick_reply():
    actions = [MessageAction(label="主選單", text="選單"), MessageAction(label="台股大盤", text="^TWII"), MessageAction(label="查台積電", text="2330"), PostbackAction(label="💖 AI 人設", data="menu:persona")]
    return QuickReply(items=[QuickReplyItem(action=a) for a in actions])

def build_flex_menu(title, items_data, alt_text):
    buttons = []
    for label, action_obj, data_str in items_data:
        style = "primary" if "finance" in data_str or "lottery" in data_str else "secondary"
        buttons.append(FlexButton(action=action_obj, style=style))
    bubble = FlexBubble(header=FlexBox(layout="vertical", contents=[FlexText(text=title, weight="bold", size="lg")]), body=FlexBox(layout="vertical", spacing="md", contents=buttons))
    return FlexMessage(alt_text=alt_text, contents=bubble)

def build_main_menu():
    items = [("💹 金融查詢", PostbackAction(label="💹 金融查詢", data="menu:finance"), "finance"), ("🎰 彩票分析", PostbackAction(label="🎰 彩票分析", data="menu:lottery"), "lottery"), ("💖 AI 角色扮演", PostbackAction(label="💖 AI 角色扮演", data="menu:persona"), "persona"), ("🌐 翻譯工具", PostbackAction(label="🌐 翻譯工具", data="menu:translate"), "translate")]
    return build_flex_menu("AI 助理主選單", items, "主選單")

def build_submenu(kind):
    menus = {
        "finance": ("💹 金融查詢", [("台股大盤", MessageAction(label="台股大盤", text="^TWII"), ""), ("美股 S&P500", MessageAction(label="美股 S&P500", text="^GSPC"), ""), ("黃金價格", MessageAction(label="黃金價格", text="金價"), ""), ("日圓匯率", MessageAction(label="日圓匯率", text="JPY"), "")]),
        "lottery": ("🎰 彩票分析", [("大樂透", MessageAction(label="大樂透", text="大樂透"), ""), ("威力彩", MessageAction(label="威力彩", text="威力彩"), ""), ("今彩539", MessageAction(label="今彩539", text="539"), "")]),
        "persona": ("💖 AI 角色扮演", [("甜美女友", MessageAction(label="甜美女友", text="甜"), ""), ("傲嬌女友", MessageAction(label="傲嬌女友", text="鹹"), ""), ("萌系女友", MessageAction(label="萌系女友", text="萌"), ""), ("酷系御姐", MessageAction(label="酷系御姐", text="酷"), "")]),
        "translate": ("🌐 翻譯工具", [("翻成英文", MessageAction(label="翻成英文", text="翻譯->英文"), ""), ("翻成日文", MessageAction(label="翻成日文", text="翻譯->日文"), ""), ("結束翻譯", MessageAction(label="結束翻譯", text="翻譯->結束"), "")])
    }
    title, items = menus.get(kind, ("無效選單", []))
    return build_flex_menu(title, items, title)

# ========== 5) LINE Event Handlers ==========
@handler.add(MessageEvent, message=TextMessageContent)
async def handle_text_message(event: MessageEvent):
    chat_id, msg, reply_token = get_chat_id(event), event.message.text.strip(), event.reply_token
    try:
        bot_info: BotInfoResponse = await line_bot_api.get_bot_info()
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if isinstance(event.source, (SourceGroup, SourceRoom)) and not msg.startswith(f"@{bot_name}"):
        return
    
    msg = re.sub(f'^@{bot_name}\\s*', '', msg)
    if not msg: return

    final_reply_text, low = "", msg.lower()
    try:
        if low in ("menu", "選單"):
            await line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=[build_main_menu()]))
            return
        elif low in ("大樂透", "威力彩", "539"): final_reply_text = get_lottery_analysis(low)
        elif low in ("金價", "黃金"): final_reply_text = get_gold_analysis()
        elif low.upper() in ("JPY", "USD", "EUR"): final_reply_text = get_currency_analysis(low)
        elif re.fullmatch(r'\^?[A-Z0-9.]{2,10}', msg) or msg.isdigit(): final_reply_text = get_stock_analysis(msg.upper())
        elif low in PERSONAS:
            key = set_user_persona(chat_id, low); p = PERSONAS[key]; final_reply_text = f"💖 已切換人設：{p['title']}\n{p['greetings']}"
        elif low.startswith("翻譯->"):
            lang = low.split("->", 1)[1].strip()
            if lang == "結束": translation_states.pop(chat_id, None); final_reply_text = "✅ 已結束翻譯模式"
            else: translation_states[chat_id] = lang; final_reply_text = f"🌐 已開啟翻譯 → {lang}"
        elif chat_id in translation_states:
            final_reply_text = await translate_text(msg, translation_states[chat_id])
        else:
            sentiment = await analyze_sentiment(msg)
            sys_prompt = build_persona_prompt(chat_id, sentiment)
            history = conversation_history.setdefault(chat_id, [])
            messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]
            final_reply_text = await groq_chat_async(messages)
            history.extend([{"role":"user","content":msg}, {"role":"assistant","content":final_reply_text}])
            conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
    except Exception as e:
        logger.error(f"指令 '{msg}' 處理失敗: {e}", exc_info=True)
        final_reply_text = "抱歉，處理時發生錯誤 😵"
    
    # --- 最終回覆邏輯 (整合 TTS) ---
    messages_to_send = [TextMessage(text=final_reply_text, quick_reply=build_quick_reply())]
    if final_reply_text and openai_client and CLOUDINARY_URL:
        audio_bytes = await text_to_speech_async(final_reply_text)
        if audio_bytes:
            public_audio_url = await upload_audio_to_cloudinary(audio_bytes)
            if public_audio_url:
                messages_to_send.append(AudioMessage(original_content_url=public_audio_url, duration=20000))
                logger.info("✅ 成功上傳 TTS 語音並加入回覆佇列。")
            
    await line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=messages_to_send))

@handler.add(MessageEvent, message=AudioMessageContent)
async def handle_audio_message(event: MessageEvent):
    reply_token = event.reply_token
    try:
        content_stream = await line_bot_api.get_message_content(event.message.id)
        audio_in = await content_stream.read()
        
        if not openai_client:
            raise RuntimeError("OpenAI client 未設定，無法處理語音。")
        
        text = await run_in_threadpool(lambda: openai_client.audio.transcriptions.create(model="whisper-1", file=("audio.m4a", audio_in)).text)

        if not text: raise RuntimeError("語音轉文字失敗")
        
        sentiment = await analyze_sentiment(text)
        sys_prompt = build_persona_prompt(get_chat_id(event), sentiment)
        final_reply_text = await groq_chat_async([{"role":"system","content":sys_prompt}, {"role":"user","content":text}])
        
        messages_to_send = [TextMessage(text=f"🎧 我聽到了：\n{text}\n\n—\n{final_reply_text}", quick_reply=build_quick_reply())]
        
        if final_reply_text and CLOUDINARY_URL:
            audio_out = await text_to_speech_async(final_reply_text)
            if audio_out:
                public_audio_url = await upload_audio_to_cloudinary(audio_out)
                if public_audio_url:
                    messages_to_send.append(AudioMessage(original_content_url=public_audio_url, duration=20000))
                    logger.info("✅ 成功上傳 TTS 語音並加入回覆佇列。")
        
        await line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=messages_to_send))

    except Exception as e:
        logger.error(f"處理語音訊息失敗: {e}", exc_info=True)
        await line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text="抱歉，我沒聽清楚，可以再說一次嗎？")]))

@handler.add(PostbackEvent)
async def handle_postback(event: PostbackEvent):
    data = event.postback.data
    if data.startswith("menu:"):
        kind = data.split(":", 1)[-1]
        await line_bot_api.reply_message(ReplyMessageRequest(reply_token=event.reply_token, messages=[build_submenu(kind)]))

# ========== 6) FastAPI Routes ==========
@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        await handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"

@router.get("/")
async def root():
    return PlainTextResponse("LINE Bot is running.")

app.include_router(router)

# ========== 7) Local run ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)