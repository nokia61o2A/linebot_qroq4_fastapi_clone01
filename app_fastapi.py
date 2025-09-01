"""
AI 醬  git@github.com-nokia61o2A:nokia61o2A/linebot_qroq4_fastapi.git
"""
import os
import re
import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
import requests
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    QuickReply, QuickReplyButton, MessageAction,
    SourceGroup, SourceRoom, PostbackEvent
)
from linebot.exceptions import LineBotApiError, InvalidSignatureError

from openai import OpenAI
from groq import Groq

# 引入自訂指令模組
from my_commands.lottery_gpt import lottery_gpt
from my_commands.gold_gpt import gold_gpt
from my_commands.platinum_gpt import platinum_gpt
from my_commands.money_gpt import money_gpt
from my_commands.one04_gpt import one04_gpt
from my_commands.partjob_gpt import partjob_gpt
from my_commands.crypto_coin_gpt import crypto_gpt
from my_commands.stock.stock_gpt import stock_gpt
from my_commands.weather_gpt import weather_gpt  # 台灣氣象分析

# ============================================
# 1. 設定 logger，取代 print
# ============================================
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# 環境變數設定
BASE_URL       = os.getenv("BASE_URL")
CHANNEL_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

# Line Bot & LLM 客戶端初始化
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler      = WebhookHandler(CHANNEL_SECRET)

# 初始化 OpenAI 客戶端（--- 使用新的 API 格式 ---）
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://free.v36.cm/v1"
)

# 初始化 Groq 客戶端
groq_client = Groq(api_key=GROQ_API_KEY)

# 使用最新的 Groq 模型
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

# 保持對話歷史
conversation_history = {}
MAX_HISTORY_LEN     = 10
auto_reply_status = {}

# FastAPI 生命週期，用於啟動時更新 Webhook
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        update_line_webhook()
    except Exception as e:
        logger.error(f"❌ 更新 Webhook 失敗: {e}", exc_info=True)
    yield

app = FastAPI(
    lifespan=lifespan,
    title="Line Bot API",
    description="Line Bot with FastAPI",
    version="1.0.0"
)

# 添加錯誤處理中間件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"請求處理失敗: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

app.mount("/static", StaticFiles(directory="static"), name="static")
router = APIRouter()

def update_line_webhook():
    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
    json_data = {"endpoint": f"{BASE_URL}/callback"}
    with httpx.Client() as client:
        res = client.put(
            "https://api.line.me/v2/bot/channel/webhook/endpoint",
            headers=headers, json=json_data
        )
        res.raise_for_status()
        logger.info(f"✅ Webhook 更新成功: {res.status_code}")

def show_loading_animation(user_id: str, seconds: int = 5):
    url = "https://api.line.me/v2/bot/chat/loading/start"
    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
    data = {"chatId": user_id, "loadingSeconds": seconds}
    try:
        resp = requests.post(url, headers=headers, json=data)
        if resp.status_code != 202:
            logger.error(f"❌ 載入動畫錯誤: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.error(f"❌ 載入動畫請求失敗: {e}", exc_info=True)

def calculate_english_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    english = [c for c in letters if ord(c) < 128]
    return len(english) / len(letters)

# ============================================
# Groq 呼叫工具（主→備 自動切換）
# ============================================
def groq_chat_completion(messages, max_tokens=800, temperature=0.7):
    """統一的 Groq 聊天完成函數，支持主備模型切換"""
    try:
        # 首先嘗試主要模型
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return completion.choices[0].message.content
    except Exception as e_primary:
        logger.error(f"主要模型 {GROQ_MODEL_PRIMARY} 失敗: {e_primary}")
        try:
            # 主要模型失敗時嘗試備用模型
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL_FALLBACK,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e_fallback:
            logger.error(f"備用模型 {GROQ_MODEL_FALLBACK} 也失敗: {e_fallback}")
            return f"抱歉，AI 服務暫時不可用。錯誤信息: {str(e_fallback)}"

# ============================================
# 情緒分析（先 OpenAI → 後 Groq 現行模型）
# ============================================
async def analyze_sentiment(text: str) -> str:
    """
    呼叫 OpenAI/Groq 判斷訊息情緒
    回傳: positive / neutral / negative / angry
    """
    try:
        # 使用新的 OpenAI API 格式
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一個情感分析助手，輸出文字情緒標籤"},
                {"role": "user", "content": f"判斷這句話的情緒：{text}\n只回傳一個標籤：positive, neutral, negative, angry"}
            ],
            max_tokens=10,
            temperature=0
        )
        return completion.choices[0].message.content.strip().lower()
    except Exception as e:
        logger.error(f"OpenAI 情感分析失敗: {e}")
        # OpenAI 失敗時使用 Groq
        messages = [
            {"role": "system", "content": "你是一個情感分析助手，輸出文字情緒標籤"},
            {"role": "user", "content": f"判斷這句話的情緒：{text}\n只回傳一個標籤：positive, neutral, negative, angry"}
        ]
        result = groq_chat_completion(messages, max_tokens=10, temperature=0)
        return result.strip().lower() if result else "neutral"

# ============================================
# 一般聊天（帶入情緒標籤的 System Prompt）
# ============================================
async def get_reply_with_sentiment(messages, sentiment: str = "neutral"):
    """用 OpenAI / Groq 回覆訊息（帶情緒提示）"""
    system_prompt = f"""
你是溫柔的 AI 女友，要根據使用者的情緒調整語氣。
情緒標籤：{sentiment}
- positive: 活潑興奮，跟著開心。
- negative: 安慰、貼心。
- angry: 安撫、冷靜。
- neutral: 正常聊天。
回覆請用繁體中文，保持女友般的口吻與自然節奏。
""".strip()

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        # 使用新的 OpenAI API 格式
        completion = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=full_messages, 
            max_tokens=800, 
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI 回覆失敗: {e}")
        # OpenAI 失敗時使用 Groq
        return groq_chat_completion(full_messages, max_tokens=800, temperature=0.7)

@router.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(400, "Invalid signature")
    except Exception as e:
        logger.error(f"Callback 處理失敗: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    return JSONResponse({"message": "ok"})

app.include_router(router)

@handler.add(MessageEvent, message=TextMessage)
def handle_message_wrapper(event):
    asyncio.create_task(handle_message(event))

async def get_async_reply(messages):
    """一般回覆函數（不帶情緒分析）"""
    try:
        # 使用新的 OpenAI API 格式
        resp = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages, 
            max_tokens=800
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI 一般回覆失敗: {e}")
        # 使用 Groq 替代方案
        return groq_chat_completion(messages, max_tokens=800, temperature=0.7)

async def handle_message(event):
    user_id = event.source.user_id
    msg = event.message.text.strip()
    reply_token = event.reply_token
    is_group = isinstance(event.source, (SourceGroup, SourceRoom))
    
    chat_id = event.source.group_id if isinstance(event.source, SourceGroup) else (
        event.source.room_id if isinstance(event.source, SourceRoom) else user_id
    )
    
    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = not is_group

    if not is_group:
        show_loading_animation(user_id)

    bot_info = line_bot_api.get_bot_info()
    bot_name = bot_info.display_name
    
    # 處理 @ 開頭的訊息
    processed_msg = msg
    if msg.startswith('@'):
        processed_msg = re.sub(r'^@\w+\s*', '', msg).strip()
    
    # 自動回覆開關指令處理
    if processed_msg.lower() == '開啟自動回答':
        auto_reply_status[chat_id] = True
        await reply_simple(reply_token, "✅ 已開啟自動回答")
        return
    elif processed_msg.lower() == '關閉自動回答':
        auto_reply_status[chat_id] = False
        await reply_simple(reply_token, "✅ 已關閉自動回答")
        return
        
    if not auto_reply_status[chat_id]:
        if not any(name in msg.lower() for name in bot_name.lower().split()):
            return
        msg_parts = re.split(r'@\w+\s*', msg, 1)
        if len(msg_parts) > 1:
            processed_msg = msg_parts[1].strip()
        else:
            auto_reply_status[chat_id] = True
            await reply_simple(reply_token, "✅ 已開啟自動回答")
            return
    else:
        if msg.startswith('@'):
            processed_msg = re.sub(r'^@\w+\s*', '', msg).strip()

    # 更新對話歷史
    conversation_history.setdefault(user_id, [])
    conversation_history[user_id].append({"role": "user", "content": processed_msg + "，請以繁體中文回答"})
    if len(conversation_history[user_id]) > MAX_HISTORY_LEN * 2:
        conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY_LEN * 2:]

    reply_text = None
    try:
        if any(k in processed_msg for k in ["威力彩", "大樂透", "539", "雙贏彩"]):
            reply_text = lottery_gpt(processed_msg)
        elif processed_msg.startswith("104:"):
            job_keyword = processed_msg[4:].strip()
            reply_text = one04_gpt(job_keyword)
        elif processed_msg.lower().startswith("大盤") or processed_msg.lower().startswith("台股"):
            reply_text = stock_gpt("大盤")
        elif processed_msg.lower().startswith("美盤") or processed_msg.lower().startswith("美股"):
            reply_text = stock_gpt("美盤")
        elif processed_msg.startswith("pt:"):
            reply_text = partjob_gpt(processed_msg[3:])
        elif processed_msg.startswith("cb:") or processed_msg.startswith("$:"):
            coin_id = processed_msg[3:].strip() if processed_msg.startswith("cb:") else processed_msg[2:].strip()
            reply_text = crypto_gpt(coin_id)
        elif any(processed_msg.lower().startswith(k) for k in ["金價", "黃金", "gold"]):
            reply_text = gold_gpt()
        elif any(processed_msg.lower().startswith(k) for k in ["鉑", "platinum"]):
            reply_text = platinum_gpt()
        elif any(processed_msg.lower().startswith(k) for k in ["日幣", "jpy"]):
            reply_text = money_gpt("JPY")
        elif any(processed_msg.lower().startswith(k) for k in ["美金", "usd"]):
            reply_text = money_gpt("USD")
        elif any(k in processed_msg for k in ["天氣", "氣象"]):
            reply_text = weather_gpt("桃園市")
        else:
            stock_code = re.fullmatch(r"\d{4,6}[A-Za-z]?", processed_msg)
            stockUS_code = re.fullmatch(r"[A-Za-z]{1,5}", processed_msg)
            if stock_code:
                reply_text = stock_gpt(stock_code.group())
            elif stockUS_code:
                reply_text = stock_gpt(stockUS_code.group())
            else:
                sentiment = await analyze_sentiment(processed_msg)
                reply_text = await get_reply_with_sentiment(
                    conversation_history[user_id][-MAX_HISTORY_LEN:], sentiment=sentiment
                )

    except Exception as e:
        logger.error(f"處理訊息時發生錯誤：{e}", exc_info=True)
        reply_text = "抱歉，伺服器發生錯誤，請稍後再試。"

    if not reply_text:
        reply_text = "抱歉，目前無法提供回應，請稍後再試。"

    reply_message = create_reply_message(reply_text, is_group, bot_name)

    try:
        line_bot_api.reply_message(reply_token, reply_message)
        conversation_history[user_id].append({"role": "assistant", "content": reply_text})
    except LineBotApiError as e:
        logger.error(f"回覆訊息失敗：{e.error.message}", exc_info=True)

def create_reply_message(reply_text: str, is_group: bool, bot_name: str) -> TextSendMessage:
    quick_reply_items = []

    if calculate_english_ratio(reply_text) > 0.1:
        quick_reply_items.append(
            QuickReplyButton(
                action=MessageAction(label="翻譯成中文", text="請將上述內容翻譯成中文")
            )
        )

    prefix = f"@{bot_name} " if is_group else ""
    common_buttons = [
        ("開啟自動回答", "開啟自動回答"),
        ("關閉自動回答", "關閉自動回答"),
        ("台股大盤", f"{prefix}大盤"),
        ("美股大盤", f"{prefix}美股"),
        ("大樂透", f"{prefix}大樂透"),
        ("威力彩", f"{prefix}威力彩"),
        ("金價", f"{prefix}金價"),
        ("日元", f"{prefix}JPY"),
        ("美元", f"{prefix}USD"),
        (f"{bot_name}", f"@{bot_name}"),
    ]
    
    for label, text in common_buttons:
        quick_reply_items.append(
            QuickReplyButton(action=MessageAction(label=label, text=text))
        )

    return TextSendMessage(text=reply_text, quick_reply=QuickReply(items=quick_reply_items))

async def reply_simple(reply_token, text):
    try:
        line_bot_api.reply_message(reply_token, TextSendMessage(text=text))
    except LineBotApiError as e:
        logger.error(f"❌ 回覆訊息失敗: {e}")

@handler.add(PostbackEvent)
async def handle_postback(event):
    logger.info(f"Postback data: {event.postback.data}")

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Service is live."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info")