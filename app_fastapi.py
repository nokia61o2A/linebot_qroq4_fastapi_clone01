import os
import re
import asyncio
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

# 自訂指令模組
from my_commands.lottery_gpt import lottery_gpt
from my_commands.gold_gpt import gold_gpt
from my_commands.platinum_gpt import platinum_gpt
from my_commands.money_gpt import money_gpt
from my_commands.one04_gpt import one04_gpt
from my_commands.partjob_gpt import partjob_gpt
from my_commands.crypto_coin_gpt import crypto_gpt
from my_commands.stock.stock_gpt import stock_gpt
from my_commands.weather_gpt import weather_gpt  # 台灣氣象分析

# 環境變數設定
BASE_URL           = os.getenv("BASE_URL")
CHANNEL_TOKEN      = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET     = os.getenv("CHANNEL_SECRET")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")

# Line Bot & LLM 客戶端初始化
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler      = WebhookHandler(CHANNEL_SECRET)
client       = OpenAI(api_key=OPENAI_API_KEY, base_url="https://free.v36.cm/v1")
groq_client  = Groq(api_key=GROQ_API_KEY)

# 保持對話歷史
conversation_history = {}
MAX_HISTORY_LEN     = 10

# FastAPI 生命週期
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        update_line_webhook()
    except Exception as e:
        print(f"❌ 更新 Webhook 失敗: {e}")
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 工具函式

def update_line_webhook():
    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
    json_data = {"endpoint": f"{BASE_URL}/callback"}
    with httpx.Client() as client:
        res = client.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=json_data)
        res.raise_for_status()
        print(f"✅ Webhook 更新成功: {res.status_code}")


def show_loading_animation(user_id: str, seconds: int = 5):
    url = "https://api.line.me/v2/bot/chat/loading/start"
    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
    data = {"chatId": user_id, "loadingSeconds": seconds}
    try:
        resp = requests.post(url, headers=headers, json=data)
        if resp.status_code != 202:
            print(f"❌ 載入動畫錯誤: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"❌ 載入動畫請求失敗: {e}")


def calculate_english_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    english = [c for c in letters if ord(c) < 128]
    return len(english) / len(letters)

# 路由與事件處理
router = APIRouter()

@router.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(400, "Invalid signature")
    except Exception as e:
        raise HTTPException(500, str(e))
    return JSONResponse({"message": "ok"})

app.include_router(router)

@handler.add(MessageEvent, message=TextMessage)
def handle_message_wrapper(event):
    asyncio.create_task(handle_message(event))


async def get_async_reply(messages):
    try:
        resp = await client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=800)
        return resp.choices[0].message.content
    except Exception:
        groq_resp = groq_client.chat.completions.create(
            model="llama3-70b-8192", messages=messages, max_tokens=2000, temperature=1.2
        )
        return groq_resp.choices[0].message.content


async def handle_message(event):
    user_id = event.source.user_id
    msg     = event.message.text.strip()
    is_group = isinstance(event.source, (SourceGroup, SourceRoom))

    if not is_group:
        show_loading_animation(user_id)

    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append({"role": "user", "content": msg + "，請以繁體中文回答"})
    if len(conversation_history[user_id]) > MAX_HISTORY_LEN * 2:
        conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY_LEN * 2:]

    try:
        # 依優先順序判斷指令
        if any(k in msg for k in ["威力彩", "大樂透", "539", "雙贏彩"]):
            reply_text = lottery_gpt(msg)
        elif msg.lower().startswith("大盤") or msg.lower().startswith("台股"):
            reply_text = stock_gpt("大盤")
        elif msg.lower().startswith("美盤") or msg.lower().startswith("美股"):
            reply_text = stock_gpt("美盤")
        elif msg.startswith("cb:") or msg.startswith("$:"):
            coin_id = msg[3:].strip() if msg.startswith("cb:") else msg[2:].strip()
            reply_text = crypto_gpt(coin_id)
        elif any(msg.lower().startswith(k) for k in ["金價", "黃金", "gold"]):
            reply_text = gold_gpt()
        elif any(msg.lower().startswith(k) for k in ["鉑", "platinum"]):
            reply_text = platinum_gpt()
        elif any(msg.lower().startswith(k) for k in ["日幣", "jpy"]):
            reply_text = money_gpt("JPY")
        elif any(msg.lower().startswith(k) for k in ["美金", "usd"]):
            reply_text = money_gpt("USD")
        elif any(k in msg for k in ["天氣", "氣象"]):
            reply_text = weather_gpt("桃園市")
        else:
            # 台股代碼或美股代號
            stock_code   = re.fullmatch(r"\d{4,6}[A-Za-z]?", msg)
            stockUS_code = re.fullmatch(r"[A-Za-z]{1,5}", msg)
            if stock_code:
                reply_text = stock_gpt(stock_code.group())
            elif stockUS_code:
                reply_text = stock_gpt(stockUS_code.group())
            else:
                reply_text = await get_async_reply(conversation_history[user_id][-MAX_HISTORY_LEN:])
    except Exception as e:
        reply_text = f"API 發生錯誤：{e}"

    if not reply_text:
        reply_text = "抱歉，目前無法提供回應，請稍後再試。"

    # 快速回覆按鈕
    english_ratio = calculate_english_ratio(reply_text)
    quick_items = []
    if english_ratio > 0.1:
        quick_items.append(QuickReplyButton(action=MessageAction(label="翻譯成中文", text="請將上述內容翻譯成中文")))
    for label, text in [("台股大盤","大盤"), ("美股大盤","美股"), ("大樂透","大樂透"), ("威力彩","威力彩"), ("金價","金價")]:
        quick_items.append(QuickReplyButton(action=MessageAction(label=label, text=text)))

    if quick_items:
        message = TextSendMessage(text=reply_text, quick_reply=QuickReply(items=quick_items))
    else:
        message = TextSendMessage(text=reply_text)

    try:
        line_bot_api.reply_message(event.reply_token, message)
    except LineBotApiError as e:
        print(f"❌ 回覆訊息失敗：{e}")

    conversation_history[user_id].append({"role": "assistant", "content": reply_text})

@handler.add(PostbackEvent)
async def handle_postback(event):
    print(f"Postback data: {event.postback.data}")

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Service is live."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port)

