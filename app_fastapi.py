"""
aibot FastAPI 應用程序初始化 (v30 - 含 Day11 零推播提醒功能 + APScheduler 安全匯入)
"""

# ============================================
# 1. 匯入 (Imports)
# ============================================
import os
import re
import asyncio
import logging
import random
from contextlib import asynccontextmanager
from typing import Dict, List
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool

from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    QuickReply, QuickReplyButton, MessageAction,
    SourceGroup, SourceRoom, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent, TextComponent,
    ButtonComponent
)
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from groq import AsyncGroq

# Peewee for Reminder DB
from peewee import *

# ============================================
# 2. Logger & 基本設定
# ============================================
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# --- APScheduler（安全匯入）---
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    APSCHED_AVAILABLE = True
except Exception:
    AsyncIOScheduler = None  # type: ignore
    APSCHED_AVAILABLE = False
    logger.warning("未安裝 APScheduler，提醒排程功能將停用。請在 requirements.txt 加 APScheduler==3.10.4")

ENABLE_REMINDER = os.getenv("ENABLE_REMINDER", "true").lower() == "true"

# ============================================
# 3. Peewee Model：Reminder
# ============================================
DB_PATH = os.getenv("REMINDER_DB", "reminders.db")
db = SqliteDatabase(DB_PATH)

class BaseModel(Model):
    class Meta:
        database = db

class Reminder(BaseModel):
    id      = AutoField()
    user_id = CharField(index=True)
    text    = CharField()
    due_at  = DateTimeField(index=True)
    sent    = BooleanField(default=False)
    due     = BooleanField(default=False)

def init_db():
    db.connect(reuse_if_open=True)
    db.create_tables([Reminder], safe=True)

# ============================================
# 4. Scheduler：只「標記到期」，不主動推播
# ============================================
def mark_due_reminders():
    now = datetime.now(timezone.utc)
    q = (Reminder
         .update(due=True)
         .where((Reminder.sent == False) &
                (Reminder.due == False) &
                (Reminder.due_at <= now)))
    n = q.execute()
    if n:
        logger.info(f"[Scheduler] 標記到期提醒 {n} 筆為 due")

_scheduler = None
def start_scheduler():
    global _scheduler
    if not ENABLE_REMINDER:
        logger.info("提醒排程已停用 (ENABLE_REMINDER=false)")
        return
    if not APSCHED_AVAILABLE:
        logger.warning("APScheduler 不可用，跳過排程器啟動")
        return
    _scheduler = AsyncIOScheduler(timezone="UTC")
    _scheduler.add_job(mark_due_reminders, "interval", seconds=60, id="mark_due_job", replace_existing=True)
    _scheduler.start()
    logger.info("✅ Reminder Scheduler started (interval=60s, tz=UTC)")

# ============================================
# 5. FastAPI 初始化
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with httpx.AsyncClient() as client:
            await update_line_webhook(client)
    except Exception as e:
        logger.error(f"❌ 啟動初始化失敗: {e}", exc_info=True)

    init_db()
    start_scheduler()
    yield

app = FastAPI(lifespan=lifespan, title="Line Bot API", version="1.0.0")
router = APIRouter()

# ============================================
# 6. LINE 與 Groq 初始化
# ============================================
BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY = map(os.getenv, ["BASE_URL", "CHANNEL_ACCESS_TOKEN", "CHANNEL_SECRET", "GROQ_API_KEY"])
if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise ValueError("缺少必要的環境變數！")

line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-8b-instant")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-70b-versatile")

# 狀態
conversation_history, MAX_HISTORY_LEN = {}, 10
auto_reply_status, user_persona = {}, {}

# ============================================
# 7. 輔助函式
# ============================================
async def update_line_webhook(client: httpx.AsyncClient):
    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
    json_data = {"endpoint": f"{BASE_URL}/callback"}
    res = await client.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=json_data, timeout=10.0)
    res.raise_for_status()
    logger.info(f"✅ Webhook 更新成功: {res.status_code}")

async def groq_chat_completion(messages, max_tokens=600, temperature=0.7):
    try:
        resp = await groq_client.chat.completions.create(model=GROQ_MODEL_PRIMARY, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API 主模型失敗: {e}")
        resp = await groq_client.chat.completions.create(model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return resp.choices[0].message.content.strip()

async def analyze_sentiment(text: str) -> str:
    messages = [{"role": "system", "content": "Analyze sentiment. Respond only: positive, neutral, negative, angry, sad, happy."},
                {"role": "user", "content": text}]
    result = await groq_chat_completion(messages, 20, 0)
    return (result or "neutral").strip().lower()

def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom): return event.source.room_id
    return event.source.user_id

def reply_simple(reply_token, text):
    try:
        line_bot_api.reply_message(reply_token, TextSendMessage(text=text))
    except LineBotApiError as e:
        logger.error(f"Reply 訊息失敗: {e}")

def parse_time_hhmm(s: str):
    h, m = map(int, s.split(":"))
    now = datetime.now(timezone.utc)
    return now.replace(hour=h, minute=m, second=0, microsecond=0)

# ============================================
# 8. LINE Webhook 處理器
# ============================================
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    user_id, chat_id = event.source.user_id, get_chat_id(event)
    msg, reply_token = event.message.text.strip(), event.reply_token
    is_group = isinstance(event.source, (SourceGroup, SourceRoom))

    if not msg: return
    if chat_id not in auto_reply_status: auto_reply_status[chat_id] = True

    low = msg.lower()

    # ====== 提醒功能 ======
    if msg.startswith("提醒我"):
        parts = msg.split(maxsplit=2)
        if len(parts) < 3:
            return reply_simple(reply_token, "格式：提醒我 21:30 內容")
        time_str, text = parts[1], parts[2]
        due_at = parse_time_hhmm(time_str)
        Reminder.create(user_id=user_id, text=text, due_at=due_at)
        return reply_simple(reply_token, f"✅ 我記下了～到 {time_str} 我會提醒你（下次你說話時回覆）")

    # 拉式提醒：使用者再說話時送出
    due_list = list(Reminder.select().where(
        (Reminder.user_id == user_id) & (Reminder.sent == False) & (Reminder.due == True)
    ))
    if due_list:
        lines = ["⏰ 到點提醒："]
        for r in due_list:
            lines.append(f"• {r.text}（原定 {r.due_at.astimezone().strftime('%H:%M')}）")
        reply_simple(reply_token, "\n".join(lines))
        (Reminder.update(sent=True).where(Reminder.id.in_([r.id for r in due_list]))).execute()
        return

    # ====== AI 基本聊天 ======
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = asyncio.run(analyze_sentiment(msg))
        system_prompt = f"你是一位AI女友，根據使用者情緒 {sentiment} 來回覆。"
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": msg}]
        final_reply = asyncio.run(groq_chat_completion(messages))
        history.extend([{"role": "user", "content": msg}, {"role": "assistant", "content": final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
    except Exception as e:
        logger.error(f"AI 回覆失敗: {e}", exc_info=True)
        final_reply = "抱歉，我剛剛走神了 😅，可以再說一次嗎？"

    return reply_simple(reply_token, final_reply)

@handler.add(PostbackEvent)
def handle_postback(event): pass

# ============================================
# 9. FastAPI Routes
# ============================================
@router.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    try: await run_in_threadpool(handler.handle, body.decode("utf-8"), signature)
    except InvalidSignatureError: raise HTTPException(400, "Invalid signature")
    return JSONResponse({"message": "ok"})

@router.get("/")
async def root(): return {"message": "Line Bot Service is live."}

app.include_router(router)