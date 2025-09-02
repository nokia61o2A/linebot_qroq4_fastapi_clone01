import os
import re
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict

import httpx
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceGroup, SourceRoom
)
from linebot.exceptions import LineBotApiError, InvalidSignatureError

from groq import Groq

#-- 繁體中文說明 --
# 新增 pypinyin 用於生成中文拼音和注音符號，pyvi 用於解析越南文音節和聲調
from pypinyin import pinyin, Style
from pyvi import ViTokenizer, ViUtils

# --- 繁體中文說明 ---
# 基礎設定：Line Bot 與 Groq API 初始化
# ------------------------------------------ #
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

BASE_URL       = os.getenv("BASE_URL")
CHANNEL_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler      = WebhookHandler(CHANNEL_SECRET)

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL_PRIMARY  = os.getenv("GROQ_MODEL_PRIMARY",  "llama-3.1-8b-instant")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-70b-versatile")

os.environ["GROQ_MODEL"] = GROQ_MODEL_PRIMARY

# --- 繁體中文說明 ---
# 匯入自訂功能模組
# ------------------------------------------ #
try:
    from my_commands.lottery_gpt import lottery_gpt
except ImportError:
    def lottery_gpt(msg): return "彩票功能暫時不可用"

try:
    from my_commands.gold_gpt import gold_gpt
except ImportError:
    def gold_gpt(msg): return "金價功能暫時不可用"

#-- 繁體中文說明 --
# 新增聊天室獨立的翻譯狀態管理
# 使用字典儲存每個聊天室的翻譯狀態，鍵為聊天室ID，值為當前語言
# ------------------------------------------ #
translation_states: Dict[str, str] = {}

def get_translation_state(chat_id: str) -> str:
    return translation_states.get(chat_id, "none")  # 預設無翻譯

def set_translation_state(chat_id: str, lang: str) -> None:
    translation_states[chat_id] = lang

def clear_translation_state(chat_id: str) -> None:
    translation_states[chat_id] = "none"

#-- 繁體中文說明 --
# 翻譯邏輯：將文字翻譯為指定語言（異步函數）
# ------------------------------------------ #
async def translate_text(text: str, target_lang: str) -> str:
    if target_lang == "none":
        return text
    try:
        prompt = f"將以下文字翻譯成{target_lang}：{text}"
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"翻譯失敗: {e}")
        return text  # 翻譯失敗時返回原文

#-- 🔥 修正版：同步翻譯處理，避免 event loop already running
def sync_translate_text(text: str, target_lang: str) -> str:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果 event loop 已經在跑，改用 thread-safe 提交 coroutine
            future = asyncio.run_coroutine_threadsafe(
                translate_text(text, target_lang), loop
            )
            return future.result()
        else:
            return loop.run_until_complete(translate_text(text, target_lang))
    except Exception as e:
        logger.error(f"sync_translate_text 錯誤: {e}")
        return text

#-- 繁體中文說明 --
# 處理訊息事件的主邏輯
# ------------------------------------------ #
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    chat_id = (
        event.source.group_id if isinstance(event.source, SourceGroup)
        else event.source.room_id if isinstance(event.source, SourceRoom)
        else event.source.user_id
    )
    user_message = event.message.text.strip()

    # 檢查是否為翻譯指令
    if user_message.startswith("/translate"):
        lang = user_message.replace("/translate", "").strip().lower()
        if lang in ["none", "zh", "en", "vi", "jp"]:
            set_translation_state(chat_id, lang)
            reply = f"已設定此聊天室的翻譯語言為: {lang if lang != 'none' else '無'}"
        else:
            reply = "支援的語言: none, zh (中文), en (英文), vi (越南文), jp (日文)"
    elif user_message.startswith("/lottery"):
        reply = lottery_gpt(user_message)
    elif user_message.startswith("/gold"):
        reply = gold_gpt(user_message)
    else:
        target_lang = get_translation_state(chat_id)
        reply = sync_translate_text(user_message, target_lang)

    # 回覆訊息
    try:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply)
        )
    except LineBotApiError as e:
        logger.error(f"回覆訊息失敗: {e}")

# --- FastAPI 初始化 ---
app = FastAPI()

# 設置靜態檔案路徑
app.mount("/static", StaticFiles(directory="static"), name="static")

# Webhook 路由處理
@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()

    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="無效的簽名")
    except LineBotApiError as e:
        logger.error(f"LineBot API 錯誤: {e}")
        raise HTTPException(status_code=500, detail="內部伺服器錯誤")

    return JSONResponse(content={"status": "OK"})

# 生命週期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("應用程式啟動中...")
    yield
    logger.info("應用程式關閉中...")

app.router.lifespan_context = lifespan

# 主程式入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)