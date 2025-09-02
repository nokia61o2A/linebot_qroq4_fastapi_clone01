import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
# ⚡️ 核心工具：用於在異步環境中運行同步程式碼
from fastapi.concurrency import run_in_threadpool

from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceGroup, SourceRoom
)
from linebot.exceptions import LineBotApiError, InvalidSignatureError

# 🔥 FIX 1: 必須使用異步版本的 Groq 客戶端 (AsyncGroq) 才能搭配 await
from groq import AsyncGroq

# --- 基礎設定 ---
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# 從環境變數讀取設定
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 檢查必要的環境變數，若缺少則直接在啟動時報錯
if not all([CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise ValueError("缺少環境變數：請設定 CHANNEL_ACCESS_TOKEN, CHANNEL_SECRET, GROQ_API_KEY")

# 初始化 API 客戶端
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# 使用 AsyncGroq
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-8b-instant")

# --- 匯入自訂功能模組 ---
try:
    from my_commands.lottery_gpt import lottery_gpt
except ImportError:
    logger.warning("無法匯入 'lottery_gpt' 模組，將使用預設功能。")
    def lottery_gpt(msg: str) -> str: return "彩票功能暫時不可用"

try:
    from my_commands.gold_gpt import gold_gpt
except ImportError:
    logger.warning("無法匯入 'gold_gpt' 模組，將使用預設功能。")
    def gold_gpt(msg: str) -> str: return "金價功能暫時不可用"

# --- 翻譯狀態管理 ---
# 注意：此狀態儲存在記憶體中，伺服器重啟後會遺失。
translation_states: Dict[str, str] = {}

def get_chat_id(event: MessageEvent) -> str:
    """從 Line event 中提取唯一的聊天室 ID (使用者、群組或房間)"""
    if isinstance(event.source, SourceGroup):
        return event.source.group_id
    if isinstance(event.source, SourceRoom):
        return event.source.room_id
    return event.source.user_id

def get_translation_state(chat_id: str) -> str:
    return translation_states.get(chat_id, "none")

def set_translation_state(chat_id: str, lang: str) -> None:
    translation_states[chat_id] = lang

# --- 翻譯核心邏輯 ---

# 🔥 FIX 2: 確保翻譯函數為 'async' 且正確使用 'await'
async def translate_text(text: str, target_lang: str) -> str:
    """使用 Groq API 異步翻譯文字"""
    if target_lang == "none" or not target_lang:
        return text

    # 優化後的 Prompt，指示模型僅輸出結果
    prompt = f"請將以下文字翻譯成'{target_lang}'，僅輸出翻譯後的結果，不要包含任何額外的說明或引號：\n\n{text}"
    try:
        # 現在 groq_client 是 AsyncGroq，可以被正確地 await
        chat_completion = await groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API 翻譯失敗: {e}")
        return f"翻譯時發生錯誤，請稍後再試。 (原文: {text})"

# --- 訊息處理主邏輯 ---
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    """處理 Line 的文字訊息事件 (此函數為同步的)"""
    chat_id = get_chat_id(event)
    user_message = event.message.text.strip()
    reply = ""

    # 指令處理 (轉為小寫以增加彈性)
    command = user_message.lower()
    if command.startswith("/translate"):
        parts = user_message.split()
        lang = parts[1].lower() if len(parts) > 1 else ""
        supported_langs = {"none": "無", "zh": "繁體中文", "en": "英文", "vi": "越南文", "jp": "日文"}
        if lang in supported_langs:
            set_translation_state(chat_id, lang)
            reply = f"已將此聊天室的翻譯模式設定為: {supported_langs[lang]}"
        else:
            reply = "支援的語言: /translate [none|zh|en|vi|jp]"
    elif command.startswith("/lottery"):
        reply = lottery_gpt(user_message)
    elif command.startswith("/gold"):
        reply = gold_gpt(user_message)
    else:
        # 非指令的一般訊息，檢查是否需要翻譯
        target_lang = get_translation_state(chat_id)
        if target_lang != "none":
            try:
                # 🔥 FIX 3: 從同步函數中安全地執行異步函數
                # 因為 handle_message 是透過 run_in_threadpool 在背景執行緒中運行的，
                # 該執行緒沒有正在運行的事件循環。
                # 因此，使用 asyncio.run() 是最直接且正確的方式來執行我們的 async translate_text。
                reply = asyncio.run(translate_text(user_message, target_lang))
            except Exception as e:
                logger.error(f"在 handle_message 中執行 asyncio.run 失敗: {e}")
                reply = "處理您的訊息時發生內部錯誤。"
        else:
            # 如果沒有設定翻譯，不進行任何回覆，避免機器人洗版
            return

    # 確保有內容才回覆
    if reply:
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=reply)
            )
        except LineBotApiError as e:
            logger.error(f"回覆 Line 訊息失敗: {e.status_code} {e.error.message}")

# --- FastAPI 應用程式設定 ---

# 🔥 FIX 4: 使用現代的 FastAPI lifespan 語法
@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    logger.info("應用程式啟動...")
    yield
    logger.info("應用程式關閉。")

app = FastAPI(lifespan=lifespan)

# 掛載靜態檔案目錄
app.mount("/static", StaticFiles(directory="static"), name="static")

# Webhook 路由
@app.post("/callback")
async def callback(request: Request):
    """Line Bot 的 Webhook 端點"""
    signature = request.headers.get("X-Line-Signature")
    if not signature:
        raise HTTPException(status_code=400, detail="缺少 X-Line-Signature 標頭")

    body = await request.body()
    body_str = body.decode('utf-8')

    try:
        # ⚡️ 核心：將同步的 handler.handle 放到獨立的執行緒中運行，
        # 這可以防止它阻塞 FastAPI 的主異步事件循環。
        await run_in_threadpool(handler.handle, body_str, signature)
    except InvalidSignatureError:
        logger.warning("無效的簽名，請檢查您的 Channel Secret。")
        raise HTTPException(status_code=400, detail="無效的簽名")
    except LineBotApiError as e:
        logger.error(f"Line Bot API 錯誤: {e.status_code} {e.error.message}")
        raise HTTPException(status_code=500, detail="Line Bot API 錯誤")
    except Exception as e:
        logger.error(f"處理 callback 時發生未知錯誤: {e}")
        raise HTTPException(status_code=500, detail="內部伺服器錯誤")

    return JSONResponse(content={"status": "OK"})

# 健康檢查路由
@app.get("/")
async def root():
    return {"message": "Line Bot is running."}

# 主程式入口 (用於本機開發)
if __name__ == "__main__":
    import uvicorn
    print("啟動開發伺服器於 http://127.0.0.1:8000")
    # 在 Render 等平台部署時，會由 gunicorn 或 uvicorn worker class 指定 host
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)