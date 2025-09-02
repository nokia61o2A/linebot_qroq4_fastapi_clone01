import os
import re
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List

import httpx
import requests
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    QuickReply, QuickReplyButton, MessageAction,
    SourceGroup, SourceRoom, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent, TextComponent,
    ButtonComponent
)
from linebot.exceptions import LineBotApiError, InvalidSignatureError

from groq import Groq

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
from my_commands.lottery_gpt import lottery_gpt
from my_commands.gold_gpt import gold_gpt
from my_commands.platinum_gpt import platinum_gpt
from my_commands.money_gpt import money_gpt
from my_commands.one04_gpt import one04_gpt
from my_commands.partjob_gpt import partjob_gpt
from my_commands.crypto_coin_gpt import crypto_gpt
from my_commands.weather_gpt import weather_gpt
from my_commands.stock.stock_gpt import stock_gpt   # ✅ 改用小寫 stock 資料夾

# ============================================
# 狀態管理
# ============================================
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
auto_reply_status: Dict[str, bool] = {}
user_persona: Dict[str, str] = {}
translation_requests: Dict[str, dict] = {}  # {user_id: {"lang": "繁體中文", "text": ""}}

# ============================================
# FastAPI 與 Webhook
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        update_line_webhook()
    except Exception as e:
        logger.error(f"❌ 啟動初始化失敗: {e}", exc_info=True)
    yield

app = FastAPI(lifespan=lifespan, title="Line Bot API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
router = APIRouter()

def update_line_webhook():
    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
    json_data = {"endpoint": f"{BASE_URL}/callback"}
    with httpx.Client() as c:
        res = c.put("https://api.line.me/v2/bot/channel/webhook/endpoint",
                    headers=headers, json=json_data, timeout=10.0)
        res.raise_for_status()
        logger.info(f"✅ Webhook 更新成功: {res.status_code}")

# ============================================
# QuickReply 與 Flex Menu
# ============================================
def build_quick_reply_items(is_group: bool, bot_name: str) -> List[QuickReplyButton]:
    items: List[QuickReplyButton] = []
    prefix = f"@{bot_name} " if is_group else ""
    items.extend([
        QuickReplyButton(action=MessageAction(label="💖 人設選單", text="人設選單")),
        QuickReplyButton(action=MessageAction(label="💰 金融選單", text="金融選單")),
        QuickReplyButton(action=MessageAction(label="🎰 彩票選單", text="彩票選單")),
        QuickReplyButton(action=MessageAction(label="🌐 翻譯選單", text="翻譯選單")),  # ✅ 新增翻譯選單
        QuickReplyButton(action=MessageAction(label="✅ 開啟自動回答", text="開啟自動回答")),
        QuickReplyButton(action=MessageAction(label="❌ 關閉自動回答", text="關閉自動回答")),
        QuickReplyButton(action=MessageAction(label="🌤️ 天氣", text=f"{prefix}天氣")),
    ])
    return items

def build_flex_menu(title: str, subtitle: str, actions: List[MessageAction]) -> FlexSendMessage:
    buttons: List[ButtonComponent] = []
    for act in actions:
        buttons.append(ButtonComponent(style="primary", height="sm", action=act, margin="md", color="#905C44"))
    bubble = BubbleContainer(
        size="mega",
        header=BoxComponent(layout="vertical", contents=[
            TextComponent(text=title, weight="bold", size="xl", color="#FFFFFF", align="center"),
            TextComponent(text=subtitle, size="sm", color="#EEEEEE", wrap=True, align="center", margin="md"),
        ], spacing="sm", paddingAll="20px", backgroundColor="#FF6B6B", cornerRadius="lg"),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm", paddingAll="20px", backgroundColor="#FFF9F2", cornerRadius="lg"),
        footer=BoxComponent(layout="vertical", contents=[
            TextComponent(text="💖 點擊按鈕快速執行", size="xs", color="#888888", align="center", margin="md")
        ], paddingAll="10px")
    )
    return FlexSendMessage(alt_text=title, contents=bubble)

def flex_menu_translate() -> FlexSendMessage:
    actions = [
        MessageAction(label="🇺🇸 翻英文", text="翻譯->英文"),
        MessageAction(label="🇹🇼 翻繁體中文", text="翻譯->繁體中文"),
        MessageAction(label="🇨🇳 翻簡體中文", text="翻譯->簡體中文"),
        MessageAction(label="🇯🇵 翻日文", text="翻譯->日文"),
        MessageAction(label="❌ 結束翻譯", text="翻譯->結束"),
    ]
    return build_flex_menu("🌐 翻譯選擇", "選擇要翻譯的目標語言", actions)

def flex_menu_finance(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    actions = [
        MessageAction(label="📈 台股大盤", text=f"{prefix}大盤"),
        MessageAction(label="📊 美股大盤", text=f"{prefix}美股"),
        MessageAction(label="💰 金價查詢", text=f"{prefix}金價"),
        MessageAction(label="💴 日元匯率", text=f"{prefix}JPY"),
        MessageAction(label="💵 美元匯率", text=f"{prefix}USD"),
    ]
    return build_flex_menu("💰 金融服務", "點擊下方按鈕快速查詢最新資訊", actions)

def flex_menu_lottery(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    actions = [
        MessageAction(label="🎰 大樂透", text=f"{prefix}大樂透"),
        MessageAction(label="🎯 威力彩", text=f"{prefix}威力彩"),
        MessageAction(label="🔢 539",   text=f"{prefix}539"),
    ]
    return build_flex_menu("🎰 彩票服務", "快速開單與最新開獎資訊", actions)

def flex_menu_persona() -> FlexSendMessage:
    actions = [
        MessageAction(label="🌸 甜美女友", text="甜"),
        MessageAction(label="😏 傲嬌女友", text="鹹"),
        MessageAction(label="✨ 萌系女友", text="萌"),
        MessageAction(label="🧊 酷系御姐", text="酷"),
    ]
    return build_flex_menu("💖 人設選擇", "切換 AI 女友的說話風格", actions)

# ============================================
# Groq 工具
# ============================================
def groq_chat_completion(messages, max_tokens=600, temperature=0.7):
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=10.0
        )
        return completion.choices[0].message.content
    except Exception as e_primary:
        logger.error(f"主要模型失敗: {e_primary}")
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL_FALLBACK,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=10.0
            )
            return completion.choices[0].message.content
        except Exception as e_fallback:
            logger.error(f"備用模型失敗: {e_fallback}")
            return "抱歉，AI 服務暫時不可用。"

async def translate_text(text: str, target_language: str) -> str:
    try:
        messages = [
            {"role": "system", "content": f"你是一位專業翻譯，請將以下文字翻譯成 {target_language}"},
            {"role": "user", "content": text}
        ]
        return groq_chat_completion(messages, max_tokens=800, temperature=0.3)
    except Exception as e:
        logger.error(f"翻譯失敗: {e}")
        return text

# ============================================
# handle_message
# ============================================
@router.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(400, "Invalid signature")
    return JSONResponse({"message": "ok"})

app.include_router(router)

@handler.add(MessageEvent, message=TextMessage)
def handle_message_wrapper(event):
    asyncio.create_task(handle_message(event))

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

    bot_name = line_bot_api.get_bot_info().display_name
    low = msg.lower()

    # --- Flex 選單觸發 ---
    if low == '人設選單':
        line_bot_api.reply_message(reply_token, flex_menu_persona()); return
    elif low == '金融選單':
        line_bot_api.reply_message(reply_token, flex_menu_finance(bot_name, is_group)); return
    elif low == '彩票選單':
        line_bot_api.reply_message(reply_token, flex_menu_lottery(bot_name, is_group)); return
    elif low == '翻譯選單':
        line_bot_api.reply_message(reply_token, flex_menu_translate()); return

    # --- 翻譯邏輯 ---
    if low.startswith("翻譯->"):
        choice = low.replace("翻譯->", "")
        if choice == "結束":
            translation_requests.pop(user_id, None)
            await reply_simple(reply_token, "✅ 已結束翻譯模式")
            return
        else:
            translation_requests[user_id] = {"lang": choice, "text": ""}
            await reply_simple(reply_token, f"🌐 已啟用翻譯模式，下一則訊息將翻譯成【{choice}】")
            return
    elif user_id in translation_requests and translation_requests[user_id]["lang"]:
        target_lang = translation_requests[user_id]["lang"]
        translated = await translate_text(msg, target_lang)
        await reply_simple(reply_token, f"🌐 翻譯結果 ({target_lang})：\n{translated}")
        return

    # --- 功能觸發判斷 ---
    reply_text = None
    if any(k in msg for k in ["威力彩","大樂透","539","雙贏彩"]):
        reply_text = lottery_gpt(msg)
    elif msg.startswith("104:"):
        reply_text = one04_gpt(msg[4:].strip())
    elif msg.startswith("pt:"):
        reply_text = partjob_gpt(msg[3:].strip())
    elif msg.startswith("cb:") or msg.startswith("$:"):
        coin = msg[3:].strip() if msg.startswith("cb:") else msg[2:].strip()
        reply_text = crypto_gpt(coin)
    elif "金價" in msg or "黃金" in msg:
        reply_text = gold_gpt()
    elif "鉑" in msg or "platinum" in msg.lower():
        reply_text = platinum_gpt()
    elif "USD" in msg or "美金" in msg:
        reply_text = money_gpt("USD")
    elif "JPY" in msg or "日幣" in msg:
        reply_text = money_gpt("JPY")
    elif "大盤" in msg or "台股" in msg:
        reply_text = stock_gpt("大盤")
    elif "美股" in msg:
        reply_text = stock_gpt("美盤")
    elif "天氣" in msg:
        reply_text = weather_gpt("台北市")
     else:
        stock_code   = re.fullmatch(r"\d{4,6}[A-Za-z]?", msg)
        stockUS_code = re.fullmatch(r"[A-Za-z]{1,5}", msg)
        if stock_code:
            reply_text = stock_gpt(stock_code.group())
        elif stockUS_code:
            reply_text = stock_gpt(stockUS_code.group())
        else:
            # --- 繁體中文說明 ---
            # 沒有觸發功能 → 進入 LLM 模式，使用人設 + 情緒
            # ------------------------------------------ #
            conversation_history.setdefault(user_id, [])
            conversation_history[user_id].append({"role": "user", "content": msg})

            # 保留最近對話，避免爆 token
            if len(conversation_history[user_id]) > MAX_HISTORY_LEN * 2:
                conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY_LEN*2:]

            # 情緒分析
            sentiment = await analyze_sentiment(msg)

            # 構建人設 prompt
            persona_prompt = build_persona_prompt(user_id, sentiment)
            full_messages = [{"role": "system", "content": persona_prompt}] + conversation_history[user_id]

            reply_text = groq_chat_completion(full_messages, max_tokens=600, temperature=0.7)

            # 存入歷史
            conversation_history[user_id].append({"role": "assistant", "content": reply_text})

    try:
        quick_items = build_quick_reply_items(is_group, bot_name)
        line_bot_api.reply_message(reply_token, TextSendMessage(text=reply_text, quick_reply=QuickReply(items=quick_items)))
    except LineBotApiError as e:
        logger.error(f"回覆訊息失敗: {e.error.message}", exc_info=True)

# ============================================
# 簡單回覆工具
# ============================================
async def reply_simple(reply_token, text: str):
    try:
        bot_name = line_bot_api.get_bot_info().display_name
        quick_items = build_quick_reply_items(is_group=False, bot_name=bot_name)
        line_bot_api.reply_message(reply_token, TextSendMessage(text=text, quick_reply=QuickReply(items=quick_items)))
    except Exception as e:
        logger.error(f"reply_simple 發生錯誤: {e}")

# ============================================
# 健康檢查
# ============================================
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