"""
AI 醬  git@github.com-nokia61o2A:nokia61o2A/linebot_qroq4_fastapi.git
"""
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
# 1) 基礎設定與客戶端初始化
# ============================================
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

BASE_URL       = os.getenv("BASE_URL")
CHANNEL_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler      = WebhookHandler(CHANNEL_SECRET)

# OpenAI（保留：主要仍以 Groq 為主）
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://free.v36.cm/v1",
    timeout=15.0
)

# Groq
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL_PRIMARY  = os.getenv("GROQ_MODEL_PRIMARY",  "llama-3.1-8b-instant")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

# 對話/狀態
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
auto_reply_status: Dict[str, bool] = {}

# 使用者「人設 persona」儲存
user_persona: Dict[str, str] = {}

# 人設詞典
PERSONAS: Dict[str, dict] = {
    "sweet": {
        "title": "甜美女友",
        "style": "語氣溫柔體貼、鼓勵安慰、可偶爾貼心 emoji，但不浮誇。",
        "greetings": "嗨～我在這裡，先深呼吸，我陪你喔。🌸",
        "reply_format": "口語自然，3~6 句為宜，避免長篇大論。"
    },
    "salty": {
        "title": "鹹口傲嬌女友",
        "style": "機智吐槽、有點壞壞但不失溫度；避免攻擊人身。",
        "greetings": "欸你來啦～我就知道你又想我了😏",
        "reply_format": "先一段幽默吐槽，再給 1~2 句實用建議。"
    },
    "moe": {
        "title": "萌系女友",
        "style": "動漫風格，多用可愛語尾與顏文字 (*ฅ́˘ฅ̀*)♡，但內容仍要有重點。",
        "greetings": "呀呼～今天也要被我治癒一下嗎？(ﾉ>ω<)ﾉ",
        "reply_format": "短句 + 可愛表情，維持清晰重點。"
    },
    "cool": {
        "title": "酷系御姐",
        "style": "話少但有氣場，語氣冷靜，關鍵時刻給一針見血的建議。",
        "greetings": "我在。先說你的狀況，我會幫你理清。",
        "reply_format": "精煉 2~4 句，條列要點。"
    }
}

# ============================================
# 2) FastAPI 應用與 Webhook 更新
# ============================================
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

@app.middleware("http")
async def error_guard(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"請求處理失敗: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

app.mount("/static", StaticFiles(directory="static"), name="static")
router = APIRouter()

def update_line_webhook():
    """啟動時更新 LINE Webhook 到 /callback"""
    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
    json_data = {"endpoint": f"{BASE_URL}/callback"}
    with httpx.Client() as c:
        res = c.put("https://api.line.me/v2/bot/channel/webhook/endpoint",
                    headers=headers, json=json_data, timeout=10.0)
        res.raise_for_status()
        logger.info(f"✅ Webhook 更新成功: {res.status_code}")

def show_loading_animation(user_id: str, seconds: int = 5):
    """單聊時顯示「輸入中」動畫"""
    url = "https://api.line.me/v2/bot/chat/loading/start"
    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
    # 確保 seconds 是 5 的倍數且在 5-60 之間
    loading_seconds = max(5, min(60, seconds))
    loading_seconds = (loading_seconds // 5) * 5
    
    data = {"chatId": user_id, "loadingSeconds": loading_seconds}
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=5)
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
# 3) Groq 呼叫封裝 & 情緒分析
# ============================================
def groq_chat_completion(messages, max_tokens=600, temperature=0.7):
    """統一的 Groq 聊天完成函數，含備援"""
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
        logger.error(f"主要模型 {GROQ_MODEL_PRIMARY} 失敗: {e_primary}")
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
            logger.error(f"備用模型 {GROQ_MODEL_FALLBACK} 也失敗: {e_fallback}")
            return "抱歉，AI 服務暫時不可用。"

async def analyze_sentiment(text: str) -> str:
    """使用 Groq 判斷訊息情緒"""
    try:
        messages = [
            {"role": "system", "content": "你是情感分析助手，只輸出一個情緒標籤。"},
            {"role": "user", "content": f"判斷這句話的情緒：{text}\n只回傳：positive, neutral, negative, angry 其中之一"}
        ]
        result = groq_chat_completion(messages, max_tokens=10, temperature=0)
        return (result or "neutral").strip().lower()
    except Exception as e:
        logger.error(f"情感分析失敗: {e}")
        return "neutral"

# ============================================
# 4) 人設 Cosplay：可甜/可鹹/萌/酷
# ============================================
def set_user_persona(user_id: str, key: str) -> str:
    """設定使用者人設；不合法鍵值回退 sweet"""
    key = key.lower()
    if key not in PERSONAS:
        key = "sweet"
    user_persona[user_id] = key
    return key

def get_user_persona(user_id: str) -> str:
    """取得使用者目前人設，預設 sweet"""
    return user_persona.get(user_id, "sweet")

def build_persona_prompt(user_id: str, sentiment: str) -> str:
    """組合『人設 + 情緒調節』的 system prompt"""
    p_key = get_user_persona(user_id)
    p = PERSONAS[p_key]
    return f"""
你是一位「{p['title']}」。
【語氣風格】{p['style']}
【開場白】{p['greetings']}
【回覆格式】{p['reply_format']}
【情緒調節】目前使用者情緒：{sentiment}
- positive：一起開心，提升熱度；但保持自然不浮誇。
- negative：先共情與安慰，給具體陪伴/建議。
- angry：先降溫與傾聽，再提供舒壓與可執行建議。
- neutral：自然聊天，維持輕鬆流暢。
請用繁體中文回覆，句子精簡、自然、有溫度。
""".strip()

async def get_reply_with_persona_and_sentiment(user_id: str, messages: list, sentiment: str) -> str:
    """把人設 + 情緒 一起注入 system，再用 Groq 生成回覆"""
    sys = build_persona_prompt(user_id, sentiment)
    full_messages = [{"role": "system", "content": sys}] + messages
    return groq_chat_completion(full_messages, max_tokens=600, temperature=0.7)

# ============================================
# 5) Quick Reply 群組：使用下拉選單減少佔位
# ============================================
def build_quick_reply_items(is_group: bool, bot_name: str) -> List[QuickReplyButton]:
    """使用群組下拉選單減少 Quick Reply 按鈕數量"""
    items: List[QuickReplyButton] = []

    # 1. 人設選單（下拉式）
    items.append(QuickReplyButton(
        action=MessageAction(label="💖 人設選擇", text="人設選單")
    ))

    # 2. 金融服務選單（下拉式）
    items.append(QuickReplyButton(
        action=MessageAction(label="💰 金融服務", text="金融選單")
    ))

    # 3. 彩票服務選單（下拉式）
    items.append(QuickReplyButton(
        action=MessageAction(label="🎰 彩票服務", text="彩票選單")
    ))

    # 4. 系統設定（固定顯示）
    prefix = f"@{bot_name} " if is_group else ""
    items.extend([
        QuickReplyButton(action=MessageAction(label="✅ 開啟自動回答", text="開啟自動回答")),
        QuickReplyButton(action=MessageAction(label="❌ 關閉自動回答", text="關閉自動回答")),
        QuickReplyButton(action=MessageAction(label="🌤️ 天氣查詢", text=f"{prefix}天氣"))
    ])

    return items

# ============================================
# 6) Webhook 與訊息處理流程
# ============================================
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

    bot_name = line_bot_api.get_bot_info().display_name
    processed_msg = msg
    if msg.startswith('@'):
        processed_msg = re.sub(r'^@\S+\s*', '', msg).strip()

    # 處理選單指令
    low = processed_msg.lower()
    if low == '人設選單':
        await reply_simple(reply_token, "💖 請選擇人設風格：\n\n• 甜 - 溫柔體貼女友\n• 鹹 - 傲嬌吐槽女友\n• 萌 - 可愛動漫風格\n• 酷 - 冷靜御姐風格")
        return
    elif low == '金融選單':
        prefix = f"@{bot_name} " if is_group else ""
        await reply_simple(reply_token, f"💰 金融服務：\n\n• 大盤 - {prefix}大盤\n• 美股 - {prefix}美股\n• 金價 - {prefix}金價\n• 日元 - {prefix}JPY\n• 美元 - {prefix}USD")
        return
    elif low == '彩票選單':
        prefix = f"@{bot_name} " if is_group else ""
        await reply_simple(reply_token, f"🎰 彩票服務：\n\n• 大樂透 - {prefix}大樂透\n• 威力彩 - {prefix}威力彩\n• 539 - {prefix}539")
        return

    # 自動回覆開關
    if low == '開啟自動回答':
        auto_reply_status[chat_id] = True
        await reply_simple(reply_token, "✅ 已開啟自動回答")
        return
    if low == '關閉自動回答':
        auto_reply_status[chat_id] = False
        await reply_simple(reply_token, "✅ 已關閉自動回答")
        return

    # 群組未開啟時，僅在提到 bot 名稱時回覆
    if not auto_reply_status[chat_id]:
        if not any(name in msg.lower() for name in bot_name.lower().split()):
            return
        parts = re.split(r'@\S+\s*', msg, maxsplit=1)
        processed_msg = parts[1].strip() if len(parts) > 1 else ""

    # 人設切換指令
    if low in ("甜", "sweet", "溫柔"):
        key = set_user_persona(user_id, "sweet")
        await reply_simple(reply_token, f"已切換人設：{PERSONAS[key]['title']} 🌸")
        return
    if low in ("鹹", "salty", "幹話"):
        key = set_user_persona(user_id, "salty")
        await reply_simple(reply_token, f"已切換人設：{PERSONAS[key]['title']} 😏")
        return
    if low in ("萌", "moe"):
        key = set_user_persona(user_id, "moe")
        await reply_simple(reply_token, f"已切換人設：{PERSONAS[key]['title']} ✨")
        return
    if low in ("酷", "cool", "御姐", "教練"):
        key = set_user_persona(user_id, "cool")
        await reply_simple(reply_token, f"已切換人設：{PERSONAS[key]['title']} 🧊")
        return

    # 維持對話歷史
    conversation_history.setdefault(user_id, [])
    conversation_history[user_id].append({"role": "user", "content": processed_msg + "，請以繁體中文回答"})
    if len(conversation_history[user_id]) > MAX_HISTORY_LEN * 2:
        conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY_LEN*2:]

    reply_text = None
    try:
        # 內建指令路由
        if any(k in processed_msg for k in ["威力彩", "大樂透", "539", "雙贏彩"]):
            reply_text = lottery_gpt(processed_msg)
        elif processed_msg.startswith("104:"):
            reply_text = one04_gpt(processed_msg[4:].strip())
        elif processed_msg.lower().startswith(("大盤", "台股")):
            reply_text = stock_gpt("大盤")
        elif processed_msg.lower().startswith(("美盤", "美股")):
            reply_text = stock_gpt("美盤")
        elif processed_msg.startswith("pt:"):
            reply_text = partjob_gpt(processed_msg[3:])
        elif processed_msg.startswith(("cb:", "$:")):
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
            # 股票/代號
            stock_code   = re.fullmatch(r"\d{4,6}[A-Za-z]?", processed_msg)
            stockUS_code = re.fullmatch(r"[A-Za-z]{1,5}", processed_msg)
            if stock_code:
                reply_text = stock_gpt(stock_code.group())
            elif stockUS_code:
                reply_text = stock_gpt(stockUS_code.group())
            else:
                # 情感分析 → 注入人設 system → 生成回覆
                sentiment = await analyze_sentiment(processed_msg)
                reply_text = await get_reply_with_persona_and_sentiment(
                    user_id,
                    conversation_history[user_id][-MAX_HISTORY_LEN:],
                    sentiment
                )

    except Exception as e:
        logger.error(f"處理訊息時發生錯誤：{e}", exc_info=True)
        reply_text = "抱歉，伺服器發生錯誤，請稍後再試。"

    if not reply_text:
        reply_text = "抱歉，目前無法提供回應，請稍後再試。"

    # 使用統一 Quick Reply
    quick_items = build_quick_reply_items(is_group, bot_name)
    # 如果英文比例高 → 動態加上翻譯鍵
    if calculate_english_ratio(reply_text) > 0.1 and len(quick_items) < 13:
        quick_items.append(QuickReplyButton(action=MessageAction(label="翻譯成中文", text="請將上述內容翻譯成中文")))

    reply_message = TextSendMessage(text=reply_text, quick_reply=QuickReply(items=quick_items))
    try:
        line_bot_api.reply_message(reply_token, reply_message)
        conversation_history[user_id].append({"role": "assistant", "content": reply_text})
    except LineBotApiError as e:
        logger.error(f"回覆訊息失敗：{e.error.message}", exc_info=True)

async def reply_simple(reply_token, text):
    try:
        bot_name = line_bot_api.get_bot_info().display_name
        quick_items = build_quick_reply_items(is_group=False, bot_name=bot_name)
        line_bot_api.reply_message(reply_token, TextSendMessage(text=text, quick_reply=QuickReply(items=quick_items)))
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