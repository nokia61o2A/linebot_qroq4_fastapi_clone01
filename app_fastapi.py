"""
aibot FastAPI 應用程序初始化
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
try:
    from my_commands.lottery_gpt import lottery_gpt
except ImportError:
    def lottery_gpt(msg): return "彩票功能暫時不可用"

try:
    from my_commands.gold_gpt import gold_gpt
except ImportError:
    def gold_gpt(): return "金價功能暫時不可用"

try:
    from my_commands.platinum_gpt import platinum_gpt
except ImportError:
    def platinum_gpt(): return "鉑金價格功能暫時不可用"

try:
    from my_commands.money_gpt import money_gpt
except ImportError:
    def money_gpt(currency): return f"{currency}匯率功能暫時不可用"

try:
    from my_commands.one04_gpt import one04_gpt
except ImportError:
    def one04_gpt(msg): return "104功能暫時不可用"

try:
    from my_commands.partjob_gpt import partjob_gpt
except ImportError:
    def partjob_gpt(msg): return "打工功能暫時不可用"

try:
    from my_commands.crypto_coin_gpt import crypto_gpt
except ImportError:
    def crypto_gpt(coin): return f"{coin}加密貨幣功能暫時不可用"

try:
    from my_commands.stock.stock_gpt import stock_gpt
except ImportError:
    def stock_gpt(code): return f"{code}股票功能暫時不可用"

# 天氣功能 - 直接在這裡實作以避免循環匯入
def weather_gpt(city: str = "台北市") -> str:
    """簡單的天氣查詢功能"""
    try:
        # 這裡可以整合實際的天氣API
        # 目前返回模擬回應
        return f"🌤️ {city}今日天氣：晴時多雲，氣溫 25-30°C，降雨機率 20%"
    except Exception as e:
        logger.error(f"天氣查詢錯誤: {e}")
        return "天氣功能暫時不可用，請稍後再試"

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
        QuickReplyButton(action=MessageAction(label="🌐 翻譯選單", text="翻譯選單")),
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
        MessageAction(label="🇰🇷 翻韓文", text="翻譯->韓文"),
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
        MessageAction(label="🪙 比特幣", text=f"{prefix}$:BTC"),
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
            return "抱歉，AI 服務暫時不可用。請稍後再試 💔"

async def translate_text(text: str, target_language: str) -> str:
    """使用 Groq API 進行翻譯"""
    try:
        messages = [
            {"role": "system", "content": f"你是一位專業翻譯師。請將使用者提供的文字準確翻譯成{target_language}。只需要回傳翻譯結果，不要額外說明。"},
            {"role": "user", "content": text}
        ]
        return groq_chat_completion(messages, max_tokens=800, temperature=0.3)
    except Exception as e:
        logger.error(f"翻譯失敗: {e}")
        return f"翻譯失敗，原文：{text}"

async def analyze_sentiment(text: str) -> str:
    """分析使用者訊息的情緒"""
    try:
        messages = [
            {"role": "system", "content": "你是情感分析專家。分析使用者訊息的情緒，只回傳以下之一：positive（積極）, neutral（中性）, negative（消極）, angry（憤怒）, sad（悲傷）, happy（快樂）"},
            {"role": "user", "content": f"分析這句話的情緒：{text}"}
        ]
        result = groq_chat_completion(messages, max_tokens=20, temperature=0)
        return (result or "neutral").strip().lower()
    except Exception as e:
        logger.error(f"情感分析失敗: {e}")
        return "neutral"

# ============================================
# 人設設定
# ============================================
PERSONAS = {
    "sweet": {
        "title": "甜美女友", 
        "style": "溫柔體貼，用詞親暱，會關心對方感受，語調甜美", 
        "greetings": "親愛的～我在這裡陪你呢 🌸💕",
        "emoji": "🌸💕😊🥰"
    },
    "salty": {
        "title": "傲嬌女友", 
        "style": "表面冷淡實則關心，會吐槽但帶著愛意，有點小壞壞", 
        "greetings": "哼！又來找我了嗎...不過我就勉為其難陪你一下吧 😏💋",
        "emoji": "😏💋🙄😤"
    },
    "moe": {
        "title": "萌系女友", 
        "style": "可愛天真，語尾詞豐富，用詞軟萌，充滿活力", 
        "greetings": "呀呼～！今天也要被我萌到嗎～(ﾉ>ω<)ﾉ ✨",
        "emoji": "✨🎀(ﾉ>ω<)ﾉ🌈"
    },
    "cool": {
        "title": "酷系御姐", 
        "style": "冷靜理性，說話直接，給人可靠感，有領導氣質", 
        "greetings": "我在這裡。有什麼需要我幫你分析的嗎？ 🧊⚡",
        "emoji": "🧊⚡💎🖤"
    },
}

def set_user_persona(user_id: str, key: str):
    if key not in PERSONAS: 
        key = "sweet"
    user_persona[user_id] = key
    return key

def get_user_persona(user_id: str):
    return user_persona.get(user_id, "sweet")

def build_persona_prompt(user_id: str, sentiment: str) -> str:
    p_key = get_user_persona(user_id)
    p = PERSONAS[p_key]
    
    # 根據情緒調整回應風格
    emotion_guide = {
        "positive": "對方心情不錯，可以更活潑一點回應",
        "happy": "對方很開心，一起分享這份喜悦",
        "neutral": "正常聊天模式",
        "negative": "對方情緒低落，給予安慰和鼓勵",
        "sad": "對方很難過，溫柔陪伴和安慰",
        "angry": "對方生氣了，冷靜傾聽並安撫情緒"
    }
    
    emotion_tip = emotion_guide.get(sentiment, "正常聊天模式")
    
    return f"""
你是一位「{p['title']}」AI女友。

【角色特質】{p['style']}
【常用表情】{p['emoji']}
【情境分析】使用者當前情緒：{sentiment} - {emotion_tip}

【回應原則】
1. 用繁體中文自然對話
2. 保持你的人設風格
3. 回應要簡潔有趣，不要太長
4. 適時使用表情符號增加親和力
5. 根據對方情緒調整說話方式

請以你的角色風格回應使用者。
""".strip()

# ============================================
# 自動回答控制
# ============================================
def handle_auto_reply_toggle(chat_id: str, msg: str) -> str:
    """處理自動回答開關"""
    if msg == "開啟自動回答":
        auto_reply_status[chat_id] = True
        return "✅ 已開啟自動回答模式，我會回應所有訊息"
    elif msg == "關閉自動回答":
        auto_reply_status[chat_id] = False
        return "❌ 已關閉自動回答模式，只有@我才會回應"
    return None

# ============================================
# 訊息處理主函數
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

    # 確定聊天室ID
    chat_id = event.source.group_id if isinstance(event.source, SourceGroup) else (
        event.source.room_id if isinstance(event.source, SourceRoom) else user_id
    )
    
    # 初始化自動回答狀態
    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = not is_group  # 私聊預設開啟，群組預設關閉

    try:
        bot_name = line_bot_api.get_bot_info().display_name
    except Exception as e:
        logger.error(f"獲取bot名稱失敗: {e}")
        bot_name = "AI助手"

    low = msg.lower()

    # --- 檢查是否需要回應 ---
    should_reply = auto_reply_status.get(chat_id, True)
    if is_group and not should_reply:
        # 群組中關閉自動回答時，只有@機器人才回應
        if not (msg.startswith(f"@{bot_name}") or bot_name.lower() in low):
            return

    # 移除@機器人的前綴
    if msg.startswith(f"@{bot_name}"):
        msg = msg[len(f"@{bot_name}"):].strip()
        low = msg.lower()

    # --- 自動回答開關控制 ---
    toggle_result = handle_auto_reply_toggle(chat_id, msg)
    if toggle_result:
        await reply_simple(reply_token, toggle_result, is_group, bot_name)
        return

    # --- Flex 選單觸發 ---
    if low == '人設選單':
        line_bot_api.reply_message(reply_token, flex_menu_persona())
        return
    elif low == '金融選單':
        line_bot_api.reply_message(reply_token, flex_menu_finance(bot_name, is_group))
        return
    elif low == '彩票選單':
        line_bot_api.reply_message(reply_token, flex_menu_lottery(bot_name, is_group))
        return
    elif low == '翻譯選單':
        line_bot_api.reply_message(reply_token, flex_menu_translate())
        return

    # --- 翻譯模式處理 ---
    if low.startswith("翻譯->"):
        choice = low.replace("翻譯->", "")
        if choice == "結束":
            translation_requests.pop(user_id, None)
            await reply_simple(reply_token, "✅ 已結束翻譯模式", is_group, bot_name)
            return
        else:
            translation_requests[user_id] = {"lang": choice, "text": ""}
            await reply_simple(reply_token, f"🌐 翻譯模式已啟用，下一則訊息將翻譯成【{choice}】", is_group, bot_name)
            return
    
    # 處理翻譯請求
    elif user_id in translation_requests and translation_requests[user_id]["lang"]:
        target_lang = translation_requests[user_id]["lang"]
        translated = await translate_text(msg, target_lang)
        await reply_simple(reply_token, f"🌐 翻譯結果 ({target_lang})：\n\n{translated}", is_group, bot_name)
        return

    # --- 人設切換 ---
    persona_changes = {
        "甜": "sweet", "sweet": "sweet",
        "鹹": "salty", "salty": "salty", 
        "萌": "moe", "moe": "moe",
        "酷": "cool", "cool": "cool"
    }
    
    if low in persona_changes:
        key = set_user_persona(user_id, persona_changes[low])
        p = PERSONAS[key]
        await reply_simple(reply_token, f"💖 已切換人設：{p['title']}\n{p['greetings']}", is_group, bot_name)
        return

    # --- 功能指令處理 ---
    reply_text = None
    
    # 彩票類
    if any(k in msg for k in ["威力彩", "大樂透", "539", "雙贏彩"]):
        reply_text = lottery_gpt(msg)
    
    # 工作類
    elif msg.startswith("104:"):
        reply_text = one04_gpt(msg[4:].strip())
    elif msg.startswith("pt:"):
        reply_text = partjob_gpt(msg[3:].strip())
    
    # 加密貨幣
    elif msg.startswith("cb:") or msg.startswith("$:"):
        coin = msg[3:].strip() if msg.startswith("cb:") else msg[2:].strip()
        reply_text = crypto_gpt(coin)
    
    # 金融類
    elif "金價" in msg or "黃金" in msg:
        reply_text = gold_gpt()
    elif "鉑" in msg or "platinum" in msg.lower():
        reply_text = platinum_gpt()
    elif "USD" in msg or "美金" in msg or "美元" in msg:
        reply_text = money_gpt("USD")
    elif "JPY" in msg or "日幣" in msg or "日元" in msg:
        reply_text = money_gpt("JPY")
    
    # 股市類
    elif "大盤" in msg or "台股" in msg:
        reply_text = stock_gpt("大盤")
    elif "美股" in msg:
        reply_text = stock_gpt("美盤")
    
    # 天氣
    elif "天氣" in msg:
        # 嘗試提取城市名稱
        city_match = re.search(r"(台北|新北|桃園|台中|台南|高雄|基隆|新竹|苗栗|彰化|南投|雲林|嘉義|屏東|宜蘭|花蓮|台東|澎湖|金門|馬祖)", msg)
        city = city_match.group(1) if city_match else "台北市"
        reply_text = weather_gpt(city)
    
    # 股票代碼檢查
    elif re.fullmatch(r"\d{4,6}[A-Za-z]?", msg):
        reply_text = stock_gpt(msg)
    elif re.fullmatch(r"[A-Za-z]{1,5}", msg) and len(msg) <= 5:
        reply_text = stock_gpt(msg)
    
    # --- 預設：AI 聊天模式 ---
    else:
        # 初始化對話記錄
        if user_id not in conversation_history:
            conversation_history[user_id] = []

        # 加入使用者訊息到歷史記錄
        conversation_history[user_id].append({"role": "user", "content": msg})

        # 限制歷史記錄長度
        if len(conversation_history[user_id]) > MAX_HISTORY_LEN * 2:
            conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY_LEN*2:]

        # 情感分析
        sentiment = await analyze_sentiment(msg)
        
        # 建立人設提示
        persona_prompt = build_persona_prompt(user_id, sentiment)
        
        # 準備完整對話
        full_messages = [{"role": "system", "content": persona_prompt}] + conversation_history[user_id]

        # 生成回應
        reply_text = groq_chat_completion(full_messages, max_tokens=600, temperature=0.7)

        # 加入助手回應到歷史記錄
        if reply_text:
            conversation_history[user_id].append({"role": "assistant", "content": reply_text})

    # --- 發送回覆 ---
    if not reply_text:
        reply_text = "抱歉，我現在有點忙，請稍後再試試 💔"

    # 建立快速回覆按鈕
    quick_items = build_quick_reply_items(is_group, bot_name)
    reply_message = TextSendMessage(text=reply_text, quick_reply=QuickReply(items=quick_items))
    
    try:
        line_bot_api.reply_message(reply_token, reply_message)
    except LineBotApiError as e:
        logger.error(f"❌ 回覆訊息失敗：{e.error.message}", exc_info=True)

# ============================================
# 簡單回覆函數
# ============================================
async def reply_simple(reply_token, text, is_group=False, bot_name="AI助手"):
    """發送簡單文字回覆"""
    try:
        quick_items = build_quick_reply_items(is_group, bot_name)
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=text, quick_reply=QuickReply(items=quick_items))
        )
    except LineBotApiError as e:
        logger.error(f"❌ 回覆訊息失敗: {e}")

# ============================================
# Postback 事件處理
# ============================================
@handler.add(PostbackEvent)
async def handle_postback(event):
    """處理 Postback 事件"""
    data = event.postback.data
    user_id = event.source.user_id
    reply_token = event.reply_token
    
    logger.info(f"收到 Postback: {data} from user: {user_id}")
    
    # 可以根據 data 內容進行不同處理
    if data.startswith("persona_"):
        persona_key = data.replace("persona_", "")
        if persona_key in PERSONAS:
            set_user_persona(user_id, persona_key)
            p = PERSONAS[persona_key]
            await reply_simple(reply_token, f"💖 已切換人設：{p['title']}\n{p['greetings']}")

# ============================================
# 健康檢查與根路由
# ============================================
@app.get("/healthz")
async def health_check():
    """健康檢查端點"""
    return {"status": "ok", "message": "Line Bot is running"}

@app.get("/")
async def root():
    """根路由"""
    return {"message": "Line Bot Service is live.", "version": "1.0.0"}

@app.get("/status")
async def status():
    """