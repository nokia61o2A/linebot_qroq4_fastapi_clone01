"""
aibot FastAPI 應用程序初始化 (最終發音標註版)
"""
import os
import re
import asyncio
import logging
import random
from contextlib import asynccontextmanager
from typing import Dict, List

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

# --- 新增：發音標註函式庫 ---
try:
    from pypinyin import pinyin, Style
    PINYIN_ENABLED = True
except ImportError:
    PINYIN_ENABLED = False
    logger.warning("未安裝 'pypinyin'，中文注音功能將不可用。")

try:
    import pykakasi
    KAKASI_ENABLED = True
except ImportError:
    KAKASI_ENABLED = False
    logger.warning("未安裝 'pykakasi'，日文羅馬拼音功能將不可用。")

try:
    from korean_romanizer.romanizer import Romanizer
    KOREAN_ROMANIZER_ENABLED = True
except ImportError:
    KOREAN_ROMANIZER_ENABLED = False
    logger.warning("未安裝 'korean-romanizer'，韓文羅馬拼音功能將不可用。")


# --- 基礎設定 ---
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise ValueError("缺少必要的環境變數！")

line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-8b-instant")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-70b-versatile")

# --- 匯入自訂功能模組 ---
# (此部分保持不變)
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

# ============================================
# 狀態管理
# ============================================
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
auto_reply_status: Dict[str, bool] = {}
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}

# ============================================
# FastAPI 與 Webhook
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with httpx.AsyncClient() as client:
            await update_line_webhook(client)
    except Exception as e:
        logger.error(f"❌ 啟動初始化失敗: {e}", exc_info=True)
    yield

app = FastAPI(lifespan=lifespan, title="Line Bot API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
router = APIRouter()

async def update_line_webhook(client: httpx.AsyncClient):
    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
    json_data = {"endpoint": f"{BASE_URL}/callback"}
    res = await client.put("https://api.line.me/v2/bot/channel/webhook/endpoint",
                           headers=headers, json=json_data, timeout=10.0)
    res.raise_for_status()
    logger.info(f"✅ Webhook 更新成功: {res.status_code}")

# ============================================
# 選單生成 (Flex & QuickReply)
# (此部分保持不變)
# ============================================
def build_quick_reply_items(is_group: bool, bot_name: str) -> List[QuickReplyButton]:
    items = [
        QuickReplyButton(action=MessageAction(label="💖 我的人設", text="我的人設")),
        QuickReplyButton(action=MessageAction(label="💰 金融選單", text="金融選單")),
        QuickReplyButton(action=MessageAction(label="🎰 彩票選單", text="彩票選單")),
        QuickReplyButton(action=MessageAction(label="🌐 翻譯選單", text="翻譯選單")),
        QuickReplyButton(action=MessageAction(label="✅ 開啟自動回答", text="開啟自動回答")),
        QuickReplyButton(action=MessageAction(label="❌ 關閉自動回答", text="關閉自動回答")),
    ]
    return items

def build_flex_menu(title: str, subtitle: str, actions: List[MessageAction]) -> FlexSendMessage:
    buttons = [ButtonComponent(style="primary", height="sm", action=act, margin="md", color="#905C44") for act in actions]
    bubble = BubbleContainer(
        size="mega",
        header=BoxComponent(layout="vertical", contents=[TextComponent(text=title, weight="bold", size="xl", color="#FFFFFF", align="center"), TextComponent(text=subtitle, size="sm", color="#EEEEEE", wrap=True, align="center", margin="md")], spacing="sm", paddingAll="20px", backgroundColor="#FF6B6B", cornerRadius="lg"),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm", paddingAll="20px", backgroundColor="#FFF9F2", cornerRadius="lg"),
        footer=BoxComponent(layout="vertical", contents=[TextComponent(text="💖 點擊按鈕快速執行", size="xs", color="#888888", align="center", margin="md")], paddingAll="10px")
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
        MessageAction(label="📈 台股大盤", text=f"{prefix}大盤"), MessageAction(label="📊 美股大盤", text=f"{prefix}美股"),
        MessageAction(label="💰 金價查詢", text=f"{prefix}金價"), MessageAction(label="💴 日元匯率", text=f"{prefix}JPY"),
        MessageAction(label="💵 美元匯率", text=f"{prefix}USD"), MessageAction(label="🪙 比特幣", text=f"{prefix}$:BTC"),
    ]
    return build_flex_menu("💰 金融服務", "點擊下方按鈕快速查詢最新資訊", actions)

def flex_menu_lottery(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    actions = [
        MessageAction(label="🎰 大樂透", text=f"{prefix}大樂透"), MessageAction(label="🎯 威力彩", text=f"{prefix}威力彩"),
        MessageAction(label="🔢 539", text=f"{prefix}539"),
    ]
    return build_flex_menu("🎰 彩票服務", "快速開單與最新開獎資訊", actions)

def flex_menu_persona() -> FlexSendMessage:
    actions = [
        MessageAction(label="🌸 甜美女友", text="甜"), MessageAction(label="😏 傲嬌女友", text="鹹"),
        MessageAction(label="✨ 萌系女友", text="萌"), MessageAction(label="🧊 酷系御姐", text="酷"),
        MessageAction(label="📚 知性學姐", text="smart"), MessageAction(label="💪 元氣少女", text="cute"),
        MessageAction(label="🎲 隨機人設", text="random"),
    ]
    return build_flex_menu("💖 人設選擇", "切換 AI 女友的說話風格", actions)

# ============================================
# 🔥 新增：發音標註功能
# ============================================
def get_phonetic_transcription(text: str, target_language: str) -> str:
    """根據目標語言生成發音標註"""
    phonetics = []
    
    if target_language in ["繁體中文", "簡體中文"] and PINYIN_ENABLED:
        try:
            hanyu_pinyin_list = pinyin(text, style=Style.NORMAL)
            hanyu_pinyin = ' '.join([item[0] for item in hanyu_pinyin_list])
            phonetics.append(f"漢語拼音: {hanyu_pinyin}")

            bopomofo_list = pinyin(text, style=Style.BOPOMOFO)
            bopomofo = ' '.join([item[0] for item in bopomofo_list])
            phonetics.append(f"注音(ㄅㄆㄇ): {bopomofo}")
        except Exception as e:
            logger.error(f"Pypinyin 錯誤: {e}")

    elif target_language == "日文" and KAKASI_ENABLED:
        try:
            kks = pykakasi.kakasi()
            result = kks.convert(text)
            romaji = ' '.join([item['romaji'] for item in result])
            phonetics.append(f"羅馬拼音: {romaji}")
        except Exception as e:
            logger.error(f"Pykakasi 錯誤: {e}")

    elif target_language == "韓文" and KOREAN_ROMANIZER_ENABLED:
        try:
            romanized_text = Romanizer(text).run()
            phonetics.append(f"羅馬拼音: {romanized_text}")
        except Exception as e:
            logger.error(f"Korean Romanizer 錯誤: {e}")
            
    return "\n".join(phonetics)

# ============================================
# Groq 工具 (異步)
# ============================================
async def groq_chat_completion(messages, max_tokens=600, temperature=0.7):
    try:
        completion = await groq_client.chat.completions.create(model=GROQ_MODEL_PRIMARY, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return completion.choices[0].message.content
    except Exception as e_primary:
        logger.error(f"主要模型失敗: {e_primary}")
        try:
            completion = await groq_client.chat.completions.create(model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature)
            return completion.choices[0].message.content
        except Exception as e_fallback:
            logger.error(f"備用模型失敗: {e_fallback}")
            return "抱歉，AI 服務暫時不可用。請稍後再試 💔"

async def translate_text(text: str, target_language: str) -> str:
    messages = [{"role": "system", "content": f"你是一位專業翻譯師。請將使用者提供的文字準確翻譯成{target_language}。只需要回傳翻譯結果，不要包含任何額外的說明或引號。"}, {"role": "user", "content": text}]
    return await groq_chat_completion(messages, max_tokens=800, temperature=0.3)

async def analyze_sentiment(text: str) -> str:
    messages = [{"role": "system", "content": "你是情感分析專家。分析使用者訊息的情緒，只回傳以下之一：positive, neutral, negative, angry, sad, happy"}, {"role": "user", "content": f"分析這句話的情緒：{text}"}]
    result = await groq_chat_completion(messages, max_tokens=20, temperature=0)
    return (result or "neutral").strip().lower()

# ============================================
# 人設設定 (此部分保持不變)
# ============================================
PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼...", "greetings": "親愛的～我在這裡陪你呢 🌸💕", "emoji": "🌸💕😊🥰"},
    "salty": {"title": "傲嬌女友", "style": "表面冷淡...", "greetings": "哼！又來找我了嗎... 😏💋", "emoji": "😏💋🙄😤"},
    "moe": {"title": "萌系女友", "style": "可愛天真...", "greetings": "呀呼～！(ﾉ>ω<)ﾉ ✨", "emoji": "✨🎀(ﾉ>ω<)ﾉ🌈"},
    "cool": {"title": "酷系御姐", "style": "冷靜理性...", "greetings": "我在這裡。需要我幫你分析嗎？ 🧊⚡", "emoji": "🧊⚡💎🖤"},
    "smart": {"title": "知性學姐", "style": "博學多聞...", "greetings": "你好，有什麼我能幫你解答的嗎？📚✨", "emoji": "📚🔍🧠💡"},
    "cute": {"title": "元氣少女", "style": "活潑開朗...", "greetings": "嗨嗨！今天也要元氣滿滿哦！💪😄", "emoji": "💪😄🌟⭐"},
}

def set_user_persona(user_id: str, key: str):
    key = "sweet" if key not in PERSONAS and key != "random" else key
    if key == "random": key = random.choice(list(PERSONAS.keys()))
    user_persona[user_id] = key
    return key

def get_user_persona(user_id: str): return user_persona.get(user_id, "sweet")
def get_persona_info(user_id: str) -> str:
    p = PERSONAS[get_user_persona(user_id)]
    return f"💖 當前人設：{p['title']}\n\n【特質】{p['style']}\n【常用表情】{p['emoji']}\n\n{p['greetings']}"

def build_persona_prompt(user_id: str, sentiment: str) -> str:
    p = PERSONAS[get_user_persona(user_id)]
    emotion_guide = {"positive": "對方心情不錯...", "happy": "對方很開心...", "neutral": "正常聊天...", "negative": "對方情緒低落...", "sad": "對方很難過...", "angry": "對方生氣了..."}
    emotion_tip = emotion_guide.get(sentiment, "正常聊天模式")
    return f"你是一位「{p['title']}」AI女友...\n【角色特質】{p['style']}...\n【情境分析】...{emotion_tip}...\n請以你的角色風格回應使用者。"

# ============================================
# 訊息處理主邏輯
# ============================================
def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom): return event.source.room_id
    return event.source.user_id

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    user_id, chat_id = event.source.user_id, get_chat_id(event)
    msg, reply_token = event.message.text.strip(), event.reply_token
    is_group = isinstance(event.source, (SourceGroup, SourceRoom))

    if chat_id not in auto_reply_status: auto_reply_status[chat_id] = True
    try: bot_name = line_bot_api.get_bot_info().display_name
    except Exception: bot_name = "AI助手"

    low = msg.lower()
    if is_group and not auto_reply_status.get(chat_id, True) and not msg.startswith(f"@{bot_name}"): return
    if msg.startswith(f"@{bot_name}"):
        msg, low = msg[len(f"@{bot_name}"):].strip(), msg.lower()

    if msg == "開啟自動回答":
        auto_reply_status[chat_id] = True
        return reply_simple(reply_token, "✅ 已開啟自動回答模式", is_group, bot_name)
    elif msg == "關閉自動回答":
        auto_reply_status[chat_id] = False
        return reply_simple(reply_token, "❌ 已關閉自動回答模式", is_group, bot_name)
    
    menu_map = {'人設選單': flex_menu_persona(), '金融選單': flex_menu_finance(bot_name, is_group), '彩票選單': flex_menu_lottery(bot_name, is_group), '翻譯選單': flex_menu_translate()}
    if low in menu_map: return line_bot_api.reply_message(reply_token, menu_map[low])
    if low in ['我的人設', '當前人設']: return reply_simple(reply_token, get_persona_info(user_id), is_group, bot_name)

    if low.startswith("翻譯->"):
        choice = msg.replace("翻譯->", "").strip()
        if choice == "結束":
            translation_states.pop(chat_id, None)
            return reply_simple(reply_token, "✅ 已結束翻譯模式", is_group, bot_name)
        else:
            translation_states[chat_id] = choice
            return reply_simple(reply_token, f"🌐 本聊天室翻譯模式已啟用，下一則訊息將翻譯成【{choice}】", is_group, bot_name)

    if chat_id in translation_states:
        target_lang = translation_states[chat_id]
        translated = asyncio.run(translate_text(msg, target_lang))
        
        # 🔥 核心修改：生成並組合發音標註
        phonetic_info = get_phonetic_transcription(translated, target_lang)
        
        final_reply = f"🌐 翻譯結果 ({target_lang})：\n\n{translated}"
        if phonetic_info:
            final_reply += f"\n\n( {phonetic_info} )"
            
        return reply_simple(reply_token, final_reply, is_group, bot_name)
    
    persona_keys = {"甜": "sweet", "鹹": "salty", "萌": "moe", "酷": "cool", "smart": "smart", "知性": "smart", "cute": "cute", "元氣": "cute", "random": "random", "隨機": "random"}
    if low in persona_keys:
        key = set_user_persona(user_id, persona_keys[low])
        p = PERSONAS[key]
        return reply_simple(reply_token, f"💖 已切換人設：{p['title']}\n{p['greetings']}", is_group, bot_name)

    reply_text = None
    if any(k in msg for k in ["威力彩", "大樂透", "539"]): reply_text = lottery_gpt(msg)
    elif msg.startswith("104:"): reply_text = one04_gpt(msg[4:].strip())
    elif msg.startswith("pt:"): reply_text = partjob_gpt(msg[3:].strip())
    elif msg.startswith(("$:", "cb:")): reply_text = crypto_gpt(msg[3:].strip())
    elif "金價" in msg or "黃金" in msg: reply_text = gold_gpt()
    elif "鉑" in msg: reply_text = platinum_gpt()
    elif "USD" in msg or "美金" in msg: reply_text = money_gpt("USD")
    elif "JPY" in msg or "日幣" in msg: reply_text = money_gpt("JPY")
    elif "大盤" in msg or "台股" in msg: reply_text = stock_gpt("大盤")
    elif "美股" in msg: reply_text = stock_gpt("美盤")
    elif re.fullmatch(r"(\d{4,6}[A-Za-z]?)|([A-Za-z]{1,5})", msg): reply_text = stock_gpt(msg)
    
    if reply_text is None:
        if user_id not in conversation_history: conversation_history[user_id] = []
        conversation_history[user_id].append({"role": "user", "content": msg})
        if len(conversation_history[user_id]) > MAX_HISTORY_LEN * 2:
            conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY_LEN*2:]

        sentiment = asyncio.run(analyze_sentiment(msg))
        persona_prompt = build_persona_prompt(user_id, sentiment)
        full_messages = [{"role": "system", "content": persona_prompt}] + conversation_history[user_id]
        reply_text = asyncio.run(groq_chat_completion(full_messages))
        if reply_text: conversation_history[user_id].append({"role": "assistant", "content": reply_text})

    reply_text = reply_text or "抱歉，我現在有點忙，請稍後再試試 💔"
    quick_items = build_quick_reply_items(is_group, bot_name)
    line_bot_api.reply_message(reply_token, TextSendMessage(text=reply_text, quick_reply=QuickReply(items=quick_items)))

def reply_simple(reply_token, text, is_group=False, bot_name="AI助手"):
    quick_items = build_quick_reply_items(is_group, bot_name)
    line_bot_api.reply_message(reply_token, TextSendMessage(text=text, quick_reply=QuickReply(items=quick_items)))

@handler.add(PostbackEvent)
def handle_postback(event):
    data, user_id = event.postback.data, event.source.user_id
    if data.startswith("persona_"):
        key = data.replace("persona_", "")
        if key in PERSONAS:
            set_user_persona(user_id, key)
            p = PERSONAS[key]
            reply_simple(event.reply_token, f"💖 已切換人設：{p['title']}\n{p['greetings']}")

# ============================================
# FastAPI 路由
# ============================================
@router.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    try:
        await run_in_threadpool(handler.handle, body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(400, "Invalid signature")
    return JSONResponse({"message": "ok"})

@router.get("/")
async def root(): return {"message": "Line Bot Service is live.", "version": "1.0.0"}
app.include_router(router)