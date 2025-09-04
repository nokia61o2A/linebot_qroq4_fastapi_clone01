# app_fastapi.py
"""
AI 醬 (v31) - FastAPI on Render
- Zero-push reminders (APScheduler + Peewee/SQLite)
- Persona cosplay (sweet/salty/moe/cool)
- Sentiment-aware replies (Groq)
- Quick Reply & Flex vertical menu
"""

# ========= 1) Imports =========
import os
import re
import random
import logging
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

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
    FlexSendMessage, BubbleContainer, BoxComponent, TextComponent, ButtonComponent
)
from linebot.exceptions import LineBotApiError, InvalidSignatureError

# Groq（非同步）
from groq import AsyncGroq

# Peewee / SQLite（提醒資料）
from peewee import SqliteDatabase, Model, AutoField, CharField, DateTimeField, BooleanField

# APScheduler（只標記到期，不主動推播）
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    SCHED_AVAILABLE = True
except Exception as _:
    SCHED_AVAILABLE = False


# ========= 2) Globals & Clients =========
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

BASE_URL       = os.getenv("BASE_URL")
CHANNEL_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise RuntimeError("缺少必要環境變數：BASE_URL / CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET / GROQ_API_KEY")

line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler      = WebhookHandler(CHANNEL_SECRET)

groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_MODEL_PRIMARY  = os.getenv("GROQ_MODEL_PRIMARY",  "llama-3.1-8b-instant")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-70b-versatile")

# 對話記憶 / 自動回覆狀態 / 人設 / 翻譯狀態
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
auto_reply_status: Dict[str, bool] = {}
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}

# 可選指令模組（缺時不會炸）
try:
    from my_commands.lottery_gpt import lottery_gpt
except Exception:
    def lottery_gpt(_): return "🎰 彩券查詢暫時不可用"
try:
    from my_commands.gold_gpt import gold_gpt
except Exception:
    def gold_gpt(): return "💰 金價查詢暫時不可用"
try:
    from my_commands.platinum_gpt import platinum_gpt
except Exception:
    def platinum_gpt(): return "🪙 鉑金查詢暫時不可用"
try:
    from my_commands.money_gpt import money_gpt
except Exception:
    def money_gpt(code): return f"💱 匯率查詢（{code}）暫時不可用"
try:
    from my_commands.partjob_gpt import partjob_gpt
except Exception:
    def partjob_gpt(_): return "🧰 兼差查詢暫時不可用"
try:
    from my_commands.crypto_coin_gpt import crypto_gpt
except Exception:
    def crypto_gpt(_): return "₿ 加密幣查詢暫時不可用"
try:
    from my_commands.one04_gpt import one04_gpt
except Exception:
    def one04_gpt(_): return "104 職缺查詢暫時不可用"
try:
    from my_commands.stock.stock_gpt import stock_gpt
except Exception:
    def stock_gpt(code): return f"📈 股票/大盤（{code}）暫時不可用"
try:
    from my_commands.weather_gpt import weather_gpt
except Exception:
    def weather_gpt(_): return "🌤️ 天氣查詢暫時不可用"


# ========= 3) DB models (SQLite) =========
DB_PATH = os.getenv("REMINDER_DB", "reminders.db")
db = SqliteDatabase(DB_PATH)

class BaseModel(Model):
    class Meta:
        database = db

class Reminder(BaseModel):
    id      = AutoField()
    chat_id = CharField(index=True)   # 依聊天室（個人/群組）區分
    text    = CharField()
    due_at  = DateTimeField(index=True)
    sent    = BooleanField(default=False)  # 已回覆
    due     = BooleanField(default=False)  # 到期（排程標記）

def init_db():
    db.connect(reuse_if_open=True)
    db.create_tables([Reminder], safe=True)
    logger.info("✅ SQLite/peewee 初始化完成")


# ========= 4) APScheduler (mark due only) =========
scheduler: Optional["AsyncIOScheduler"] = None

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

def start_scheduler():
    global scheduler
    if not SCHED_AVAILABLE:
        logger.warning("APScheduler 未安裝，無法啟動排程（提醒仍可手動查詢）")
        return
    scheduler = AsyncIOScheduler()
    scheduler.add_job(mark_due_reminders, "interval", seconds=60, id="mark_due_job", replace_existing=True)
    scheduler.start()
    logger.info("✅ APScheduler 啟動，60 秒掃描到期提醒")


# ========= 5) FastAPI app & startup =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    start_scheduler()
    try:
        async with httpx.AsyncClient() as c:
            headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
            payload = {"endpoint": f"{BASE_URL}/callback"}
            res = await c.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=payload, timeout=10.0)
            res.raise_for_status()
            logger.info(f"✅ Webhook 更新成功: {res.status_code}")
    except Exception as e:
        logger.error(f"❌ Webhook 更新失敗：{e}", exc_info=True)

    yield

app = FastAPI(lifespan=lifespan, title="Line Bot API", version="1.0.0")
router = APIRouter()
app.mount("/static", StaticFiles(directory="static"), name="static")


# ========= 6) Persona / Menus / Helpers =========
PERSONAS: Dict[str, dict] = {
    "sweet": {
        "title": "甜美女友",
        "style": "語氣溫柔體貼、鼓勵安慰，偶爾貼心 emoji，但不浮誇。",
        "greetings": "嗨～我在這裡，先深呼吸，我陪你喔。🌸",
        "emoji": "🌸💕😊🥰",
    },
    "salty": {
        "title": "傲嬌女友",
        "style": "機智吐槽、有點壞壞但不失溫度；避免人身攻擊。",
        "greetings": "欸你來啦～我就知道你又想我了😏",
        "emoji": "😏😒🙄",
    },
    "moe": {
        "title": "萌系女友",
        "style": "動漫風，可愛語尾與顏文字；內容仍保重點。",
        "greetings": "呀呼～今天也要被我治癒一下嗎？(ﾉ>ω<)ﾉ",
        "emoji": "✨🎀(ﾉ>ω<)ﾉ⭐",
    },
    "cool": {
        "title": "酷系御姐",
        "style": "話少但有氣場；冷靜分析，建議精準。",
        "greetings": "我在。先說你的狀況，我會幫你理清。",
        "emoji": "🧊⚡️🖤",
    },
}

LANGUAGE_MAP = {
    "英文": "English", "日文": "Japanese", "韓文": "Korean",
    "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"
}

def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom): return event.source.room_id
    return event.source.user_id

def build_quick_reply_items(is_group: bool, bot_name: str) -> List[QuickReplyButton]:
    # <= 13 items（LINE 限制）
    return [
        QuickReplyButton(action=MessageAction(label="🌸 甜", text="甜")),
        QuickReplyButton(action=MessageAction(label="😏 鹹", text="鹹")),
        QuickReplyButton(action=MessageAction(label="🎀 萌", text="萌")),
        QuickReplyButton(action=MessageAction(label="🧊 酷", text="酷")),
        QuickReplyButton(action=MessageAction(label="💖 人設選單", text="人設選單")),
        QuickReplyButton(action=MessageAction(label="💰 金融選單", text="金融選單")),
        QuickReplyButton(action=MessageAction(label="🎰 彩票選單", text="彩票選單")),
        QuickReplyButton(action=MessageAction(label="🌐 翻譯選單", text="翻譯選單")),
        QuickReplyButton(action=MessageAction(label="⏰ 新增提醒", text="提醒教學")),
        QuickReplyButton(action=MessageAction(label="✅ 開啟自動回答", text="開啟自動回答")),
        QuickReplyButton(action=MessageAction(label="❌ 關閉自動回答", text="關閉自動回答")),
    ]

def build_flex_menu(title: str, subtitle: str, actions: List[MessageAction]) -> FlexSendMessage:
    buttons = [ButtonComponent(style="primary", height="sm", action=act, margin="md", color="#00B900") for act in actions]
    bubble = BubbleContainer(
        header=BoxComponent(
            layout="vertical",
            contents=[
                TextComponent(text=title, weight="bold", size="xl", color="#000000", align="center"),
                TextComponent(text=subtitle, size="sm", color="#666666", wrap=True, align="center", margin="md"),
            ],
            backgroundColor="#FFFFFF",
        ),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm", paddingAll="12px", backgroundColor="#FAFAFA"),
    )
    return FlexSendMessage(alt_text=title, contents=bubble)

def flex_menu_finance(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    actions = [
        MessageAction(label="🇹🇼 台股大盤", text=f"{prefix}大盤"),
        MessageAction(label="🇺🇸 美股大盤", text=f"{prefix}美股"),
        MessageAction(label="💰 金價查詢", text=f"{prefix}金價"),
        MessageAction(label="💴 日元匯率", text=f"{prefix}JPY"),
        MessageAction(label="📊 查個股 (例: 2330)", text=f"{prefix}2330"),
    ]
    return build_flex_menu("💰 金融服務", "快速查詢最新金融資訊", actions)

def flex_menu_lottery(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    actions = [
        MessageAction(label="🎰 大樂透", text=f"{prefix}大樂透"),
        MessageAction(label="🎯 威力彩",  text=f"{prefix}威力彩"),
        MessageAction(label="🔢 539",    text=f"{prefix}539"),
    ]
    return build_flex_menu("🎰 彩票服務", "最新開獎資訊", actions)

def flex_menu_translate() -> FlexSendMessage:
    actions = [
        MessageAction(label="🇺🇸 翻英文",       text="翻譯->英文"),
        MessageAction(label="🇻🇳 翻越南文",     text="翻譯->越南文"),
        MessageAction(label="🇯🇵 翻日文",       text="翻譯->日文"),
        MessageAction(label="🇰🇷 翻韓文",       text="翻譯->韓文"),
        MessageAction(label="🇹🇼 翻繁體中文",   text="翻譯->繁體中文"),
        MessageAction(label="❌ 結束翻譯",     text="翻譯->結束"),
    ]
    return build_flex_menu("🌐 翻譯選擇", "選擇目標語言", actions)

def flex_menu_persona() -> FlexSendMessage:
    actions = [
        MessageAction(label="🌸 甜美女友", text="甜"),
        MessageAction(label="😏 傲嬌女友", text="鹹"),
        MessageAction(label="🎀 萌系女友", text="萌"),
        MessageAction(label="🧊 酷系御姐", text="酷"),
        MessageAction(label="🎲 隨機人設", text="random"),
    ]
    return build_flex_menu("💖 人設選擇", "切換 AI 女友的說話風格", actions)

def set_user_persona(chat_id: str, key: str) -> str:
    if key == "random": key = random.choice(list(PERSONAS.keys()))
    if key not in PERSONAS: key = "sweet"
    user_persona[chat_id] = key
    return key

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    key = user_persona.get(chat_id, "sweet")
    p = PERSONAS[key]
    emotion_tip = {
        "positive": "對方心情不錯，可以更活潑一點回應",
        "happy":    "對方很開心，一起分享這份喜悅",
        "neutral":  "正常聊天模式",
        "negative": "對方情緒低落，給予安慰與鼓勵",
        "sad":      "對方很難過，請溫柔陪伴與安慰",
        "angry":    "對方生氣，先冷靜傾聽並安撫情緒，再給建議",
    }.get(sentiment, "正常聊天模式")
    return f"""
你是一位「{p['title']}」AI 女友。角色特質：「{p['style']}」。
當前使用者情緒：{sentiment} → {emotion_tip}
請用繁體中文回覆，語氣自然、有溫度，句子精煉，適度使用表情符號（{p['emoji']}），避免太長篇。
""".strip()

def calculate_english_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters: return 0.0
    english = [c for c in letters if ord(c) < 128]
    return len(english) / len(letters)

async def groq_chat_completion(messages, max_tokens=600, temperature=0.7) -> str:
    try:
        r = await groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY, messages=messages,
            max_tokens=max_tokens, temperature=temperature
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e1:
        logger.error(f"Groq 主模型失敗：{e1}")
        r = await groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK, messages=messages,
            max_tokens=max_tokens, temperature=temperature
        )
        return (r.choices[0].message.content or "").strip()

async def analyze_sentiment(text: str) -> str:
    msgs = [
        {"role": "system", "content": "Analyze sentiment. Reply with one token: positive, neutral, negative, angry, sad, happy."},
        {"role": "user", "content": text}
    ]
    res = await groq_chat_completion(msgs, max_tokens=20, temperature=0)
    return (res or "neutral").split()[0].strip().lower()

async def translate_text(text: str, target_language: str) -> str:
    sys = "You are a professional translation engine. Output ONLY the translated text."
    usr = f"""{{
  "source_language": "auto-detect",
  "target_language": "{target_language}",
  "text_to_translate": "{text}"
}}"""
    return await groq_chat_completion(
        [{"role":"system","content":sys},{"role":"user","content":usr}],
        max_tokens=800, temperature=0.3
    )

def reply_simple(reply_token, text, is_group, bot_name):
    """所有文字回覆都走這裡 → 一律帶 QuickReply"""
    try:
        items = build_quick_reply_items(is_group, bot_name)
        if calculate_english_ratio(text) > 0.1 and len(items) < 13:
            items.append(QuickReplyButton(action=MessageAction(label="翻譯成中文", text="翻譯->繁體中文")))
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=text, quick_reply=QuickReply(items=items))
        )
    except LineBotApiError as e:
        logger.error(f"回覆失敗：{e}", exc_info=True)


# ========= 7) LINE Webhook =========
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    asyncio.create_task(handle_message_async(event))

async def handle_message_async(event: MessageEvent):
    chat_id = get_chat_id(event)
    user_id = event.source.user_id
    msg     = (event.message.text or "").strip()
    reply_token = event.reply_token
    is_group = isinstance(event.source, (SourceGroup, SourceRoom))
    bot_name = (line_bot_api.get_bot_info().display_name if hasattr(line_bot_api, "get_bot_info") else "AI醬")

    if not msg: return
    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True

    low = msg.lower()

    # --- 開關自動回答 ---
    if low == "開啟自動回答":
        auto_reply_status[chat_id] = True
        return reply_simple(reply_token, "✅ 已開啟自動回答模式", is_group, bot_name)
    if low == "關閉自動回答":
        auto_reply_status[chat_id] = False
        return reply_simple(reply_token, "❌ 已關閉自動回答模式", is_group, bot_name)

    # 群組未開啟 → 需 @botname 觸發
    if is_group and not auto_reply_status.get(chat_id, True) and not low.startswith(f"@{bot_name}".lower()):
        return
    if low.startswith(f"@{bot_name}".lower()):
        msg = msg[len(f"@{bot_name}"):].strip()
        low = msg.lower()

    # --- 菜單快捷 ---
    menu_map = {
        "金融選單": lambda: flex_menu_finance(bot_name, is_group),
        "彩票選單": lambda: flex_menu_lottery(bot_name, is_group),
        "翻譯選單": lambda: flex_menu_translate(),
        "我的人設": lambda: flex_menu_persona(),
        "人設選單": lambda: flex_menu_persona(),
        "提醒教學": lambda: None,
    }
    if low in menu_map:
        if low == "提醒教學":
            return reply_simple(
                reply_token,
                "⏰ 新增提醒：\n輸入格式：\n\n提醒我 HH:MM 內容\n\n例：提醒我 21:30 量血壓\n\n系統會在時間到後標記，到你**下一次說話時**一次回覆（不耗推播額度）。",
                is_group, bot_name
            )
        return line_bot_api.reply_message(reply_token, menu_map[low]())

    # --- 翻譯模式開關 ---
    if low.startswith("翻譯->"):
        choice = msg.split("->", 1)[1].strip()
        if choice == "結束":
            translation_states.pop(chat_id, None)
            return reply_simple(reply_token, "✅ 已結束翻譯模式", is_group, bot_name)
        translation_states[chat_id] = choice
        return reply_simple(reply_token, f"🌐 本聊天室翻譯模式：{choice}，直接貼文字我會翻譯。", is_group, bot_name)

    # --- 先處理「拉式提醒」：有 due 的就優先回覆 ---
    due_list = list(Reminder.select().where(
        (Reminder.chat_id == chat_id) & (Reminder.sent == False) & (Reminder.due == True)
    ))
    if due_list:
        lines = ["⏰ 到點提醒："]
        for r in due_list:
            local_time = r.due_at.astimezone().strftime("%H:%M")
            lines.append(f"• {r.text}（原定 {local_time}）")
        # 彙整回覆也一樣用 reply_simple → 帶 QuickReply
        reply_simple(reply_token, "\n".join(lines), is_group, bot_name)
        (Reminder.update(sent=True).where(Reminder.id.in_([r.id for r in due_list]))).execute()
        return

    # --- 新增提醒（零推播）：提醒我 HH:MM 內容 ---
    if low.startswith("提醒我"):
        parts = msg.split(maxsplit=2)
        if len(parts) < 3:
            return reply_simple(reply_token, "格式：提醒我 21:30 內容", is_group, bot_name)
        time_str, text = parts[1], parts[2]
        try:
            h, m = map(int, time_str.split(":"))
            now = datetime.now(timezone.utc)
            due_at = now.replace(hour=h, minute=m, second=0, microsecond=0)
            # 如果今天已過，視為明天同時間
            if due_at <= now:
                due_at = due_at.replace(day=now.day) + timedelta(days=1)
        except Exception:
            return reply_simple(reply_token, "時間格式錯誤，請用 HH:MM（如 21:30）", is_group, bot_name)

        Reminder.create(chat_id=chat_id, text=text, due_at=due_at)
        return reply_simple(
            reply_token,
            f"✅ 我記下了～到 {due_at.astimezone().strftime('%H:%M')} 我會提醒你（在你**下一次說話時**回覆，不耗推播額度）",
            is_group, bot_name
        )

    # --- 翻譯流程 ---
    if chat_id in translation_states:
        display_lang = translation_states[chat_id]
        target_lang  = LANGUAGE_MAP.get(display_lang, display_lang)
        try:
            translated = await translate_text(msg, target_lang)
            return reply_simple(
                reply_token,
                f"🌐 翻譯結果（{display_lang}）：\n\n{translated}",
                is_group, bot_name
            )
        except Exception as e:
            logger.error(f"翻譯失敗：{e}", exc_info=True)
            return reply_simple(reply_token, "翻譯服務暫時忙線，再試一次可以嗎？", is_group, bot_name)

    # --- 指令路由 ---
    if any(k in msg for k in ["威力彩", "大樂透", "539", "雙贏彩"]):
        return reply_simple(reply_token, lottery_gpt(msg), is_group, bot_name)
    if msg.startswith("104:"):
        return reply_simple(reply_token, one04_gpt(msg[4:].strip()), is_group, bot_name)
    if any(msg.lower().startswith(k) for k in ["金價", "黃金", "gold"]):
        return reply_simple(reply_token, gold_gpt(), is_group, bot_name)
    if any(msg.lower().startswith(k) for k in ["鉑", "platinum"]):
        return reply_simple(reply_token, platinum_gpt(), is_group, bot_name)
    if any(msg.lower().startswith(k) for k in ["日幣", "jpy"]):
        return reply_simple(reply_token, money_gpt("JPY"), is_group, bot_name)
    if any(msg.lower().startswith(k) for k in ["美金", "usd"]):
        return reply_simple(reply_token, money_gpt("USD"), is_group, bot_name)
    if any(k in msg for k in ["天氣", "氣象"]):
        return reply_simple(reply_token, weather_gpt("桃園市"), is_group, bot_name)
    if msg.startswith("pt:"):
        return reply_simple(reply_token, partjob_gpt(msg[3:].strip()), is_group, bot_name)
    if msg.startswith(("cb:", "$:")):
        coin_id = msg[3:].strip() if msg.startswith("cb:") else msg[2:].strip()
        return reply_simple(reply_token, crypto_gpt(coin_id), is_group, bot_name)

    # 股票 / 大盤
    if low in ("大盤", "台股", "台股大盤"):
        return reply_simple(reply_token, stock_gpt("大盤"), is_group, bot_name)
    if low in ("美股", "美盤", "美股大盤"):
        return reply_simple(reply_token, stock_gpt("美盤"), is_group, bot_name)
    if re.fullmatch(r"\d{4,6}[A-Za-z]?", msg) or re.fullmatch(r"[A-Za-z]{1,5}", msg):
        return reply_simple(reply_token, stock_gpt(msg.upper()), is_group, bot_name)

    # --- 人設切換 ---
    persona_keys = {"甜": "sweet", "鹹": "salty", "萌": "moe", "酷": "cool", "random": "random", "隨機": "random"}
    if low in persona_keys:
        key = set_user_persona(chat_id, persona_keys[low])
        p = PERSONAS[key]
        return reply_simple(
            reply_token,
            f"已切換人設：{p['title']} ！\n\n【風格】{p['style']}\n{p['greetings']}",
            is_group, bot_name
        )

    # --- 走 LLM：情感 + 人設 ---
    conversation_history.setdefault(chat_id, [])
    conversation_history[chat_id].append({"role": "user", "content": msg})
    conversation_history[chat_id] = conversation_history[chat_id][-MAX_HISTORY_LEN*2:]

    try:
        sentiment = await analyze_sentiment(msg)
        sys = build_persona_prompt(chat_id, sentiment)
        messages = [{"role":"system","content":sys}] + conversation_history[chat_id][-MAX_HISTORY_LEN:]  # 帶入短歷史
        ai_reply = await groq_chat_completion(messages, max_tokens=600, temperature=0.7)
        conversation_history[chat_id].append({"role":"assistant", "content": ai_reply})
        conversation_history[chat_id] = conversation_history[chat_id][-MAX_HISTORY_LEN*2:]
        return reply_simple(reply_token, ai_reply, is_group, bot_name)
    except Exception as e:
        logger.error(f"AI 回覆失敗：{e}", exc_info=True)
        return reply_simple(reply_token, "抱歉，我剛剛走神了 😅 再跟我說一次？", is_group, bot_name)


@handler.add(PostbackEvent)
def handle_postback(_: PostbackEvent):
    # 預留（目前未用）
    return


# ========= 8) FastAPI routes =========
@router.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    try:
        await run_in_threadpool(handler.handle, body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(400, "Invalid signature")
    except Exception as e:
        logger.error(f"Callback 處理失敗：{e}", exc_info=True)
        raise HTTPException(500, "Internal error")
    return JSONResponse({"message": "ok"})

@router.get("/healthz")
async def healthz():
    return {"status": "ok"}

@router.get("/")
async def root():
    return {"message": "Service is live."}

app.include_router(router)


# ========= 9) Uvicorn local run =========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info")