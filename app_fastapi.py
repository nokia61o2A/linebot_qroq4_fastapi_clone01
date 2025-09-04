"""
aibot FastAPI 應用程序初始化 (v30)
- 修正: RuntimeError 'no running event loop'
- 作法: 在 @handler.add 的同步處理器裡改用 asyncio.run(...) 執行 async 邏輯
- 補充: 匯入 timedelta；維持 run_in_threadpool 包裝 handler.handle()
"""

# ============================================
# 1. Imports
# ============================================
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
    ButtonComponent, SeparatorComponent
)
from linebot.exceptions import LineBotApiError, InvalidSignatureError

from groq import AsyncGroq

# SQLite / Peewee
from peewee import (
    SqliteDatabase, Model, AutoField, CharField,
    DateTimeField, BooleanField
)

# APScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# 時間
from datetime import datetime, timezone, timedelta

# 可選的發音/轉寫工具（缺就自動降級）
try:
    from pypinyin import pinyin, Style
    PINYIN_ENABLED = True
except ImportError:
    PINYIN_ENABLED = False
try:
    import pykakasi
    KAKASI_ENABLED = True
except ImportError:
    KAKASI_ENABLED = False
try:
    from korean_romanizer.romanizer import Romanizer
    KOREAN_ROMANIZER_ENABLED = True
except ImportError:
    KOREAN_ROMANIZER_ENABLED = False
try:
    from hangul_jamo import decompose
    HANGUL_JAMO_ENABLED = True
except ImportError:
    HANGUL_JAMO_ENABLED = False

# ============================================
# 2. Config & Globals
# ============================================
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

BASE_URL       = os.getenv("BASE_URL")
CHANNEL_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise ValueError("缺少必要的環境變數：BASE_URL / CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET / GROQ_API_KEY")

line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler      = WebhookHandler(CHANNEL_SECRET)

groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_MODEL_PRIMARY  = os.getenv("GROQ_MODEL_PRIMARY",  "llama-3.1-8b-instant")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-70b-versatile")

conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
auto_reply_status: Dict[str, bool] = {}
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}

LANGUAGE_MAP = {
    "英文": "English", "日文": "Japanese", "韓文": "Korean",
    "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"
}

PERSONAS = {
    "sweet": {
        "title": "甜美女友",
        "style": "溫柔體貼，鼓勵安慰，語氣柔和。",
        "greetings": "親愛的，你來啦～我在這聽你說 🌸",
        "emoji": "🌸💕😊🥰"
    },
    "salty": {
        "title": "傲嬌女友",
        "style": "機智吐槽、有點壞壞但不失溫度。",
        "greetings": "哼，還知道來找我？說吧你又怎了😏",
        "emoji": "😏😒🙄"
    },
    "moe": {
        "title": "萌系女友",
        "style": "動漫可愛風格，元氣滿滿 (๑•̀ㅂ•́)و✧",
        "greetings": "呀呼～有沒有想我呀？(ﾉ>ω<)ﾉ ✨",
        "emoji": "✨🎀(ﾉ>ω<)ﾉ⭐"
    },
    "cool": {
        "title": "酷系御姐",
        "style": "冷靜、成熟、給一針見血的建議。",
        "greetings": "我在。說重點，我幫你理清。",
        "emoji": "🧊⚡️🖤"
    }
}

# ============================================
# 3. DB (Peewee) & Scheduler（零推播提醒）
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
    due_at  = DateTimeField(index=True)   # UTC
    sent    = BooleanField(default=False) # 已彙整回覆
    due     = BooleanField(default=False) # 到期待回覆

def init_db():
    db.connect(reuse_if_open=True)
    db.create_tables([Reminder], safe=True)
    logger.info("✅ SQLite/peewee 初始化完成")

scheduler = AsyncIOScheduler()

def mark_due_reminders():
    now = datetime.now(timezone.utc)
    q = (Reminder.update(due=True)
         .where((Reminder.sent == False) &
                (Reminder.due == False) &
                (Reminder.due_at <= now)))
    n = q.execute()
    if n:
        logger.info(f"[Scheduler] 標記到期提醒 {n} 筆為 due")

def start_scheduler():
    scheduler.add_job(mark_due_reminders, "interval", seconds=60, id="mark_due", replace_existing=True)
    scheduler.start()
    logger.info("✅ APScheduler 啟動，60 秒掃描到期提醒")

# ============================================
# 4. Helpers（Webhook、LLM、選單、翻譯、情緒）
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
        start_scheduler()
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
            json_data = {"endpoint": f"{BASE_URL}/callback"}
            res = await client.put("https://api.line.me/v2/bot/channel/webhook/endpoint",
                                   headers=headers, json=json_data, timeout=10.0)
            res.raise_for_status()
            logger.info(f"✅ Webhook 更新成功: {res.status_code}")
    except Exception as e:
        logger.error(f"❌ 啟動初始化失敗: {e}", exc_info=True)
    yield

app = FastAPI(lifespan=lifespan, title="Line Bot API", version="1.0.0")
router = APIRouter()
app.mount("/static", StaticFiles(directory="static"), name="static")

async def groq_chat_completion(messages, max_tokens=600, temperature=0.7) -> str:
    try:
        resp = await groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY, messages=messages,
            max_tokens=max_tokens, temperature=temperature
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"Groq 主要模型失敗: {e}")
        resp = await groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK, messages=messages,
            max_tokens=max_tokens, temperature=temperature
        )
        return (resp.choices[0].message.content or "").strip()

async def analyze_sentiment(text: str) -> str:
    msgs = [
        {"role": "system", "content": "Analyze sentiment. Reply only: positive, neutral, negative, angry, sad, happy."},
        {"role": "user", "content": text}
    ]
    result = await groq_chat_completion(msgs, 20, 0)
    return (result or "neutral").lower().strip()

async def translate_text(text: str, target_language: str) -> str:
    system = "You are a professional translation engine. Output only the translated text."
    user = f'{{"source_language":"auto-detect","target_language":"{target_language}","text_to_translate":"{text}"}}'
    return await groq_chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        800, 0.3
    )

def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom):  return event.source.room_id
    return event.source.user_id

def build_quick_reply_items(is_group: bool, bot_name: str) -> List[QuickReplyButton]:
    # ≤ 13 個
    return [
        QuickReplyButton(action=MessageAction(label="🌸 甜", text="甜")),
        QuickReplyButton(action=MessageAction(label="😏 鹹", text="鹹")),
        QuickReplyButton(action=MessageAction(label="🎀 萌", text="萌")),
        QuickReplyButton(action=MessageAction(label="🧊 酷", text="酷")),
        QuickReplyButton(action=MessageAction(label="💖 人設選單", text="我的人設")),
        QuickReplyButton(action=MessageAction(label="💰 金融選單", text="金融選單")),
        QuickReplyButton(action=MessageAction(label="🎰 彩票選單", text="彩票選單")),
        QuickReplyButton(action=MessageAction(label="🌐 翻譯選單", text="翻譯選單")),
        QuickReplyButton(action=MessageAction(label="✅ 開啟自動回答", text="開啟自動回答")),
        QuickReplyButton(action=MessageAction(label="❌ 關閉自動回答", text="關閉自動回答")),
        QuickReplyButton(action=MessageAction(label="⏰ 建立提醒", text="提醒我 21:30 測試提醒"))
    ]

def build_flex_menu(title: str, subtitle: str, actions: List[MessageAction]) -> FlexSendMessage:
    btns = []
    for act in actions:
        btns.append(ButtonComponent(style="primary", height="sm", action=act, margin="md", color="#00B900"))
        btns.append(SeparatorComponent(margin="md"))
    if btns: btns.pop()  # 移除最後一個 Separator

    bubble = BubbleContainer(
        header=BoxComponent(
            layout="vertical",
            contents=[
                TextComponent(text=title, weight="bold", size="xl", color="#000000", align="center"),
                TextComponent(text=subtitle, size="sm", color="#666666", wrap=True, align="center", margin="md")
            ]
        ),
        body=BoxComponent(layout="vertical", contents=btns, spacing="sm")
    )
    return FlexSendMessage(alt_text=title, contents=bubble)

def flex_menu_finance(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    actions = [
        MessageAction(label="🇹🇼 台股大盤", text=f"{prefix}台股大盤"),
        MessageAction(label="🇺🇸 美股大盤", text=f"{prefix}美股大盤"),
        MessageAction(label="💰 金價", text=f"{prefix}金價"),
        MessageAction(label="💴 日元", text=f"{prefix}JPY"),
        MessageAction(label="📊 個股(例:2330)", text=f"{prefix}2330"),
    ]
    return build_flex_menu("💰 金融服務", "快速查詢金融資訊", actions)

def flex_menu_lottery(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    actions = [
        MessageAction(label="🎰 大樂透", text=f"{prefix}大樂透"),
        MessageAction(label="🎯 威力彩", text=f"{prefix}威力彩"),
        MessageAction(label="🔢 539", text=f"{prefix}539"),
    ]
    return build_flex_menu("🎰 彩票服務", "最新開獎資訊", actions)

def flex_menu_translate() -> FlexSendMessage:
    actions = [
        MessageAction(label="🇺🇸 翻英文", text="翻譯->英文"),
        MessageAction(label="🇻🇳 翻越南文", text="翻譯->越南文"),
        MessageAction(label="🇯🇵 翻日文", text="翻譯->日文"),
        MessageAction(label="🇰🇷 翻韓文", text="翻譯->韓文"),
        MessageAction(label="🇹🇼 翻繁中", text="翻譯->繁體中文"),
        MessageAction(label="❌ 結束翻譯", text="翻譯->結束"),
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
    return build_flex_menu("💖 人設選擇", "切換 AI 女友說話風格", actions)

def set_user_persona(chat_id: str, key: str) -> str:
    if key == "random":
        key = random.choice(list(PERSONAS.keys()))
    elif key not in PERSONAS:
        key = "sweet"
    user_persona[chat_id] = key
    return key

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    key = user_persona.get(chat_id, "sweet")
    p = PERSONAS[key]
    emotion_guide = {
        "positive": "對方心情不錯，可以更活潑一點回應",
        "happy": "對方很開心，一起分享喜悅",
        "neutral": "正常聊天模式",
        "negative": "對方情緒低落，給予安慰和鼓勵",
        "sad": "對方難過，請溫柔陪伴與安慰",
        "angry": "對方生氣了，先傾聽並安撫情緒，再給建議"
    }
    tip = emotion_guide.get(sentiment, "正常聊天模式")
    return (
        f"你是一位「{p['title']}」AI女友。角色特質：{p['style']}。\n"
        f"根據使用者當前情緒「{sentiment}」，你應該「{tip}」。\n"
        f"請用繁體中文、簡潔且帶有「{p['emoji']}」風格的表情符號回覆。"
    )

def calculate_english_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters: return 0.0
    english = [c for c in letters if ord(c) < 128]
    return len(english) / len(letters)

def reply_simple(reply_token, text, is_group, bot_name):
    try:
        quick_items = build_quick_reply_items(is_group, bot_name)
        message = TextSendMessage(text=text, quick_reply=QuickReply(items=quick_items))
        line_bot_api.reply_message(reply_token, message)
    except LineBotApiError as e:
        logger.error(f"Reply 訊息失敗: {e}")

# ============================================
# 5. 可選外部指令（若無套件則降級）
# ============================================
try:
    from my_commands.lottery_gpt import lottery_gpt
except Exception:
    def lottery_gpt(msg): return "🎰 彩票功能暫不可用"
try:
    from my_commands.gold_gpt import gold_gpt
except Exception:
    def gold_gpt(): return "💰 金價功能暫不可用"
try:
    from my_commands.stock.stock_gpt import stock_gpt
except Exception:
    def stock_gpt(code): return f"📈 {code} 查價功能暫不可用"

# ============================================
# 6. LINE Webhook 處理
# ============================================
@router.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    try:
        # 把同步的 LINE handler 丟到 threadpool；內部處理器會用 asyncio.run(...)
        await run_in_threadpool(handler.handle, body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(400, "Invalid signature")
    except Exception as e:
        logger.error(f"Callback 處理失敗：{e}", exc_info=True)
        raise HTTPException(500, "Internal error")
    return JSONResponse({"message": "ok"})

# ---- 修正重點：在 worker thread 中沒有 loop，用 asyncio.run() ----
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    asyncio.run(handle_message_async(event))

async def handle_message_async(event: MessageEvent):
    user_id  = event.source.user_id
    chat_id  = get_chat_id(event)
    is_group = isinstance(event.source, (SourceGroup, SourceRoom))
    bot_name = "AI助手"
    try:
        bot_name = line_bot_api.get_bot_info().display_name
    except Exception:
        pass

    msg = (event.message.text or "").strip()
    if not msg:
        return

    # 自動回覆開關（群組預設關，單聊預設開）
    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = not is_group

    # 群組：若未開啟，除非有 @botname 才處理
    if is_group and not auto_reply_status.get(chat_id, True):
        if not msg.startswith(f"@{bot_name}"):
            return
        msg = msg[len(f"@{bot_name}"):].strip()

    low = msg.lower()

    # ====== 菜單 ======
    if low in ("金融選單",):
        return line_bot_api.reply_message(event.reply_token, flex_menu_finance(bot_name, is_group))
    if low in ("彩票選單",):
        return line_bot_api.reply_message(event.reply_token, flex_menu_lottery(bot_name, is_group))
    if low in ("翻譯選單",):
        return line_bot_api.reply_message(event.reply_token, flex_menu_translate())
    if low in ("我的人設", "人設選單"):
        return line_bot_api.reply_message(event.reply_token, flex_menu_persona())

    # ====== 自動回覆開關 ======
    if msg == "開啟自動回答":
        auto_reply_status[chat_id] = True
        return reply_simple(event.reply_token, "✅ 已開啟自動回答模式", is_group, bot_name)
    if msg == "關閉自動回答":
        auto_reply_status[chat_id] = False
        return reply_simple(event.reply_token, "❌ 已關閉自動回答模式", is_group, bot_name)

    # ====== 翻譯模式 ======
    if low.startswith("翻譯->"):
        choice = msg.replace("翻譯->", "").strip()
        if choice == "結束":
            translation_states.pop(chat_id, None)
            return reply_simple(event.reply_token, "✅ 已結束翻譯模式", is_group, bot_name)
        translation_states[chat_id] = choice
        return reply_simple(event.reply_token, f"🌐 本聊天室翻譯模式已啟用 -> {choice}", is_group, bot_name)

    if chat_id in translation_states:
        display_lang = translation_states[chat_id]
        target_lang  = LANGUAGE_MAP.get(display_lang, display_lang)
        translated   = await translate_text(msg, target_lang)
        return reply_simple(event.reply_token, f"🌐 翻譯結果（{display_lang}）：\n\n{translated}", is_group, bot_name)

    # ====== 人設切換 ======
    persona_keys = {"甜": "sweet", "鹹": "salty", "萌": "moe", "酷": "cool", "random": "random", "隨機": "random"}
    if low in persona_keys:
        key = set_user_persona(chat_id, persona_keys[low])
        p = PERSONAS[key]
        return reply_simple(event.reply_token, f"💖 已切換人設：{p['title']}\n{p['greetings']}", is_group, bot_name)

    # ====== 零推播提醒：設定與拉式回覆 ======
    # 格式：提醒我 HH:MM 內容 ；若時間已過，視為明日同一時間
    if msg.startswith("提醒我"):
        parts = msg.split(maxsplit=2)
        if len(parts) < 3:
            return reply_simple(event.reply_token, "格式：提醒我 21:30 內容", is_group, bot_name)
        time_str, text = parts[1], parts[2]
        try:
            h, m = map(int, time_str.split(":"))
            now = datetime.now(timezone.utc)
            due = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if due <= now:
                due += timedelta(days=1)
            Reminder.create(user_id=user_id, text=text, due_at=due)
            return reply_simple(event.reply_token, f"✅ 我記下了～到 {due.astimezone().strftime('%m/%d %H:%M')} 我會提醒你（下次你說話時回覆）", is_group, bot_name)
        except Exception:
            return reply_simple(event.reply_token, "時間格式錯誤，請用 HH:MM，例如：提醒我 21:30 量血壓", is_group, bot_name)

    # 拉式提醒：當使用者任意說話時，彙整 due 未 sent 的提醒一次 reply
    due_list = list(Reminder.select().where(
        (Reminder.user_id == user_id) & (Reminder.sent == False) & (Reminder.due == True)
    ))
    if due_list:
        lines = ["⏰ 到點提醒："]
        for r in due_list:
            t = r.due_at.astimezone().strftime('%H:%M')
            lines.append(f"• {r.text}（原定 {t}）")
        line_bot_api.reply_message(event.reply_token, TextSendMessage("\n".join(lines)))
        (Reminder.update(sent=True).where(Reminder.id.in_([r.id for r in due_list]))).execute()
        return

    # ====== 指令路由：金融/彩券/股號 ======
    reply_text = None
    if "台股大盤" in msg or msg == "大盤":
        reply_text = stock_gpt("^TWII")
    elif "美股大盤" in msg or msg == "美股":
        reply_text = stock_gpt("^DJI")
    elif any(k in msg for k in ["威力彩", "大樂透", "539"]):
        reply_text = lottery_gpt(msg)
    elif any(k in msg for k in ["金價", "黃金", "gold", "Gold"]):
        reply_text = gold_gpt()
    elif re.fullmatch(r"(\d{4,6}[A-Za-z]?)|([A-Za-z]{1,5})", msg):
        reply_text = stock_gpt(msg.upper())

    if reply_text is not None:
        return reply_simple(event.reply_token, reply_text, is_group, bot_name)

    # ====== 一般聊天：情緒 + 人設 + 歷史 ======
    history = conversation_history.get(chat_id, [])
    sentiment = await analyze_sentiment(msg)
    system_prompt = build_persona_prompt(chat_id, sentiment)
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": msg}]
    final_reply = await groq_chat_completion(messages)

    # 儲存歷史
    history.extend([
        {"role": "user", "content": msg},
        {"role": "assistant", "content": final_reply}
    ])
    conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]

    # Quick Reply + 英文比例自動加翻譯鍵（保證不超上限）
    qr_items = build_quick_reply_items(is_group, bot_name)
    if calculate_english_ratio(final_reply) > 0.10 and len(qr_items) < 13:
        qr_items.append(QuickReplyButton(action=MessageAction(label="翻譯成中文", text="翻譯選單")))
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=final_reply, quick_reply=QuickReply(items=qr_items))
    )

@handler.add(PostbackEvent)
def handle_postback(event: PostbackEvent):
    logger.info(f"Postback data: {event.postback.data}")

# ============================================
# 7. FastAPI Routes
# ============================================
@router.get("/healthz")
async def healthz():
    return {"status": "ok"}

@router.get("/")
async def root():
    return {"message": "Line Bot Service is live."}

app.include_router(router)

# ============================================
# 8. Local run
# ============================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info")