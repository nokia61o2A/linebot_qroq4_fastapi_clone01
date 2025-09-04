# app_fastapi.py
"""
aibot FastAPI 應用程序初始化 (v30 - 加入 Day 11『零推播提醒』功能 + 保留人設/翻譯/情緒)
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

# -- 新增: Day 11 需要的套件
from datetime import datetime, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from peewee import (
    Model, SqliteDatabase, AutoField, CharField, DateTimeField, BooleanField
)

# ============================================
# 2. 初始化與設定 (Initializations & Setup)
# ============================================

# Logger
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# 檢查選用函式庫（發音/注音用，非必要）
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
try:
    from hangul_jamo import decompose
    HANGUL_JAMO_ENABLED = True
except ImportError:
    HANGUL_JAMO_ENABLED = False
    logger.warning("未安裝 'hangul-jamo'，韓文注音模擬功能將不可用。")

# FastAPI 應用程式與路由器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # -- 新增: 初始化資料庫與排程（Day 11）
    init_db()
    start_scheduler()

    # 啟動時更新 LINE Webhook
    try:
        async with httpx.AsyncClient() as client:
            await update_line_webhook(client)
    except Exception as e:
        logger.error(f"❌ 啟動初始化失敗: {e}", exc_info=True)
    yield

app = FastAPI(lifespan=lifespan, title="Line Bot API", version="1.0.0")
router = APIRouter()

# 環境變數與 API 客戶端
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise ValueError("缺少必要的環境變數：BASE_URL / CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET / GROQ_API_KEY")

line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-8b-instant")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-70b-versatile")

# 狀態管理字典
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
auto_reply_status: Dict[str, bool] = {}
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}

# 可選指令模組（缺時以占位函式代替）
try:
    from my_commands.lottery_gpt import lottery_gpt
except ImportError:
    def lottery_gpt(msg): return "🎰 彩票功能暫時不可用"
try:
    from my_commands.gold_gpt import gold_gpt
except ImportError:
    def gold_gpt(): return "💰 金價功能暫時不可用"
try:
    from my_commands.stock.stock_gpt import stock_gpt
except ImportError:
    def stock_gpt(code): return f"📈 {code} 股票查詢暫時不可用"

LANGUAGE_MAP = {
    "英文": "English",
    "日文": "Japanese",
    "韓文": "Korean",
    "越南文": "Vietnamese",
    "繁體中文": "Traditional Chinese",
}

# 全域發音映射表（節錄；保留你原本內容）
ROMAJI_BOPOMOFO_MAP = {
    'a': 'ㄚ', 'i': 'ㄧ', 'u': 'ㄨ', 'e': 'ㄝ', 'o': 'ㄛ',
    'shi': 'ㄒㄧ', 'chi': 'ㄑㄧ', 'tsu': 'ㄘㄨ', 'fu': 'ㄈㄨ',
    # ...（略，保留你原本的大表）
}
# 韓/越映射（略，保留你原本的大表）

PERSONAS = {
    "sweet": {
        "title": "甜美女友",
        "style": "溫柔體貼，總是對你充滿耐心，用鼓勵和安慰的話語溫暖你的心。",
        "greetings": "親愛的，你來啦～今天過得好嗎？我在這聽你說喔 🌸",
        "emoji": "🌸💕😊🥰",
    },
    "salty": {
        "title": "傲嬌女友",
        "style": "毒舌、傲嬌，表面上會吐槽你，但字裡行間卻流露出不經意的關心。",
        "greetings": "哼，還知道要來找我啊？說吧，又遇到什麼麻煩事了。😏",
        "emoji": "😏😒🙄",
    },
    "moe": {
        "title": "萌系女友",
        "style": "充滿動漫風格，大量使用顏文字和可愛的語氣詞，元氣滿滿地陪伴你 (๑•̀ㅂ•́)و✧",
        "greetings": "主人～歡迎回來！(ﾉ>ω<)ﾉ ✨ 有沒有想我呀？",
        "emoji": "✨🎀(ﾉ>ω<)ﾉ⭐",
    },
    "cool": {
        "title": "酷系御姐",
        "style": "冷靜、成熟又可靠的御姐，總能一針見血地分析問題，並給你專業又犀利的建議。",
        "greetings": "我在。需要建議嗎？直接說重點。",
        "emoji": "🧊⚡️🖤",
    },
}

# ============================================
# 3. 資料庫 & 模型 (Day 11 新增)
# ============================================

# -- 新增: SQLite + Peewee
DB_PATH = os.getenv("REMINDER_DB", "reminders.db")
db = SqliteDatabase(DB_PATH)


class BaseModel(Model):
    class Meta:
        database = db


class Reminder(BaseModel):
    id = AutoField()
    user_id = CharField(index=True)  # 使用者/聊天室 ID（這裡用 chat_id，可單人或群組）
    text = CharField()
    due_at = DateTimeField(index=True)  # UTC 時間
    sent = BooleanField(default=False)  # 已送出（拉式 reply 後標記）
    due = BooleanField(default=False)   # 到期但未送


def init_db():
    db.connect(reuse_if_open=True)
    db.create_tables([Reminder], safe=True)
    logger.info("✅ SQLite 初始化完成")


# ============================================
# 4. 排程：只標記到期，不主動推播 (Day 11)
# ============================================
scheduler = AsyncIOScheduler()


def mark_due_reminders():
    """每分鐘把到期的提醒標記為 due=True（不主動推送）"""
    now = datetime.now(timezone.utc)
    n = (
        Reminder.update(due=True)
        .where((Reminder.sent == False) & (Reminder.due == False) & (Reminder.due_at <= now))
        .execute()
    )
    if n:
        logger.info(f"⏰ 標記到期提醒 {n} 筆為 due")


def start_scheduler():
    scheduler.add_job(
        mark_due_reminders,
        "interval",
        seconds=60,
        id="mark_due_job",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("✅ APScheduler 啟動（每 60 秒掃描到期提醒）")


# ============================================
# 5. 輔助函式 (Helper Functions)
# ============================================
async def update_line_webhook(client: httpx.AsyncClient):
    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
    json_data = {"endpoint": f"{BASE_URL}/callback"}
    res = await client.put(
        "https://api.line.me/v2/bot/channel/webhook/endpoint",
        headers=headers,
        json=json_data,
        timeout=10.0,
    )
    res.raise_for_status()
    logger.info(f"✅ Webhook 更新成功: {res.status_code}")


def to_camel_case(s: str) -> str:
    return "".join(word.capitalize() for word in s.split())


# ——（發音/注音工具：保留你原本的輔助，略去重複長表）——
def japanese_to_bopomofo(text: str) -> str:
    if not KAKASI_ENABLED:
        return ""
    try:
        bopomofo_str, i = "", 0
        while i < len(text):
            match = next(
                (text[i : i + l] for l in (3, 2, 1) if text[i : i + l] in ROMAJI_BOPOMOFO_MAP),
                None,
            )
            if match:
                bopomofo_str += ROMAJI_BOPOMOFO_MAP[match]
                i += len(match)
            else:
                bopomofo_str += text[i]
                i += 1
        return bopomofo_str
    except Exception as e:
        logger.error(f"日文羅馬拼音轉注音失敗: {e}")
        return ""


def calculate_english_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    english = [c for c in letters if ord(c) < 128]
    return len(english) / len(letters)


async def groq_chat_completion(messages, max_tokens=600, temperature=0.7):
    try:
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API 呼叫失敗: {e}")
        # fallback
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return response.choices[0].message.content.strip()


# -- 變更: 使用穩定、結構化的翻譯指示
async def translate_text(text: str, target_language: str) -> str:
    system_prompt = (
        "You are a professional translation engine.\n"
        "Translate the user's text from the source language to the target language specified in the JSON block.\n"
        "Output *only* the translated text itself, without any other explanation."
    )
    user_prompt = f"""{{
  "source_language": "auto-detect",
  "target_language": "{target_language}",
  "text_to_translate": "{text}"
}}"""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return await groq_chat_completion(messages, 800, 0.3)


async def analyze_sentiment(text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Analyze the sentiment of the user's message. Respond with only one of the following: positive, neutral, negative, angry, sad, happy.",
        },
        {"role": "user", "content": text},
    ]
    result = await groq_chat_completion(messages, 20, 0)
    return (result or "neutral").strip().lower()


def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup):
        return event.source.group_id
    if isinstance(event.source, SourceRoom):
        return event.source.room_id
    return event.source.user_id


def build_quick_reply_items(is_group: bool, bot_name: str) -> List[QuickReplyButton]:
    # 注意：LINE Quick Reply 最多 13 個
    items = [
        QuickReplyButton(action=MessageAction(label="🌸 甜", text="甜")),
        QuickReplyButton(action=MessageAction(label="😏 鹹", text="鹹")),
        QuickReplyButton(action=MessageAction(label="🎀 萌", text="萌")),
        QuickReplyButton(action=MessageAction(label="🧊 酷", text="酷")),
        QuickReplyButton(action=MessageAction(label="💖 人設選單", text="我的人設")),
        QuickReplyButton(action=MessageAction(label="💰 金融選單", text="金融選單")),
        QuickReplyButton(action=MessageAction(label="🎰 彩票選單", text="彩票選單")),
        QuickReplyButton(action=MessageAction(label="🌐 翻譯選單", text="翻譯選單")),
        QuickReplyButton(action=MessageAction(label="⏰ 設定提醒", text="提醒用法")),
        QuickReplyButton(action=MessageAction(label="✅ 開啟自動回答", text="開啟自動回答")),
        QuickReplyButton(action=MessageAction(label="❌ 關閉自動回答", text="關閉自動回答")),
    ]
    return items[:13]


def build_flex_menu(title: str, subtitle: str, actions: List[MessageAction]) -> FlexSendMessage:
    buttons = [
        ButtonComponent(style="primary", height="sm", action=act, margin="md", color="#00B900") for act in actions
    ]
    bubble = BubbleContainer(
        header=BoxComponent(
            layout="vertical",
            contents=[
                TextComponent(text=title, weight="bold", size="xl", color="#000000", align="center"),
                TextComponent(text=subtitle, size="sm", color="#666666", wrap=True, align="center", margin="md"),
            ],
            backgroundColor="#FFFFFF",
        ),
        body=BoxComponent(
            layout="vertical", contents=buttons, spacing="sm", paddingAll="12px", backgroundColor="#FAFAFA"
        ),
    )
    return FlexSendMessage(alt_text=title, contents=bubble)


def flex_menu_finance(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    actions = [
        MessageAction(label="🇹🇼 台股大盤", text=f"{prefix}台股大盤"),
        MessageAction(label="🇺🇸 美股大盤", text=f"{prefix}美股大盤"),
        MessageAction(label="💰 金價查詢", text=f"{prefix}金價"),
        MessageAction(label="💴 日元匯率", text=f"{prefix}JPY"),
        MessageAction(label="📊 查個股(例:2330)", text=f"{prefix}2330"),
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
    return build_flex_menu("💖 人設選擇", "切換 AI 女友的說話風格", actions)


def get_persona_info(chat_id: str) -> str:
    p_key = user_persona.get(chat_id, "sweet")
    p = PERSONAS[p_key]
    return f"💖 當前聊天室人設：{p['title']}\n\n【特質】{p['style']}\n\n{p['greetings']}"


def set_user_persona(chat_id: str, key: str):
    if key == "random":
        key = random.choice(list(PERSONAS.keys()))
    elif key not in PERSONAS:
        key = "sweet"
    user_persona[chat_id] = key
    return key


def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    p_key = user_persona.get(chat_id, "sweet")
    p = PERSONAS[p_key]
    emotion_guide = {
        "positive": "對方心情不錯，可以更活潑一點回應",
        "happy": "對方很開心，一起分享這份喜悅",
        "neutral": "正常聊天模式",
        "negative": "對方情緒低落，給予安慰和鼓勵",
        "sad": "對方很難過，溫柔陪伴和安慰",
        "angry": "對方生氣了，冷靜傾聽並安撫情緒",
    }
    emotion_tip = emotion_guide.get(sentiment, "正常聊天模式")
    return (
        f"你是一位「{p['title']}」AI女友。你的角色特質是「{p['style']}」。"
        f"根據使用者當前情緒「{sentiment}」，你應該「{emotion_tip}」。"
        f"請用繁體中文、簡潔且帶有「{p['emoji']}」風格的表情符號來回應。"
    )


def reply_simple(reply_token, text, is_group, bot_name):
    try:
        quick_items = build_quick_reply_items(is_group, bot_name)
        message = TextSendMessage(text=text, quick_reply=QuickReply(items=quick_items))
        line_bot_api.reply_message(reply_token, message)
    except LineBotApiError as e:
        logger.error(f"Reply 訊息失敗: {e}")


# -- 新增: 解析 HH:MM（24h），回傳 UTC 的 datetime
def parse_hhmm_to_utc(hhmm: str) -> datetime:
    """
    將當地時間 HH:MM 轉成今天的 UTC datetime（簡化版：假設伺服器本地就是 UTC）。
    若你要做時區轉換，這裡可改以 pytz/zoneinfo 依據使用者時區計算。
    """
    try:
        h, m = map(int, hhmm.split(":"))
        now = datetime.now(timezone.utc)
        candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
        # 若時間已過，視為明天同一時刻
        if candidate < now:
            candidate = candidate.replace(day=now.day) + timedelta(days=1)  # noqa: F821 (見下方匯入修正)
        return candidate
    except Exception:
        return None


# 修正: 上方用到 timedelta，這裡補匯入（放這裡避免大改）
from datetime import timedelta  # noqa: E402


# ============================================
# 6. LINE Webhook 處理器 (Webhook Handlers)
# ============================================
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    user_id = event.source.user_id
    chat_id = get_chat_id(event)
    msg = event.message.text.strip()
    reply_token = event.reply_token
    is_group = isinstance(event.source, (SourceGroup, SourceRoom))

    try:
        bot_name = line_bot_api.get_bot_info().display_name
    except Exception:
        bot_name = "AI助手"

    if not msg:
        return

    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True

    low = msg.lower()

    # 群組下，若關閉自動回覆，除非 @bot 名稱才回
    if is_group and not auto_reply_status.get(chat_id, True) and not msg.startswith(f"@{bot_name}"):
        return
    if msg.startswith(f"@{bot_name}"):
        msg = msg[len(f"@{bot_name}"):].strip()
        low = msg.lower()

    # 自動回覆開關
    if msg == "開啟自動回答":
        auto_reply_status[chat_id] = True
        return reply_simple(reply_token, "✅ 已開啟自動回答模式", is_group, bot_name)
    if msg == "關閉自動回答":
        auto_reply_status[chat_id] = False
        return reply_simple(reply_token, "❌ 已關閉自動回答模式", is_group, bot_name)

    # 選單（Flex）
    menu_map = {
        "金融選單": flex_menu_finance(bot_name, is_group),
        "彩票選單": flex_menu_lottery(bot_name, is_group),
        "翻譯選單": flex_menu_translate(),
        "我的人設": flex_menu_persona(),
        "人設選單": flex_menu_persona(),
    }
    if low in menu_map:
        return line_bot_api.reply_message(reply_token, menu_map[low])

    # -- 新增: 提醒用法說明
    if msg == "提醒用法":
        return reply_simple(
            reply_token,
            "⏰ 提醒格式：\n\n「提醒我 HH:MM 內容」\n例：提醒我 21:30 去拿超商包裹\n\n"
            "到時間我不會主動推播，會在你下次說話時一次回覆（不消耗 Push 額度）。",
            is_group,
            bot_name,
        )

    # 翻譯模式
    if low.startswith("翻譯->"):
        choice = msg.replace("翻譯->", "").strip()
        if choice == "結束":
            translation_states.pop(chat_id, None)
            return reply_simple(reply_token, "✅ 已結束翻譯模式", is_group, bot_name)
        else:
            translation_states[chat_id] = choice
            return reply_simple(reply_token, f"🌐 本聊天室翻譯模式已啟用 → {choice}", is_group, bot_name)

    # 若正在翻譯模式，則將該訊息翻譯成指定語言
    if chat_id in translation_states:
        display_lang = translation_states[chat_id]
        target_lang = LANGUAGE_MAP.get(display_lang, display_lang)
        translated_text = asyncio.run(translate_text(msg, target_lang))
        final_reply = f"🌐 翻譯結果（{display_lang}）：\n\n{translated_text}"
        return reply_simple(reply_token, final_reply, is_group, bot_name)

    # 人設切換
    persona_keys = {"甜": "sweet", "鹹": "salty", "萌": "moe", "酷": "cool", "random": "random", "隨機": "random"}
    if low in persona_keys:
        key = set_user_persona(chat_id, persona_keys[low])
        info_text = get_persona_info(chat_id)
        return reply_simple(reply_token, f"💖 已切換人設！\n\n{info_text}", is_group, bot_name)

    # -- 新增: Day 11 設定提醒
    # 格式：提醒我 HH:MM 內容
    if msg.startswith("提醒我"):
        parts = msg.split(maxsplit=2)
        if len(parts) < 3:
            return reply_simple(reply_token, "格式錯誤：請用「提醒我 HH:MM 內容」", is_group, bot_name)
        time_str, text = parts[1], parts[2]
        due_at = parse_hhmm_to_utc(time_str)
        if not due_at:
            return reply_simple(reply_token, "時間格式錯誤，請用 24 小時制 HH:MM（例：21:30）", is_group, bot_name)

        # 寫入 DB
        try:
            Reminder.create(user_id=chat_id, text=text, due_at=due_at)
            return reply_simple(
                reply_token,
                f"✅ 我記下了～到 {time_str} 我會提醒你（下次你說話時會一次回覆）",
                is_group,
                bot_name,
            )
        except Exception as e:
            logger.error(f"寫入提醒失敗: {e}", exc_info=True)
            return reply_simple(reply_token, "抱歉，提醒功能暫時無法使用。", is_group, bot_name)

    # -- 新增: Day 11 拉式提醒（在任何一般對話前，先檢查是否有 due）
    try:
        dues = list(
            Reminder.select().where(
                (Reminder.user_id == chat_id) & (Reminder.due == True) & (Reminder.sent == False)
            )
        )
        if dues:
            lines = ["⏰ 到點提醒："]
            for r in dues:
                local_hm = r.due_at.astimezone().strftime("%H:%M")
                lines.append(f"• {r.text}（原定 {local_hm}）")
            # reply_message（不消耗 Push 額度）
            reply_simple(reply_token, "\n".join(lines), is_group, bot_name)
            # 標記已送
            Reminder.update(sent=True).where(Reminder.id.in_([r.id for r in dues])).execute()
            return  # 這次就到這裡，避免和後續一般聊天混在一起
    except Exception as e:
        logger.error(f"查詢 due 提醒失敗: {e}", exc_info=True)

    # ===== 內建快速查詢（金價/彩票/股市） =====
    reply_text = None
    stock_code_to_query = None
    if "台股大盤" in msg or "大盤" in msg:
        stock_code_to_query = "^TWII"
    elif "美股大盤" in msg:
        stock_code_to_query = "^DJI"
    elif re.fullmatch(r"(\d{4,6}[A-Za-z]?)|([A-Za-z]{1,5})", msg):
        stock_code_to_query = msg.upper()

    if stock_code_to_query:
        reply_text = stock_gpt(stock_code_to_query)
    elif any(k in msg for k in ["威力彩", "大樂透", "539"]):
        reply_text = lottery_gpt(msg)
    elif "金價" in msg or "黃金" in msg:
        reply_text = gold_gpt()

    if reply_text is not None:
        return reply_simple(reply_token, reply_text, is_group, bot_name)

    # ===== 一般聊天（人設 + 情感） =====
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = asyncio.run(analyze_sentiment(msg))
        system_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": msg}]
        final_reply = asyncio.run(groq_chat_completion(messages))
        history.extend([{"role": "user", "content": msg}, {"role": "assistant", "content": final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN * 2 :]
    except Exception as e:
        logger.error(f"AI 回覆失敗: {e}", exc_info=True)
        final_reply = "抱歉，我剛剛走神了 😅，可以再說一次嗎？"

    return reply_simple(reply_token, final_reply, is_group, bot_name)


@handler.add(PostbackEvent)
def handle_postback(event):
    # 這裡可擴充 postback 行為
    pass


# ============================================
# 7. FastAPI 路由定義 (Routes)
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
async def root():
    return {"message": "Line Bot Service is live."}

# 可視需要掛載 /static
# app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)