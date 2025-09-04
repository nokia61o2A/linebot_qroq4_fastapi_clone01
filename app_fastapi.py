"""
aibot FastAPI 應用程序 (群組穩定 Quick Reply + 零推播提醒 + 人設 + 情緒分析)
- 修正：
  1) 統一回覆出口（文字/Flex 都掛 Quick Reply，且 <= 13 顆）
  2) 移除 asyncio 依賴，避免 no running event loop
  3) APScheduler 改 BackgroundScheduler，同步穩定
"""

# =========================
# 1) Imports
# =========================
import os
import re
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import httpx
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    QuickReply, QuickReplyButton, MessageAction,
    SourceGroup, SourceRoom, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent, TextComponent,
    ButtonComponent
)

# Groq（同步）
from groq import Groq

# Peewee / APScheduler（同步）
from peewee import *
from apscheduler.schedulers.background import BackgroundScheduler

# =========================
# 2) 基本設定
# =========================
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

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL_PRIMARY  = os.getenv("GROQ_MODEL_PRIMARY",  "llama-3.1-8b-instant")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-70b-versatile")

# 對話與狀態
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
auto_reply_status: Dict[str, bool] = {}
user_persona: Dict[str, str] = {}

# =========================
# 3) 人設與工具
# =========================
PERSONAS: Dict[str, dict] = {
    "sweet": {
        "title": "甜美女友",
        "style": "溫柔體貼，鼓勵安慰；口語自然，適度使用 🌸💕。",
        "greetings": "親愛的，你來啦～我在這裡，跟我說說吧。🌸",
        "emoji": "🌸💕😊"
    },
    "salty": {
        "title": "傲嬌女友",
        "style": "機智吐槽、有點壞壞但不傷人；先鬧你再關心你。",
        "greetings": "哼，還知道來找我？說吧，哪裡需要我救場。😏",
        "emoji": "😏🙄"
    },
    "moe": {
        "title": "萌系女友",
        "style": "動漫語感 + 顏文字 (ﾉ>ω<)ﾉ，內容仍要有重點。",
        "greetings": "呀呼～我來為你打氣啦！(๑•̀ㅂ•́)و✧",
        "emoji": "✨🎀(ﾉ>ω<)ﾉ"
    },
    "cool": {
        "title": "酷系御姐",
        "style": "冷靜精煉，關鍵建議一針見血；少量表情但有溫度。",
        "greetings": "我在。把狀況說清楚，我幫你理出解法。",
        "emoji": "🧊⚡️"
    },
}

def set_user_persona(chat_id: str, key: str) -> str:
    key = key.lower()
    if key == "random":
        key = random.choice(list(PERSONAS.keys()))
    if key not in PERSONAS:
        key = "sweet"
    user_persona[chat_id] = key
    return key

def get_user_persona(chat_id: str) -> str:
    return user_persona.get(chat_id, "sweet")

def build_persona_system(chat_id: str, sentiment: str) -> str:
    p = PERSONAS[get_user_persona(chat_id)]
    emotion_tip = {
        "positive":"一起開心，但自然不浮誇",
        "happy":"分享喜悅、保持活力",
        "neutral":"自然輕鬆地對談",
        "negative":"先共情安慰，給具體陪伴與小步建議",
        "sad":"溫柔陪伴、多肯定；提供可行的小事",
        "angry":"降溫、傾聽，再協助拆解問題"
    }.get(sentiment, "自然輕鬆地對談")

    return f"""
你是一位「{p['title']}」AI 女友。
【風格】{p['style']}
【情緒調節】使用者情緒：{sentiment} → {emotion_tip}
請用繁體中文、精煉友善、加入少量表情（{p['emoji']}），回答 2~6 句。
""".strip()

def groq_chat(messages: List[dict], max_tokens=600, temperature=0.7) -> str:
    try:
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"[Groq primary] {e}")
        r = groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (r.choices[0].message.content or "").strip()

def analyze_sentiment(text: str) -> str:
    messages = [
        {"role":"system","content":"請只輸出一個標籤：positive, neutral, negative, angry, happy, sad"},
        {"role":"user","content":text}
    ]
    out = (groq_chat(messages, max_tokens=8, temperature=0) or "neutral").lower()
    out = re.sub(r"[^a-z]", "", out)
    return out if out in {"positive","neutral","negative","angry","happy","sad"} else "neutral"

# =========================
# 4) Quick Reply：統一出口（<= 13 顆）
# =========================
MAX_QR = 13

def build_quick_reply_items(is_group: bool, bot_name: str) -> List[QuickReplyButton]:
    prefix = f"@{bot_name} " if is_group else ""
    items: List[QuickReplyButton] = [
        QuickReplyButton(action=MessageAction(label="🌸 甜", text="甜")),
        QuickReplyButton(action=MessageAction(label="😏 鹹", text="鹹")),
        QuickReplyButton(action=MessageAction(label="🎀 萌", text="萌")),
        QuickReplyButton(action=MessageAction(label="🧊 酷", text="酷")),
        QuickReplyButton(action=MessageAction(label="💖 人設選單", text="人設選單")),
        QuickReplyButton(action=MessageAction(label="💰 金融選單", text="金融選單")),
        QuickReplyButton(action=MessageAction(label="🎰 彩票選單", text="彩票選單")),
        QuickReplyButton(action=MessageAction(label="✅ 開啟自動回答", text="開啟自動回答")),
        QuickReplyButton(action=MessageAction(label="❌ 關閉自動回答", text="關閉自動回答")),
        QuickReplyButton(action=MessageAction(label="🌤️ 天氣", text=f"{prefix}天氣")),
        QuickReplyButton(action=MessageAction(label="📈 台股大盤", text=f"{prefix}台股大盤")),
    ]
    return items[:MAX_QR]

def _attach_qr(message, is_group: bool, bot_name: str):
    message.quick_reply = QuickReply(items=build_quick_reply_items(is_group, bot_name))
    return message

def reply_text_with_qr(reply_token: str, text: str, is_group: bool, bot_name: str):
    try:
        line_bot_api.reply_message(reply_token, _attach_qr(TextSendMessage(text=text), is_group, bot_name))
    except LineBotApiError as e:
        logger.error(f"Reply Text 失敗: {e.error.message}")

def reply_flex_with_qr(reply_token: str, flex: FlexSendMessage, is_group: bool, bot_name: str):
    try:
        line_bot_api.reply_message(reply_token, _attach_qr(flex, is_group, bot_name))
    except LineBotApiError as e:
        logger.error(f"Reply Flex 失敗: {e.error.message}")

# =========================
# 5) Flex 選單
# =========================
def build_flex_menu(title: str, subtitle: str, actions: List[MessageAction]) -> FlexSendMessage:
    buttons = [ButtonComponent(style="primary", height="sm", action=a, margin="md", color="#00B900") for a in actions]
    bubble = BubbleContainer(
        header=BoxComponent(layout="vertical", contents=[
            TextComponent(text=title, weight="bold", size="xl", color="#000000", align="center"),
            TextComponent(text=subtitle, size="sm", color="#666666", wrap=True, align="center", margin="md"),
        ], backgroundColor="#FFFFFF"),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm", paddingAll="12px", backgroundColor="#FAFAFA"),
    )
    return FlexSendMessage(alt_text=title, contents=bubble)

def flex_menu_persona() -> FlexSendMessage:
    acts = [
        MessageAction(label="🌸 甜美女友", text="甜"),
        MessageAction(label="😏 傲嬌女友", text="鹹"),
        MessageAction(label="🎀 萌系女友", text="萌"),
        MessageAction(label="🧊 酷系御姐", text="酷"),
        MessageAction(label="🎲 隨機人設", text="random"),
    ]
    return build_flex_menu("💖 人設選擇", "切換 AI 女友的說話風格", acts)

def flex_menu_finance(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    acts = [
        MessageAction(label="🇹🇼 台股大盤", text=f"{prefix}台股大盤"),
        MessageAction(label="🇺🇸 美股大盤", text=f"{prefix}美股大盤"),
        MessageAction(label="💰 金價查詢", text=f"{prefix}金價"),
        MessageAction(label="📊 查個股(例:2330)", text=f"{prefix}2330"),
    ]
    return build_flex_menu("💰 金融服務", "快速查詢金融資訊", acts)

def flex_menu_lottery(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    acts = [
        MessageAction(label="🎰 大樂透", text=f"{prefix}大樂透"),
        MessageAction(label="🎯 威力彩", text=f"{prefix}威力彩"),
        MessageAction(label="🔢 539",   text=f"{prefix}539"),
    ]
    return build_flex_menu("🎰 彩券服務", "最新開獎資訊", acts)

# =========================
# 6) 零推播提醒（SQLite + APScheduler）
# =========================
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
    due     = BooleanField(default=False)  # 到期但尚未回覆（pull）

def init_db():
    db.connect(reuse_if_open=True)
    db.create_tables([Reminder], safe=True)
    logger.info("✅ SQLite/peewee 初始化完成")

scheduler = BackgroundScheduler()

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
    scheduler.add_job(mark_due_reminders, "interval", seconds=60, id="mark_due", replace_existing=True)
    scheduler.start()
    logger.info("✅ APScheduler 啟動，60 秒掃描到期提醒")

def parse_time_hhmm(s: str) -> datetime:
    """把 HH:MM 轉成今天的 UTC 時間（若已過，順延到明天）"""
    h, m = map(int, s.split(":"))
    now_local = datetime.now()
    due_local = now_local.replace(hour=h, minute=m, second=0, microsecond=0)
    if due_local < now_local:
        due_local += timedelta(days=1)
    return due_local.astimezone(timezone.utc)

# =========================
# 7) FastAPI App
# =========================
app = FastAPI(title="Line Bot API")
router = APIRouter()

@app.on_event("startup")
def on_startup():
    # 更新 Webhook
    try:
        headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
        json_data = {"endpoint": f"{BASE_URL}/callback"}
        with httpx.Client(timeout=10.0) as c:
            r = c.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=json_data)
            r.raise_for_status()
            logger.info(f"✅ Webhook 更新成功: {r.status_code}")
    except Exception as e:
        logger.error(f"❌ Webhook 更新失敗: {e}", exc_info=True)

    # DB + Scheduler
    init_db()
    start_scheduler()

app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# 8) 指令與路由（你可接回自己的 my_commands）
# =========================
def stock_gpt(code: str) -> str:
    return f"（示意）查詢 {code} 的行情與走勢…"

def lottery_gpt(msg: str) -> str:
    return "（示意）最新開獎/機率/冷熱號…"

def gold_gpt() -> str:
    return "（示意）最新國際金價走勢…"

def weather_gpt(city: str) -> str:
    return f"（示意）{city} 天氣：晴時多雲，降雨 10%…"

# =========================
# 9) LINE Handlers（同步）
# =========================
def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup):
        return event.source.group_id
    if isinstance(event.source, SourceRoom):
        return event.source.room_id
    return event.source.user_id

@router.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(400, "Invalid signature")
    except Exception as e:
        logger.error(f"Callback 處理失敗：{e}", exc_info=True)
        raise HTTPException(500, "Internal error")
    return JSONResponse({"message":"ok"})

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    user_id = event.source.user_id
    chat_id = get_chat_id(event)
    msg     = (event.message.text or "").strip()
    reply_token = event.reply_token
    is_group = isinstance(event.source, (SourceGroup, SourceRoom))

    try:
        bot_name = line_bot_api.get_bot_info().display_name
    except:
        bot_name = "AI助手"

    if not msg:
        return

    # 初始化群組自動回覆狀態（預設：單聊 True / 群組 False）
    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = not is_group

    # 未開啟時：僅在提到 @bot 時開啟並提示
    if is_group and not auto_reply_status[chat_id]:
        if f"@{bot_name}" in msg:
            auto_reply_status[chat_id] = True
            reply_text_with_qr(reply_token, "✅ 已開啟本群的自動回覆。\n可使用：人設選單 / 金融選單 / 彩票選單。", is_group, bot_name)
        return

    low = msg.lower()

    # ---- 系統控制 ----
    if msg == "開啟自動回答":
        auto_reply_status[chat_id] = True
        return reply_text_with_qr(reply_token, "✅ 已開啟自動回答", is_group, bot_name)
    if msg == "關閉自動回答":
        auto_reply_status[chat_id] = False
        return reply_text_with_qr(reply_token, "❌ 已關閉自動回答（在群組可 @我 重新開啟）", is_group, bot_name)

    # ---- Flex 選單 ----
    menu_map = {
        "人設選單":  flex_menu_persona(),
        "我的人設":  flex_menu_persona(),
        "金融選單":  flex_menu_finance(bot_name, is_group),
        "彩票選單":  flex_menu_lottery(bot_name, is_group),
    }
    if msg in menu_map:
        return reply_flex_with_qr(reply_token, menu_map[msg], is_group, bot_name)

    # ---- 人設切換 ----
    persona_keys = {"甜":"sweet","鹹":"salty","萌":"moe","酷":"cool","random":"random","隨機":"random"}
    if msg in persona_keys:
        key = set_user_persona(chat_id, persona_keys[msg])
        p   = PERSONAS[key]
        return reply_text_with_qr(reply_token, f"已切換人設：{p['title']}  ✅\n{p['greetings']}", is_group, bot_name)

    # ---- 設定提醒：提醒我 HH:MM 內容 ----
    if msg.startswith("提醒我"):
        parts = msg.split(maxsplit=2)
        if len(parts) < 3 or not re.fullmatch(r"\d{1,2}:\d{2}", parts[1]):
            return reply_text_with_qr(reply_token, "格式：提醒我 21:30 測血壓", is_group, bot_name)
        time_str, text = parts[1], parts[2]
        due_at_utc = parse_time_hhmm(time_str)
        Reminder.create(user_id=user_id, text=text, due_at=due_at_utc)
        return reply_text_with_qr(reply_token, f"✅ 我記下了～到 {time_str} 我會提醒你（下次你說話時回覆）", is_group, bot_name)

    # ---- 拉式提醒：有 due 就彙整回覆 ----
    due_list = list(Reminder.select().where(
        (Reminder.user_id == user_id) & (Reminder.sent == False) & (Reminder.due == True)
    ))
    if due_list:
        lines = ["⏰ 到點提醒："]
        for r in due_list:
            hhmm = r.due_at.astimezone().strftime("%H:%M")
            lines.append(f"• {r.text}（原定 {hhmm}）")
        (Reminder.update(sent=True).where(Reminder.id.in_([r.id for r in due_list]))).execute()
        return reply_text_with_qr(reply_token, "\n".join(lines), is_group, bot_name)

    # ---- 內建指令（示意）----
    if "天氣" in msg:
        return reply_text_with_qr(reply_token, weather_gpt("桃園市"), is_group, bot_name)
    if "金價" in msg or "黃金" in msg:
        return reply_text_with_qr(reply_token, gold_gpt(), is_group, bot_name)
    if any(k in msg for k in ["大樂透","威力彩","539"]):
        return reply_text_with_qr(reply_token, lottery_gpt(msg), is_group, bot_name)

    m_code = re.fullmatch(r"(\d{4,6}[A-Za-z]?)|([A-Za-z]{1,5})", msg)
    if msg in ("台股大盤","大盤"):
        return reply_text_with_qr(reply_token, stock_gpt("^TWII"), is_group, bot_name)
    if msg in ("美股大盤","美盤"):
        return reply_text_with_qr(reply_token, stock_gpt("^DJI"), is_group, bot_name)
    if m_code:
        return reply_text_with_qr(reply_token, stock_gpt(m_code.group()), is_group, bot_name)

    # ---- 一般對話：人設 + 情緒 ----
    history = conversation_history.get(chat_id, [])
    sentiment = analyze_sentiment(msg)
    sys = build_persona_system(chat_id, sentiment)
    messages = [{"role":"system","content":sys}] + history + [{"role":"user","content":msg}]
    reply = groq_chat(messages)

    history.extend([{"role":"user","content":msg},{"role":"assistant","content":reply}])
    conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]

    return reply_text_with_qr(reply_token, reply, is_group, bot_name)

@handler.add(PostbackEvent)
def handle_postback(event: PostbackEvent):
    logger.info(f"Postback: {event.postback.data}")

# =========================
# 10) 其他路由
# =========================
@router.get("/healthz")
def healthz():
    return {"status":"ok"}

@router.get("/")
def root():
    return {"message":"Service is live."}

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info")