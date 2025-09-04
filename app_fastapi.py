"""
app_fastapi.py
- 群聊 Flex 選單 + Quick Reply 顯示修正版
- 人設 cosplay / 翻譯模式 / 情感分析 / 常用 quick buttons
"""

# ========== 1) Imports ==========
import os
import re
import random
import logging
import asyncio
from typing import Dict, List
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    QuickReply, QuickReplyButton, MessageAction,
    SourceGroup, SourceRoom, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent,
    TextComponent, ButtonComponent
)

from groq import AsyncGroq

# ========== 2) Setup ==========
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise RuntimeError("缺少必要環境變數（BASE_URL / CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET / GROQ_API_KEY）")

line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-8b-instant")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-70b-versatile")

# 對話/狀態
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
auto_reply_status: Dict[str, bool] = {}        # 每個 chat 的「是否自動回」
user_persona: Dict[str, str] = {}              # 每個 chat 的人設
translation_states: Dict[str, str] = {}        # 每個 chat 的翻譯目標語言

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰，不浮誇。", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji":"🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度。", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji":"😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字，仍要有重點。", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji":"✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議。", "greetings": "我在。說重點。", "emoji":"🧊⚡️"}
}

LANGUAGE_MAP = {
    "英文": "English", "日文": "Japanese", "韓文": "Korean",
    "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"
}

# ========== 3) FastAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with httpx.AsyncClient() as c:
            headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
            payload = {"endpoint": f"{BASE_URL}/callback"}
            r = await c.put("https://api.line.me/v2/bot/channel/webhook/endpoint",
                            headers=headers, json=payload, timeout=10.0)
            r.raise_for_status()
            logger.info(f"✅ Webhook 更新成功: {r.status_code}")
    except Exception as e:
        logger.error(f"Webhook 更新失敗: {e}", exc_info=True)
    yield

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.0.0")
router = APIRouter()

# ========== 4) Helpers ==========
def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom):  return event.source.room_id
    return event.source.user_id

async def groq_chat(messages, max_tokens=600, temperature=0.7):
    try:
        resp = await groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY, messages=messages,
            max_tokens=max_tokens, temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq 主要模型失敗: {e}")
        resp = await groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK, messages=messages,
            max_tokens=max_tokens, temperature=temperature
        )
        return resp.choices[0].message.content.strip()

async def analyze_sentiment(text: str) -> str:
    msgs = [
        {"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
        {"role":"user","content":text}
    ]
    out = await groq_chat(msgs, max_tokens=10, temperature=0)
    return (out or "neutral").strip().lower()

async def translate_text(text: str, target_lang_display: str) -> str:
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text."
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{text}"}}'
    return await groq_chat([{"role":"system","content":sys},{"role":"user","content":usr}], 800, 0.2)

def set_user_persona(chat_id: str, key: str):
    if key == "random": key = random.choice(list(PERSONAS.keys()))
    if key not in PERSONAS: key = "sweet"
    user_persona[chat_id] = key
    return key

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    key = user_persona.get(chat_id, "sweet")
    p = PERSONAS[key]
    return (
        f"你是一位「{p['title']}」。風格：{p['style']}\n"
        f"使用者情緒：{sentiment}；請調整語氣（開心→一起開心；難過/生氣→先共情、安撫再給建議；中性→自然聊天）。\n"
        f"回覆使用繁體中文，精煉自然，帶少量表情 {p['emoji']}。"
    )

# ---------- Quick Reply 相關 ----------
def make_quick_reply_items(is_group: bool, bot_name: str) -> List[QuickReplyButton]:
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
    ]

def reply_with_quick_bar(reply_token: str, text: str, is_group: bool, bot_name: str):
    items = make_quick_reply_items(is_group, bot_name)
    msg = TextSendMessage(text=text, quick_reply=QuickReply(items=items))
    line_bot_api.reply_message(reply_token, msg)

# ---------- Flex 選單 ----------
def build_flex_menu(title: str, subtitle: str, actions: List[MessageAction]) -> FlexSendMessage:
    buttons = [
        ButtonComponent(style="primary", height="sm", action=a, margin="md", color="#00B900")
        for a in actions
    ]
    bubble = BubbleContainer(
        header=BoxComponent(layout="vertical", contents=[
            TextComponent(text=title, weight="bold", size="xl", color="#000000", align="center"),
            TextComponent(text=subtitle, size="sm", color="#666666", wrap=True, align="center", margin="md"),
        ], backgroundColor="#FFFFFF"),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm", paddingAll="12px", backgroundColor="#FAFAFA"),
    )
    return FlexSendMessage(alt_text=title, contents=bubble)

def flex_menu_finance(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    acts = [
        MessageAction(label="🇹🇼 台股大盤", text=f"{prefix}台股大盤"),
        MessageAction(label="🇺🇸 美股大盤", text=f"{prefix}美股大盤"),
        MessageAction(label="💰 金價",   text=f"{prefix}金價"),
        MessageAction(label="💴 日元",   text=f"{prefix}JPY"),
        MessageAction(label="📊 個股(例:2330)", text=f"{prefix}2330"),
    ]
    return build_flex_menu("💰 金融服務", "快速查行情", acts)

def flex_menu_lottery(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    acts = [
        MessageAction(label="🎰 大樂透", text=f"{prefix}大樂透"),
        MessageAction(label="🎯 威力彩", text=f"{prefix}威力彩"),
        MessageAction(label="🔢 539",   text=f"{prefix}539"),
    ]
    return build_flex_menu("🎰 彩票服務", "開獎/趨勢", acts)

def flex_menu_translate() -> FlexSendMessage:
    acts = [
        MessageAction(label="🇺🇸 英文", text="翻譯->英文"),
        MessageAction(label="🇯🇵 日文", text="翻譯->日文"),
        MessageAction(label="🇰🇷 韓文", text="翻譯->韓文"),
        MessageAction(label="🇻🇳 越南文", text="翻譯->越南文"),
        MessageAction(label="🇹🇼 繁中", text="翻譯->繁體中文"),
        MessageAction(label="❌ 結束翻譯", text="翻譯->結束"),
    ]
    return build_flex_menu("🌐 翻譯選擇", "選擇目標語言", acts)

def flex_menu_persona() -> FlexSendMessage:
    acts = [
        MessageAction(label="🌸 甜美女友", text="甜"),
        MessageAction(label="😏 傲嬌女友", text="鹹"),
        MessageAction(label="🎀 萌系女友", text="萌"),
        MessageAction(label="🧊 酷系御姐", text="酷"),
        MessageAction(label="🎲 隨機人設", text="random"),
    ]
    return build_flex_menu("💖 人設選擇", "切換 AI 女友風格", acts)

# ========== 5) LINE Handlers ==========
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    # 用同步入口 → 開新 task 跑 async 主邏輯，避免阻塞
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(handle_message_async(event))
    except RuntimeError:
        # 若無 running loop（某些 WSGI/測試環境），改用 asyncio.run
        asyncio.run(handle_message_async(event))

async def handle_message_async(event: MessageEvent):
    user_id, chat_id = event.source.user_id, get_chat_id(event)
    msg_raw: str = event.message.text.strip()
    reply_token = event.reply_token
    is_group = isinstance(event.source, (SourceGroup, SourceRoom))
    try:
        bot_name = line_bot_api.get_bot_info().display_name
    except Exception:
        bot_name = "AI 助手"

    if not msg_raw: return
    if chat_id not in auto_reply_status: auto_reply_status[chat_id] = True

    # 群組需 @bot 或開啟自動回覆
    if is_group and not auto_reply_status.get(chat_id, True) and not msg_raw.startswith(f"@{bot_name}"):
        return
    if msg_raw.startswith(f"@{bot_name}"):
        msg = msg_raw[len(f"@{bot_name}"):].strip()
    else:
        msg = msg_raw

    low = msg.lower()

    # 基本開關
    if low == "開啟自動回答":
        auto_reply_status[chat_id] = True
        return reply_with_quick_bar(reply_token, "✅ 已開啟自動回答", is_group, bot_name)
    if low == "關閉自動回答":
        auto_reply_status[chat_id] = False
        return reply_with_quick_bar(reply_token, "❌ 已關閉自動回答（群組需 @我 才回）", is_group, bot_name)

    # === Flex + QuickReply 二連發 ===
    if low in ("金融選單", "彩票選單", "翻譯選單", "我的人設", "人設選單"):
        flex = {
            "金融選單":  flex_menu_finance(bot_name, is_group),
            "彩票選單":  flex_menu_lottery(bot_name, is_group),
            "翻譯選單":  flex_menu_translate(),
            "我的人設":  flex_menu_persona(),
            "人設選單":  flex_menu_persona(),
        }[low]
        tip = TextSendMessage(text="👇 選一個功能開始吧",
                              quick_reply=QuickReply(items=make_quick_reply_items(is_group, bot_name)))
        line_bot_api.reply_message(reply_token, [flex, tip])
        return

    # 翻譯模式開關
    if low.startswith("翻譯->"):
        lang = msg.split("->", 1)[1].strip()
        if lang == "結束":
            translation_states.pop(chat_id, None)
            return reply_with_quick_bar(reply_token, "✅ 已結束翻譯模式", is_group, bot_name)
        translation_states[chat_id] = lang
        return reply_with_quick_bar(reply_token, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。", is_group, bot_name)

    # 人設切換
    persona_keys = {"甜":"sweet","鹹":"salty","萌":"moe","酷":"cool","random":"random","隨機":"random"}
    if low in persona_keys:
        key = set_user_persona(chat_id, persona_keys[low])
        p = PERSONAS[key]
        txt = f"💖 已切換人設：{p['title']}\n\n【特質】{p['style']}\n{p['greetings']}"
        return reply_with_quick_bar(reply_token, txt, is_group, bot_name)

    # 翻譯模式處理
    if chat_id in translation_states:
        tgt = translation_states[chat_id]
        try:
            out = await translate_text(msg, tgt)
        except Exception as e:
            logger.error(f"翻譯失敗: {e}", exc_info=True)
            out = "翻譯暫時失效，等我回神再來一次 🙏"
        return reply_with_quick_bar(reply_token, f"🌐 ({tgt})\n{out}", is_group, bot_name)

    # 其他：一般對話（人設 + 情感）
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys = build_persona_prompt(chat_id, sentiment)
        messages = [{"role":"system","content":sys}] + history + [{"role":"user","content":msg}]
        final = await groq_chat(messages)
        history.extend([{"role":"user","content":msg},{"role":"assistant","content":final}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
    except Exception as e:
        logger.error(f"AI 回覆失敗: {e}", exc_info=True)
        final = "抱歉我剛剛走神了 😅 再說一次讓我補上！"

    return reply_with_quick_bar(reply_token, final, is_group, bot_name)

@handler.add(PostbackEvent)
def handle_postback(event):  # 可按需擴充
    pass

# ========== 6) FastAPI Routes ==========
@router.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    try:
        # 用 threadpool 呼叫同步 handler；裡面再自管 async 任務
        await run_in_threadpool(handler.handle, body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(400, "Invalid signature")
    except Exception as e:
        logger.error(f"Callback 處理失敗：{e}", exc_info=True)
        raise HTTPException(500, "Internal error")
    return JSONResponse({"message":"ok"})

@router.get("/")
async def root():
    return {"message":"Service is live."}

app.include_router(router)

# ========== 7) Local run ==========
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info")