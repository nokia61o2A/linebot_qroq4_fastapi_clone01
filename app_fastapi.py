# app_fastapi.py
# =============================================================================
# LINE Bot + FastAPI (金價 / 股票 / 彩票／翻譯／TTS／單聊 Loading 動畫)
# -----------------------------------------------------------------------------
# 功能重點：
# - 彩票呼叫你自己的模組 my_commands/lottery_gpt.py（支援部分彩種）
# - 其餘彩種 fallback 使用 TaiwanLotteryCrawler 庫
# =============================================================================

import os
import re
import io
import json
import time
import random
import logging
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime

import requests
import httpx
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup

from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, AudioSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    QuickReply, QuickReplyButton, MessageAction,
    PostbackAction, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent,
    TextComponent, ButtonComponent
)

from gtts import gTTS
import cloudinary
import cloudinary.uploader
import uvicorn

# === 導入 TaiwanLotteryCrawler 庫 ===
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    _LT_CRAWLER_OK = True
    logging.info("✅ TaiwanLotteryCrawler 模組載入成功")
except Exception as e:
    _LT_CRAWLER_OK = False
    logging.warning(f"⚠️ TaiwanLotteryCrawler 載入失敗：{e}")

# === 導入你原有的分析模組 my_commands/lottery_gpt.py ===
try:
    from my_commands.lottery_gpt import lottery_gpt as ext_lottery_gpt
    _EXT_LOTTERY_OK = True
except Exception as e:
    _EXT_LOTTERY_OK = False
    logging.warning(f"⚠️ 外掛 lottery_gpt 模組載入失敗：{e}")

# ========= Logging =========
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(asctime)s:%(message)s"
)
log = logging.getLogger("app")

# ========= ENV =========
BASE_URL = os.getenv("BASE_URL")  # e.g. https://your-domain/callback
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")  # e.g. https://api.openai.com/v1 或自建代理

if not BASE_URL or not CHANNEL_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("請設定環境變數：BASE_URL、CHANNEL_ACCESS_TOKEN、CHANNEL_SECRET")

# ========= LINE SDK =========
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# ========= Cloudinary（可選，用於語音上傳）=========
CLOUD_OK = False
try:
    if os.getenv("CLOUDINARY_URL"):
        cloudinary.config(cloudinary_url=os.getenv("CLOUDINARY_URL"))
    else:
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            secure=True
        )
    if cloudinary.config().cloud_name:
        CLOUD_OK = True
        log.info("✅ Cloudinary 配置成功")
except Exception as e:
    log.warning(f"⚠️ Cloudinary 初始化失敗：{e}")

# ========= AI Clients（OpenAI/Groq，雙引擎）=========
openai_client = None
if OPENAI_API_KEY:
    try:
        import openai as openai_lib
        if OPENAI_API_BASE:
            openai_client = openai_lib.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
            log.info(f"✅ OpenAI Client (base={OPENAI_API_BASE})")
        else:
            openai_client = openai_lib.OpenAI(api_key=OPENAI_API_KEY)
            log.info("✅ OpenAI Client (official)")
    except Exception as e:
        log.warning(f"OpenAI 初始化失敗：{e}")

from groq import Groq
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        log.info("✅ Groq Client 初始化成功")
    except Exception as e:
        log.warning(f"Groq 初始化失敗：{e}")

# 強制採用當前可用的 Groq 模型（避免 404 / decommission）
GROQ_MODEL_PRIMARY = "llama-3.1-8b-instant"

# ========= 全域狀態 =========
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125 Safari/537.36"
}
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"

conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY = 10
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}
auto_reply_status: Dict[str, bool] = {}
tts_enabled: Dict[str, bool] = {}
tts_lang: Dict[str, str] = {}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼", "greet": "我在這🌸", "emoji": "🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽", "greet": "你又來啦？😏", "emoji": "😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣", "greet": "呀呼～(ﾉ>ω<)ﾉ", "emoji": "✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉", "greet": "我在。說重點。", "emoji": "🧊⚡️"},
}
PERSONA_ALIAS = {"甜": "sweet", "鹹": "salty", "萌": "moe", "酷": "cool", "random": "random"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 應用啟動")
    try:
        async with httpx.AsyncClient() as c:
            headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
            payload = {"endpoint": f"{BASE_URL}/callback"}
            r = await c.put(
                "https://api.line.me/v2/bot/channel/webhook/endpoint",
                headers=headers, json=payload, timeout=10
            )
            r.raise_for_status()
            log.info("✅ Webhook 更新成功")
    except Exception as e:
        log.warning(f"⚠️ Webhook 更新失敗：{e}")
    yield
    log.info("👋 應用關閉")

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="5.0.0")
router = APIRouter()

# ========= Loading 動畫（僅單人聊天有效）=========
def send_loading_animation(user_id: str, seconds: int = 5):
    try:
        url = "https://api.line.me/v2/bot/chat/loading/start"
        headers = {
            "Authorization": f"Bearer {CHANNEL_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {"chatId": user_id, "loadingSeconds": max(1, min(15, int(seconds)))}
        resp = requests.post(url, headers=headers, json=payload, timeout=5)
        resp.raise_for_status()
        log.info(f"✅ Loading 動畫觸發成功 chatId={user_id}")
    except Exception as e:
        log.warning(f"⚠️ Loading 動畫觸發失敗：{e}")

# ========= QuickReply（依 TTS 與翻譯模式動態顯示）=========
def quick_bar(chat_id: Optional[str] = None) -> QuickReply:
    items: List[QuickReplyButton] = [
        QuickReplyButton(action=MessageAction(label="主選單", text="選單")),
        QuickReplyButton(action=MessageAction(label="台股大盤", text="台股大盤")),
        QuickReplyButton(action=MessageAction(label="美股大盤", text="美股大盤")),
        QuickReplyButton(action=MessageAction(label="黃金價格", text="金價")),
        QuickReplyButton(action=MessageAction(label="日圓匯率", text="JPY")),
        QuickReplyButton(action=MessageAction(label="查 2330", text="2330")),
        QuickReplyButton(action=MessageAction(label="查 NVDA", text="NVDA")),
        QuickReplyButton(action=PostbackAction(label="💖 AI 人設", data="menu:persona")),
        QuickReplyButton(action=PostbackAction(label="🎰 彩票選單", data="menu:lottery")),
    ]
    if chat_id and tts_enabled.get(chat_id, False):
        items.insert(7, QuickReplyButton(action=MessageAction(label="語音 關", text="TTS OFF")))
    else:
        items.insert(7, QuickReplyButton(action=MessageAction(label="語音 開✅", text="TTS ON")))

    if chat_id and chat_id in translation_states:
        items.append(QuickReplyButton(action=MessageAction(label="結束翻譯", text="翻譯->結束")))
    else:
        items.append(QuickReplyButton(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate")))

    return QuickReply(items=items)

# ========= sender.name（翻譯模式顯示「翻譯模式（中↔英）」）=========
def display_sender_name(chat_id: str) -> Tuple[str, Optional[str]]:
    if chat_id in translation_states:
        target = translation_states.get(chat_id) or ""
        mapping = {"英文": "中→英", "日文": "中→日", "繁體中文": "→ 繁中", "中英雙向": "中↔英"}
        arrow = mapping.get(target, f"→ {target}") if target else ""
        name = f"翻譯模式（{arrow}）" if arrow else "翻譯模式"
        return name, None
    return "AI 助理", None

# ========= 後續：TTS、AI、翻譯、股票、金價、彩票分析 等功能續寫……
# （Page 2/2 接續）  
# ========= TTS 與預設 =========
def ensure_defaults(chat_id: str):
    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True
    if chat_id not in tts_enabled:
        tts_enabled[chat_id] = False
    if chat_id not in tts_lang:
        tts_lang[chat_id] = "zh-TW"
    if chat_id not in user_persona:
        user_persona[chat_id] = "sweet"

def tts_make_url(text: str, lang_code: str) -> Tuple[Optional[str], int]:
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        data = buf.getvalue()
        if not CLOUD_OK:
            return None, 0
        res = cloudinary.uploader.upload(
            data, resource_type="video",
            folder="line-bot-tts",
            public_id=f"say_{int(time.time()*1000)}",
            overwrite=True
        )
        url = res.get("secure_url")
        dur = max(1000, int(len(data)/32))
        return url, dur if url else (None, 0)
    except Exception as e:
        log.error(f"TTS 生成/上傳失敗：{e}")
        return None, 0

# ========= Flex 主選單與子選單（移除多餘分隔線）=========
def flex_main(chat_id: Optional[str] = None) -> FlexSendMessage:
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[
            TextComponent(text="AI 助理主選單", weight="bold", size="lg")
        ]),
        body=BoxComponent(
            layout="vertical", spacing="md",
            contents=[
                TextComponent(text="請選擇功能：", size="md"),
                ButtonComponent(action=PostbackAction(label="💹 金融查詢", data="menu:finance"),
                                style="primary", color="#5E86C1"),
                ButtonComponent(action=PostbackAction(label="🎰 彩票分析", data="menu:lottery"),
                                style="primary", color="#5EC186"),
                ButtonComponent(action=PostbackAction(label="💖 AI 角色", data="menu:persona"),
                                style="secondary"),
                ButtonComponent(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate"),
                                style="secondary"),
                ButtonComponent(action=PostbackAction(label="⚙️ 系統設定", data="menu:settings"),
                                style="secondary"),
            ]
        )
    )
    return FlexSendMessage(alt_text="主選單", contents=bubble, quick_reply=quick_bar(chat_id))

def flex_submenu(kind: str, chat_id: Optional[str] = None) -> FlexSendMessage:
    title, buttons = "子選單", []
    if kind == "finance":
        title = "💹 金融查詢"
        buttons = [
            ButtonComponent(action=MessageAction(label="台股大盤", text="台股大盤")),
            ButtonComponent(action=MessageAction(label="美股大盤", text="美股大盤")),
            ButtonComponent(action=MessageAction(label="黃金價格", text="金價")),
            ButtonComponent(action=MessageAction(label="日圓匯率", text="JPY")),
            ButtonComponent(action=MessageAction(label="查 2330", text="2330")),
            ButtonComponent(action=MessageAction(label="查 NVDA", text="NVDA")),
        ]
    elif kind == "lottery":
        title = "🎰 彩票分析"
        buttons = [
            ButtonComponent(action=MessageAction(label="大樂透", text="大樂透")),
            ButtonComponent(action=MessageAction(label="威力彩", text="威力彩")),
            ButtonComponent(action=MessageAction(label="今彩539", text="今彩539")),
            ButtonComponent(action=MessageAction(label="雙贏彩", text="雙贏彩")),
            ButtonComponent(action=MessageAction(label="3星彩", text="3星彩")),
            ButtonComponent(action=MessageAction(label="4星彩", text="4星彩")),
            ButtonComponent(action=MessageAction(label="38樂合彩", text="38樂合彩")),
            ButtonComponent(action=MessageAction(label="39樂合彩", text="39樂合彩")),
            ButtonComponent(action=MessageAction(label="49樂合彩", text="49樂合彩")),
        ]
    elif kind == "persona":
        title = "💖 AI 角色"
        buttons = [
            ButtonComponent(action=MessageAction(label="甜美女友", text="甜")),
            ButtonComponent(action=MessageAction(label="傲嬌女友", text="鹹")),
            ButtonComponent(action=MessageAction(label="萌系女友", text="萌")),
            ButtonComponent(action=MessageAction(label="酷系御姐", text="酷")),
            ButtonComponent(action=MessageAction(label="隨機", text="random")),
        ]
    elif kind == "translate":
        title = "🌐 翻譯工具"
        buttons = [
            ButtonComponent(action=MessageAction(label="翻英文", text="翻譯->英文")),
            ButtonComponent(action=MessageAction(label="翻日文", text="翻譯->日文")),
            ButtonComponent(action=MessageAction(label="翻繁中", text="翻譯->繁體中文")),
            ButtonComponent(action=MessageAction(label="中↔英", text="翻譯->中英雙向")),
            ButtonComponent(action=MessageAction(label="結束翻譯", text="翻譯->結束")),
        ]
    elif kind == "settings":
        title = "⚙️ 系統設定"
        buttons = [
            ButtonComponent(action=MessageAction(label="開啟自動回答", text="開啟自動回答")),
            ButtonComponent(action=MessageAction(label="關閉自動回答", text="關閉自動回答")),
        ]
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[
            TextComponent(text=title, weight="bold", size="lg")
        ]),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm")
    )
    return FlexSendMessage(alt_text=title, contents=bubble, quick_reply=quick_bar(chat_id))

# ========= 其它功能：股票報告、金價、匯率已在 Page 1 中定義 ≈ 略 … ========

# ========= 彩票報告函式（略重複版，已在 Page 1 定義：lottery_report_all）===========
# （此處假設已載入你自己的模組 ext_lottery_gpt 與 fallback 函式 lottery_report_all）

# ========= 事件處理：MessageEvent =========
@handler.add(MessageEvent, message=TextMessage)
def on_message(event: MessageEvent):
    chat_id = (
        event.source.group_id if isinstance(event.source, SourceGroup) else
        event.source.room_id  if isinstance(event.source, SourceRoom)  else
        event.source.user_id
    )
    ensure_defaults(chat_id)

    text = (event.message.text or "").strip()
    if not text:
        return

    should = isinstance(event.source, SourceUser) or auto_reply_status.get(chat_id, True)
    if not should:
        return

    if isinstance(event.source, SourceUser):
        send_loading_animation(chat_id, seconds=4)

    low = text.lower()

    try:
        # 主選單
        if low in ("menu", "選單", "主選單"):
            line_bot_api.reply_message(event.reply_token, flex_main(chat_id))
            return

        # TTS 切換
        if low in ("tts on", "tts on✅"):
            tts_enabled[chat_id] = True
            reply_text_audio_flex(event.reply_token, chat_id, "已開啟語音播報 ✅", None, 0)
            return
        if low in ("tts off", "tts off❌", "tts off✖"):
            tts_enabled[chat_id] = False
            reply_text_audio_flex(event.reply_token, chat_id, "已關閉語音播報", None, 0)
            return

        # 金價查詢
        if low in ("金價", "黃金", "黃金價格"):
            msg, sell, buy, ts = get_bot_gold()
            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                audio, dur = tts_make_url(msg, tts_lang[chat_id])
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # 匯率 JPY→TWD
        if low == "jpy":
            msg = jpy_twd()
            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                audio, dur = tts_make_url(msg, tts_lang[chat_id])
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # 股票查詢
        if low in ("台股大盤", "大盤", "美股大盤", "美盤", "美股") \
           or _TW_CODE_RE.match(text.upper()) \
           or (_US_CODE_RE.match(text.upper()) and text.upper() != "JPY"):
            msg = stock_report(text)
            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                audio, dur = tts_make_url(msg, tts_lang[chat_id])
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # 彩票觸發（支援所有彩種）
        lottery_names = ("大樂透", "威力彩", "今彩539", "539", "雙贏彩", "3星彩", "4星彩", "38樂合彩", "39樂合彩", "49樂合彩")
        if text in lottery_names:
            mapping = {"539": "今彩539"}
            kind = mapping.get(text, text)

            if _EXT_LOTTERY_OK and kind in ("大樂透", "威力彩", "今彩539"):
                try:
                    msg = ext_lottery_gpt(kind)
                except Exception as e:
                    log.warning(f"外掛分析模組失敗：{e}")
                    msg = lottery_report_all(kind)
            else:
                msg = lottery_report_all(kind)

            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                audio, dur = tts_make_url(msg, tts_lang[chat_id])
            reply_text_audio_flex(event.reply_token, chat_id, msg, audio, dur)
            return

        # 自動回覆開關
        if text in ("開啟自動回答", "關閉自動回答"):
            auto_reply_status[chat_id] = (text == "開啟自動回答")
            reply_text_audio_flex(event.reply_token, chat_id,
                                  f"自動回答：{'開啟' if auto_reply_status[chat_id] else '關閉'}",
                                  None, 0)
            return

        # 人設切換
        if text in PERSONA_ALIAS:
            key = PERSONA_ALIAS[text]
            if key == "random":
                key = random.choice(list(PERSONAS.keys()))
            user_persona[chat_id] = key
            p = PERSONAS[key]
            reply_text_audio_flex(event.reply_token, chat_id,
                                  f"💖 角色切換：{p['title']}\n{p['greet']}",
                                  None, 0)
            return

        # 翻譯模式切換
        if text.startswith("翻譯->"):
            lang = text.split("->", 1)[1]
            if lang in ("結束", "結束翻譯"):
                translation_states.pop(chat_id, None)
                reply_text_audio_flex(event.reply_token, chat_id, "✅ 已結束翻譯模式", None, 0)
            else:
                if lang in ("英文", "日文", "繁體中文", "中英雙向"):
                    translation_states[chat_id] = lang
                    label = "中↔英" if lang == "中英雙向" else f"→ {lang}"
                    reply_text_audio_flex(event.reply_token, chat_id, f"🈯 已開啟翻譯模式（{label}）", None, 0)
                else:
                    reply_text_audio_flex(event.reply_token, chat_id, "未支援的翻譯目標。", None, 0)
            return

        # 翻譯模式內容
        if chat_id in translation_states:
            mode = translation_states[chat_id]
            if mode == "中英雙向":
                out = translate_bilingual(text)
            else:
                out = translate_text(text, mode)
            audio, dur = (None, 0)
            if tts_enabled[chat_id]:
                lang_code = tts_lang[chat_id]
                if mode == "中英雙向":
                    ascii_ratio = sum(1 for ch in out if ord(ch) < 128) / max(1, len(out))
                    lang_code = "en" if ascii_ratio > 0.6 else "zh-TW"
                audio, dur = tts_make_url(out, lang_code)
            reply_text_audio_flex(event.reply_token, chat_id, out, audio, dur)
            return

        # 一般聊天（帶人設）
        key = user_persona.get(chat_id, "sweet")
        p = PERSONAS[key]
        sys_prompt = f"你是「{p['title']}」。風格：{p['style']}。用繁體中文，自然精煉，適量表情 {p['emoji']}。"
        hist = conversation_history.get(chat_id, [])
        msgs = [{"role": "system", "content": sys_prompt}] + hist + [{"role": "user", "content": text}]
        out = ai_chat(msgs)
        hist.extend([{"role": "user", "content": text}, {"role": "assistant", "content": out}])
        conversation_history[chat_id] = hist[-MAX_HISTORY * 2:]

        audio, dur = (None, 0)
        if tts_enabled[chat_id]:
            audio, dur = tts_make_url(out, tts_lang[chat_id])
        reply_text_audio_flex(event.reply_token, chat_id, out, audio, dur)

    except LineBotApiError as e:
        log.error(f"LINE 回覆失敗：{e}")
        try:
            reply_text_audio_flex(event.reply_token, chat_id, "⚠️ LINE 回覆失敗，請稍後再試。", None, 0)
        except Exception:
            pass
    except Exception as e:
        log.error(f"處理訊息錯誤：{e}", exc_info=True)
        try:
            reply_text_audio_flex(event.reply_token, chat_id, "😵‍💫 發生錯誤，請稍後再試。", None, 0)
        except Exception:
            pass

# ========= 事件處理：PostbackEvent =========
@handler.add(PostbackEvent)
def on_postback(event: PostbackEvent):
    data = (event.postback.data or "")
    sub = data[5:] if data.startswith("menu:") else ""
    chat_id = (
        event.source.group_id if isinstance(event.source, SourceGroup) else
        event.source.room_id  if isinstance(event.source, SourceRoom)  else
        event.source.user_id
    )
    try:
        line_bot_api.reply_message(
            event.reply_token,
            [flex_submenu(sub or "finance", chat_id),
             TextSendMessage(text="請選擇 👇", quick_reply=quick_bar(chat_id))]
        )
    except Exception as e:
        log.error(f"Postback 失敗：{e}")

@router.post("/callback")
async def callback(request: Request):
    sig = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode("utf-8")
    try:
        handler.handle(body, sig)
        return JSONResponse({"status": "ok"})
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        log.error(f"/callback 失敗：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="internal error")

@router.get("/")
async def root():
    return PlainTextResponse("LINE Bot running.", status_code=200)

@router.get("/healthz")
async def health():
    return PlainTextResponse("ok", status_code=200)

app.include_router(router)

# ========= Local run =========
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)