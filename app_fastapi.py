# app_fastapi.py  v1.4.1  (Render-friendly, no-Redis)
# 變更摘要：
# - [FIX] 翻譯模式最高優先：開啟後任何訊息皆先翻譯，只輸出譯文
# - [NEW] 一次性行內翻譯：en:/英文:/EN>/ja:/日文:/zh:/繁中: 等前綴立即翻譯（stateless）
# - [CHG] Render 建議使用單一 worker（--workers 1）；程式仍保留記憶體 TTL，避免卡死
# - [CHG] get_chat_id 強化、翻譯指令解析更寬鬆；完整註解

import os
import re
import io
import random
import logging
import pkg_resources
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

# --- HTTP / 解析 ---
import requests
import httpx
from bs4 import BeautifulSoup

# --- 資料處理 / 金融（沿用） ---
import pandas as pd
import yfinance as yf

# --- FastAPI / LINE SDK v3 ---
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from linebot.v3.exceptions import InvalidSignatureError
# 事件/訊息型別還是在 webhooks（複數）
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    AudioMessageContent,
    PostbackEvent,
)

# WebhookHandler 在 webhook（單數）模組
from linebot.v3.webhook import WebhookHandler
from linebot.v3.messaging import (
    Configuration, ApiClient, AsyncMessagingApi, ReplyMessageRequest,
    TextMessage, AudioMessage, ImageMessage, FlexMessage, FlexBubble, FlexBox,
    FlexText, FlexButton, QuickReply, QuickReplyItem, MessageAction, PostbackAction,
    BotInfoResponse,
)

# --- Cloudinary（可選） ---
import cloudinary
import cloudinary.uploader

# --- 語音 ---
from gtts import gTTS

# --- LLM ---
from groq import AsyncGroq, Groq
import openai

# --- Matplotlib（可選） ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
try:
    import mplfinance as mpf
    HAS_MPLFIN = True
except Exception:
    HAS_MPLFIN = False

# ====== 基本設定 ======
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

logger.info("Installed versions:")
for pkg in ["line-bot-sdk","fastapi","uvicorn","groq","openai","requests","pandas",
            "beautifulsoup4","httpx","yfinance","cloudinary","gTTS","matplotlib","mplfinance"]:
    try:
        version = pkg_resources.get_distribution(pkg).version
        logger.info(f"{pkg}: {version}")
    except pkg_resources.DistributionNotFound:
        logger.warning(f"{pkg}: not installed")

BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto").lower()  # auto/openai/gtts

if not CHANNEL_TOKEN or not CHANNEL_SECRET:
    raise RuntimeError("缺少必要環境變數：CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET")

# Cloudinary 可選
if CLOUDINARY_URL:
    try:
        cloudinary.config(
            cloud_name=re.search(r"@(.+)", CLOUDINARY_URL).group(1),
            api_key=re.search(r"//(\d+):", CLOUDINARY_URL).group(1),
            api_secret=re.search(r":([A-Za-z0-9_-]+)@", CLOUDINARY_URL).group(1),
        )
        logger.info("Cloudinary OK")
    except Exception as e:
        logger.error(f"Cloudinary 設定失敗: {e}")
        CLOUDINARY_URL = None

# LINE / LLM
configuration = Configuration(access_token=CHANNEL_TOKEN)
async_api_client = ApiClient(configuration=configuration)
line_bot_api = AsyncMessagingApi(api_client=async_api_client)
handler = WebhookHandler(CHANNEL_SECRET)

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
sync_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.warning(f"初始化 OpenAI 失敗：{e}")

# LLM 模型
GROQ_MODEL_PRIMARY  = os.getenv("GROQ_MODEL_PRIMARY",  "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

# ====== 狀態 ======
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10

# [FIX] 翻譯模式記憶體 + TTL（Render 無 Redis）
translation_states: Dict[str, str] = {}                  # chat_id -> 顯示語名（中文）
translation_states_ttl: Dict[str, datetime] = {}         # chat_id -> 到期時間
TRANSLATE_TTL_SECONDS = int(os.getenv("TRANSLATE_TTL_SECONDS", "7200"))  # 2h

auto_reply_status: Dict[str, bool] = {}
user_persona: Dict[str, str] = {}

PERSONAS = {
    "sweet": {"title":"甜美女友","style":"溫柔體貼，鼓勵安慰","greetings":"親愛的～我在這裡聽你說 🌸","emoji":"🌸💕😊"},
    "salty": {"title":"傲嬌女友","style":"機智吐槽，壞壞但有溫度","greetings":"你又來啦？說吧，哪裡卡住了。😏","emoji":"😏🙄"},
    "moe":   {"title":"萌系女友","style":"動漫語氣＋可愛顏文字","greetings":"呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ","emoji":"✨🎀"},
    "cool":  {"title":"酷系御姐","style":"冷靜精煉，關鍵建議","greetings":"我在。說重點。","emoji":"🧊⚡️"},
}
LANGUAGE_MAP = {
    "英文":"English","日文":"Japanese","韓文":"Korean","越南文":"Vietnamese",
    "繁體中文":"Traditional Chinese","中文":"Traditional Chinese",
    "en":"English","ja":"Japanese","jp":"Japanese","ko":"Korean","vi":"Vietnamese","zh":"Traditional Chinese"
}
PERSONA_ALIAS = {"甜":"sweet","鹹":"salty","萌":"moe","酷":"cool","random":"random"}

# [FIX] 翻譯指令解析（多種箭頭/空白/別名）
TRANSLATE_CMD = re.compile(
    r"^(?:翻譯|翻译|翻成)\s*(?:->|→|>)?\s*(英文|English|日文|Japanese|韓文|Korean|越南文|Vietnamese|繁體中文|中文)\s*$",
    re.IGNORECASE
)

# [NEW] 一次性行內翻譯前綴（stateless）：en:/英文:/EN>/ja:/日文:/zh:/繁中:
INLINE_TRANSLATE = re.compile(
    r"^(en|eng|英文|ja|jp|日文|zh|繁中|中文)\s*[:：>]\s*(.+)$",
    re.IGNORECASE
)

# ====== 小工具 ======
def _now() -> datetime: return datetime.utcnow()

# ------------------- 修正版：chat_id 取用（完整覆蓋此函式） -------------------
def get_chat_id(event: MessageEvent) -> str:
    """
    取得可穩定識別對話的 chat_id。
    - 先讀 attribute：userId/user_id、groupId/group_id、roomId/room_id
    - 若物件支援 to_dict()，再從 dict 兜底一次（有些 SDK 版本屬性讀不到，但 dict 有）
    - 最後保底：用 type + source 的字串雜湊，避免回傳 'user:unknown'
    為了讓翻譯模式在「下一則訊息」讀得到，我們需要兩次訊息得到**同一把 key**。
    """
    source = event.source

    # 1) 先嘗試直讀屬性（不同版本/環境屬性名可能不同）
    stype = getattr(source, "type", None) or getattr(source, "_type", None)
    uid = getattr(source, "userId", None) or getattr(source, "user_id", None)
    gid = getattr(source, "groupId", None) or getattr(source, "group_id", None)
    rid = getattr(source, "roomId", None)  or getattr(source, "room_id", None)

    # 2) 如果有 to_dict()，再兜底一次（很多 v3 型別都支援）
    try:
        if hasattr(source, "to_dict"):
            d = source.to_dict() or {}
            stype = stype or d.get("type")
            uid = uid or d.get("userId")  or d.get("user_id")
            gid = gid or d.get("groupId") or d.get("group_id")
            rid = rid or d.get("roomId")  or d.get("room_id")
    except Exception:
        pass

    # 3) 依群組/聊天室/私訊優先序回傳
    if gid: return f"group:{gid}"
    if rid: return f"room:{rid}"
    if uid: return f"user:{uid}"

    # 4) 最後保底，避免 'user:unknown' 造成下次 key 不同
    #    使用 source 的字串表現做 hash（不含機敏資訊）
    key_fallback = f"{stype or 'unknown'}:{abs(hash(str(source)))%10_000_000}"
    return key_fallback
# ------------------- /修正版：chat_id 取用 -------------------

# 這三個函式原本就有，但這裡加入更明確的 log（可覆蓋）
def _tstate_set(chat_id: str, lang_display: str):
    translation_states[chat_id] = lang_display
    translation_states_ttl[chat_id] = _now() + timedelta(seconds=TRANSLATE_TTL_SECONDS)
    logger.info(f"[TranslateMode] SET chat_id={chat_id} -> {lang_display} (ttl={TRANSLATE_TTL_SECONDS}s)")

def _tstate_get(chat_id: str) -> Optional[str]:
    exp = translation_states_ttl.get(chat_id)
    if exp and _now() > exp:
        translation_states.pop(chat_id, None)
        translation_states_ttl.pop(chat_id, None)
        logger.info(f"[TranslateMode] EXPIRE chat_id={chat_id}")
        return None
    val = translation_states.get(chat_id)
    logger.info(f"[TranslateMode] GET chat_id={chat_id} -> {val}")
    return val

def _tstate_clear(chat_id: str):
    translation_states.pop(chat_id, None)
    translation_states_ttl.pop(chat_id, None)
    logger.info(f"[TranslateMode] CLEAR chat_id={chat_id}")

def build_quick_reply() -> QuickReply:
    return QuickReply(items=[
        QuickReplyItem(action=MessageAction(label="主選單", text="選單")),
        QuickReplyItem(action=MessageAction(label="台股大盤", text="台股大盤")),
        QuickReplyItem(action=MessageAction(label="美股大盤", text="美股大盤")),
        QuickReplyItem(action=MessageAction(label="黃金價格", text="金價")),
        QuickReplyItem(action=MessageAction(label="查台積電", text="2330")),
        QuickReplyItem(action=MessageAction(label="查輝達", text="NVDA")),
        QuickReplyItem(action=MessageAction(label="查日圓", text="JPY")),
        QuickReplyItem(action=PostbackAction(label="💖 AI 人設", data="menu:persona")),
        QuickReplyItem(action=PostbackAction(label="🎰 彩票選單", data="menu:lottery")),
        QuickReplyItem(action=MessageAction(label="結束翻譯", text="翻譯->結束")),
    ])

def build_main_menu() -> FlexMessage:
    items = [
        ("💹 金融查詢", PostbackAction(label="💹 金融查詢", data="menu:finance")),
        ("🎰 彩票分析", PostbackAction(label="🎰 彩票分析", data="menu:lottery")),
        ("💖 AI 角色扮演", PostbackAction(label="💖 AI 角色扮演", data="menu:persona")),
        ("🌐 翻譯工具", PostbackAction(label="🌐 翻譯工具", data="menu:translate")),
    ]
    buttons = [FlexButton(action=i[1], style="primary" if idx<2 else "secondary") for idx,i in enumerate(items)]
    bubble = FlexBubble(
        header=FlexBox(layout="vertical", contents=[FlexText(text="AI 助理主選單", weight="bold", size="lg")]),
        body=FlexBox(layout="vertical", spacing="md", contents=buttons),
    )
    return FlexMessage(alt_text="主選單", contents=bubble)

def build_submenu(kind: str) -> FlexMessage:
    menus = {
        "translate": ("🌐 翻譯工具", [
            ("翻成英文", MessageAction(label="翻成英文", text="翻譯->英文")),
            ("翻成日文", MessageAction(label="翻成日文", text="翻譯->日文")),
            ("翻成繁中", MessageAction(label="翻成繁中", text="翻譯->繁體中文")),
            ("結束翻譯模式", MessageAction(label="結束翻譯模式", text="翻譯->結束")),
        ])
    }
    title, items = menus.get(kind, ("無效選單", []))
    rows, row = [], []
    for _, action in items:
        row.append(FlexButton(action=action, style="primary"))
        if len(row)==2: rows.append(FlexBox(layout="horizontal", spacing="sm", contents=row)); row=[]
    if row: rows.append(FlexBox(layout="horizontal", spacing="sm", contents=row))
    bubble = FlexBubble(
        header=FlexBox(layout="vertical", contents=[FlexText(text=title, weight="bold", size="lg")]),
        body=FlexBox(layout="vertical", spacing="md", contents=rows or [FlexText(text="（尚無項目）")]),
    )
    return FlexMessage(alt_text=title, contents=bubble)

async def reply_text_with_tts_and_extras(reply_token: str, text: str, extras: Optional[List]=None):
    if not text: text = "（無內容）"
    messages = [TextMessage(text=text, quick_reply=build_quick_reply())]
    if extras: messages.extend(extras)
    # 可選：回覆同時附 TTS
    if CLOUDINARY_URL:
        try:
            audio_bytes = await text_to_speech_async(text)
            if audio_bytes:
                res = await run_in_threadpool(lambda: cloudinary.uploader.upload(
                    io.BytesIO(audio_bytes), resource_type="video", folder="line-bot-tts", format="mp3"))
                url = res.get("secure_url")
                if url:
                    est = max(3000, min(30000, len(text)*60))
                    messages.append(AudioMessage(original_content_url=url, duration=est))
        except Exception as e:
            logger.warning(f"TTS 附加失敗：{e}")
    await line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=messages))

# ====== LLM 包裝 ======
def get_analysis_reply(messages: List[dict]) -> str:
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=1500
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI 失敗：{e}")
    try:
        resp = sync_groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY, messages=messages, temperature=0.7, max_tokens=2000
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq 主模型失敗：{e}")
        resp = sync_groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK, messages=messages, temperature=0.9, max_tokens=1500
        )
        return resp.choices[0].message.content

async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
    if not async_groq_client:
        return await run_in_threadpool(lambda: get_analysis_reply(messages))
    resp = await async_groq_client.chat.completions.create(
        model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    return resp.choices[0].message.content.strip()

async def analyze_sentiment(text: str) -> str:
    msgs = [{"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
            {"role":"user","content":text}]
    try:
        out = await groq_chat_async(msgs, max_tokens=10, temperature=0)
        return (out or "neutral").strip().lower()
    except Exception:
        return "neutral"

async def translate_text(text: str, target_lang_display: str) -> str:
    """只輸出譯文（嚴格）"""
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text with no extra words."
    clean = re.sub(r"[\u200B-\u200D\uFEFF]", "", text).strip()
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{clean}"}}'
    return await groq_chat_async([{"role":"system","content":sys},{"role":"user","content":usr}], 800, 0.2)

def set_user_persona(chat_id: str, key: str):
    key_mapped = PERSONA_ALIAS.get(key, key)
    if key_mapped == "random": key_mapped = random.choice(list(PERSONAS.keys()))
    if key_mapped not in PERSONAS: key_mapped = "sweet"
    user_persona[chat_id] = key_mapped
    return key_mapped

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    key = user_persona.get(chat_id, "sweet"); p = PERSONAS[key]
    return (f"你是一位「{p['title']}」。風格：{p['style']}。\n"
            f"使用者情緒：{sentiment}。\n"
            f"回覆請精煉自然，使用繁體中文，帶少量表情 {p['emoji']}.")

# ====== 金價/股票（沿用，略過細節註解） ======
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"
DEFAULT_HEADERS = {"User-Agent":"Mozilla/5.0","Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}

def parse_bot_gold_text(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser"); text = soup.get_text(" ", strip=True)
    m_time = re.search(r"掛牌時間[:：]\s*([0-9]{4}/[0-9]{2}/[0-9]{2}\s+[0-9]{2}:[0-9]{2})", text)
    listed_at = m_time.group(1) if m_time else None
    m_sell = re.search(r"本行賣出\s*([0-9,]+(?:\.[0-9]+)?)", text)
    m_buy  = re.search(r"本行買進\s*([0-9,]+(?:\.[0-9]+)?)", text)
    if not (m_sell and m_buy): raise RuntimeError("找不到『本行賣出/本行買進』欄位")
    sell = float(m_sell.group(1).replace(",","")); buy = float(m_buy.group(1).replace(",",""))
    return {"listed_at":listed_at,"sell_twd_per_g":sell,"buy_twd_per_g":buy,"source":BOT_GOLD_URL}

def get_bot_gold_quote() -> dict:
    r = requests.get(BOT_GOLD_URL, headers=DEFAULT_HEADERS, timeout=10); r.raise_for_status()
    return parse_bot_gold_text(r.text)

# ====== 事件處理 ======
@handler.add(MessageEvent, message=TextMessageContent)
async def handle_text_message(event: MessageEvent):
    chat_id   = get_chat_id(event)
    msg_raw   = (event.message.text or "").strip()
    reply_tok = event.reply_token
    if not msg_raw: return

    # 取得 bot 名稱（支援 @提及）
    try:
        bot_info: BotInfoResponse = await line_bot_api.get_bot_info()
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if chat_id not in auto_reply_status: auto_reply_status[chat_id] = True
    is_group_or_room = getattr(event.source, "type", "") in ("group","room")
    if is_group_or_room and not auto_reply_status.get(chat_id, True) and not msg_raw.startswith(f"@{bot_name}"):
        return

    msg = msg_raw
    if msg_raw.startswith(f"@{bot_name}"):
        msg = re.sub(f'^@{re.escape(bot_name)}\\s*','',msg_raw).strip()
    if not msg: return

    # --- 1) 指令：翻譯模式開/關 ---
    m = TRANSLATE_CMD.match(msg)
    if m:
        lang_token = m.group(1)
        rev = {"English":"英文","Japanese":"日文","Korean":"韓文","Vietnamese":"越南文","繁體中文":"繁體中文","中文":"繁體中文"}
        lang_display = rev.get(lang_token, lang_token)
        _tstate_set(chat_id, lang_display)
        await reply_text_with_tts_and_extras(reply_tok, f"🌐 已開啟翻譯 → {lang_display}，請直接輸入要翻的內容。")
        return

    if msg.startswith("翻譯->"):
        lang = msg.split("->",1)[1].strip()
        if lang == "結束":
            _tstate_clear(chat_id)
            await reply_text_with_tts_and_extras(reply_tok, "✅ 已結束翻譯模式")
        else:
            _tstate_set(chat_id, lang)
            await reply_text_with_tts_and_extras(reply_tok, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")
        return

    # --- 2) [NEW] 一次性行內翻譯（stateless，最高優先）---
    im = INLINE_TRANSLATE.match(msg)
    if im:
        lang_key, text_to_translate = im.group(1).lower(), im.group(2)
        # 正規化語言
        lang_display = {
            "en":"英文","eng":"英文","英文":"英文",
            "ja":"日文","jp":"日文","日文":"日文",
            "zh":"繁體中文","繁中":"繁體中文","中文":"繁體中文",
        }.get(lang_key, "英文")
        out = await translate_text(text_to_translate, lang_display)
        await reply_text_with_tts_and_extras(reply_tok, out)
        return

    # --- 3) ✅ 翻譯模式最高優先 ---
    current_lang = _tstate_get(chat_id)
    if current_lang:
        try:
            out = await translate_text(msg, current_lang)
            await reply_text_with_tts_and_extras(reply_tok, out)
        except Exception as e:
            logger.error(f"翻譯失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_tok, "抱歉，翻譯目前不可用。")
        return

    # --- 4) 其他路由（示例） ---
    low = msg.lower()
    if low in ("menu","選單","主選單"):
        await line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_tok, messages=[build_main_menu()]))
        return

    if low in ("金價","黃金"):
        try:
            data = get_bot_gold_quote()
            ts, sell, buy = data.get("listed_at") or "（未標示）", data["sell_twd_per_g"], data["buy_twd_per_g"]
            spread = sell - buy
            txt = (f"**金價（台灣銀行）**\n- 掛牌時間：{ts}\n- 賣出(1g)：{sell:,.0f} 元\n- 買進(1g)：{buy:,.0f} 元\n"
                   f"- 價差：{spread:,.0f} 元\n來源：{BOT_GOLD_URL}")
            await reply_text_with_tts_and_extras(reply_tok, txt)
        except Exception as e:
            await reply_text_with_tts_and_extras(reply_tok, "抱歉，目前無法取得金價。")
        return

    # --- 5) 一般聊天（人設） ---
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]
        final_reply = await groq_chat_async(messages)
        history.extend([{"role":"user","content":msg},{"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
        await reply_text_with_tts_and_extras(reply_tok, final_reply)
    except Exception as e:
        logger.error(f"聊天回覆失敗: {e}", exc_info=True)
        await reply_text_with_tts_and_extras(reply_tok, "抱歉我剛剛走神了 😅 再說一次讓我補上！")

# 語音（沿用你的既有流程即可；此處省略與翻譯無關的細節）
@handler.add(MessageEvent, message=AudioMessageContent)
async def handle_audio_message(event: MessageEvent):
    reply_tok = event.reply_token
    await reply_text_with_tts_and_extras(reply_tok, "（語音處理沿用原實作；與翻譯功能無關）")

# Postback
@handler.add(PostbackEvent)
async def handle_postback(event: PostbackEvent):
    data = event.postback.data or ""
    if data.startswith("menu:"):
        kind = data.split(":",1)[-1]
        await line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=event.reply_token, messages=[build_submenu(kind)])
        )

# ====== FastAPI ======
@asynccontextmanager
async def lifespan(app: FastAPI):
    if BASE_URL:
        async with httpx.AsyncClient() as c:
            for endpoint in ("https://api-data.line.me/v2/bot/channel/webhook/endpoint",
                             "https://api.line.me/v2/bot/channel/webhook/endpoint"):
                try:
                    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type":"application/json"}
                    payload = {"endpoint": f"{BASE_URL}/callback"}
                    r = await c.put(endpoint, headers=headers, json=payload, timeout=10.0)
                    r.raise_for_status()
                    logger.info(f"Webhook 更新成功: {endpoint} {r.status_code}")
                    break
                except Exception as e:
                    logger.warning(f"Webhook 更新失敗：{e}")
    yield

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.4.1")
router = APIRouter()

@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        await handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Callback 失敗：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse({"status":"ok"})

@router.get("/")
async def root():
    return PlainTextResponse("LINE Bot is running.", status_code=200)

@router.get("/healthz")
async def healthz():
    return PlainTextResponse("ok", status_code=200)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    # Render 建議：--workers 1，避免記憶體狀態分裂
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)