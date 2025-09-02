"""
aibot FastAPI 應用程序初始化 (v8 - 修正 NameError 與 Korean Romanizer 錯誤)
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

# ============================================
# 2. 初始化與設定 (Initializations & Setup)
# ============================================

# Logger
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# 檢查選用函式庫
try:
    from pypinyin import pinyin, Style
    PINYIN_ENABLED = True
except ImportError: PINYIN_ENABLED = False; logger.warning("未安裝 'pypinyin'，中文注音功能將不可用。")
try:
    import pykakasi
    KAKASI_ENABLED = True
except ImportError: KAKASI_ENABLED = False; logger.warning("未安裝 'pykakasi'，日文羅馬拼音功能將不可用。")
try:
    from korean_romanizer.romanizer import Romanizer
    KOREAN_ROMANIZER_ENABLED = True
except ImportError: KOREAN_ROMANIZER_ENABLED = False; logger.warning("未安裝 'korean-romanizer'，韓文羅馬拼音功能將不可用。")
try:
    from hangul_jamo import decompose
    HANGUL_JAMO_ENABLED = True
except ImportError: HANGUL_JAMO_ENABLED = False; logger.warning("未安裝 'hangul-jamo'，韓文注音模擬功能將不可用。")

# FastAPI 應用程式與路由器
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with httpx.AsyncClient() as client: await update_line_webhook(client)
    except Exception as e:
        logger.error(f"❌ 啟動初始化失敗: {e}", exc_info=True)
    yield

app = FastAPI(lifespan=lifespan, title="Line Bot API", version="1.0.0")
router = APIRouter() # 🔥 核心修正: 提前定義 router

# 環境變數與 API 客戶端
BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY = map(os.getenv, ["BASE_URL", "CHANNEL_ACCESS_TOKEN", "CHANNEL_SECRET", "GROQ_API_KEY"])
if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]): raise ValueError("缺少必要的環境變數！")

line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-8b-instant")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-70b-versatile")

# 狀態管理字典
conversation_history, MAX_HISTORY_LEN = {}, 10
auto_reply_status, user_persona, translation_states = {}, {}, {}

# 全域發音映射表
ROMAJI_BOPOMOFO_MAP = {'a': 'ㄚ', 'i': 'ㄧ', 'u': 'ㄨ', 'e': 'ㄝ', 'o': 'ㄛ', 'ka': 'ㄎㄚ', 'ki': 'ㄎㄧ', 'ku': 'ㄎㄨ', 'ke': 'ㄎㄝ', 'ko': 'ㄎㄛ', 'sa': 'ㄙㄚ', 'shi': 'ㄒㄧ', 'su': 'ㄙㄨ', 'se': 'ㄙㄝ', 'so': 'ㄙㄛ', 'ta': 'ㄊㄚ', 'chi': 'ㄑㄧ', 'tsu': 'ㄘㄨ', 'te': 'ㄊㄝ', 'to': 'ㄊㄛ', 'na': 'ㄋㄚ', 'ni': 'ㄋㄧ', 'nu': 'ㄋㄨ', 'ne': 'ㄋㄝ', 'no': 'ㄋㄛ', 'ha': 'ㄏㄚ', 'hi': 'ㄏㄧ', 'fu': 'ㄈㄨ', 'he': 'ㄏㄝ', 'ho': 'ㄏㄛ', 'ma': 'ㄇㄚ', 'mi': 'ㄇㄧ', 'mu': 'ㄇㄨ', 'me': 'ㄇㄝ', 'mo': 'ㄇㄛ', 'ya': 'ㄧㄚ', 'yu': 'ㄧㄨ', 'yo': 'ㄧㄛ', 'ra': 'ㄌㄚ', 'ri': 'ㄌㄧ', 'ru': 'ㄌㄨ', 're': 'ㄌㄝ', 'ro': 'ㄌㄛ', 'wa': 'ㄨㄚ', 'wo': 'ㄛ', 'n': 'ㄣ', 'ga': 'ㄍㄚ', 'gi': 'ㄍㄧ', 'gu': 'ㄍㄨ', 'ge': 'ㄍㄝ', 'go': 'ㄍㄛ', 'za': 'ㄗㄚ', 'ji': 'ㄐㄧ', 'zu': 'ㄗㄨ', 'ze': 'ㄗㄝ', 'zo': 'ㄗㄛ', 'da': 'ㄉㄚ', 'di': 'ㄉㄧ', 'dzu': 'ㄉㄨ', 'de': 'ㄉㄝ', 'do': 'ㄉㄛ', 'ba': 'ㄅㄚ', 'bi': 'ㄅㄧ', 'bu': 'ㄅㄨ', 'be': 'ㄅㄝ', 'bo': 'ㄅㄛ', 'pa': 'ㄆㄚ', 'pi': 'ㄆㄧ', 'pu': 'ㄆㄨ', 'pe': 'ㄆㄝ', 'po': 'ㄆㄛ', 'kya': 'ㄎㄧㄚ', 'kyu': 'ㄎㄧㄨ', 'kyo': 'ㄎㄧㄛ', 'sha': 'ㄕㄚ', 'shu': 'ㄕㄨ', 'sho': 'ㄕㄛ', 'cha': 'ㄑㄚ', 'chu': 'ㄑㄨ', 'cho': 'ㄑㄛ', 'nya': 'ㄋㄧㄚ', 'nyu': 'ㄋㄧㄨ', 'nyo': 'ㄋㄧㄛ', 'hya': 'ㄏㄧㄚ', 'hyu': 'ㄏㄧㄨ', 'hyo': 'ㄏㄧㄛ', 'mya': 'ㄇㄧㄚ', 'myu': 'ㄇㄧㄨ', 'myo': 'ㄇㄧㄛ', 'rya': 'ㄌㄧㄚ', 'ryu': 'ㄌㄧㄨ', 'ryo': 'ㄌㄧㄛ', 'gya': 'ㄍㄧㄚ', 'gyu': 'ㄍㄧㄨ', 'gyo': 'ㄍㄧㄛ', 'ja': 'ㄐㄧㄚ', 'ju': 'ㄐㄧㄨ', 'jo': 'ㄐㄧㄛ', 'bya': 'ㄅㄧㄚ', 'byu': 'ㄅㄧㄨ', 'byo': 'ㄅㄧㄛ', 'pya': 'ㄆㄧㄚ', 'pyu': 'ㄆㄧㄨ', 'pyo': 'ㄆㄧㄛ'}
KOREAN_BOPOMOFO_MAP = { 'ㄱ': 'ㄍ', 'ㄲ': 'ㄍ', 'ㄴ': 'ㄋ', 'ㄷ': 'ㄉ', 'ㄸ': 'ㄉ', 'ㄹ': 'ㄌ', 'ㅁ': 'ㄇ', 'ㅂ': 'ㄅ', 'ㅃ': 'ㄅ', 'ㅅ': 'ㄙ', 'ㅆ': 'ㄙ', 'ㅇ': '', 'ㅈ': 'ㄗ', 'ㅉ': 'ㄗ', 'ㅊ': 'ㄘ', 'ㅋ': 'ㄎ', 'ㅌ': 'ㄊ', 'ㅍ': 'ㄆ', 'ㅎ': 'ㄏ', 'ㅏ': 'ㄚ', 'ㅐ': 'ㄝ', 'ㅑ': 'ㄧㄚ', 'ㅒ': 'ㄧㄝ', 'ㅓ': 'ㄛ', 'ㅔ': 'ㄝ', 'ㅕ': 'ㄧㄛ', 'ㅖ': 'ㄧㄝ', 'ㅗ': 'ㄛ', 'ㅘ': 'ㄨㄚ', 'ㅙ': 'ㄨㄝ', 'ㅚ': 'ㄨㄝ', 'ㅛ': 'ㄧㄛ', 'ㅜ': 'ㄨ', 'ㅝ': 'ㄨㄛ', 'ㅞ': 'ㄨㄝ', 'ㅟ': 'ㄨㄧ', 'ㅠ': 'ㄧㄨ', 'ㅡ': 'ㄜ', 'ㅢ': 'ㅢ', 'ㅣ': 'ㄧ', 'ㄳ': 'ㄍ', 'ㄵ': 'ㄣ', 'ㄶ': 'ㄣ', 'ㄺ': 'ㄌ', 'ㄻ': 'ㄌ', 'ㄼ': 'ㄌ', 'ㄽ': 'ㄌ', 'ㄾ': 'ㄌ', 'ㄿ': 'ㄌ', 'ㅀ': 'ㄌ', 'ㅄ': 'ㄅ' }

# ============================================
# 3. 輔助函式 (Helper Functions)
# ============================================
async def update_line_webhook(client: httpx.AsyncClient):
    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
    json_data = {"endpoint": f"{BASE_URL}/callback"}
    res = await client.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=json_data, timeout=10.0)
    res.raise_for_status()
    logger.info(f"✅ Webhook 更新成功: {res.status_code}")

def japanese_to_bopomofo(text: str) -> str:
    if not KAKASI_ENABLED: return ""
    try:
        kks, romaji = pykakasi.kakasi(), ''.join([item.get('romaji', item.get('orig', '')) for item in kks.convert(text)])
        bopomofo_str, i = "", 0
        while i < len(romaji):
            match = None
            for length in (3, 2, 1):
                sub = romaji[i:i+length]
                if sub in ROMAJI_BOPOMOFO_MAP: match = sub; break
            if match:
                if i > 0 and romaji[i-1] == match[0] and romaji[i-1] not in "aiueon":
                     bopomofo_str += " " + ROMAJI_BOPOMOFO_MAP[match]
                else: bopomofo_str += ROMAJI_BOPOMOFO_MAP[match]
                i += len(match)
            else: bopomofo_str += romaji[i]; i += 1
        return bopomofo_str.strip()
    except Exception as e:
        logger.error(f"日文轉注音失敗: {e}"); return ""

def korean_to_bopomofo(text: str) -> str:
    if not HANGUL_JAMO_ENABLED: return ""
    try:
        decomposed, bopomofo_sentence = decompose(text), []
        for char in decomposed: bopomofo_sentence.append(KOREAN_BOPOMOFO_MAP.get(char, char))
        result = "".join(bopomofo_sentence)
        return re.sub(r'([ㄍㄋㄉㄌㄇㄅㄙㄗㄘㄎㄊㄆㄏ][ㄚㄛㄜㄝㄧㄨㄩ]+[ㄍㄣㄉㄌㄇㄅㄥ]?)', r'\1 ', result).strip()
    except Exception as e:
        logger.error(f"韓文轉注音失敗: {e}"); return ""

def get_phonetic_guides(text: str, target_language: str) -> Dict[str, str]:
    guides = {}
    if target_language == "日文":
        if KAKASI_ENABLED:
            try:
                kks, result = pykakasi.kakasi(), []
                for item in kks.convert(text): result.append(item.get('romaji', item['orig']))
                guides['romaji'] = ''.join(result)
                guides['bopomofo'] = japanese_to_bopomofo(text)
            except Exception as e: logger.error(f"日文發音處理失敗: {e}")
    elif target_language == "韓文":
        if KOREAN_ROMANIZER_ENABLED:
            try:
                # 🔥 核心修正: 使用 .romanize() 而不是 .run()
                guides['romaji'] = Romanizer(text).romanize()
            except Exception as e: logger.error(f"韓文羅馬拼音處理失敗: {e}")
        if HANGUL_JAMO_ENABLED:
            guides['bopomofo'] = korean_to_bopomofo(text)
    elif target_language in ["繁體中文", "簡體中文"] and PINYIN_ENABLED:
        try:
            guides['pinyin'] = ' '.join(p[0] for p in pinyin(text, style=Style.NORMAL))
            guides['bopomofo'] = ' '.join(p[0] for p in pinyin(text, style=Style.BOPOMOFO))
        except Exception as e: logger.error(f"中文發音處理失敗: {e}")
    return guides

async def groq_chat_completion(messages, max_tokens=600, temperature=0.7):
    # (此函式與前版相同)
    pass

async def translate_text(text: str, target_language: str) -> str:
    # (此函式與前版相同)
    pass

def get_chat_id(event: MessageEvent) -> str:
    # (此函式與前版相同)
    pass

def reply_simple(reply_token, text, is_group=False, bot_name="AI助手"):
    # (此函式與前版相同)
    pass

# (所有 build_flex_menu, build_quick_reply_items, persona 相關函式都與前版相同，此處省略)

# ============================================
# 4. LINE Webhook 處理器 (Webhook Handlers)
# ============================================
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    user_id, chat_id = event.source.user_id, get_chat_id(event)
    msg, reply_token = event.message.text.strip(), event.reply_token
    is_group = isinstance(event.source, (SourceGroup, SourceRoom))
    try: bot_name = line_bot_api.get_bot_info().display_name
    except: bot_name = "AI助手"

    if chat_id not in auto_reply_status: auto_reply_status[chat_id] = True
    low = msg.lower()
    if is_group and not auto_reply_status.get(chat_id, True) and not msg.startswith(f"@{bot_name}"): return
    if msg.startswith(f"@{bot_name}"):
        msg, low = msg[len(f"@{bot_name}"):].strip(), msg.lower()

    # 指令處理...
    if msg == "開啟自動回答":
        auto_reply_status[chat_id] = True; return reply_simple(reply_token, "✅ 已開啟自動回答模式", is_group, bot_name)
    elif msg == "關閉自動回答":
        auto_reply_status[chat_id] = False; return reply_simple(reply_token, "❌ 已關閉自動回答模式", is_group, bot_name)
    
    # ... 其他指令與AI聊天邏輯 (與前版相同)
    if chat_id in translation_states:
        if not msg: return
        target_lang = translation_states[chat_id]
        translated_text = asyncio.run(translate_text(msg, target_lang))
        guides = get_phonetic_guides(translated_text, target_lang)
        
        final_reply = f"🌐 翻譯結果 ({target_lang})：\n\n"
        if target_lang in ["日文", "韓文"]:
            display_text = translated_text
            if guides.get('romaji'): display_text += f" (羅馬拼音: {guides['romaji']})"
            if guides.get('bopomofo'): display_text += f" (ㄅㄆㄇ: {guides['bopomofo']})"
            final_reply += display_text
        elif target_lang in ["繁體中文", "簡體中文"]:
            final_reply += translated_text
            phonetic_parts = []
            if guides.get('pinyin'): phonetic_parts.append(f"漢語拼音: {guides['pinyin']}")
            if guides.get('bopomofo'): phonetic_parts.append(f"注音(ㄅㄆㄇ): {guides['bopomofo']}")
            if phonetic_parts: final_reply += f"\n\n( {', '.join(phonetic_parts)} )"
        else:
            final_reply += translated_text
        return reply_simple(reply_token, final_reply, is_group, bot_name)

    # (此處省略了其餘未修改的指令判斷、AI聊天等邏輯)
    reply_text = "抱歉，我現在有點忙，請稍後再試試 💔" # Fallback
    line_bot_api.reply_message(reply_token, TextSendMessage(text=reply_text))


@handler.add(PostbackEvent)
def handle_postback(event):
    # (此函式與前版相同)
    pass

# ============================================
# 5. FastAPI 路由定義 (Routes)
# ============================================
@router.post("/callback")
async def callback(request: Request):
    body = await request.body(); signature = request.headers.get("X-Line-Signature", "")
    try: await run_in_threadpool(handler.handle, body.decode("utf-8"), signature)
    except InvalidSignatureError: raise HTTPException(400, "Invalid signature")
    return JSONResponse({"message": "ok"})

@router.get("/healthz")
async def health_check(): return {"status": "ok"}

@router.get("/")
async def root(): return {"message": "Line Bot Service is live.", "version": "1.0.0"}

# ============================================
# 6. 應用程式掛載 (App Mounting)
# ============================================
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)