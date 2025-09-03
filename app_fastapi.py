"""
aibot FastAPI 應用程序初始化 (v29 - 優化翻譯指令以應對模型不穩定)
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
router = APIRouter()

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

try:
    from my_commands.lottery_gpt import lottery_gpt
except ImportError:
    def lottery_gpt(msg): return "彩票功能暫時不可用"
try:
    from my_commands.gold_gpt import gold_gpt
except ImportError:
    def gold_gpt(): return "金價功能暫時不可用"
try:
    from my_commands.stock.stock_gpt import stock_gpt
except ImportError:
    def stock_gpt(code): return f"{code}股票功能暫時不可用"

LANGUAGE_MAP = {
    "英文": "English", "日文": "Japanese", "韓文": "Korean",
    "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"
}

# 全域發音映射表與人設
ROMAJI_BOPOMOFO_MAP = {'a': 'ㄚ', 'i': 'ㄧ', 'u': 'ㄨ', 'e': 'ㄝ', 'o': 'ㄛ', 'ka': 'ㄎㄚ', 'ki': 'ㄎㄧ', 'ku': 'ㄎㄨ', 'ke': 'ㄎㄝ', 'ko': 'ㄎㄛ', 'sa': 'ㄙㄚ', 'shi': 'ㄒㄧ', 'su': 'ㄙㄨ', 'se': 'ㄙㄝ', 'so': 'ㄙㄛ', 'ta': 'ㄊㄚ', 'chi': 'ㄑㄧ', 'tsu': 'ㄘㄨ', 'te': 'ㄊㄝ', 'to': 'ㄊㄛ', 'na': 'ㄋㄚ', 'ni': 'ㄋㄧ', 'nu': 'ㄋㄨ', 'ne': 'ㄋㄝ', 'no': 'ㄋㄛ', 'ha': 'ㄏㄚ', 'hi': 'ㄏㄧ', 'fu': 'ㄈㄨ', 'he': 'ㄏㄝ', 'ho': 'ㄏㄛ', 'ma': 'ㄇㄚ', 'mi': 'ㄇㄧ', 'mu': 'ㄇㄨ', 'me': 'ㄇㄝ', 'mo': 'ㄇㄛ', 'ya': 'ㄧㄚ', 'yu': 'ㄧㄨ', 'yo': 'ㄧㄛ', 'ra': 'ㄌㄚ', 'ri': 'ㄌㄧ', 'ru': 'ㄌㄨ', 're': 'ㄌㄝ', 'ro': 'ㄌㄛ', 'wa': 'ㄨㄚ', 'wo': 'ㄛ', 'n': 'ㄣ', 'ga': 'ㄍㄚ', 'gi': 'ㄍㄧ', 'gu': 'ㄍㄨ', 'ge': 'ㄍㄝ', 'go': 'ㄍㄛ', 'za': 'ㄗㄚ', 'ji': 'ㄐㄧ', 'zu': 'ㄗㄨ', 'ze': 'ㄗㄝ', 'zo': 'ㄗㄛ', 'da': 'ㄉㄚ', 'di': 'ㄉㄧ', 'dzu': 'ㄉㄨ', 'de': 'ㄉㄝ', 'do': 'ㄉㄛ', 'ba': 'ㄅㄚ', 'bi': 'ㄅㄧ', 'bu': 'ㄅㄨ', 'be': 'ㄅㄝ', 'bo': 'ㄅㄛ', 'pa': 'ㄆㄚ', 'pi': 'ㄆㄧ', 'pu': 'ㄆㄨ', 'pe': 'ㄆㄝ', 'po': 'ㄆㄛ', 'kya': 'ㄎㄧㄚ', 'kyu': 'ㄎㄧㄨ', 'kyo': 'ㄎㄧㄛ', 'sha': 'ㄕㄚ', 'shu': 'ㄕㄨ', 'sho': 'ㄕㄛ', 'cha': 'ㄑㄚ', 'chu': 'ㄑㄨ', 'cho': 'ㄑㄛ', 'nya': 'ㄋㄧㄚ', 'nyu': 'ㄋㄧㄨ', 'nyo': 'ㄋㄧㄛ', 'hya': 'ㄏㄧㄚ', 'hyu': 'ㄏㄧㄨ', 'hyo': 'ㄏㄧㄛ', 'mya': 'ㄇㄧㄚ', 'myu': 'ㄇㄧㄨ', 'myo': 'ㄇㄧㄛ', 'rya': 'ㄌㄧㄚ', 'ryu': 'ㄌㄧㄨ', 'ryo': 'ㄌㄧㄛ', 'gya': 'ㄍㄧㄚ', 'gyu': 'ㄍㄧㄨ', 'gyo': 'ㄍㄧㄛ', 'ja': 'ㄐㄧㄚ', 'ju': 'ㄐㄧㄨ', 'jo': 'ㄐㄧㄛ', 'bya': 'ㄅㄧㄚ', 'byu': 'ㄅㄧㄨ', 'byo': 'ㄅㄧㄛ', 'pya': 'ㄆㄧㄚ', 'pyu': 'ㄆㄧㄨ', 'pyo': 'ㄆㄧㄛ'}
KOREAN_BOPOMOFO_MAP = { 'ㄱ': 'ㄍ', 'ㄲ': 'ㄍ', 'ㄴ': 'ㄋ', 'ㄷ': 'ㄉ', 'ㄸ': 'ㄉ', 'ㄹ': 'ㄌ', 'ㅁ': 'ㄇ', 'ㅂ': 'ㄅ', 'ㅃ': 'ㄅ', 'ㅅ': 'ㄙ', 'ㅆ': 'ㄙ', 'ㅇ': '', 'ㅈ': 'ㄗ', 'ㅉ': 'ㄗ', 'ㅊ': 'ㄘ', 'ㅋ': 'ㄎ', 'ㅌ': 'ㄊ', 'ㅍ': 'ㄆ', 'ㅎ': 'ㄏ', 'ㅏ': 'ㄚ', 'ㅐ': 'ㄝ', 'ㅑ': 'ㄧㄚ', 'ㅒ': 'ㄧㄝ', 'ㅓ': 'ㄛ', 'ㅔ': 'ㄝ', 'ㅕ': 'ㄧㄛ', 'ㅖ': 'ㄧㄝ', 'ㅗ': 'ㄛ', 'ㅘ': 'ㄨㄚ', 'ㅙ': 'ㄨㄝ', 'ㅚ': 'ㄨㄝ', 'ㅛ': 'ㄧㄛ', 'ㅜ': 'ㄨ', 'ㅝ': 'ㄨㄛ', 'ㅞ': 'ㄨㄝ', 'ㅟ': 'ㄨㄧ', 'ㅠ': 'ㄧㄨ', 'ㅡ': 'ㄜ', 'ㅢ': 'ㅢ', 'ㅣ': 'ㄧ', 'ㄳ': 'ㄍ', 'ㄵ': 'ㄣ', 'ㄶ': 'ㄣ', 'ㄺ': 'ㄌ', 'ㄻ': 'ㄌ', 'ㄼ': 'ㄌ', 'ㄽ': 'ㄌ', 'ㄾ': 'ㄌ', 'ㄿ': 'ㄌ', 'ㅀ': 'ㄌ', 'ㅄ': 'ㄅ' }
VIET_CHAR_DECOMPOSE = {'ă': ('a', ''), 'â': ('a', ''), 'ê': ('e', ''), 'ô': ('o', ''), 'ơ': ('o', ''), 'ư': ('u', ''), 'à': ('a', 'huyền'), 'ằ': ('a', 'huyền'), 'ầ': ('a', 'huyền'), 'è': ('e', 'huyền'), 'ề': ('e', 'huyền'), 'ì': ('i', 'huyền'), 'ò': ('o', 'huyền'), 'ồ': ('o', 'huyền'), 'ờ': ('o', 'huyền'), 'ù': ('u', 'huyền'), 'ừ': ('u', 'huyền'), 'ỳ': ('y', 'huyền'), 'á': ('a', 'sắc'), 'ắ': ('a', 'sắc'), 'ấ': ('a', 'sắc'), 'é': ('e', 'sắc'), 'ế': ('e', 'sắc'), 'í': ('i', 'sắc'), 'ó': ('o', 'sắc'), 'ố': ('o', 'sắc'), 'ớ': ('o', 'sắc'), 'ú': ('u', 'sắc'), 'ứ': ('u', 'sắc'), 'ý': ('y', 'sắc'), 'ả': ('a', 'hỏi'), 'ẳ': ('a', 'hỏi'), 'ẩ': ('a', 'hỏi'), 'ẻ': ('e', 'hỏi'), 'ể': ('e', 'hỏi'), 'ỉ': ('i', 'hỏi'), 'ỏ': ('o', 'hỏi'), 'ổ': ('o', 'hỏi'), 'ở': ('o', 'hỏi'), 'ủ': ('u', 'hỏi'), 'ử': ('u', 'hỏi'), 'ỷ': ('y', 'hỏi'), 'ã': ('a', 'ngã'), 'ẵ': ('a', 'ngã'), 'ẫ': ('a', 'ngã'), 'ẽ': ('e', 'ngã'), 'ễ': ('e', 'ngã'), 'ĩ': ('i', 'ngã'), 'õ': ('o', 'ngã'), 'ỗ': ('o', 'ngã'), 'ỡ': ('o', 'ngã'), 'ũ': ('u', 'ngã'), 'ữ': ('u', 'ngã'), 'ỹ': ('y', 'ngã'), 'ạ': ('a', 'nặng'), 'ặ': ('a', 'nặng'), 'ậ': ('a', 'nặng'), 'ẹ': ('e', 'nặng'), 'ệ': ('e', 'nặng'), 'ị': ('i', 'nặng'), 'ọ': ('o', 'nặng'), 'ộ': ('o', 'nặng'), 'ợ': ('o', 'nặng'), 'ụ': ('u', 'nặng'), 'ự': ('u', 'nặng'), 'ỵ': ('y', 'nặng')}
VIET_TONE_BOPOMOFO = {'ngang': '', 'huyền': 'ˋ', 'sắc': 'ˊ', 'hỏi': 'ˇ', 'ngã': 'ˇ', 'nặng': '˙'}
VIET_TONE_SEPARATOR = {'ngang': '/', 'huyền': '＼/', 'sắc': '／/', 'hỏi': '?/', 'ngã': '~/', 'nặng': './'}
VIET_VOWELS = {'a': 'ㄚ', 'ă': 'ㄚ', 'â': 'ㄜ', 'e': 'ㄝ', 'ê': 'ㄝ', 'i': 'ㄧ', 'y': 'ㄧ', 'o': 'ㄛ', 'ô': 'ㄛ', 'ơ': 'ㄜ', 'u': 'ㄨ', 'ư': 'ư', 'ia': 'ㄧㄚ', 'ya': 'ㄧㄚ', 'iê': 'ㄧㄝ', 'yê': 'ㄧㄝ', 'ua': 'ㄨㄚ', 'uô': 'ㄨㄛ', 'ưa': 'ưa', 'ươ': 'ươ', 'uy': 'ㄨㄧ', 'uâ': 'ㄨㄜ', 'ua': 'ㄨㄚ', 'oă': 'ㄨㄚ', 'oe': 'ㄨㄝ', 'ay': 'ㄞ', 'ây': 'ㄟ', 'ao': 'ㄠ', 'au': 'ㄠ', 'âu': 'ㄡ', 'eo': 'ㄝㄛ', 'êu': 'ㄧㄡ', 'iu': 'ㄧㄨ', 'oi': 'ㄛㄧ', 'ôi': 'ㄛㄧ', 'ơi': 'ㄜㄧ', 'ui': 'ㄨㄧ', 'ưi': 'ưi', 'uyu': 'ㄨㄧㄨ', 'oai': 'ㄨㄞ', 'oay': 'ㄨㄞ', 'uây': 'ㄨㄟ', 'ươi': 'ươi'}
VIET_INITIALS = {'b': 'ㄅ', 'ch': 'ㄑ', 'd': 'ㄗ', 'đ': 'ㄉ', 'g': 'ㄍ', 'gh': 'ㄍ', 'h': 'ㄏ', 'k': 'ㄍ', 'kh': 'ㄎ', 'l': 'ㄌ', 'm': 'ㄇ', 'n': 'ㄋ', 'nh': 'ㄋ', 'p': 'ㄆ', 'ph': 'ㄈ', 'qu': 'ㄎㄨ', 'r': 'ㄖ', 's': 'ㄕ', 't': 'ㄊ', 'th': 'ㄊ', 'tr': 'ㄐ', 'v': 'ㄨ', 'x': 'ㄒ'}
VIET_FINALS = {'c': 'ㄍ', 'ch': 'ㄍ', 'm': 'ㄇ', 'n': 'ㄣ', 'ng': 'ㄥ', 'nh': 'ㄣ', 'p': 'ㄅ', 't': 'ㄉ', 'u': 'ㄨ', 'y': 'ㄧ', 'i': 'ㄧ', 'o': 'ㄛ'}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼，總是對你充滿耐心，用鼓勵和安慰的話語溫暖你的心。", "greetings": "親愛的，你來啦～今天過得好嗎？我在這聽你說喔 🌸", "emoji": "🌸💕😊🥰"},
    "salty": {"title": "傲嬌女友", "style": "毒舌、傲嬌，表面上會吐槽你，但字裡行間卻流露出不經意的關心。", "greetings": "哼，還知道要來找我啊？說吧，又遇到什麼麻煩事了。😏", "emoji": "😏😒🙄"},
    "moe": {"title": "萌系女友", "style": "充滿動漫風格，大量使用顏文字和可愛的語氣詞，元氣滿滿地陪伴你 (๑•̀ㅂ•́)و✧", "greetings": "主人～歡迎回來！(ﾉ>ω<)ﾉ ✨ 有沒有想我呀？", "emoji": "✨🎀(ﾉ>ω<)ﾉ⭐"},
    "cool": {"title": "酷系御姐", "style": "冷靜、成熟又可靠的御姐，總能一針見血地分析問題，並給你專業又犀利的建議。", "greetings": "我在。需要建議嗎？直接說重點。", "emoji": "🧊⚡️🖤"}
}

# ============================================
# 3. 輔助函式 (Helper Functions)
# ============================================
async def update_line_webhook(client: httpx.AsyncClient):
    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
    json_data = {"endpoint": f"{BASE_URL}/callback"}
    res = await client.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=json_data, timeout=10.0)
    res.raise_for_status(); logger.info(f"✅ Webhook 更新成功: {res.status_code}")

def to_camel_case(s: str) -> str:
    return ''.join(word.capitalize() for word in s.split())

def japanese_to_bopomofo(text: str) -> str:
    if not KAKASI_ENABLED: return ""
    try:
        bopomofo_str, i = "", 0
        while i < len(text):
            match = next((text[i:i+l] for l in (3, 2, 1) if text[i:i+l] in ROMAJI_BOPOMOFO_MAP), None)
            if match: bopomofo_str += ROMAJI_BOPOMOFO_MAP[match]; i += len(match)
            else: bopomofo_str += text[i]; i += 1
        return bopomofo_str
    except Exception as e: logger.error(f"日文羅馬拼音轉注音失敗: {e}"); return ""

def korean_to_bopomofo(text: str) -> str:
    if not HANGUL_JAMO_ENABLED: return ""
    try: return "".join([KOREAN_BOPOMOFO_MAP.get(char, char) for char in decompose(text)])
    except Exception as e: logger.error(f"韓文轉注音失敗: {e}"); return ""

def vietnamese_to_bopomofo(text: str) -> str:
    words = text.lower().split()
    output_parts = []
    
    for i, word in enumerate(words):
        original_word = word
        base_word, tone = "", 'ngang'
        for char in word:
            if char in VIET_CHAR_DECOMPOSE:
                base_char, tone_name = VIET_CHAR_DECOMPOSE[char]
                base_word += base_char
                if tone_name: tone = tone_name
            else: base_word += char
        
        bopomofo_syllable, initial, final, vowel = "", "", "", ""
        for l in (3, 2, 1):
            if base_word.startswith(tuple(VIET_INITIALS.keys())) and base_word[0:l] in VIET_INITIALS:
                initial = VIET_INITIALS[base_word[0:l]]; base_word = base_word[l:]; break
        for l in (2, 1):
             if base_word.endswith(tuple(VIET_FINALS.keys())) and base_word[-l:] in VIET_FINALS:
                 final = VIET_FINALS[base_word[-l:]]; base_word = base_word[:-l]; break
        vowel = VIET_VOWELS.get(base_word, "")
        bopomofo_syllable = initial + vowel + final + VIET_TONE_BOPOMOFO[tone]
        
        separator = ""
        if i < len(words) - 1:
            next_word, next_tone = words[i+1], 'ngang'
            for char in next_word:
                if char in VIET_CHAR_DECOMPOSE:
                    _, tone_name = VIET_CHAR_DECOMPOSE[char]
                    if tone_name: next_tone = tone_name; break
            separator = VIET_TONE_SEPARATOR.get(next_tone, '/')
        output_parts.append(bopomofo_syllable + separator)
    return ''.join(output_parts).strip('/')

def get_phonetic_guides(text: str, target_language: str) -> Dict[str, str]:
    guides = {}
    if target_language == "日文" and KAKASI_ENABLED:
        try:
            kks = pykakasi.kakasi(); result = kks.convert(text)
            romaji_parts = []; bopomofo_parts = []
            for item in result:
                if item['hepburn'].isalpha():
                    romaji_parts.append(item['hepburn'])
                    bopomofo_parts.append(japanese_to_bopomofo(item['hepburn']))
            guides['romaji'] = ','.join(p.capitalize() for p in romaji_parts)
            guides['bopomofo'] = '/'.join(bopomofo_parts)
        except Exception as e: logger.error(f"日文發音處理失敗: {e}")
    elif target_language == "韓文":
        if KOREAN_ROMANIZER_ENABLED:
            try:
                romaji_text = Romanizer(text).romanize()
                guides['romaji'] = ','.join(p.capitalize() for p in romaji_text.split())
            except Exception as e: logger.error(f"韓文羅馬拼音處理失敗: {e}")
        if HANGUL_JAMO_ENABLED: guides['bopomofo'] = korean_to_bopomofo(text)
    elif target_language == "越南文":
        guides['romaji'] = text 
        guides['bopomofo'] = vietnamese_to_bopomofo(text)
    elif target_language in ["繁體中文", "簡體中文"] and PINYIN_ENABLED:
        try:
            pinyin_full = ' '.join(p[0] for p in pinyin(text, style=Style.NORMAL))
            bopomofo_full = ' '.join(p[0] for p in pinyin(text, style=Style.BOPOMOFO))
            guides['pinyin'] = to_camel_case(pinyin_full); guides['bopomofo'] = bopomofo_full
        except Exception as e: logger.error(f"中文發音處理失敗: {e}")
    return guides

async def groq_chat_completion(messages, max_tokens=600, temperature=0.7):
    try:
        response = await groq_client.chat.completions.create(model=GROQ_MODEL_PRIMARY, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API 呼叫失敗: {e}"); response = await groq_client.chat.completions.create(model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return response.choices[0].message.content.strip()

# <--- 修改點: 使用更穩定、更結構化的指令格式來要求AI進行翻譯
async def translate_text(text: str, target_language: str) -> str:
    system_prompt = f"""You are a professional translation engine.
Translate the user's text from the source language to the target language specified in the JSON block.
Output *only* the translated text itself, without any other explanation."""
    
    user_prompt = f"""
{{
  "source_language": "auto-detect",
  "target_language": "{target_language}",
  "text_to_translate": "{text}"
}}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return await groq_chat_completion(messages, 800, 0.3)


async def analyze_sentiment(text: str) -> str:
    messages = [{"role": "system", "content": "Analyze the sentiment of the user's message. Respond with only one of the following: positive, neutral, negative, angry, sad, happy."}, {"role": "user", "content": text}]
    result = await groq_chat_completion(messages, 20, 0); return (result or "neutral").strip().lower()

def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom): return event.source.room_id
    return event.source.user_id

def build_quick_reply_items(is_group: bool, bot_name: str) -> List[QuickReplyButton]:
    return [
        QuickReplyButton(action=MessageAction(label="🌸 甜", text="甜")), QuickReplyButton(action=MessageAction(label="😏 鹹", text="鹹")),
        QuickReplyButton(action=MessageAction(label="🎀 萌", text="萌")), QuickReplyButton(action=MessageAction(label="🧊 酷", text="酷")),
        QuickReplyButton(action=MessageAction(label="💖 人設選單", text="我的人設")), QuickReplyButton(action=MessageAction(label="💰 金融選單", text="金融選單")),
        QuickReplyButton(action=MessageAction(label="🎰 彩票選單", text="彩票選單")), QuickReplyButton(action=MessageAction(label="🌐 翻譯選單", text="翻譯選單")),
        QuickReplyButton(action=MessageAction(label="✅ 開啟自動回答", text="開啟自動回答")), QuickReplyButton(action=MessageAction(label="❌ 關閉自動回答", text="關閉自動回答"))
    ]

def build_flex_menu(title: str, subtitle: str, actions: List[MessageAction]) -> FlexSendMessage:
    buttons = [ButtonComponent(style="primary", height="sm", action=act, margin="md", color="#00B900") for act in actions]
    bubble = BubbleContainer(
        header=BoxComponent(layout="vertical", contents=[TextComponent(text=title, weight="bold", size="xl", color="#000000", align="center"), TextComponent(text=subtitle, size="sm", color="#666666", wrap=True, align="center", margin="md")], backgroundColor="#FFFFFF"), 
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm", paddingAll="12px", backgroundColor="#FAFAFA")
    )
    return FlexSendMessage(alt_text=title, contents=bubble)

def flex_menu_finance(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    actions = [
        MessageAction(label="🇹🇼 台股大盤", text=f"{prefix}台股大盤"),
        MessageAction(label="🇺🇸 美股大盤", text=f"{prefix}美股大盤"),
        MessageAction(label="💰 金價查詢", text=f"{prefix}金價"),
        MessageAction(label="💴 日元匯率", text=f"{prefix}JPY"),
        MessageAction(label="📊 查詢個股 (例: 2330)", text=f"{prefix}2330")
    ]
    return build_flex_menu("💰 金融服務", "快速查詢最新金融資訊", actions)
def flex_menu_lottery(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""; actions = [MessageAction(label="🎰 大樂透", text=f"{prefix}大樂透"), MessageAction(label="🎯 威力彩", text=f"{prefix}威力彩"), MessageAction(label="🔢 539", text=f"{prefix}539")]; return build_flex_menu("🎰 彩票服務", "最新開獎資訊", actions)

def flex_menu_translate() -> FlexSendMessage:
    actions = [
        MessageAction(label="🇺🇸 翻英文", text="翻譯->英文"), 
        MessageAction(label="🇻🇳 翻越南文", text="翻譯->越南文"),
        MessageAction(label="🇯🇵 翻日文", text="翻譯->日文"), 
        MessageAction(label="🇰🇷 翻韓文", text="翻譯->韓文"), 
        MessageAction(label="🇹🇼 翻繁體中文", text="翻譯->繁體中文"), 
        MessageAction(label="❌ 結束翻譯", text="翻譯->結束")
    ]
    return build_flex_menu("🌐 翻譯選擇", "選擇目標語言", actions)

def flex_menu_persona() -> FlexSendMessage:
    actions = [MessageAction(label="🌸 甜美女友", text="甜"), MessageAction(label="😏 傲嬌女友", text="鹹"), MessageAction(label="🎀 萌系女友", text="萌"), MessageAction(label="🧊 酷系御姐", text="酷"), MessageAction(label="🎲 隨機人設", text="random")]; return build_flex_menu("💖 人設選擇", "切換 AI 女友的說話風格", actions)

def get_persona_info(chat_id: str) -> str:
    p_key = user_persona.get(chat_id, "sweet"); p = PERSONAS[p_key]; return f"💖 當前聊天室人設：{p['title']}\n\n【特質】{p['style']}\n\n{p['greetings']}"
def set_user_persona(chat_id: str, key: str):
    if key == "random": key = random.choice(list(PERSONAS.keys()))
    elif key not in PERSONAS: key = "sweet"
    user_persona[chat_id] = key; return key
def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    p_key = user_persona.get(chat_id, "sweet"); p = PERSONAS[p_key]; emotion_guide = {"positive": "對方心情不錯，可以更活潑一點回應", "happy": "對方很開心，一起分享這份喜悦", "neutral": "正常聊天模式", "negative": "對方情緒低落，給予安慰和鼓勵", "sad": "對方很難過，溫柔陪伴和安慰", "angry": "對方生氣了，冷靜傾聽並安撫情緒"}; emotion_tip = emotion_guide.get(sentiment, "正常聊天模式"); return f"你是一位「{p['title']}」AI女友。你的角色特質是「{p['style']}」。根據使用者當前情緒「{sentiment}」，你應該「{emotion_tip}」。請用繁體中文、簡潔且帶有「{p['emoji']}」風格的表情符號來回應。"

def reply_simple(reply_token, text, is_group, bot_name):
    try:
        quick_items = build_quick_reply_items(is_group, bot_name)
        message = TextSendMessage(text=text, quick_reply=QuickReply(items=quick_items))
        line_bot_api.reply_message(reply_token, message)
    except LineBotApiError as e: logger.error(f"Reply 訊息失敗: {e}")

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

    if not msg: return
    if chat_id not in auto_reply_status: auto_reply_status[chat_id] = True
    
    low = msg.lower()
    if is_group and not auto_reply_status.get(chat_id, True) and not msg.startswith(f"@{bot_name}"): return
    if msg.startswith(f"@{bot_name}"): msg, low = msg[len(f"@{bot_name}"):].strip(), low[len(f"@{bot_name}"):].strip()

    if msg == "開啟自動回答": auto_reply_status[chat_id] = True; return reply_simple(reply_token, "✅ 已開啟自動回答模式", is_group, bot_name)
    if msg == "關閉自動回答": auto_reply_status[chat_id] = False; return reply_simple(reply_token, "❌ 已關閉自動回答模式", is_group, bot_name)
    
    menu_map = {'金融選單': flex_menu_finance(bot_name, is_group), '彩票選單': flex_menu_lottery(bot_name, is_group), '翻譯選單': flex_menu_translate(), '我的人設': flex_menu_persona(), '人設選單': flex_menu_persona()}
    if low in menu_map: return line_bot_api.reply_message(reply_token, menu_map[low])
    
    if low.startswith("翻譯->"):
        choice = msg.replace("翻譯->", "").strip()
        if choice == "結束": translation_states.pop(chat_id, None); return reply_simple(reply_token, "✅ 已結束翻譯模式", is_group, bot_name)
        else: translation_states[chat_id] = choice; return reply_simple(reply_token, f"🌐 本聊天室翻譯模式已啟用 -> {choice}", is_group, bot_name)

    if chat_id in translation_states:
        display_lang = translation_states[chat_id]
        prompt_lang = LANGUAGE_MAP.get(display_lang, display_lang)
        
        translated_text = asyncio.run(translate_text(msg, prompt_lang))
        guides = get_phonetic_guides(translated_text, display_lang)
        final_reply = f"🌐 翻譯結果 ({display_lang})：\n\n{translated_text}"
        
        phonetic_parts = []
        if guides.get('romaji'):
            if display_lang == '越南文':
                phonetic_parts.append(f"耳空字: {guides['romaji']}")
            else:
                phonetic_parts.append(f"羅馬拼音: {guides['romaji']}")
        if guides.get('pinyin'): phonetic_parts.append(f"漢語拼音: {guides['pinyin']}")
        if guides.get('bopomofo'):
            if display_lang in ["繁體中文", "簡體中文"]:
                bopomofo_text = '/'.join(guides['bopomofo'].split())
                phonetic_parts.append(f"注音: {bopomofo_text}")
            else:
                phonetic_parts.append(f"注音: {guides['bopomofo']}")

        if phonetic_parts: final_reply += f"\n\n( {', '.join(phonetic_parts)} )"
        return reply_simple(reply_token, final_reply, is_group, bot_name)

    persona_keys = {"甜": "sweet", "鹹": "salty", "萌": "moe", "酷": "cool", "random": "random", "隨機": "random"}
    if low in persona_keys:
        key = set_user_persona(chat_id, persona_keys[low]); p = PERSONAS[key]
        info_text = get_persona_info(chat_id)
        return reply_simple(reply_token, f"💖 已切換人設！\n\n{info_text}", is_group, bot_name)

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

    try:
        history = conversation_history.get(chat_id, []); sentiment = asyncio.run(analyze_sentiment(msg))
        system_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": msg}]
        final_reply = asyncio.run(groq_chat_completion(messages))
        history.extend([{"role": "user", "content": msg}, {"role": "assistant", "content": final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
    except Exception as e:
        logger.error(f"AI 回覆失敗: {e}", exc_info=True); final_reply = "抱歉，我剛剛走神了 😅，可以再說一次嗎？"
    
    return reply_simple(reply_token, final_reply, is_group, bot_name)

@handler.add(PostbackEvent)
def handle_postback(event): pass

# ============================================
# 5. FastAPI 路由定義 (Routes)
# ============================================
@router.post("/callback")
async def callback(request: Request):
    body = await request.body(); signature = request.headers.get("X-Line-Signature", "")
    try: await run_in_threadpool(handler.handle, body.decode("utf-8"), signature)
    except InvalidSignatureError: raise HTTPException(400, "Invalid signature")
    return JSONResponse({"message": "ok"})

@router.get("/")
async def root(): return {"message": "Line Bot Service is live."}
app.include_router(router)