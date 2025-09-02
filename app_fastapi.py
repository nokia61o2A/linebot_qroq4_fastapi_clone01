"""
aibot FastAPI 應用程序初始化 (v11 - Reply-Then-Push 最終穩定版)
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

# 自訂功能模組
try: from my_commands.lottery_gpt import lottery_gpt
except ImportError: def lottery_gpt(msg): return "彩票功能暫時不可用"
try: from my_commands.gold_gpt import gold_gpt
except ImportError: def gold_gpt(): return "金價功能暫時不可用"
try: from my_commands.stock.stock_gpt import stock_gpt
except ImportError: def stock_gpt(code): return f"{code}股票功能暫時不可用"

# 全域發音映射表與人設
ROMAJI_BOPOMOFO_MAP = {'a': 'ㄚ', 'i': 'ㄧ', 'u': 'ㄨ', 'e': 'ㄝ', 'o': 'ㄛ', 'ka': 'ㄎㄚ', 'ki': 'ㄎㄧ', 'ku': 'ㄎㄨ', 'ke': 'ㄎㄝ', 'ko': 'ㄎㄛ', 'sa': 'ㄙㄚ', 'shi': 'ㄒㄧ', 'su': 'ㄙㄨ', 'se': 'ㄙㄝ', 'so': 'ㄙㄛ', 'ta': 'ㄊㄚ', 'chi': 'ㄑㄧ', 'tsu': 'ㄘㄨ', 'te': 'ㄊㄝ', 'to': 'ㄊㄛ', 'na': 'ㄋㄚ', 'ni': 'ㄋㄧ', 'nu': 'ㄋㄨ', 'ne': 'ㄋㄝ', 'no': 'ㄋㄛ', 'ha': 'ㄏㄚ', 'hi': 'ㄏㄧ', 'fu': 'ㄈㄨ', 'he': 'ㄏㄝ', 'ho': 'ㄏㄛ', 'ma': 'ㄇㄚ', 'mi': 'ㄇㄧ', 'mu': 'ㄇㄨ', 'me': 'ㄇㄝ', 'mo': 'ㄇㄛ', 'ya': 'ㄧㄚ', 'yu': 'ㄧㄨ', 'yo': 'ㄧㄛ', 'ra': 'ㄌㄚ', 'ri': 'ㄌㄧ', 'ru': 'ㄌㄨ', 're': 'ㄌㄝ', 'ro': 'ㄌㄛ', 'wa': 'ㄨㄚ', 'wo': 'ㄛ', 'n': 'ㄣ', 'ga': 'ㄍㄚ', 'gi': 'ㄍㄧ', 'gu': 'ㄍㄨ', 'ge': 'ㄍㄝ', 'go': 'ㄍㄛ', 'za': 'ㄗㄚ', 'ji': 'ㄐㄧ', 'zu': 'ㄗㄨ', 'ze': 'ㄗㄝ', 'zo': 'ㄗㄛ', 'da': 'ㄉㄚ', 'di': 'ㄉㄧ', 'dzu': 'ㄉㄨ', 'de': 'ㄉㄝ', 'do': 'ㄉㄛ', 'ba': 'ㄅㄚ', 'bi': 'ㄅㄧ', 'bu': 'ㄅㄨ', 'be': 'ㄅㄝ', 'bo': 'ㄅㄛ', 'pa': 'ㄆㄚ', 'pi': 'ㄆㄧ', 'pu': 'ㄆㄨ', 'pe': 'ㄆㄝ', 'po': 'ㄆㄛ', 'kya': 'ㄎㄧㄚ', 'kyu': 'ㄎㄧㄨ', 'kyo': 'ㄎㄧㄛ', 'sha': 'ㄕㄚ', 'shu': 'ㄕㄨ', 'sho': 'ㄕㄛ', 'cha': 'ㄑㄚ', 'chu': 'ㄑㄨ', 'cho': 'ㄑㄛ', 'nya': 'ㄋㄧㄚ', 'nyu': 'ㄋㄧㄨ', 'nyo': 'ㄋㄧㄛ', 'hya': 'ㄏㄧㄚ', 'hyu': 'ㄏㄧㄨ', 'hyo': 'ㄏㄧㄛ', 'mya': 'ㄇㄧㄚ', 'myu': 'ㄇㄧㄨ', 'myo': 'ㄇㄧㄛ', 'rya': 'ㄌㄧㄚ', 'ryu': 'ㄌㄧㄨ', 'ryo': 'ㄌㄧㄛ', 'gya': 'ㄍㄧㄚ', 'gyu': 'ㄍㄧㄨ', 'gyo': 'ㄍㄧㄛ', 'ja': 'ㄐㄧㄚ', 'ju': 'ㄐㄧㄨ', 'jo': 'ㄐㄧㄛ', 'bya': 'ㄅㄧㄚ', 'byu': 'ㄅㄧㄨ', 'byo': 'ㄅㄧㄛ', 'pya': 'ㄆㄧㄚ', 'pyu': 'ㄆㄧㄨ', 'pyo': 'ㄆㄧㄛ'}
KOREAN_BOPOMOFO_MAP = { 'ㄱ': 'ㄍ', 'ㄲ': 'ㄍ', 'ㄴ': 'ㄋ', 'ㄷ': 'ㄉ', 'ㄸ': 'ㄉ', 'ㄹ': 'ㄌ', 'ㅁ': 'ㄇ', 'ㅂ': 'ㄅ', 'ㅃ': 'ㄅ', 'ㅅ': 'ㄙ', 'ㅆ': 'ㄙ', 'ㅇ': '', 'ㅈ': 'ㄗ', 'ㅉ': 'ㄗ', 'ㅊ': 'ㄘ', 'ㅋ': 'ㄎ', 'ㅌ': 'ㄊ', 'ㅍ': 'ㄆ', 'ㅎ': 'ㄏ', 'ㅏ': 'ㄚ', 'ㅐ': 'ㄝ', 'ㅑ': 'ㄧㄚ', 'ㅒ': 'ㄧㄝ', 'ㅓ': 'ㄛ', 'ㅔ': 'ㄝ', 'ㅕ': 'ㄧㄛ', 'ㅖ': 'ㄧㄝ', 'ㅗ': 'ㄛ', 'ㅘ': 'ㄨㄚ', 'ㅙ': 'ㄨㄝ', 'ㅚ': 'ㄨㄝ', 'ㅛ': 'ㄧㄛ', 'ㅜ': 'ㄨ', 'ㅝ': 'ㄨㄛ', 'ㅞ': 'ㄨㄝ', 'ㅟ': 'ㄨㄧ', 'ㅠ': 'ㄧㄨ', 'ㅡ': 'ㄜ', 'ㅢ': 'ㅢ', 'ㅣ': 'ㄧ', 'ㄳ': 'ㄍ', 'ㄵ': 'ㄣ', 'ㄶ': 'ㄣ', 'ㄺ': 'ㄌ', 'ㄻ': 'ㄌ', 'ㄼ': 'ㄌ', 'ㄽ': 'ㄌ', 'ㄾ': 'ㄌ', 'ㄿ': 'ㄌ', 'ㅀ': 'ㄌ', 'ㅄ': 'ㄅ' }
PERSONAS = {"sweet": {"title": "甜美女友", "style": "溫柔體貼，用詞親暱，會關心對方感受，語調甜美", "greetings": "親愛的～我在這裡陪你呢 🌸💕", "emoji": "🌸💕😊🥰"},"salty": {"title": "傲嬌女友", "style": "表面冷淡實則關心，會吐槽但帶著愛意，有點小壞壞", "greetings": "哼！又來找我了嗎... 😏💋", "emoji": "😏💋🙄😤"},"moe": {"title": "萌系女友", "style": "可愛天真，語尾詞豐富，用詞軟萌，充滿活力", "greetings": "呀呼～！(ﾉ>ω<)ﾉ ✨", "emoji": "✨🎀(ﾉ>ω<)ﾉ🌈"},"cool": {"title": "酷系御姐", "style": "冷靜理性，說話直接，給人可靠感，有領導氣質", "greetings": "我在這裡。需要我幫你分析嗎？ 🧊⚡", "emoji": "🧊⚡💎🖤"},"smart": {"title": "知性學姐", "style": "博學多聞，用詞優雅，喜歡分享知識，有耐心", "greetings": "你好，有什麼我能幫你解答的嗎？📚✨", "emoji": "📚🔍🧠💡"},"cute": {"title": "元氣少女", "style": "活潑開朗，充滿正能量，說話直率，喜歡鼓勵人", "greetings": "嗨嗨！今天也要元氣滿滿哦！💪😄", "emoji": "💪😄🌟⭐"}}

# ============================================
# 3. 輔助函式 (Helper Functions)
# ============================================
async def update_line_webhook(client: httpx.AsyncClient):
    # ... (此函式保持不變)
    headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
    json_data = {"endpoint": f"{BASE_URL}/callback"}
    res = await client.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=json_data, timeout=10.0)
    res.raise_for_status(); logger.info(f"✅ Webhook 更新成功: {res.status_code}")

# ... (所有發音函式 japanese_to_bopomofo, korean_to_bopomofo, get_phonetic_guides 保持不變)
def japanese_to_bopomofo(text: str) -> str:
    if not KAKASI_ENABLED: return ""
    try:
        kks = pykakasi.kakasi(); result = kks.convert(text); romaji = "".join([item.get('romaji', item.get('orig', '')) for item in result])
        bopomofo_str, i = "", 0
        while i < len(romaji):
            match = next((romaji[i:i+l] for l in (3, 2, 1) if romaji[i:i+l] in ROMAJI_BOPOMOFO_MAP), None)
            if match: bopomofo_str += ROMAJI_BOPOMOFO_MAP[match]; i += len(match)
            else: bopomofo_str += romaji[i]; i += 1
        return bopomofo_str
    except Exception as e: logger.error(f"日文轉注音失敗: {e}"); return ""
def korean_to_bopomofo(text: str) -> str:
    if not HANGUL_JAMO_ENABLED: return ""
    try: return "".join([KOREAN_BOPOMOFO_MAP.get(char, char) for char in decompose(text)])
    except Exception as e: logger.error(f"韓文轉注音失敗: {e}"); return ""
def get_phonetic_guides(text: str, target_language: str) -> Dict[str, str]:
    guides = {}
    if target_language == "日文" and KAKASI_ENABLED:
        try: kks = pykakasi.kakasi(); guides['romaji'] = "".join([item['hepburn'] for item in kks.convert(text)]); guides['bopomofo'] = japanese_to_bopomofo(text)
        except Exception as e: logger.error(f"日文發音處理失敗: {e}")
    elif target_language == "韓文":
        if KOREAN_ROMANIZER_ENABLED:
            try: guides['romaji'] = Romanizer(text).romanize()
            except Exception as e: logger.error(f"韓文羅馬拼音處理失敗: {e}")
        if HANGUL_JAMO_ENABLED: guides['bopomofo'] = korean_to_bopomofo(text)
    elif target_language in ["繁體中文", "簡體中文"] and PINYIN_ENABLED:
        try: guides['pinyin'] = ' '.join(p[0] for p in pinyin(text, style=Style.NORMAL)); guides['bopomofo'] = ' '.join(p[0] for p in pinyin(text, style=Style.BOPOMOFO))
        except Exception as e: logger.error(f"中文發音處理失敗: {e}")
    return guides

async def groq_chat_completion(messages, max_tokens=600, temperature=0.7):
    # ... (此函式保持不變)
    try:
        response = await groq_client.chat.completions.create(model=GROQ_MODEL_PRIMARY, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API 呼叫失敗: {e}"); response = await groq_client.chat.com_pletions.create(model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return response.choices[0].message.content.strip()

async def translate_text(text: str, target_language: str) -> str:
    # ... (此函式保持不變)
    messages = [{"role": "system", "content": f"You are a professional translator. Translate the following text to {target_language}. Output only the translated text itself."}, {"role": "user", "content": text}]
    return await groq_chat_completion(messages, 800, 0.3)

async def analyze_sentiment(text: str) -> str:
    # ... (此函式保持不變)
    messages = [{"role": "system", "content": "Analyze the sentiment of the user's message. Respond with only one of the following: positive, neutral, negative, angry, sad, happy."}, {"role": "user", "content": text}]
    result = await groq_chat_completion(messages, 20, 0); return (result or "neutral").strip().lower()

def get_chat_id(event: MessageEvent) -> str:
    # ... (此函式保持不變)
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom): return event.source.room_id
    return event.source.user_id

# ... (所有 Flex Menu 和 Persona 相關的 build/get/set 函式保持不變)
def build_quick_reply_items(is_group: bool, bot_name: str) -> List[QuickReplyButton]:
    return [QuickReplyButton(action=MessageAction(label="💖 我的人設", text="我的人設")), QuickReplyButton(action=MessageAction(label="💰 金融選單", text="金融選單")), QuickReplyButton(action=MessageAction(label="🎰 彩票選單", text="彩票選單")), QuickReplyButton(action=MessageAction(label="🌐 翻譯選單", text="翻譯選單")), QuickReplyButton(action=MessageAction(label="✅ 開啟自動回答", text="開啟自動回答")), QuickReplyButton(action=MessageAction(label="❌ 關閉自動回答", text="關閉自動回答"))]
def build_flex_menu(title: str, subtitle: str, actions: List[MessageAction]) -> FlexSendMessage:
    buttons = [ButtonComponent(style="primary", height="sm", action=act, margin="md", color="#905C44") for act in actions]; bubble = BubbleContainer(header=BoxComponent(layout="vertical", contents=[TextComponent(text=title, weight="bold", size="xl", color="#FFFFFF", align="center"), TextComponent(text=subtitle, size="sm", color="#EEEEEE", wrap=True, align="center", margin="md")], backgroundColor="#FF6B6B"), body=BoxComponent(layout="vertical", contents=buttons, spacing="sm", paddingAll="12px", backgroundColor="#FFF9F2")); return FlexSendMessage(alt_text=title, contents=bubble)
def flex_menu_finance(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""; actions = [MessageAction(label="📈 台股大盤", text=f"{prefix}大盤"), MessageAction(label="💰 金價查詢", text=f"{prefix}金價"), MessageAction(label="💴 日元匯率", text=f"{prefix}JPY")]; return build_flex_menu("💰 金融服務", "快速查詢最新資訊", actions)
def flex_menu_lottery(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""; actions = [MessageAction(label="🎰 大樂透", text=f"{prefix}大樂透"), MessageAction(label="🎯 威力彩", text=f"{prefix}威力彩"), MessageAction(label="🔢 539", text=f"{prefix}539")]; return build_flex_menu("🎰 彩票服務", "最新開獎資訊", actions)
def flex_menu_translate() -> FlexSendMessage:
    actions = [MessageAction(label="🇺🇸 翻英文", text="翻譯->英文"), MessageAction(label="🇹🇼 翻繁體中文", text="翻譯->繁體中文"), MessageAction(label="🇯🇵 翻日文", text="翻譯->日文"), MessageAction(label="🇰🇷 翻韓文", text="翻譯->韓文"), MessageAction(label="❌ 結束翻譯", text="翻譯->結束")]; return build_flex_menu("🌐 翻譯選擇", "選擇目標語言", actions)
def flex_menu_persona() -> FlexSendMessage:
    actions = [MessageAction(label="🌸 甜美女友", text="甜"), MessageAction(label="😏 傲嬌女友", text="鹹"), MessageAction(label="🎲 隨機人設", text="random")]; return build_flex_menu("💖 人設選擇", "切換 AI 女友的說話風格", actions)
def get_persona_info(user_id: str) -> str:
    p_key = user_persona.get(user_id, "sweet"); p = PERSONAS[p_key]; return f"💖 當前人設：{p['title']}\n\n【特質】{p['style']}\n\n{p['greetings']}"
def set_user_persona(user_id: str, key: str):
    if key == "random": key = random.choice(list(PERSONAS.keys()))
    elif key not in PERSONAS: key = "sweet"
    user_persona[user_id] = key; return key
def build_persona_prompt(user_id: str, sentiment: str) -> str:
    p_key = user_persona.get(user_id, "sweet"); p = PERSONAS[p_key]; emotion_guide = {"positive": "對方心情不錯，可以更活潑一點回應", "happy": "對方很開心，一起分享這份喜悦", "neutral": "正常聊天模式", "negative": "對方情緒低落，給予安慰和鼓勵", "sad": "對方很難過，溫柔陪伴和安慰", "angry": "對方生氣了，冷靜傾聽並安撫情緒"}; emotion_tip = emotion_guide.get(sentiment, "正常聊天模式"); return f"你是一位「{p['title']}」AI女友。你的角色特質是「{p['style']}」。根據使用者當前情緒「{sentiment}」，你應該「{emotion_tip}」。請用繁體中文、簡潔且帶有「{p['emoji']}」風格的表情符號來回應。"

# 🔥 核心修正: 新增 Push 訊息的輔助函式
def push_simple(chat_id, text, is_group, bot_name):
    """使用 Push API 發送帶有快速按鈕的訊息"""
    try:
        quick_items = build_quick_reply_items(is_group, bot_name)
        message = TextSendMessage(text=text, quick_reply=QuickReply(items=quick_items))
        line_bot_api.push_message(chat_id, message)
    except LineBotApiError as e:
        logger.error(f"Push 訊息失敗: {e}")

def reply_simple(reply_token, text, is_group, bot_name):
    """使用 Reply API 發送帶有快速按鈕的訊息 (用於快速回應)"""
    try:
        quick_items = build_quick_reply_items(is_group, bot_name)
        message = TextSendMessage(text=text, quick_reply=QuickReply(items=quick_items))
        line_bot_api.reply_message(reply_token, message)
    except LineBotApiError as e:
        logger.error(f"Reply 訊息失敗: {e}")

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

    # --- 系統與選單指令 (快速回應) ---
    if msg == "開啟自動回答": auto_reply_status[chat_id] = True; return reply_simple(reply_token, "✅ 已開啟自動回答模式", is_group, bot_name)
    if msg == "關閉自動回答": auto_reply_status[chat_id] = False; return reply_simple(reply_token, "❌ 已關閉自動回答模式", is_group, bot_name)
    
    menu_map = {'金融選單': flex_menu_finance(bot_name, is_group), '彩票選單': flex_menu_lottery(bot_name, is_group), '翻譯選單': flex_menu_translate(), '人設選單': flex_menu_persona()}
    if low in menu_map: return line_bot_api.reply_message(reply_token, menu_map[low])
    if low in ['我的人設', '當前人設']: return reply_simple(reply_token, get_persona_info(user_id), is_group, bot_name)
    
    if low.startswith("翻譯->"):
        choice = msg.replace("翻譯->", "").strip()
        if choice == "結束": translation_states.pop(chat_id, None); return reply_simple(reply_token, "✅ 已結束翻譯模式", is_group, bot_name)
        else: translation_states[chat_id] = choice; return reply_simple(reply_token, f"🌐 本聊天室翻譯模式已啟用 -> {choice}", is_group, bot_name)

    # 🔥 核心修正: 將耗時操作改為 Reply-Then-Push 模式
    # --- 翻譯模式處理 (耗時) ---
    if chat_id in translation_states:
        # 1. 立即回覆「處理中」
        line_bot_api.reply_message(reply_token, TextSendMessage(text=f"好的，正在為您翻譯成 {translation_states[chat_id]}... ✍️"))
        
        # 2. 執行耗時的翻譯與發音分析
        target_lang = translation_states[chat_id]
        translated_text = asyncio.run(translate_text(msg, target_lang))
        guides = get_phonetic_guides(translated_text, target_lang)

        final_reply = f"🌐 翻譯結果 ({target_lang})：\n\n{translated_text}"
        phonetic_parts = []
        if guides.get('romaji'): phonetic_parts.append(f"羅馬拼音: {guides['romaji']}")
        if guides.get('pinyin'): phonetic_parts.append(f"漢語拼音: {guides['pinyin']}")
        if guides.get('bopomofo'): phonetic_parts.append(f"注音: {guides['bopomofo']}")
        if phonetic_parts: final_reply += f"\n\n( {', '.join(phonetic_parts)} )"
        
        # 3. 推送最終結果
        return push_simple(chat_id, final_reply, is_group, bot_name)

    # --- 人設切換 (快速回應) ---
    persona_keys = {"甜": "sweet", "鹹": "salty", "萌": "moe", "酷": "cool", "smart": "smart", "知性": "smart", "cute": "cute", "元氣": "cute", "random": "random", "隨機": "random"}
    if low in persona_keys:
        key = set_user_persona(user_id, persona_keys[low]); p = PERSONAS[key]
        return reply_simple(reply_token, f"💖 已切換人設：{p['title']}\n{p['greetings']}", is_group, bot_name)

    # --- 功能型指令 (可能耗時，但暫時用同步處理) ---
    reply_text = None
    if any(k in msg for k in ["威力彩", "大樂透", "539"]): reply_text = lottery_gpt(msg)
    elif "金價" in msg or "黃金" in msg: reply_text = gold_gpt()
    elif re.fullmatch(r"(\d{4,6}[A-Za-z]?)|([A-Za-z]{1,5})", msg): reply_text = stock_gpt(msg)
    
    # 如果功能型指令有結果，就直接回覆
    if reply_text is not None:
        return reply_simple(reply_token, reply_text, is_group, bot_name)

    # --- AI 聊天回覆 (耗時) ---
    # 1. 立即回覆「處理中」
    line_bot_api.reply_message(reply_token, TextSendMessage(text="好的，請稍候，我正在思考中... 🤔"))
    
    # 2. 執行耗時的 AI 生成
    try:
        history = conversation_history.get(chat_id, []); sentiment = asyncio.run(analyze_sentiment(msg)); system_prompt = build_persona_prompt(user_id, sentiment)
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": msg}]
        final_reply = asyncio.run(groq_chat_completion(messages))
        history.extend([{"role": "user", "content": msg}, {"role": "assistant", "content": final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
    except Exception as e:
        logger.error(f"AI 回覆失敗: {e}", exc_info=True); final_reply = "抱歉，我剛剛走神了 😅，可以再說一次嗎？"

    # 3. 推送最終結果
    return push_simple(chat_id, final_reply, is_group, bot_name)

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