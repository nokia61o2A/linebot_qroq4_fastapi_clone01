# ========== 1) Imports ==========
import os
import re
import random
import logging
import asyncio
from typing import Dict, List
from contextlib import asynccontextmanager

import httpx
import pandas as pd
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

from groq import AsyncGroq, Groq
import openai

# ========== 2) Setup ==========
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # 新增 OpenAI Key

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY, OPENAI_API_KEY]):
    raise RuntimeError("缺少必要環境變數（BASE_URL / CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET / GROQ_API_KEY / OPENAI_API_KEY）")

line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# 非同步 Groq Client (用於對話)
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
# 同步 Groq/OpenAI Client (用於黃金分析的 threadpool)
sync_groq_client = Groq(api_key=GROQ_API_KEY)
openai.api_key = OPENAI_API_KEY

GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-8b-instant")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-70b-versatile")

# 對話/狀態
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
auto_reply_status: Dict[str, bool] = {}
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}

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

# --- AI 對話相關 ---
async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
    """非同步版本的 Groq 呼叫，用於即時對話"""
    try:
        resp = await async_groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY, messages=messages,
            max_tokens=max_tokens, temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq 主要模型失敗: {e}")
        resp = await async_groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK, messages=messages,
            max_tokens=max_tokens, temperature=temperature
        )
        return resp.choices[0].message.content.strip()

async def analyze_sentiment(text: str) -> str:
    msgs = [
        {"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
        {"role":"user","content":text}
    ]
    out = await groq_chat_async(msgs, max_tokens=10, temperature=0)
    return (out or "neutral").strip().lower()

async def translate_text(text: str, target_lang_display: str) -> str:
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text."
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{text}"}}'
    return await groq_chat_async([{"role":"system","content":sys},{"role":"user","content":usr}], 800, 0.2)

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

# --- 黃金價格分析 (從 gold_gpt.py 整合) ---
def get_gold_analysis_reply(messages):
    """同步版本的 AI 呼叫，用於黃金分析"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages)
        return response.choices[0].message.content
    except openai.OpenAIError as openai_err:
        logger.error(f"OpenAI API 失敗: {openai_err}")
        try:
            response = sync_groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=1000,
                temperature=1.2
            )
            return response.choices[0].message.content
        except Exception as groq_err:
            logger.error(f"Groq API 備援也失敗: {groq_err}")
            return f"抱歉，AI分析師目前連線不穩定，請稍後再試。"

def fetch_and_process_gold_data():
    """抓取並處理台銀黃金牌價年度資料"""
    url = "https://rate.bot.com.tw/gold/chart/year/TWD"
    df_list = pd.read_html(url)
    df = df_list[0]
    df = df[["日期", "本行賣出價格"]].copy()
    df.index = pd.to_datetime(df["日期"], format="%Y/%m/%d")
    df.sort_index(inplace=True)
    return df

def generate_gold_content_msg():
    """生成黃金分析的 AI 提示"""
    gold_prices_df = fetch_and_process_gold_data()
    max_price = gold_prices_df['本行賣出價格'].max()
    min_price = gold_prices_df['本行賣出價格'].min()
    last_price = gold_prices_df['本行賣出價格'].iloc[-1]
    last_date = gold_prices_df.index[-1].strftime("%Y-%m-%d")
    
    # 讓資料更簡潔，只取最近30天
    recent_data = gold_prices_df.tail(30).to_string()

    content_msg = (
        f'你是一位專業的金價分析師，請根據以下近一年的台灣銀行黃金牌價數據(台幣計價)，撰寫一份專業、簡潔且易懂的趨勢分析報告。\n'
        f'--- 資料摘要 ---\n'
        f'最新日期: {last_date}\n'
        f'最新價格: {last_price}\n'
        f'年度最高價: {max_price}\n'
        f'年度最低價: {min_price}\n'
        f'--- 最近30天數據 ---\n'
        f'{recent_data}\n'
        f'--- 分析要求 ---\n'
        f'1. 開頭先明確指出「{last_date} 的最新賣出牌價為 {last_price} 元」。\n'
        f'2. 根據數據分析近一週、近一個月及近一年的價格趨勢（例如：波動、上漲、下跌）。\n'
        f'3. 提及年度高點與低點，並簡單說明其意義。\n'
        f'4. 最後給出一個簡短的總結與後市展望（保持中立客觀，可提及國際情勢影響等因素）。\n'
        f'5. 全程使用繁體中文，語氣專業，結構清晰。'
    )
    return content_msg

def get_gold_analysis():
    """執行完整的黃金分析流程"""
    logger.info("開始執行黃金價格分析...")
    content_msg = generate_gold_content_msg()
    msg = [{
        "role": "system",
        "content": "你是一位專業的金價分析師, 使用以下數據來撰寫專業、簡潔、易懂的分析報告。"
    }, {
        "role": "user",
        "content": content_msg
    }]
    reply_data = get_gold_analysis_reply(msg)
    logger.info("黃金價格分析完成。")
    return reply_data

# --- UI 元件 ---
def make_quick_reply_items(is_group: bool, bot_name: str) -> List[QuickReplyButton]:
    # ... (此函式內容不變)
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
    # ... (此函式內容不變)
    items = make_quick_reply_items(is_group, bot_name)
    msg = TextSendMessage(text=text, quick_reply=QuickReply(items=items))
    line_bot_api.reply_message(reply_token, msg)

def build_flex_menu(title: str, subtitle: str, actions: List[MessageAction]) -> FlexSendMessage:
    # ... (此函式內容不變)
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
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(handle_message_async(event))
    except RuntimeError:
        asyncio.run(handle_message_async(event))

async def handle_message_async(event: MessageEvent):
    user_id, chat_id = event.source.user_id, get_chat_id(event)
    msg_raw: str = event.message.text.strip()
    reply_token = event.reply_token
    is_group = isinstance(event.source, (SourceGroup, SourceRoom))
    try:
        bot_info = await run_in_threadpool(line_bot_api.get_bot_info)
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if not msg_raw: return
    if chat_id not in auto_reply_status: auto_reply_status[chat_id] = True

    if is_group and not auto_reply_status.get(chat_id, True) and not msg_raw.startswith(f"@{bot_name}"):
        return
    msg = msg_raw[len(f"@{bot_name}"):].strip() if msg_raw.startswith(f"@{bot_name}") else msg_raw

    low = msg.lower()

    # --- 功能觸發區 ---
    # 開關自動回答
    if low in ("開啟自動回答", "關閉自動回答"):
        is_on = low == "開啟自動回答"
        auto_reply_status[chat_id] = is_on
        text = "✅ 已開啟自動回答" if is_on else "❌ 已關閉自動回答（群組需 @我 才回）"
        return reply_with_quick_bar(reply_token, text, is_group, bot_name)

    # Flex 選單
    if low in ("金融選單", "彩票選單", "翻譯選單", "我的人設", "人設選單"):
        flex_map = {
            "金融選單": flex_menu_finance(bot_name, is_group),
            "彩票選單": flex_menu_lottery(bot_name, is_group),
            "翻譯選單": flex_menu_translate(),
            "我的人設": flex_menu_persona(),
            "人設選單": flex_menu_persona(),
        }
        flex = flex_map[low]
        tip = TextSendMessage(text="👇 選一個功能開始吧", quick_reply=QuickReply(items=make_quick_reply_items(is_group, bot_name)))
        line_bot_api.reply_message(reply_token, [flex, tip])
        return

    # 翻譯模式
    if low.startswith("翻譯->"):
        lang = msg.split("->", 1)[1].strip()
        if lang == "結束":
            translation_states.pop(chat_id, None)
            return reply_with_quick_bar(reply_token, "✅ 已結束翻譯模式", is_group, bot_name)
        translation_states[chat_id] = lang
        return reply_with_quick_bar(reply_token, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。", is_group, bot_name)

    # 人設切換
    persona_keys = {"甜":"sweet", "鹹":"salty", "萌":"moe", "酷":"cool", "random":"random", "隨機":"random"}
    if low in persona_keys:
        key = set_user_persona(chat_id, persona_keys[low])
        p = PERSONAS[key]
        txt = f"💖 已切換人設：{p['title']}\n\n【特質】{p['style']}\n{p['greetings']}"
        return reply_with_quick_bar(reply_token, txt, is_group, bot_name)
        
    # **【新增】黃金價格分析觸發**
    if low in ("金價", "黃金"):
        try:
            # 先回覆處理中訊息，避免使用者等待過久
            line_bot_api.reply_message(reply_token, TextSendMessage(text="正在為您分析最新金價趨勢，請稍候... 🔍"))
            
            # 使用 threadpool 執行耗時的同步任務
            analysis_report = await run_in_threadpool(get_gold_analysis)
            
            # 分析完成後，用 push_message 將結果傳送給使用者
            # (因為 reply_token 只能用一次，且可能已過期)
            line_bot_api.push_message(chat_id, TextSendMessage(text=analysis_report))
        except Exception as e:
            logger.error(f"黃金分析流程失敗: {e}", exc_info=True)
            line_bot_api.push_message(chat_id, TextSendMessage(text="抱歉，金價分析服務暫時無法使用，請稍後再試。"))
        return

    # --- 模式處理區 ---
    # 翻譯模式處理
    if chat_id in translation_states:
        tgt = translation_states[chat_id]
        try:
            out = await translate_text(msg, tgt)
        except Exception as e:
            logger.error(f"翻譯失敗: {e}", exc_info=True)
            out = "翻譯暫時失效，等我回神再來一次 🙏"
        return reply_with_quick_bar(reply_token, f"🌐 ({tgt})\n{out}", is_group, bot_name)

    # 一般對話（人設 + 情感）
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]
        final_reply = await groq_chat_async(messages)
        
        history.extend([{"role":"user","content":msg}, {"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
    except Exception as e:
        logger.error(f"AI 回覆失敗: {e}", exc_info=True)
        final_reply = "抱歉我剛剛走神了 😅 再說一次讓我補上！"

    return reply_with_quick_bar(reply_token, final_reply, is_group, bot_name)


@handler.add(PostbackEvent)
def handle_postback(event):
    pass

# ========== 6) FastAPI Routes ==========
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
    return JSONResponse({"message":"ok"})

@router.get("/")
async def root():
    return {"message":"Service is live."}

app.include_router(router)

# ========== 7) Local run ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)