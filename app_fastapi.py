# ========== 1) Imports ==========
import os
import re
import random
import logging
import asyncio
import time
from typing import Dict, List
from contextlib import asynccontextmanager

import httpx
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    QuickReply, QuickReplyButton, MessageAction,
    SourceUser, SourceGroup, SourceRoom, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent,
    TextComponent, ButtonComponent
)

from groq import AsyncGroq, Groq
from openai import OpenAI

# ========== 2) Setup ==========
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise RuntimeError("缺少必要環境變數（BASE_URL / CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET / GROQ_API_KEY）")

# OpenAI API Key 改為可選
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None
    logger.warning("未設定 OPENAI_API_KEY，分析將僅使用 Groq 作為後備。")

line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
sync_groq_client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

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

# ========== 4) Currency Analysis Functions ==========
def get_analysis_reply(messages):
    """統一的分析回覆函式，支援 OpenAI 和 Groq"""
    try:
        if not openai_client: 
            raise Exception("OpenAI client not available")
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-1106", 
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as openai_err:
        logger.error(f"OpenAI API 失敗: {openai_err}")
        try:
            response = sync_groq_client.chat.completions.create(
                model=GROQ_MODEL_PRIMARY, 
                messages=messages, 
                max_tokens=1000, 
                temperature=1.2
            )
            return response.choices[0].message.content
        except Exception as groq_err:
            logger.error(f"Groq 主要模型失敗: {groq_err}")
            try:
                response = sync_groq_client.chat.completions.create(
                    model=GROQ_MODEL_FALLBACK, 
                    messages=messages, 
                    max_tokens=1000, 
                    temperature=1.2
                )
                return response.choices[0].message.content
            except Exception as fallback_err:
                logger.error(f"Groq 備援也失敗: {fallback_err}")
                return "抱歉，AI分析師目前連線不穩定，請稍後再試。"

def fetch_currency_rates(kind):
    """擷取匯率資料"""
    url = f"https://rate.bot.com.tw/xrt/quote/day/{kind}"
    
    max_retries = 3
    retry_count = 0
    retry_delay = 2

    while retry_count < max_retries:
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                table = soup.find('table', class_='table table-striped table-bordered table-condensed table-hover')
                if not table:
                    logger.warning(f"找不到 {kind} 匯率資料的表格。")
                    return None

                rows = table.find('tbody').find_all('tr')

                time_rates = []
                spot_rates = []
                selling_rates = []

                for row in rows:
                    columns = row.find_all('td')
                    if len(columns) >= 5:
                        time_rate = columns[0].text.strip()
                        spot_rate = columns[2].text.strip()
                        selling_rate = columns[3].text.strip()

                        time_rates.append(time_rate)
                        spot_rates.append(spot_rate)
                        selling_rates.append(selling_rate)

                df = pd.DataFrame({
                    '日期時間': time_rates,
                    '即期匯率': spot_rates,
                    '本行賣出價格': selling_rates
                })

                return df
            else:
                logger.warning(f"HTTP 請求失敗，狀態碼: {response.status_code}")
                
        except (requests.ConnectionError, requests.Timeout, requests.RequestException) as e:
            logger.error(f"網路連接錯誤 (嘗試 {retry_count+1}/{max_retries}): {str(e)}")
            
        retry_count += 1
        if retry_count < max_retries:
            logger.info(f"等待 {retry_delay} 秒後重試...")
            time.sleep(retry_delay)
            retry_delay *= 2

    logger.error("所有重試都失敗，返回預設值")
    return pd.DataFrame({
        '日期時間': ['N/A'],
        '即期匯率': ['0.0'],
        '本行賣出價格': ['0.0']
    })

def generate_currency_content_msg(kind):
    """生成匯率分析報告內容"""
    money_prices_df = fetch_currency_rates(kind)
    if money_prices_df is None or money_prices_df.empty or money_prices_df['日期時間'].iloc[0] == 'N/A':
        return "無法獲取匯率資料，但服務仍在運行中。請稍後再試。"

    money_prices_df['本行賣出價格'] = pd.to_numeric(money_prices_df['本行賣出價格'], errors='coerce')
    max_price = money_prices_df['本行賣出價格'].max()
    min_price = money_prices_df['本行賣出價格'].min()
    last_date = money_prices_df['日期時間'].iloc[-1]

    content_msg = f'你現在是一位專業的{kind}幣種分析師，使用以下資料來撰寫分析報告：\n'
    content_msg += f'{money_prices_df.tail(30)} 顯示最近的30筆資料，\n'
    content_msg += f'最新日期時間: {last_date}，最高價: {max_price}，最低價: {min_price}。\n'
    content_msg += '請給出完整的趨勢分析報告，顯示每日匯率（日期時間、匯率）（幣種/台幣），使用繁體中文。'

    return content_msg

def get_currency_analysis(kind):
    """取得匯率分析報告"""
    logger.info(f"開始執行{kind}匯率分析...")
    content_msg = generate_currency_content_msg(kind)
    
    if content_msg == "無法獲取匯率資料，但服務仍在運行中。請稍後再試。":
        return content_msg

    msg = [{
        "role": "system",
        "content": f"你現在是一位專業的{kind}幣種分析師，使用以下資料來撰寫分析報告。"
    }, {
        "role": "user",
        "content": content_msg
    }]

    reply_data = get_analysis_reply(msg)
    logger.info(f"{kind}匯率分析完成。")
    return reply_data

def fetch_and_process_gold_data():
    """擷取和處理黃金資料"""
    url = "https://rate.bot.com.tw/gold/chart/year/TWD"
    df_list = pd.read_html(url)
    df = df_list[0]
    df = df[["日期", "本行賣出價格"]].copy()
    df.index = pd.to_datetime(df["日期"], format="%Y/%m/%d")
    df.sort_index(inplace=True)
    return df

def generate_gold_content_msg():
    """生成黃金分析報告內容"""
    gold_prices_df = fetch_and_process_gold_data()
    max_price, min_price = gold_prices_df['本行賣出價格'].max(), gold_prices_df['本行賣出價格'].min()
    last_price = gold_prices_df['本行賣出價格'].iloc[-1]
    last_date = gold_prices_df.index[-1].strftime("%Y-%m-%d")
    recent_data = gold_prices_df.tail(30).to_string()
    return (f'你是一位專業的金價分析師，請根據以下近一年的台灣銀行黃金牌價數據(台幣計價)，撰寫一份專業、簡潔且易懂的趨勢分析報告。\n'
            f'--- 資料摘要 ---\n最新日期: {last_date}\n最新價格: {last_price}\n年度最高價: {max_price}\n年度最低價: {min_price}\n'
            f'--- 最近30天數據 ---\n{recent_data}\n--- 分析要求 ---\n'
            f'1. 開頭先明確指出「{last_date} 的最新賣出牌價為 {last_price} 元」。\n'
            f'2. 根據數據分析近一週、近一個月及近一年的價格趨勢（例如：波動、上漲、下跌）。\n'
            f'3. 提及年度高點與低點，並簡單說明其意義。\n'
            f'4. 最後給出一個簡短的總結與後市展望（保持中立客觀，可提及國際情勢影響等因素）。\n'
            f'5. 全程使用繁體中文，語氣專業，結構清晰。')

def get_gold_analysis():
    """取得黃金分析報告"""
    logger.info("開始執行黃金價格分析...")
    content_msg = generate_gold_content_msg()
    msg = [{"role": "system", "content": "你是一位專業的金價分析師, 使用以下數據來撰寫專業、簡潔、易懂的分析報告。"}, {"role": "user", "content": content_msg}]
    reply_data = get_analysis_reply(msg)
    logger.info("黃金價格分析完成。")
    return reply_data

# ========== 5) Other Helper Functions ==========
def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom):  return event.source.room_id
    return event.source.user_id

async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
    try:
        resp = await async_groq_client.chat.completions.create(model=GROQ_MODEL_PRIMARY, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq 主要模型失敗: {e}")
        resp = await async_groq_client.chat.completions.create(model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return resp.choices[0].message.content.strip()

async def analyze_sentiment(text: str) -> str:
    msgs = [{"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."}, {"role":"user","content":text}]
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
    return (f"你是一位「{p['title']}」。風格：{p['style']}\n"
            f"使用者情緒：{sentiment}；請調整語氣（開心→一起開心；難過/生氣→先共情、安撫再給建議；中性→自然聊天）。\n"
            f"回覆使用繁體中文，精煉自然，帶少量表情 {p['emoji']}。")

def make_quick_reply_items(is_group: bool, bot_name: str) -> List[QuickReplyButton]:
    return [QuickReplyButton(action=MessageAction(label=l, text=t)) for l, t in [("🌸 甜", "甜"), ("😏 鹹", "鹹"), ("🎀 萌", "萌"), ("🧊 酷", "酷"), ("💖 人設選單", "我的人設"), ("💰 金融選單", "金融選單"), ("🎰 彩票選單", "彩票選單"), ("🌐 翻譯選單", "翻譯選單"), ("✅ 開啟自動回答", "開啟自動回答"), ("❌ 關閉自動回答", "關閉自動回答")]]

def reply_with_quick_bar(reply_token: str, text: str, is_group: bool, bot_name: str):
    items = make_quick_reply_items(is_group, bot_name)
    msg = TextSendMessage(text=text, quick_reply=QuickReply(items=items))
    line_bot_api.reply_message(reply_token, msg)

def build_flex_menu(title: str, subtitle: str, actions: List[MessageAction]) -> FlexSendMessage:
    buttons = [ButtonComponent(style="primary", height="sm", action=a, margin="md", color="#00B900") for a in actions]
    bubble = BubbleContainer(header=BoxComponent(layout="vertical", contents=[TextComponent(text=title, weight="bold", size="xl", color="#000000", align="center"), TextComponent(text=subtitle, size="sm", color="#666666", wrap=True, align="center", margin="md")], backgroundColor="#FFFFFF"), body=BoxComponent(layout="vertical", contents=buttons, spacing="sm", paddingAll="12px", backgroundColor="#FAFAFA"))
    return FlexSendMessage(alt_text=title, contents=bubble)

def flex_menu_finance(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    acts = [MessageAction(label=l, text=f"{prefix}{t}") for l, t in [("🇹🇼 台股大盤", "台股大盤"), ("🇺🇸 美股大盤", "美股大盤"), ("💰 金價", "金價"), ("💴 日元", "日元"), ("🇪🇺 歐元", "歐元"), ("🇬🇧 英鎊", "英鎊"), ("📊 個股(例:2330)", "2330")]]
    return build_flex_menu("💰 金融服務", "快速查行情", acts)

def flex_menu_lottery(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    acts = [MessageAction(label=l, text=f"{prefix}{t}") for l, t in [("🎰 大樂透", "大樂透"), ("🎯 威力彩", "威力彩"), ("🔢 539", "539")]]
    return build_flex_menu("🎰 彩票服務", "開獎/趨勢", acts)

def flex_menu_translate() -> FlexSendMessage:
    acts = [MessageAction(label=l, text=t) for l, t in [("🇺🇸 英文", "翻譯->英文"), ("🇯🇵 日文", "翻譯->日文"), ("🇰🇷 韓文", "翻譯->韓文"), ("🇻🇳 越南文", "翻譯->越南文"), ("🇹🇼 繁中", "翻譯->繁體中文"), ("❌ 結束翻譯", "翻譯->結束")]]
    return build_flex_menu("🌐 翻譯選擇", "選擇目標語言", acts)

def flex_menu_persona() -> FlexSendMessage:
    acts = [MessageAction(label=l, text=t) for l, t in [("🌸 甜美女友", "甜"), ("😏 傲嬌女友", "鹹"), ("🎀 萌系女友", "萌"), ("🧊 酷系御姐", "酷"), ("🎲 隨機人設", "random")]]
    return build_flex_menu("💖 人設選擇", "切換 AI 女友風格", acts)

# ========== 6) LINE Handlers ==========
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(handle_message_async(event))
    except RuntimeError:
        asyncio.run(handle_message_async(event))

async def handle_message_async(event: MessageEvent):
    chat_id, msg_raw = get_chat_id(event), event.message.text.strip()
    reply_token, is_group = event.reply_token, not isinstance(event.source, SourceUser)
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
    if not msg: return

    low = msg.lower()

    if low in ("開啟自動回答", "關閉自動回答"):
        is_on = low == "開啟自動回答"
        auto_reply_status[chat_id] = is_on
        text = "✅ 已開啟自動回答" if is_on else "❌ 已關閉自動回答（群組需 @我 才回）"
        return reply_with_quick_bar(reply_token, text, is_group, bot_name)

    if low in ("金融選單", "彩票選單", "翻譯選單", "我的人設", "人設選單"):
        flex_map = {"金融選單": flex_menu_finance(bot_name, is_group), "彩票選單": flex_menu_lottery(bot_name, is_group), "翻譯選單": flex_menu_translate(), "我的人設": flex_menu_persona(), "人設選單": flex_menu_persona()}
        flex = flex_map[low]
        tip = TextSendMessage(text="👇 選一個功能開始吧", quick_reply=QuickReply(items=make_quick_reply_items(is_group, bot_name)))
        return line_bot_api.reply_message(reply_token, [flex, tip])

    if low.startswith("翻譯->"):
        lang = msg.split("->", 1)[1].strip()
        if lang == "結束":
            translation_states.pop(chat_id, None)
            return reply_with_quick_bar(reply_token, "✅ 已結束翻譯模式", is_group, bot_name)
        translation_states[chat_id] = lang
        return reply_with_quick_bar(reply_token, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。", is_group, bot_name)

    persona_keys = {"甜":"sweet", "鹹":"salty", "萌":"moe", "酷":"cool", "random":"random", "隨機":"random"}
    if low in persona_keys:
        key = set_user_persona(chat_id, persona_keys[low])
        p = PERSONAS[key]
        txt = f"💖 已切換人設：{p['title']}\n\n【特質】{p['style']}\n{p['greetings']}"
        return reply_with_quick_bar(reply_token, txt, is_group, bot_name)
        
    # 金融分析功能
    if low in ("金價", "黃金"):
        try:
            analysis_report = await run_in_threadpool(get_gold_analysis)
            line_bot_api.reply_message(reply_token, TextSendMessage(text=analysis_report))
        except Exception as e:
            logger.error(f"黃金分析流程失敗: {e}", exc_info=True)
            line_bot_api.reply_message(reply_token, TextSendMessage(text="抱歉，金價分析服務暫時無法使用，請稍後再試。"))
        return

    # 匯率分析功能
    currency_map = {
        "日元": "JPY", "日幣": "JPY", "jpy": "JPY",
        "美元": "USD", "美金": "USD", "usd": "USD", 
        "歐元": "EUR", "eur": "EUR",
        "英鎊": "GBP", "gbp": "GBP",
        "人民幣": "CNY", "cny": "CNY"
    }
    
    if low in currency_map:
        currency_code = currency_map[low]
        try:
            analysis_report = await run_in_threadpool(get_currency_analysis, currency_code)
            line_bot_api.reply_message(reply_token, TextSendMessage(text=analysis_report))
        except Exception as e:
            logger.error(f"{currency_code}匯率分析流程失敗: {e}", exc_info=True)
            line_bot_api.reply_message(reply_token, TextSendMessage(text=f"抱歉，{currency_code}匯率分析服務暫時無法使用，請稍後再試。"))
        return

    if chat_id in translation_states:
        tgt = translation_states[chat_id]
        try:
            out = await translate_text(msg, tgt)
        except Exception as e:
            logger.error(f"翻譯失敗: {e}", exc_info=True)
            out = "翻譯暫時失效，等我回神再來一次 🙏"
        return reply_with_quick_bar(reply_token, f"🌐 ({tgt})\n{out}", is_group, bot_name)

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

# ========== 7) FastAPI Routes ==========
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

# ========== 8) Local run ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi