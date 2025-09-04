# ========== 1) Imports ==========
import os
import re
import random
import logging
import asyncio
from typing import Dict, List
from contextlib import asynccontextmanager
import time

# 匯率/運彩爬蟲需要的套件
import requests
from bs4 import BeautifulSoup

import httpx
import pandas as pd
from fastapi import FastAPI, AProuter, Request, HTTPException
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
import openai

# --- 載入自訂的彩票爬蟲模組 ---
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    LOTTERY_ENABLED = True
except ImportError as e:
    logging.warning(f"無法載入彩票模組，彩票功能將停用。請確認 TaiwanLottery.py 與 my_commands/CaiyunfangweiCrawler.py 存在。錯誤: {e}")
    LOTTERY_ENABLED = False


# ========== 2) Setup ==========
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise RuntimeError("缺少必要環境變數 (BASE_URL, CHANNEL_ACCESS_TOKEN, CHANNEL_SECRET, GROQ_API_KEY)")

line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
sync_groq_client = Groq(api_key=GROQ_API_KEY)
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None
    logger.warning("未設定 OPENAI_API_KEY，分析功能將僅使用 Groq。")

GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

if LOTTERY_ENABLED:
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()

# 對話狀態、PERSONAS, LANGUAGE_MAP 等
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
auto_reply_status: Dict[str, bool] = {}
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}
PERSONAS = { "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰，不浮誇。", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji":"🌸💕😊"}, "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度。", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji":"😏🙄"}, "moe":   {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字，仍要有重點。", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji":"✨🎀"}, "cool":  {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議。", "greetings": "我在。說重點。", "emoji":"🧊⚡️"}}
LANGUAGE_MAP = { "英文": "English", "日文": "Japanese", "韓文": "Korean", "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"}

# ========== 3) FastAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with httpx.AsyncClient() as c:
            headers = {"Authorization": f"Bearer {CHANNEL_TOKEN}", "Content-Type": "application/json"}
            payload = {"endpoint": f"{BASE_URL}/callback"}
            r = await c.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=payload, timeout=10.0)
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

# --- AI & 分析相關函式 ---
def get_analysis_reply(messages):
    try:
        if not openai_client: raise Exception("OpenAI client not initialized.")
        response = openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        return response.choices[0].message.content
    except Exception as openai_err:
        logger.warning(f"OpenAI API 失敗: {openai_err}")
        try:
            response = sync_groq_client.chat.completions.create(model=GROQ_MODEL_PRIMARY, messages=messages, max_tokens=2000, temperature=1.0)
            return response.choices[0].message.content
        except Exception as groq_err:
            logger.warning(f"Groq 主要模型失敗: {groq_err}")
            try:
                response = sync_groq_client.chat.completions.create(model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=1500, temperature=1.2)
                return response.choices[0].message.content
            except Exception as fallback_err:
                logger.error(f"所有 AI API 都失敗: {fallback_err}")
                return "抱歉，AI分析師目前連線不穩定，請稍後再試。"

async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
    try:
        resp = await async_groq_client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages, max_tokens=max_tokens, temperature=temperature)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq Async 主要模型失敗: {e}")
        resp = await async_groq_client.chat.completions.create(model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return resp.choices[0].message.content.strip()

def get_gold_analysis():
    logger.info("開始執行黃金價格分析...")
    url = "https://rate.bot.com.tw/gold/chart/year/TWD"
    df_list = pd.read_html(url)
    df = df_list[0]
    df = df[["日期", "本行賣出價格"]].copy()
    df.index = pd.to_datetime(df["日期"], format="%Y/%m/%d")
    df.sort_index(inplace=True)
    
    max_price = df['本行賣出價格'].max()
    min_price = df['本行賣出價格'].min()
    last_price = df['本行賣出價格'].iloc[-1]
    last_date = df.index[-1].strftime("%Y-%m-%d")
    recent_data = df.tail(30).to_string()
    
    content_msg = (f'你是一位專業的金價分析師，請根據以下近一年的台灣銀行黃金牌價數據(台幣計價)，撰寫一份專業、簡潔且易懂的趨勢分析報告。\n'
                   f'--- 資料摘要 ---\n最新日期: {last_date}\n最新價格: {last_price}\n年度最高價: {max_price}\n年度最低價: {min_price}\n'
                   f'--- 最近30天數據 ---\n{recent_data}\n--- 分析要求 ---\n'
                   f'1. 開頭先明確指出「{last_date} 的最新賣出牌價為 {last_price} 元」。\n'
                   f'2. 根據數據分析近一週、近一個月及近一年的價格趨勢。\n'
                   f'3. 提及年度高點與低點，並簡單說明其意義。\n'
                   f'4. 最後給出一個簡短的總結與後市展望（保持中立客觀）。\n'
                   f'5. 全程使用繁體中文，語氣專業，結構清晰。')
    
    msg = [{"role": "system", "content": "你是一位專業的金價分析師。"}, {"role": "user", "content": content_msg}]
    return get_analysis_reply(msg)

def fetch_currency_rates(kind: str):
    url = f"https://rate.bot.com.tw/xrt/quote/day/{kind}"
    max_retries, retry_count, retry_delay = 3, 0, 2
    while retry_count < max_retries:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                df_list = pd.read_html(response.text)
                if df_list:
                    df = df_list[0]
                    df = df.iloc[:, [0, 4]]
                    df.columns = ['掛牌時間', '即期賣出']
                    logger.info(f"成功擷取 {kind} 匯率資料。")
                    return df
            else:
                logger.warning(f"HTTP 請求 {kind} 失敗，狀態碼: {response.status_code}")
        except requests.RequestException as e:
            logger.warning(f"網路連接錯誤 (嘗試 {retry_count+1}/{max_retries}): {e}")
        
        retry_count += 1
        if retry_count < max_retries:
            time.sleep(retry_delay)
            retry_delay *= 2
    
    logger.error(f"所有重試均失敗，無法獲取 {kind} 匯率資料。")
    return None

def get_currency_analysis(kind: str):
    logger.info(f"開始執行 {kind} 匯率分析...")
    currency_df = fetch_currency_rates(kind)
    if currency_df is None or currency_df.empty:
        return f"抱歉，目前無法獲取 {kind} 的匯率資料，請稍後再試。"

    currency_df['即期賣出'] = pd.to_numeric(currency_df['即期賣出'], errors='coerce')
    currency_df.dropna(subset=['即期賣出'], inplace=True)
    
    if currency_df.empty: return f"抱歉，獲取的 {kind} 資料格式有誤，暫時無法分析。"
        
    max_price, min_price = currency_df['即期賣出'].max(), currency_df['即期賣出'].min()
    last_price, last_time = currency_df['即期賣出'].iloc[0], currency_df['掛牌時間'].iloc[0]

    content_msg = (f'你是一位專業的日圓(JPY)匯率分析師，請根據以下今日台灣銀行日圓的即期賣出價數據(JPY/TWD)，撰寫一份專業、簡潔且易懂的趨勢分析報告。\n'
                   f'--- 今日數據摘要 ---\n最新時間: {last_time}\n最新匯率: {last_price}\n今日最高價: {max_price}\n今日最低價: {min_price}\n'
                   f'--- 今日所有報價紀錄 ---\n{currency_df.to_string()}\n--- 分析要求 ---\n'
                   f'1. 開頭明確指出「截至 {last_time} 的最新日圓即期賣出價為 {last_price}」。\n'
                   f'2. 根據今日的價格波動（最高、最低、最新價），分析今日盤中趨勢。\n'
                   f'3. 提出簡短的結論，例如「日圓今天呈現波動走升/走貶/盤整格局」。\n'
                   f'4. 可選：基於常識，簡要提及可能影響日圓匯率的總體經濟因素（例如：日本央行政策、美金走勢等）。\n'
                   f'5. 全程使用繁體中文，語氣專業，避免不確定的預測。')

    msg = [{"role": "system", "content": f"你是一位專業的 {kind} 幣種分析師。"}, {"role": "user", "content": content_msg}]
    return get_analysis_reply(msg)

def lotto_exercise():
    try:
        params = {'sport': 'NBA', 'date': '2024-05-16', 'names': ['洛杉磯湖人', '金州勇士'], 'limit': 6}
        headers = {'X-JBot-Token': 'FREE_TOKEN_WITH_20_TIMES_PRE_DAY'}
        url = 'https://api.sportsbot.tech/v2/records'
        res = requests.get(url, headers=headers, params=params, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logger.error(f"運彩資料獲取失敗: {e}")
        return f"運彩資料獲取失敗: {e}"

def get_lottery_analysis(lottery_type_input: str):
    logger.info(f"開始執行 {lottery_type_input} 彩票分析...")
    lottery_type = lottery_type_input.lower()

    if "威力" in lottery_type: last_lotto = lottery_crawler.super_lotto()
    elif "大樂" in lottery_type: last_lotto = lottery_crawler.lotto649()
    elif "539" in lottery_type: last_lotto = lottery_crawler.daily_cash()
    elif "運彩" in lottery_type: last_lotto = lotto_exercise()
    else: return f"抱歉，暫不支援 {lottery_type_input} 類型的分析。"

    if "運彩" not in lottery_type:
        try:
            caiyunfangwei_info = caiyunfangwei_crawler.get_caiyunfangwei()
            content_msg = (f'你現在是一位專業的樂透彩分析師, 使用{lottery_type_input}的資料來撰寫分析報告:\n'
                           f'近幾期號碼資訊:\n{last_lotto}\n'
                           f'顯示今天國歷/農歷日期：{caiyunfangwei_info.get("今天日期", "未知")}\n'
                           f'今日歲次：{caiyunfangwei_info.get("今日歲次", "未知")}\n'
                           f'財神方位：{caiyunfangwei_info.get("財神方位", "未知")}\n'
                           '最冷號碼，最熱號碼\n請給出完整的趨勢分析報告，最近所有每次開號碼,'
                           '並給3組與彩類同數位數字隨機號和不含特別號(如果有的彩種,)\n'
                           '第1組最冷組合:給與該彩種開獎同數位數字隨機號和(數字小到大)，威力彩多顯示二區才顯示，其他彩種不含二區\n'
                           '第2組最熱組合:給與該彩種開獎同數位數字隨機號和(數字小到大)，威力彩多顯示二區才顯示，其他彩種不含二區\n'
                           '第3組隨機組合:給與該彩種開獎同數位數字隨機號和(數字小到大)，威力彩多顯示二區才顯示，其他彩種不含二區\n'
                           '請寫詳細的數字，1不要省略\n{發財的吉祥句20字內要有勵志感}\n'
                           'example:   ***財神方位提示***\n國歷：2024/06/19（星期三）\n農曆甲辰年五月十四號\n根據財神方位 :東北\n'
                           '使用台灣繁體中文。')
        except Exception as e:
            logger.error(f"獲取財神方位失敗: {e}")
            content_msg = (f'你現在是一位專業的樂透彩分析師, 使用{lottery_type_input}的資料來撰寫分析報告:\n'
                           f'近幾期號碼資訊:\n{last_lotto}\n'
                           '財神方位資訊暫時無法獲取\n'
                           '請給出完整的趨勢分析報告，並給3組隨機號碼組合\n'
                           '使用台灣繁體中文。')
    else:
        content_msg = (f'你現在是一位專業的運彩分析師, 使用{lottery_type_input}的資料來撰寫分析報告:\n'
                       f'近幾運彩資料資訊:\n{last_lotto}\n'
                       '{發財的吉祥句20字內要有勵志感}\n'
                       '使用台灣用詞的繁體中文。')
    
    msg = [{"role": "system", "content": f"你現在是一位專業的彩券分析師, 使用{lottery_type_input}近期的號碼進行分析，生成一份專業的趨勢分析報告。"}, {"role": "user", "content": content_msg}]
    return get_analysis_reply(msg)

# --- UI & 對話 Helpers ---
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
    bubble = BubbleContainer(header=BoxComponent(layout="vertical", contents=[TextComponent(text=title, weight="bold", size="xl", align="center"), TextComponent(text=subtitle, size="sm", color="#666666", wrap=True, align="center", margin="md")]), body=BoxComponent(layout="vertical", contents=buttons, spacing="sm", paddingAll="12px"))
    return FlexSendMessage(alt_text=title, contents=bubble)

def flex_menu_finance(bot_name: str, is_group: bool) -> FlexSendMessage:
    prefix = f"@{bot_name} " if is_group else ""
    acts = [MessageAction(label=l, text=f"{prefix}{t}") for l, t in [("🇹🇼 台股大盤", "台股大盤"), ("🇺🇸 美股大盤", "美股大盤"), ("💰 金價", "金價"), ("💴 日元", "JPY"), ("📊 個股(例:2330)", "2330")]]
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


# ========== 5) LINE Handlers ==========
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

    # --- 命令 & 功能觸發區 (按優先級排列) ---
    
    # 選單優先
    if low in ("金融選單", "彩票選單", "翻譯選單", "我的人設", "人設選單"):
        flex_map = {
            "金融選單": flex_menu_finance(bot_name, is_group), 
            "彩票選單": flex_menu_lottery(bot_name, is_group), 
            "翻譯選單": flex_menu_translate(), 
            "我的人設": flex_menu_persona(), 
            "人設選單": flex_menu_persona()
        }
        flex = flex_map[low]
        tip = TextSendMessage(text="👇 選一個功能開始吧", quick_reply=QuickReply(items=make_quick_reply_items(is_group, bot_name)))
        return line_bot_api.reply_message(reply_token, [flex, tip])

    # 特定分析命令
    LOTTERY_KEYWORDS = ["大樂透", "威力彩", "539", "運彩"]
    if msg in LOTTERY_KEYWORDS:
        if not LOTTERY_ENABLED:
            return line_bot_api.reply_message(reply_token, TextSendMessage(text="抱歉，彩票分析功能目前設定不完整，暫時無法使用。"))
        try:
            analysis_report = await run_in_threadpool(get_lottery_analysis, msg)
            return line_bot_api.reply_message(reply_token, TextSendMessage(text=analysis_report))
        except Exception as e:
            logger.error(f"彩票分析流程失敗: {e}", exc_info=True)
            return line_bot_api.reply_message(reply_token, TextSendMessage(text=f"抱歉，分析 {msg} 時發生錯誤。"))

    if low in ("金價", "黃金"):
        try:
            analysis_report = await run_in_threadpool(get_gold_analysis)
            return line_bot_api.reply_message(reply_token, TextSendMessage(text=analysis_report))
        except Exception as e:
            logger.error(f"黃金分析流程失敗: {e}", exc_info=True)
            return line_bot_api.reply_message(reply_token, TextSendMessage(text="抱歉，金價分析服務暫時無法使用。"))
    
    if low == "jpy":
        try:
            analysis_report = await run_in_threadpool(get_currency_analysis, "JPY")
            return line_bot_api.reply_message(reply_token, TextSendMessage(text=analysis_report))
        except Exception as e:
            logger.error(f"日圓分析流程失敗: {e}", exc_info=True)
            return line_bot_api.reply_message(reply_token, TextSendMessage(text="抱歉，日圓匯率分析服務暫時無法使用。"))

    # 設定類命令
    if low in ("開啟自動回答", "關閉自動回答"):
        is_on = low == "開啟自動回答"
        auto_reply_status[chat_id] = is_on
        text = "✅ 已開啟自動回答" if is_on else "❌ 已關閉自動回答（群組需 @我 才回）"
        return reply_with_quick_bar(reply_token, text, is_group, bot_name)

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

    # --- 模式處理 & 一般對話 (最後的預設行為) ---
    if chat_id in translation_states:
        try:
            out = await translate_text(msg, translation_states[chat_id])
            return reply_with_quick_bar(reply_token, f"🌐 ({translation_states[chat_id]})\n{out}", is_group, bot_name)
        except Exception as e:
            logger.error(f"翻譯失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "翻譯暫時失效，等我回神再來一次 🙏", is_group, bot_name)

    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]
        final_reply = await groq_chat_async(messages)
        history.extend([{"role":"user","content":msg}, {"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
        return reply_with_quick_bar(reply_token, final_reply, is_group, bot_name)
    except Exception as e:
        logger.error(f"AI 回覆失敗: {e}", exc_info=True)
        return reply_with_quick_bar(reply_token, "抱歉我剛剛走神了 😅 再說一次讓我補上！", is_group, bot_name)

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