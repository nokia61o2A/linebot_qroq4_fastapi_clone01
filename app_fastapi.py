# ========== 1) Imports ==========
import os
import re
import random
import logging
import asyncio
from typing import Dict, List
from contextlib import asynccontextmanager
import time

import requests
from bs4 import BeautifulSoup
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
    SourceUser, SourceGroup, SourceRoom, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent,
    TextComponent, ButtonComponent
)

from groq import AsyncGroq, Groq
import openai

# --- 【新增】載入自訂的彩票爬蟲模組 ---
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

# ... (環境變數等設定保持不變)
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise RuntimeError("缺少必要環境變數")

line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
sync_groq_client = Groq(api_key=GROQ_API_KEY)
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

# --- 【新增】初始化彩票爬蟲 ---
if LOTTERY_ENABLED:
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()

# ... (對話狀態、PERSONAS, LANGUAGE_MAP 等保持不變)
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
    # ... (此函式內容不變)
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
    # ... (此函式內容不變)
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom):  return event.source.room_id
    return event.source.user_id

# --- AI & 分析相關函式 ---
def get_analysis_reply(messages):
    # ... (共用的分析函式，內容不變)
    try:
        if not openai_client: raise Exception("OpenAI client not initialized.")
        response = openai_client.chat.completions.create(model="gpt-3.5-turbo-1106", messages=messages)
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

# ... (黃金分析、匯率分析函式保持不變)
def get_gold_analysis():
    logger.info("開始執行黃金價格分析...")
    # ... (省略內部程式碼以節省空間)
    gold_prices_df = pd.read_html("https://rate.bot.com.tw/gold/chart/year/TWD")[0]
    gold_prices_df = gold_prices_df[["日期", "本行賣出價格"]].copy()
    gold_prices_df.index = pd.to_datetime(gold_prices_df["日期"], format="%Y/%m/%d")
    gold_prices_df.sort_index(inplace=True)
    max_price, min_price = gold_prices_df['本行賣出價格'].max(), gold_prices_df['本行賣出價格'].min()
    last_price, last_date = gold_prices_df['本行賣出價格'].iloc[-1], gold_prices_df.index[-1].strftime("%Y-%m-%d")
    content_msg = (f'你是一位專業的金價分析師...\n' # 省略詳細 prompt
                   f'最新日期: {last_date}, 最新價格: {last_price}...\n'
                   f'{gold_prices_df.tail(30).to_string()}')
    msg = [{"role": "system", "content": "你是一位專業的金價分析師。"}, {"role": "user", "content": content_msg}]
    return get_analysis_reply(msg)

def get_currency_analysis(kind: str):
    logger.info(f"開始執行 {kind} 匯率分析...")
    # ... (省略內部程式碼以節省空間)
    url = f"https://rate.bot.com.tw/xrt/quote/day/{kind}"
    try:
        currency_df = pd.read_html(requests.get(url, timeout=10).text)[0]
        currency_df = currency_df.iloc[:, [0, 4]]
        currency_df.columns = ['掛牌時間', '即期賣出']
        currency_df['即期賣出'] = pd.to_numeric(currency_df['即期賣出'], errors='coerce')
        currency_df.dropna(subset=['即期賣出'], inplace=True)
        max_price, min_price, last_price = currency_df['即期賣出'].max(), currency_df['即期賣出'].min(), currency_df['即期賣出'].iloc[0]
        last_time = currency_df['掛牌時間'].iloc[0]
        content_msg = (f'你是一位專業的日圓(JPY)匯率分析師...\n' # 省略詳細 prompt
                       f'最新時間: {last_time}, 最新匯率: {last_price}...\n'
                       f'{currency_df.to_string()}')
        msg = [{"role": "system", "content": f"你是一位專業的 {kind} 幣種分析師。"}, {"role": "user", "content": content_msg}]
        return get_analysis_reply(msg)
    except Exception as e:
        logger.error(f"無法獲取 {kind} 匯率資料: {e}")
        return f"抱歉，目前無法獲取 {kind} 的匯率資料，請稍後再試。"

# --- 【新增】彩票分析 (整合 lottery_gpt.py) ---
def lotto_exercise():
    try:
        # 請注意：此 API token 可能有每日使用限制
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
    """執行完整的彩票分析流程"""
    logger.info(f"開始執行 {lottery_type_input} 彩票分析...")
    lottery_type = lottery_type_input.lower()

    if "威力" in lottery_type:
        last_lotto = lottery_crawler.super_lotto()
    elif "大樂" in lottery_type:
        last_lotto = lottery_crawler.lotto649()
    elif "539" in lottery_type:
        last_lotto = lottery_crawler.daily_cash()
    # 可以根據需要加入更多彩種
    elif "運彩" in lottery_type:
        last_lotto = lotto_exercise()
    else:
        return f"抱歉，暫不支援 {lottery_type_input} 類型的分析。"

    content_msg = ""
    # 根據您的邏輯建立 content_msg
    if "運彩" not in lottery_type:
        try:
            caiyunfangwei_info = caiyunfangwei_crawler.get_caiyunfangwei()
            content_msg = (f'你現在是一位專業的樂透彩分析師, 使用{lottery_type_input}的資料來撰寫分析報告:\n'
                           f'近幾期號碼資訊:\n{last_lotto}\n'
                           f'顯示今天國歷/農歷日期：{caiyunfangwei_info.get("今天日期", "未知")}\n'
                           f'今日歲次：{caiyunfangwei_info.get("今日歲次", "未知")}\n'
                           f'財神方位：{caiyunfangwei_info.get("財神方位", "未知")}\n'
                           '最冷號碼，最熱號碼\n'
                           '請給出完整的趨勢分析報告，最近所有每次開號碼,'
                           '並給3組與彩類同數位數字隨機號和不含特別號(如果有的彩種,)\n'
                           '第1組最冷組合:給與該彩種開獎同數位數字隨機號和(數字小到大)，威力彩多顯示二區才顯示，其他彩種不含二區\n'
                           '第2組最熱組合:給與該彩種開獎同數位數字隨機號和(數字小到大)，威力彩多顯示二區才顯示，其他彩種不含二區\n'
                           '第3組隨機組合:給與該彩種開獎同數位數字隨機號和(數字小到大)，威力彩多顯示二區才顯示，其他彩種不含二區\n'
                           '請寫詳細的數字，1不要省略\n'
                           '{發財的吉祥句20字內要有勵志感}\n'
                           'example:   ***財神方位提示***\n國歷：2024/06/19（星期三）\n農曆甲辰年五月十四號\n根據財神方位 :東北\n'
                           '使用台灣繁體中文。')
        except Exception:
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
    
    msg = [{
        "role": "system",
        "content": f"你現在是一位專業的彩券分析師, 使用{lottery_type_input}近期的號碼進行分析，生成一份專業的趨勢分析報告。"
    }, {
        "role": "user",
        "content": content_msg
    }]

    reply_data = get_analysis_reply(msg)
    logger.info(f"{lottery_type_input} 彩票分析完成。")
    return reply_data

# ... (其他 UI Helpers 保持不變)
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
# ... (其他選單函式不變)

# ========== 5) LINE Handlers ==========
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    # ... (此函式內容不變)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(handle_message_async(event))
    except RuntimeError:
        asyncio.run(handle_message_async(event))

async def handle_message_async(event: MessageEvent):
    # ... (此函式大部分內容不變)
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
    
    # --- 【新增】彩票分析觸發 ---
    # 定義彩票關鍵字，方便管理
    LOTTERY_KEYWORDS = ["大樂透", "威力彩", "539", "運彩"]
    # 檢查訊息是否為彩票關鍵字 (忽略大小寫)
    if msg in LOTTERY_KEYWORDS:
        if not LOTTERY_ENABLED:
            return line_bot_api.reply_message(reply_token, TextSendMessage(text="抱歉，彩票分析功能目前設定不完整，暫時無法使用。"))
        try:
            analysis_report = await run_in_threadpool(get_lottery_analysis, msg) # 將原始訊息傳入
            line_bot_api.reply_message(reply_token, TextSendMessage(text=analysis_report))
        except Exception as e:
            logger.error(f"彩票分析流程失敗: {e}", exc_info=True)
            line_bot_api.reply_message(reply_token, TextSendMessage(text=f"抱歉，分析 {msg} 時發生錯誤。"))
        return

    # --- 其他功能觸發區 ---
    # ... (其他功能觸發, 如金融分析、選單, 翻譯, 人設等保持不變)
    if low in ("金價", "黃金"):
        try:
            analysis_report = await run_in_threadpool(get_gold_analysis)
            line_bot_api.reply_message(reply_token, TextSendMessage(text=analysis_report))
        except Exception as e:
            logger.error(f"黃金分析流程失敗: {e}", exc_info=True)
            line_bot_api.reply_message(reply_token, TextSendMessage(text="抱歉，金價分析服務暫時無法使用。"))
        return
    if low == "jpy":
        try:
            analysis_report = await run_in_threadpool(get_currency_analysis, "JPY")
            line_bot_api.reply_message(reply_token, TextSendMessage(text=analysis_report))
        except Exception as e:
            logger.error(f"日圓分析流程失敗: {e}", exc_info=True)
            line_bot_api.reply_message(reply_token, TextSendMessage(text="抱歉，日圓匯率分析服務暫時無法使用。"))
        return

    # ... (一般對話 & 模式處理，保持不變)
    # 此處省略重複的程式碼，請保留您原有的`if low in ("開啟自動回答"...)`及之後的所有邏輯

# ========== 6) FastAPI Routes ==========
# ... (此區塊邏輯保持不變)
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
# ... (其他 routes 不變)

# ========== 7) Local run ==========
# ... (此區塊邏輯保持不變)