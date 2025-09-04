# ========== 1) Imports ==========
import os
import re
import random
import logging
import asyncio
from typing import Dict, List
from contextlib import asynccontextmanager
import time
from io import StringIO
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import httpx
import pandas as pd
import html5lib

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

# --- 【靈活載入】載入自訂的彩票爬蟲模組 ---
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    LOTTERY_ENABLED = True
except ImportError:
    logging.warning("無法載入彩票模組，彩票功能將停用。")
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

if LOTTERY_ENABLED:
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()

# 狀態字典
conversation_history: Dict[str, List[dict]] = {}
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}
auto_reply_status: Dict[str, bool] = {}

PERSONAS = { "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰", "emoji":"🌸💕😊"}, "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度", "emoji":"😏🙄"}, "moe":   {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字", "emoji":"✨🎀"}, "cool":  {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議", "emoji":"🧊⚡️"}}
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
            response = sync_groq_client.chat.completions.create(model=GROQ_MODEL_PRIMARY, messages=messages, max_tokens=2000, temperature=0.8)
            return response.choices[0].message.content
        except Exception as groq_err:
            logger.warning(f"Groq 主要模型失敗: {groq_err}")
            try:
                response = sync_groq_client.chat.completions.create(model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=1500, temperature=1.0)
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
    try:
        url = "https://rate.bot.com.tw/gold?Lang=zh-TW"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        df_list = pd.read_html(StringIO(response.text), flavor='html5lib')
        df = df_list[0]
        # 取得 TWD 計價的牌價
        gold_price = df[df['商品'] == '黃金牌價']['本行賣出'].values[0]
        content_msg = (f"你是一位金融快報記者，請根據最新的台灣銀行黃金牌價提供一則簡短報導。\n"
                       f"最新數據：黃金（1公克）對台幣（TWD）的賣出價為 {gold_price} 元。\n"
                       f"報導要求：\n1. 開頭直接點出最新價格。\n2. 簡要分析此價格在近期市場中的位置（例如：處於高點、低點、或盤整）。\n3. 提及可能影響金價的因素（例如：通膨預期、美元走勢、避險情緒）。\n4. 語氣中立客觀，使用繁體中文。")
        msg = [{"role": "system", "content": "你是一位專業的金融記者。"}, {"role": "user", "content": content_msg}]
        return get_analysis_reply(msg)
    except Exception as e:
        logger.error(f"黃金價格爬取或分析失敗: {e}", exc_info=True)
        return "抱歉，目前無法獲取黃金價格，請稍後再試。"

# 【 crucial fix 】改用您指定的 open.er-api.com API
def get_currency_analysis(target_currency: str):
    logger.info(f"開始執行 {target_currency} 匯率分析 (使用新 API)...")
    try:
        # 基礎貨幣設為 TWD，目標是獲取 JPY->TWD 的匯率
        base_currency = 'TWD'
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("result") == "success":
            rate = data["rates"].get(base_currency)
            if rate is None:
                return f"抱歉，API中找不到 {base_currency} 的匯率資訊。"
            
            # 1 JPY = ? TWD，所以 1 TWD = 1/rate JPY
            twd_per_jpy = 1 / rate
            
            content_msg = (f"你是一位外匯分析師，請根據最新即時匯率撰寫一則簡短的日圓(JPY)匯率快訊。\n"
                           f"最新數據：1 日圓 (JPY) 可以兌換 {twd_per_jpy:.4f} 新台幣 (TWD)。\n"
                           f"分析要求：\n1. 直接報告目前的匯率。\n2. 根據此匯率水平，簡要說明現在去日本旅遊或換匯是相對划算還是昂貴。\n3. 提供一句給換匯族的實用建議。\n4. 語氣輕鬆易懂，使用繁體中文。")
            msg = [{"role": "system", "content": "你是一位專業的外匯分析師。"}, {"role": "user", "content": content_msg}]
            return get_analysis_reply(msg)
        else:
            return "抱歉，獲取匯率資料失敗，請稍後再試。"
            
    except requests.RequestException as e:
        logger.error(f"API 連接錯誤，無法獲取 {target_currency} 匯率: {e}")
        return f"抱歉，連接外匯 API 時發生網路錯誤，請稍後再試。"
    except Exception as e:
        logger.error(f"處理 {target_currency} API 資料時發生錯誤: {e}", exc_info=True)
        return f"抱歉，處理外匯資料時發生內部錯誤，請稍後再試。"

# --- (彩票, UI, 對話 Helpers 等函式保持不變) ---

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
    
    # --- 命令 & 功能觸發區 ---
    if low in ("金融選單", "彩票選單", "翻譯選單", "我的人設", "人設選單"):
        # (此處省略 Flex Menu 建立邏輯，與上一版相同)
        pass

    LOTTERY_KEYWORDS = ["大樂透", "威力彩", "539", "運彩"]
    if msg in LOTTERY_KEYWORDS:
        if not LOTTERY_ENABLED:
            return line_bot_api.reply_message(reply_token, TextSendMessage(text="抱歉，彩票分析功能目前設定不完整。"))
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

    # (設定類命令、模式處理、一般對話等邏輯，與上一版相同，此處省略)

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