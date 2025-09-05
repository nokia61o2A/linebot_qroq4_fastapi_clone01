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

# --- 數據處理與爬蟲 ---
import requests
from bs4 import BeautifulSoup
import httpx
import pandas as pd
import html5lib
import yfinance as yf

# --- FastAPI 與 LINE Bot SDK ---
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

# --- AI 相關 ---
from groq import AsyncGroq, Groq
import openai

# --- 【靈活載入】載入自訂的彩票與股票爬蟲模組 ---
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    LOTTERY_ENABLED = True
except ImportError:
    logging.warning("無法載入彩票模組，彩票功能將停用。")
    LOTTERY_ENABLED = False

try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
    STOCK_ENABLED = True
except ImportError as e:
    logging.warning(f"無法載入股票模組，股票功能將停用。錯誤: {e}")
    STOCK_ENABLED = False

# ========== 2) Setup ==========
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# --- 環境變數 ---
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise RuntimeError("缺少必要環境變數")

# --- API 用戶端初始化 ---
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

# --- 狀態字典與常數 ---
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}
auto_reply_status: Dict[str, bool] = {}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji":"🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji":"😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji":"✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議", "greetings": "我在。說重點。", "emoji":"🧊⚡️"}
}
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

# --- 金融 & 彩票分析 ---
def get_gold_analysis():
    logger.info("開始執行黃金價格分析...")
    try:
        url = "https://rate.bot.com.tw/gold?Lang=zh-TW"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'}
        response =

