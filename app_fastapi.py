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
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    QuickReply, QuickReplyButton, MessageAction,
    PostbackAction, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent,
    TextComponent, ButtonComponent, SeparatorComponent
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
    raise RuntimeError("缺少必要環境變數 (BASE_URL, CHANNEL_ACCESS_TOKEN, CHANNEL_SECRET, GROQ_API_KEY)")

# --- API 用戶端初始化 ---
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# AI 客戶端初始化
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
sync_groq_client = Groq(api_key=GROQ_API_KEY)
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None
    logger.warning("未設定 OPENAI_API_KEY，AI 分析功能將僅使用 Groq。")

# 【 CRUCIAL FIX 】更新為當前有效的 Groq 模型 (請定期檢查 Groq 官網是否有更新)
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

if LOTTERY_ENABLED:
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()

# --- 狀態字典與常數 ---
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10 # 保持對話歷史的長度，避免超限
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
    # 應用啟動時更新 LINE Bot Webhook Endpoint
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
    """獲取聊天 ID，用於區分不同用戶或群組的對話歷史"""
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom):  return event.source.room_id
    return event.source.user_id

# --- AI & 分析相關函式 ---
def get_analysis_reply(messages: List[Dict[str, str]]):
    """
    統一的 AI 回覆獲取函式，優先使用 OpenAI，失敗則回退到 Groq 的主要模型，再失敗則回退到 Groq 的備用模型。
    """
    if openai_client:
        try:
            # 使用最新的 gpt-4o 模型，如果沒有權限，請改回 gpt-3.5-turbo 或 gpt-3.5-turbo-0125
            response = openai_client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=2000, temperature=0.7)
            return response.choices[0].message.content
        except Exception as openai_err:
            logger.warning(f"OpenAI API 失敗: {openai_err}")
            # 如果 OpenAI 失敗，嘗試 Groq
            pass # 繼續執行 Groq 的邏輯

    if sync_groq_client:
        try:
            response = sync_groq_client.chat.completions.create(model=GROQ_MODEL_PRIMARY, messages=messages, max_tokens=2000, temperature=0.8)
            return response.choices[0].message.content
        except Exception as groq_primary_err:
            logger.warning(f"Groq 主要模型失敗 ({GROQ_MODEL_PRIMARY}): {groq_primary_err}")
            # 如果主要 Groq 模型失敗，嘗試備用 Groq 模型
            try:
                response = sync_groq_client.chat.completions.create(model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=1500, temperature=1.0)
                return response.choices[0].message.content
            except Exception as groq_fallback_err:
                logger.error(f"所有 AI API 都失敗 (Groq 備用模型 {GROQ_MODEL_FALLBACK} 失敗): {groq_fallback_err}")
    else:
        logger.error("Groq client 未初始化，請檢查 GROQ_API_KEY。")

    return "抱歉，AI 分析師目前連線不穩定，請稍後再試。"


async def groq_chat_async(messages: List[Dict[str, str]], max_tokens: int = 600, temperature: float = 0.7):
    """
    異步 Groq 回覆函式，主要用於情感分析等輕量級任務，確保非阻塞。
    """
    try:
        resp = await async_groq_client.chat.completions.create(
            model=GROQ_MODEL_FALLBACK, # 輕量級任務直接使用備用模型
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq Async 模型 ({GROQ_MODEL_FALLBACK}) 失敗: {e}", exc_info=True)
        # 拋出異常讓上層處理，或者返回一個錯誤訊息
        raise e # 繼續拋出異常，讓 handle_message_async 捕獲


# --- 金融 & 彩票分析 ---

def get_gold_ai_analysis_report():
    """
    從台灣銀行網站抓取即時和近期黃金數據，並調用 AI 生成分析報告。
    """
    logger.info("開始獲取黃金數據並生成 AI 分析報告...")
    current_price_url = "https://rate.bot.com.tw/gold?Lang=zh-TW"
    history_chart_url = "https://rate.bot.com.tw/gold/chart/year/TWD" # 用於歷史趨勢判斷

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'}

    current_gold_data = {}
    historical_summary = "無法獲取歷史數據摘要。"

    # 1. 獲取即時黃金牌價
    try:
        response = requests.get(current_price_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find("table", {"class": "table-striped"})
        if table:
            rows = table.find("tbody").find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) > 1 and "黃金牌價" in cells[0].text:
                    current_gold_data['sell_price'] = cells[4].text.strip()
                    current_gold_data['buy_price'] = cells[3].text.strip()
                    break
        if not current_gold_data:
            raise ValueError("在即時牌價頁面找不到黃金牌價數據。")

        current_gold_data['update_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"成功獲取即時黃金牌價: {current_gold_data}")

    except Exception as e:
        logger.error(f"獲取即時黃金牌價失敗: {e}", exc_info=True)
        return "抱歉，目前無法獲取即時黃金牌價，請稍後再試。"

    # 2. 獲取歷史數據並生成摘要 (近30天)
    try:
        df_list = pd.read_html(history_chart_url)
        df = df_list[0]
        df = df[["日期", "本行賣出價格"]].copy()
        df.columns = ["Date", "Sell_Price"]
        df['Sell_Price'] = pd.to_numeric(df['Sell_Price'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d")
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        recent_df = df[df.index >= (datetime.now() - timedelta(days=30))]
        if not recent_df.empty:
            max_price_30d = recent_df['Sell_Price'].max()
            min_price_30d = recent_df['Sell_Price'].min()
            avg_price_30d = recent_df['Sell_Price'].mean()

            # 計算今日與30天前的價格變化（需要確保數據足夠）
            if len(df) >= 30:
                price_30_days_ago = df['Sell_Price'].iloc[-30]
                current_sell_price_num = float(current_gold_data['sell_price'].replace(',', ''))
                change_30d = current_sell_price_num - price_30_days_ago
                change_percent_30d = (change_30d / price_30_days_ago) * 100 if price_30_days_ago else 0

                historical_summary = (
                    f"近30天黃金賣出價最高為 {max_price_30d:.2f} 元，最低為 {min_price_30d:.2f} 元，平均約 {avg_price_30d:.2f} 元。\n"
                    f"相較30天前，價格變化約 {change_30d:.2f} 元 ({change_percent_30d:.2f}%)。"
                )
            else:
                historical_summary = (
                    f"近期黃金賣出價最高為 {max_price_30d:.2f} 元，最低為 {min_price_30d:.2f} 元，平均約 {avg_price_30d:.2f} 元。"
                )
        else:
            historical_summary = "近30天歷史數據不足。"
        logger.info(f"黃金歷史數據摘要: {historical_summary}")

    except Exception as e:
        logger.error(f"獲取或處理歷史黃金數據失敗: {e}", exc_info=True)
        historical_summary = "無法獲取歷史數據摘要。"


    # 3. 構造 AI 提示詞
    content_msg = (
        f"你是一位專業的黃金市場分析師。請根據以下資訊撰寫一份簡潔、專業的黃金價格分析報告。\n"
        f"**最新即時黃金牌價 (台灣銀行)**:\n"
        f"  - 1 公克黃金賣出價 (對台幣): {current_gold_data.get('sell_price', 'N/A')} 元\n"
        f"  - 1 公克黃金買入價 (對台幣): {current_gold_data.get('buy_price', 'N/A')} 元\n"
        f"  - 更新時間: {current_gold_data.get('update_time', 'N/A')}\n\n"
        f"**近期市場摘要 (近30天)**:\n"
        f"{historical_summary}\n\n"
        f"**分析要求**:\n"
        f"1. 直接點出目前的賣出價格。\n"
        f"2. 簡要分析當前價格是處於近期高點、低點還是盤整，並結合近30天數據摘要。\n"
        f"3. 提及可能影響金價的短期因素（如美元走勢、通膨預期、地緣政治事件、利率政策）。\n"
        f"4. 對於一般投資者或消費者，提供一句精簡的**黃金買賣建議**。\n"
        f"5. 語氣專業、客觀，並使用台灣繁體中文，內容控制在 250 字以內。"
    )

    messages = [
        {"role": "system", "content": "你是一位專業的黃金市場分析師，善於從數據中提煉關鍵資訊並提供簡潔的市場洞察。"},
        {"role": "user", "content": content_msg}
    ]

    return get_analysis_reply(messages)


def get_currency_analysis(target_currency: str):
    logger.info(f"開始執行 {target_currency} 匯率分析...")
    try:
        base_currency = 'TWD'
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("result") == "success":
            rate = data["rates"].get(base_currency)
            if rate is None: return f"抱歉，API中找不到 {base_currency} 的匯率資訊。"
            twd_per_jpy = rate
            
            # AI 分析匯率的 Prompt
            content_msg = (f"你是一位外匯分析師，請根據最新即時匯率撰寫一則簡短的日圓(JPY)匯率快訊。\n"
                           f"最新數據：1 日圓 (JPY) 可以兌換 {twd_per_jpy:.5f} 新台幣 (TWD)。\n"
                           f"分析要求：\n1. 直接報告目前的匯率。\n2. 根據此匯率水平，簡要說明現在去日本旅遊或換匯是相對划算還是昂貴。\n3. 提供一句給換匯族的實用建議。\n4. 語氣輕鬆易懂，使用繁體中文。")
            
            msg = [{"role": "system", "content": "你是一位專業的外匯分析師。"}, {"role": "user", "content": content_msg}]
            return get_analysis_reply(msg)
        else:
            return f"抱歉，獲取匯率資料失敗：{data.get('error-type', '未知錯誤')}"
    except Exception as e:
        logger.error(f"處理 {target_currency} API 資料時發生錯誤: {e}", exc_info=True)
        return f"抱歉，處理外匯資料時發生內部錯誤，請稍後再試。"

def get_lottery_analysis(lottery_type_input: str):
    logger.info(f"開始執行 {lottery_type_input} 彩票分析...")
    if not LOTTERY_ENABLED:
        return "抱歉，彩票分析功能目前設定不完整或模組未載入。"

    lottery_type = lottery_type_input.lower()
    last_lotto = None
    if "威力" in lottery_type: last_lotto = lottery_crawler.super_lotto()
    elif "大樂" in lottery_type: last_lotto = lottery_crawler.lotto649()
    elif "539" in lottery_type: last_lotto = lottery_crawler.daily_cash()
    else: return f"抱歉，暫不支援 {lottery_type_input} 類型的分析。"

    if not last_lotto:
        return f"抱歉，無法獲取 {lottery_type_input} 的近期開獎資訊。"

    try:
        caiyunfangwei_info = caiyunfangwei_crawler.get_caiyunfangwei()
        caiyun_msg = (
            f"***財神方位提示***\n國歷：{caiyunfangwei_info.get('今天日期', '未知')}\n"
            f"農曆：{caiyunfangwei_info.get('今日農曆', '未知')}\n" # 修正鍵值
            f"今日歲次：{caiyunfangwei_info.get('今日歲次', '未知')}\n"
            f"財神方位：{caiyunfangwei_info.get('財神方位', '未知')}\n"
        )
    except Exception as e:
        logger.error(f"獲取財神方位失敗: {e}")
        caiyun_msg = "財神方位資訊暫時無法獲取。"
    
    # 構造 AI Prompt
    content_msg = (
        f'你現在是一位專業的樂透彩分析師，請使用{lottery_type_input}的近期開獎資料來撰寫一份趨勢分析報告，並綜合考慮以下資訊：\n'
        f'**近期開獎號碼資訊:**\n{last_lotto}\n\n'
        f'**今日財運資訊:**\n{caiyun_msg}\n\n'
        f'**分析要求**:\n'
        f'1. 報告近期號碼的趨勢 (例如：熱門號碼、冷門號碼、特定區間號碼出現頻率)。\n'
        f'2. 根據趨勢，提供3組與該彩種開獎數字位數相同的隨機號碼組合（數字小到大排列）。\n'
        f'   - 第1組: 最冷門數字組合。\n'
        f'   - 第2組: 最熱門數字組合。\n'
        f'   - 第3組: 純隨機數字組合。\n'
        f'(如果彩種有特別號/二區，請單獨顯示二區號碼，其他彩種不含二區。)\n'
        f'3. 最後提供一句鼓舞人心的發財吉祥話 (20字內要有勵志感)。\n'
        f'4. 請使用台灣繁體中文，並在報告中顯示詳細數字，不要省略。\n'
    )

    msg = [{"role": "system", "content": f"你現在是一位專業的彩券分析師, 善於從歷史數據中找出趨勢。"}, {"role": "user", "content": content_msg}]
    return get_analysis_reply(msg)

def get_stock_name_from_yahoo(stock_symbol: str) -> str:
    """
    從 Yahoo Finance 獲取股票名稱。
    """
    try:
        ticker = yf.Ticker(stock_symbol)
        info = ticker.info
        name = info.get('longName') or info.get('shortName')
        if name:
            logger.info(f"從 Yahoo Finance 獲取股票名稱成功: {stock_symbol} -> {name}")
            return name
    except Exception as e:
        logger.warning(f"從 Yahoo Finance 獲取 {stock_symbol} 名稱失敗: {e}")
    return stock_symbol # 失敗時返回原始代碼

def remove_full_width_spaces(data):
    return data.replace('\u3000', ' ') if isinstance(data, str) else data

def get_stock_analysis(stock_id_input: str):
    logger.info(f"開始執行 {stock_id_input} 股票分析...")
    stock_id = stock_id_input
    stock_name = stock_id_input # 預設名稱

    user_input_upper = stock_id_input.upper()
    
    # 處理大盤指數
    if user_input_upper in ["台股大盤", "大盤"]:
        stock_id = "^TWII"
        stock_name = "台灣加權指數"
    elif user_input_upper in ["美股大盤", "美盤", "美股"]:
        stock_id = "^GSPC"
        stock_name = "S&P 500 指數"
    elif re.match(r'^\d{4,6}[A-Z]?$', user_input_upper): # 台灣股票代碼
        stock_id = f"{user_input_upper}.TW"
        stock_name = get_stock_name_from_yahoo(stock_id) or user_input_upper
    else: # 可能是美股代碼或無法識別的代碼
        stock_id = user_input_upper
        stock_name = get_stock_name_from_yahoo(stock_id) or user_input_upper

    try:
        # 使用 YahooStock 獲取即時報價，並檢查是否成功獲取
        newprice_stock = YahooStock(stock_id)
        if not newprice_stock.name and stock_id not in ["^TWII", "^GSPC"]: # 如果 YahooStock 沒拿到名字，再試圖從 API 拿
             fetched_name = get_stock_name_from_yahoo(stock_id)
             if fetched_name != stock_id: # 如果確實拿到了不同的名字
                 newprice_stock.name = fetched_name
                 stock_name = fetched_name # 更新 stock_name
        
        # 再次檢查 YahooStock 是否有獲取到有效的數據
        if not newprice_stock.currentPrice and stock_id not in ["^TWII", "^GSPC", "^GSPC"]:
            # 對於普通的股票，如果沒有價格，可能代碼有誤
            return f"抱歉，無法獲取 {stock_name} ({stock_id_input}) 的即時資訊，請確認股票代碼是否正確。"

        price_data = stock_price(stock_id)
        news_data = str(stock_news(stock_name))
        news_data = remove_full_width_spaces(news_data)[:1024] # 限制新聞長度

        content_msg = (f'你現在是一位專業的證券分析師, 你會依據以下資料來進行分析並給出一份完整的分析報告:\n'
                       f'**股票代碼:** {stock_id}, **股票名稱:** {newprice_stock.name if newprice_stock.name else stock_name}\n'
                       f'**即時報價 (部分數據可能為延遲或收盤價):**\n {vars(newprice_stock)}\n'
                       f'**近期價格資訊 (近30天):**\n {price_data}\n')

        if stock_id not in ["^TWII", "^GSPC"]: # 大盤指數通常沒有基本面和配息
            stock_value_data = stock_fundamental(stock_id)
            stock_vividend_data = stock_dividend(stock_id)
            content_msg += f'**每季營收資訊：**\n {stock_value_data if stock_value_data is not None else "無法取得"}\n'
            content_msg += f'**配息資料：**\n {stock_vividend_data if stock_vividend_data is not None else "無法取得"}\n'

        content_msg += f'**近期新聞資訊:** \n {news_data if news_data else "無相關新聞"}\n'
        content_msg += f'請給我 {newprice_stock.name if newprice_stock.name else stock_name} 近期的趨勢報告。請以詳細、嚴謹及專業的角度撰寫此報告，並提及重要的數字，請使用台灣地區的繁體中文回答。'
        
        stock_link = f"https://finance.yahoo.com/quote/{stock_id}"
        
        system_prompt = (f"你現在是一位專業的證券分析師。請基於近期的股價走勢、基本面分析、新聞資訊等進行綜合分析。\n"
                         f"請提供以下內容：\n- **股名(股號)** ,現價(現漲跌幅),現價的資料的取得時間\n- 股價走勢\n- 基本面分析\n- 技術面分析\n- 消息面\n- 籌碼面\n- 推薦購買區間\n- 預計停利點\n- 建議買入張數\n- 市場趨勢\n- 配息分析\n- 綜合分析\n"
                         f"然後生成一份專業的趨勢分析報告。\n"
                         f"最後，請提供一個正確的股票連結：[股票資訊連結]({stock_link})。\n"
                         f"回應請使用繁體中文並格式化為 Markdown。")

        msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content_msg}]
        return get_analysis_reply(msg)

    except Exception as e:
        logger.error(f"股票分析流程失敗: {e}", exc_info=True)
        return f"抱歉，分析 {stock_id_input} 時發生錯誤，請確認股票代碼是否正確或稍後再試。錯誤訊息: {e}"


# --- UI & 對話 Helpers ---
async def analyze_sentiment(text: str) -> str:
    """分析用戶輸入的情緒，以調整 AI 的語氣"""
    msgs = [{"role":"system","content":"Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."}, {"role":"user","content":text}]
    out = await groq_chat_async(msgs, max_tokens=10, temperature=0)
    return (out or "neutral").strip().lower()

async def translate_text(text: str, target_lang_display: str) -> str:
    """調用 AI 進行翻譯"""
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text."
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{text}"}}'
    
    # 這裡也使用 get_analysis_reply 來確保備援機制
    messages = [{"role":"system","content":sys},{"role":"user","content":usr}]
    return get_analysis_reply(messages)


def set_user_persona(chat_id: str, key: str):
    """設定或隨機切換 AI 的人設"""
    if key == "random": key = random.choice(list(PERSONAS.keys()))
    if key not in PERSONAS: key = "sweet" # 預設為甜美女友
    user_persona[chat_id] = key
    return key

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    """根據人設和用戶情緒構建 AI 的系統提示詞"""
    key = user_persona.get(chat_id, "sweet")
    p = PERSONAS[key]
    return (f"你是一位「{p['title']}」。風格：{p['style']}\n"
            f"使用者情緒：{sentiment}；請調整語氣（開心→一起開心；難過/生氣→先共情、安撫再給建議；中性→自然聊天）。\n"
            f"回覆使用繁體中文，精煉自然，帶少量表情 {p['emoji']}。")

def build_quick_reply() -> QuickReply:
    """構建快速回覆按鈕"""
    return QuickReply(items=[
        QuickReplyButton(action=MessageAction(label="主選單", text="選單")),
        QuickReplyButton(action=MessageAction(label="台股大盤", text="台股大盤")),
        QuickReplyButton(action=MessageAction(label="美股大盤", text="美股大盤")),
        QuickReplyButton(action=MessageAction(label="查台積電", text="2330")),
        QuickReplyButton(action=MessageAction(label="查輝達", text="NVDA")),
        QuickReplyButton(action=MessageAction(label="查日圓", text="JPY")),
        QuickReplyButton(action=PostbackAction(label="💖 AI 人設", data="menu:persona")),
        QuickReplyButton(action=PostbackAction(label="🎰 彩票選單", data="menu:lottery")),
    ])

def reply_with_quick_bar(reply_token: str, text: str):
    """回覆訊息並帶上快速回覆按鈕"""
    line_bot_api.reply_message(
        reply_token,
        TextSendMessage(text=text, quick_reply=build_quick_reply())
    )

def build_main_menu_flex() -> FlexSendMessage:
    """構建主選單的 Flex Message"""
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[TextComponent(text="AI 助理主選單", weight="bold", size="lg")]),
        body=BoxComponent(
            layout="vertical", spacing="md",
            contents=[
                TextComponent(text="請選擇功能分類：", size="sm"),
                SeparatorComponent(margin="md"),
                ButtonComponent(action=PostbackAction(label="💹 金融查詢", data="menu:finance"), style="primary", color="#5E86C1"),
                ButtonComponent(action=PostbackAction(label="🎰 彩票分析", data="menu:lottery"), style="primary", color="#5EC186"),
                ButtonComponent(action=PostbackAction(label="💖 AI 角色扮演", data="menu:persona"), style="secondary"),
                ButtonComponent(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate"), style="secondary"),
                ButtonComponent(action=PostbackAction(label="⚙️ 系統設定", data="menu:settings"), style="secondary"),
            ]
        )
    )
    return FlexSendMessage(alt_text="主選單", contents=bubble)

def build_submenu_flex(kind: str) -> FlexSendMessage:
    """構建子選單的 Flex Message"""
    title = "子選單"
    buttons = []
    if kind == "finance":
        title = "💹 金融查詢"
        buttons = [
            ButtonComponent(action=MessageAction(label="台股大盤", text="台股大盤")),
            ButtonComponent(action=MessageAction(label="美股大盤", text="美股大盤")),
            ButtonComponent(action=MessageAction(label="黃金價格", text="金價")),
            ButtonComponent(action=MessageAction(label="日圓匯率", text="JPY")),
            ButtonComponent(action=MessageAction(label="查 2330 台積電", text="2330")),
            ButtonComponent(action=MessageAction(label="查 NVDA 輝達", text="NVDA")),
        ]
    elif kind == "lottery":
        title = "� 彩票分析"
        buttons = [
            ButtonComponent(action=MessageAction(label="大樂透", text="大樂透")),
            ButtonComponent(action=MessageAction(label="威力彩", text="威力彩")),
            ButtonComponent(action=MessageAction(label="今彩539", text="539")),
        ]
    elif kind == "persona":
        title = "💖 AI 角色扮演"
        buttons = [
            ButtonComponent(action=MessageAction(label="甜美女友", text="甜")),
            ButtonComponent(action=MessageAction(label="傲嬌女友", text="鹹")),
            ButtonComponent(action=MessageAction(label="萌系女友", text="萌")),
            ButtonComponent(action=MessageAction(label="酷系御姐", text="酷")),
            ButtonComponent(action=MessageAction(label="隨機切換", text="random")),
        ]
    elif kind == "translate":
        title = "🌐 翻譯工具"
        buttons = [
            ButtonComponent(action=MessageAction(label="翻成英文", text="翻譯->英文")),
            ButtonComponent(action=MessageAction(label="翻成日文", text="翻譯->日文")),
            ButtonComponent(action=MessageAction(label="翻成繁中", text="翻譯->繁體中文")),
            ButtonComponent(action=MessageAction(label="結束翻譯模式", text="翻譯->結束")),
        ]
    elif kind == "settings":
        title = "⚙️ 系統設定"
        buttons = [
            ButtonComponent(action=MessageAction(label="開啟自動回答 (群組)", text="開啟自動回答")),
            ButtonComponent(action=MessageAction(label="關閉自動回答 (群組)", text="關閉自動回答")),
        ]

    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[TextComponent(text=title, weight="bold", size="lg")]),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm")
    )
    return FlexSendMessage(alt_text=title, contents=bubble)

# ========== 5) LINE Handlers ==========
@handler.add(MessageEvent, message=TextMessage)
def on_message_text(event: MessageEvent):
    """處理接收到的文字訊息"""
    try:
        asyncio.run(handle_message_async(event))
    except Exception as e:
        logger.error(f"處理訊息失敗: {e}", exc_info=True)

@handler.add(PostbackEvent)
def on_postback(event: PostbackEvent):
    """處理接收到的 Postback 事件 (來自 Flex Message 的按鈕)"""
    data = (event.postback.data or "").strip()
    if data.startswith("menu:"):
        kind = data.split(":", 1)[-1]
        line_bot_api.reply_message(
            event.reply_token,
            [build_submenu_flex(kind), TextSendMessage(text="請選擇一項服務", quick_reply=build_quick_reply())]
        )
        return

async def handle_message_async(event: MessageEvent):
    """核心訊息處理邏輯 (異步)"""
    chat_id, msg_raw = get_chat_id(event), event.message.text.strip()
    reply_token, is_group = event.reply_token, not isinstance(event.source, SourceUser)

    try:
        bot_info = await run_in_threadpool(line_bot_api.get_bot_info)
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手" # 獲取 Bot 名稱失敗時的備用名稱

    if not msg_raw: return # 空訊息不處理
    
    # 群組模式下，如果未開啟自動回答，則必須 @Bot 才會回應
    if is_group and not auto_reply_status.get(chat_id, True) and not msg_raw.startswith(f"@{bot_name}"):
        return
    
    # 移除 @Bot 名稱 (如果存在)
    msg = msg_raw[len(f"@{bot_name}"):].strip() if msg_raw.startswith(f"@{bot_name}") else msg_raw
    if not msg: return # 移除 @Bot 名稱後如果為空，則不處理

    low = msg.lower() # 轉小寫方便判斷指令
    
    def is_stock_query(text: str) -> bool:
        """判斷是否為股票查詢指令"""
        text_upper = text.upper()
        if text_upper in ["台股大盤", "大盤", "美股大盤", "美盤", "美股"]: return True
        if re.match(r'^\d{4,6}[A-Z]?$', text_upper): return True # 台灣股票代碼 (4-6位數字，可選英文字母)
        if re.match(r'^[A-Z]{1,5}$', text_upper) and text_upper not in ["JPY", "USD", "EUR"]: return True # 美股代碼 (1-5位大寫字母，排除貨幣代碼)
        return False

    # --- 命令 & 功能觸發區 (按優先級排列) ---
    
    if low in ("menu", "選單", "主選單"):
        return line_bot_api.reply_message(reply_token, build_main_menu_flex())

    LOTTERY_KEYWORDS = ["大樂透", "威力彩", "539", "今彩539"]
    if msg in LOTTERY_KEYWORDS:
        try:
            analysis_report = await run_in_threadpool(get_lottery_analysis, msg)
            return reply_with_quick_bar(reply_token, analysis_report)
        except Exception as e:
            logger.error(f"彩票分析流程失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")

    if is_stock_query(msg):
        if not STOCK_ENABLED:
            return reply_with_quick_bar(reply_token, "抱歉，股票分析模組目前設定不完整或載入失敗。")
        try:
            analysis_report = await run_in_threadpool(get_stock_analysis, msg)
            return reply_with_quick_bar(reply_token, analysis_report)
        except Exception as e:
            logger.error(f"股票分析流程失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")

    if low in ("金價", "黃金"):
        try:
            # 直接調用整合後的黃金分析函式
            analysis_report = await run_in_threadpool(get_gold_ai_analysis_report)
            return reply_with_quick_bar(reply_token, analysis_report)
        except Exception as e:
            logger.error(f"黃金分析流程失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "抱歉，黃金分析服務暫時無法使用。")
    
    if low == "jpy":
        try:
            analysis_report = await run_in_threadpool(get_currency_analysis, "JPY")
            return reply_with_quick_bar(reply_token, analysis_report)
        except Exception as e:
            logger.error(f"日圓分析流程失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "抱歉，日圓匯率分析服務暫時無法使用。")

    if low in ("開啟自動回答", "關閉自動回答"):
        is_on = low == "開啟自動回答"
        auto_reply_status[chat_id] = is_on
        text = "✅ 已開啟自動回答" if is_on else "❌ 已關閉自動回答（群組需 @我 才回）"
        return reply_with_quick_bar(reply_token, text)

    if low.startswith("翻譯->"):
        lang = msg.split("->", 1)[1].strip()
        if lang == "結束":
            translation_states.pop(chat_id, None)
            return reply_with_quick_bar(reply_token, "✅ 已結束翻譯模式")
        translation_states[chat_id] = lang
        return reply_with_quick_bar(reply_token, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")

    persona_keys = {"甜":"sweet", "鹹":"salty", "萌":"moe", "酷":"cool", "random":"random"}
    if low in persona_keys:
        key = set_user_persona(chat_id, persona_keys[low])
        p = PERSONAS[key]
        txt = f"💖 已切換人設：{p['title']}\n\n{p['greetings']}"
        return reply_with_quick_bar(reply_token, txt)

    # --- 模式處理 & 一般對話 (最後的預設行為) ---
    if chat_id in translation_states:
        try:
            out = await translate_text(msg, translation_states[chat_id])
            return reply_with_quick_bar(reply_token, f"🌐 ({translation_states[chat_id]})\n{out}")
        except Exception as e:
            logger.error(f"翻譯失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "翻譯暫時失效，等我回神再來一次 🙏")

    # 預設的 AI 對話
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        
        # 為了傳遞給 get_analysis_reply，需要處理成正確的格式
        messages_for_ai = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]
        
        final_reply = get_analysis_reply(messages_for_ai) # 注意這裡不再是 await groq_chat_async
        
        history.extend([{"role":"user","content":msg}, {"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:] # 只保留最近的N輪對話
        return reply_with_quick_bar(reply_token, final_reply)
    except Exception as e:
        logger.error(f"AI 一般回覆失敗: {e}", exc_info=True)
        return reply_with_quick_bar(reply_token, "抱歉我剛剛走神了 😅 再說一次讓我補上！")

# ========== 6) FastAPI Routes ==========
@router.post("/callback")
async def callback(request: Request):
    """LINE Bot Webhook 回調接口"""
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        await run_in_threadpool(handler.handle, body.decode("utf-8"), signature)
    except InvalidSignatureError:
        logger.error("Invalid signature. Please check your channel access token/channel secret.")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Callback 處理失敗：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse({"status": "ok"})

@router.get("/")
async def root():
    """根路徑，用於健康檢查或顯示基本訊息"""
    return PlainTextResponse("LINE Bot is running.", status_code=200)
    
@router.get("/healthz")
async def healthz():
    """健康檢查接口，用於 Render.com 檢查服務狀態"""
    return PlainTextResponse("ok")

app.include_router(router)

# ========== 7) Local run ==========
if __name__ == "__main__":
    import uvicorn
    # 設置環境變數，僅用於本地測試
    os.environ["BASE_URL"] = "http://localhost:8000" # 本地測試的 URL
    os.environ["CHANNEL_ACCESS_TOKEN"] = "YOUR_LINE_CHANNEL_ACCESS_TOKEN" # 替換為你的 Line Access Token
    os.environ["CHANNEL_SECRET"] = "YOUR_LINE_CHANNEL_SECRET"     # 替換為你的 Line Channel Secret
    os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"         # 替換為你的 Groq API Key
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"       # 替換為你的 OpenAI API Key (可選)

    # 重新初始化客戶端，確保本地測試時能讀取到環境變數
    CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
    CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    line_bot_api = LineBotApi(CHANNEL_TOKEN)
    handler = WebhookHandler(CHANNEL_SECRET)
    async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
    sync_groq_client = Groq(api_key=GROQ_API_KEY)
    if OPENAI_API_KEY:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai_client = None


    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Local ser�