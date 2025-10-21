import os
import re
import random
import logging
import asyncio
import requests
from datetime import datetime
from typing import Tuple
from bs4 import BeautifulSoup
import yfinance as yf  # 依賴：pip install yfinance websockets
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.routing import APIRouter
from contextlib import asynccontextmanager
import uvicorn
from linebot.exceptions import InvalidSignatureError
# --- 繁體中文解：[修正] 匯入 V3 所需的 Configuration 和 ApiClient ---
from linebot.v3.messaging import (
    MessagingApi, ReplyMessageRequest, TextMessage, PushMessageRequest,
    Configuration, ApiClient
)
from linebot.v3.webhook import WebhookParser  # v3 Parser
from linebot.v3.webhooks import MessageEvent, TextMessageContent, AudioMessageContent, PostbackEvent  # v3 Events
import openai
from groq import AsyncGroq
import httpx
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# ── 全域變數與 Mock 定義 ────────────────────────────────────────────────────
PERSONA_ALIAS = {"sweet": "sweet", "random": "random"}  # 人設別名
PERSONAS = {
    "sweet": {"title": "甜美助手", "style": "溫柔親切", "emoji": "😊"}
}  # 預設人設
user_persona = {}  # 每個聊天的人設字典
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
TRANSLATE_CMD = re.compile(r'^翻譯\s*(.*)$')  # 翻譯指令正則
INLINE_TRANSLATE = re.compile(r'^(en|ja|zh|英文|日文|中文)\s+(.+)$')  # 內聯翻譯正則
LOTTERY_OK = True  # 彩票模組旗標
conversation_history = {}  # 聊天歷史字典
MAX_HISTORY_LEN = 10  # 歷史長度限制
OPENAI_OK = False  # 全域旗標（在 lifespan 中設定）
GROQ_OK = False
OPENAI_LAST_REASON = "uninitialized"
GROQ_LAST_REASON = "uninitialized"
DISABLE_GROQ = os.getenv("DISABLE_GROQ", "false").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 繁體中文解：請再次確認 Render.com 上的 KEY 是 CHANNEL_TOKEN ---
CHANNEL_TOKEN = os.getenv("CHANNEL_TOKEN", "dummy")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET", "dummy")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# --- 繁體中文解：[修正] line_bot_api 改為在各函數內部動態建立 ---
# --- 繁體中文解：[修正] 建立 V3 所需的全域 Configuration 物件 ---
configuration = Configuration(access_token=CHANNEL_TOKEN) if CHANNEL_TOKEN != "dummy" else None

# --- 繁體中文解：[修正] WebhookParser 在 V3 只需要 CHANNEL_SECRET。
# --- 繁體中文解：並檢查 CHANNEL_SECRET 是否為 dummy ---
parser = WebhookParser(CHANNEL_SECRET) if CHANNEL_SECRET != "dummy" else None

# Mock 客戶端
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
GROQ_MODEL_FALLBACK = "llama-3.1-8b-instant"

# Mock 函數（之後替換為真實實作）
def get_chat_id(event): 
    return str(event.source.user_id) if hasattr(event.source, 'user_id') else "test_chat"

async def _tstate_set(chat_id, lang): 
    logger.debug(f"[翻譯狀態] 設定 {chat_id[:10]}... 為 {lang}")
    pass  # 翻譯狀態

def _tstate_get(chat_id): 
    logger.debug(f"[翻譯狀態] 取得 {chat_id[:10]}...")
    return None

def _tstate_clear(chat_id): 
    logger.debug(f"[翻譯狀態] 清除 {chat_id[:10]}...")
    pass

async def reply_text_with_tts_and_extras(reply_tok, text, event=None):
    # --- 繁體中文解：[修正] 檢查 V3 configuration 物件是否存在 ---
    if configuration is not None:
        try:
            # --- 繁體中文解：[修正] 使用 with...as... 語法動態建立 V3 ApiClient 和 MessagingApi ---
            async with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                logger.debug(f"準備回覆 (reply_token: {reply_tok[:10]}...): {text[:50]}...")
                request = ReplyMessageRequest(reply_token=reply_tok, messages=[TextMessage(text=text)])
                # --- 繁體中文解：[修正] V3 的 API 呼叫現在是異步(async)的，所以使用 await ---
                await line_bot_api.reply_message(request)
                logger.debug(f"已成功回覆文字：{text[:50]}...")
        except Exception as e:
            logger.error(f"回覆訊息失敗 (Token: {reply_tok[:10]}...)，嘗試使用 push：{e}")
            if event:
                try:
                    # --- 繁體中文解：[修正] 推播備援也需要動態建立 V3 ApiClient ---
                    async with ApiClient(configuration) as api_client:
                        line_bot_api = MessagingApi(api_client)
                        chat_id = get_chat_id(event)
                        logger.debug(f"推播備援：準備推播至 {chat_id[:20]}...")
                        push_request = PushMessageRequest(to=chat_id, messages=[TextMessage(text=text)])
                        await line_bot_api.push_message(push_request)
                        logger.info(f"推播備援成功至 {chat_id[:20]}...")
                except Exception as push_e:
                    logger.error(f"推播備援失敗：{push_e}")
            else:
                logger.error("推播備援失敗：缺少 event 參數")
    else:
        print(f"[MOCK] 回覆：{text}")

async def reply_menu_with_hint(reply_tok, menu, hint=""): 
    # --- 繁體中文解：[修正] 檢查 V3 configuration 物件 ---
    if configuration is not None:
        # 選單需自訂（QuickReply 在 v3 為 FlexMessage 或 other）
        logger.info(f"準備回覆選單 (Token: {reply_tok[:10]}...)")
        # --- 繁體中文解：[修正] 這裡未來實作時，也需要使用 with ApiClient... 方式呼叫 line_bot_api ---
        print("已發送選單（v3 需調整）")
    else:
        print("[MOCK] 已發送選單")

def build_main_menu(): 
    logger.debug("呼叫：build_main_menu()")
    return []  # 真實：v3 QuickReply 或 FlexMessage

def build_submenu(kind): 
    logger.debug(f"呼叫：build_submenu(kind={kind})")
    return []

async def translate_text(text, lang): 
    logger.debug(f"呼叫：translate_text(text={text[:20]}..., lang={lang})")
    return f"翻譯結果：{text} → {lang}"

async def analyze_sentiment(msg): 
    logger.debug(f"呼叫：analyze_sentiment(msg={msg[:20]}...)")
    return "neutral"

async def groq_chat_async(messages):
    logger.debug(f"呼叫：groq_chat_async() (messages len: {len(messages)})")
    if async_groq_client and GROQ_OK:
        try:
            resp = await async_groq_client.chat.completions.create(
                model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=500
            )
            logger.debug("Groq 呼叫成功")
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq 呼叫失敗：{e}")
            return "AI 分析服務暫時不可用，請稍後再試。"
    logger.debug("Groq 未啟用，回傳模擬 LLM 回應")
    return "模擬 LLM：你好！這是自由回應模式～（設定 GROQ_API_KEY 以使用真實 LLM）"

async def speech_to_text_async(audio): 
    logger.debug(f"呼叫：speech_to_text_async() (audio len: {len(audio)})")
    return "模擬轉錄文字：這是語音內容"

def run_in_threadpool(func, *args):
    logger.debug(f"呼叫：run_in_threadpool(func={func.__name__})")
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return loop.run_in_executor(executor, lambda: func(*args))

def run_lottery_analysis(msg): 
    logger.debug(f"呼叫：run_lottery_analysis(msg={msg})")
    return f"彩票分析：{msg} 建議號碼 1-2-3-4-5-6（模擬資料）"

class YahooStock: 
    def __init__(self, id): 
        logger.debug(f"呼叫：YahooStock(id={id})")
        self.name = f"股票 {id}（模擬名稱）"

def stock_price(id): 
    logger.debug(f"呼叫：stock_price(id={id})")
    return pd.DataFrame({"Close": [100.0, 101.0, 99.0]})

def stock_news(hint): 
    logger.debug(f"呼叫：stock_news(hint={hint})")
    return ["模擬新聞：股票上漲中"]

def stock_fundamental(id): 
    logger.debug(f"呼叫：stock_fundamental(id={id})")
    return "模擬基本面：EPS 5.0，營收成長 10%"

def stock_dividend(id): 
    logger.debug(f"呼叫：stock_dividend(id={id})")
    return "模擬配息：2.5%"

def get_analysis_reply(messages): 
    logger.debug(f"呼叫：get_analysis_reply() (messages len: {len(messages)})")
    return "模擬分析：建議買進，目標價 110 元"

def log_provider_status(): 
    logger.info(f"供應商狀態：OpenAI={OPENAI_OK}, Groq={GROQ_OK}")

# ── 人設與 Prompt 建構 ──────────────────────────────────────────────────────
def set_user_persona(chat_id: str, key: str):
    """設定使用者人設"""
    key_mapped = PERSONA_ALIAS.get(key, key)
    if key_mapped == "random": 
        key_mapped = random.choice(list(PERSONAS.keys()))
    if key_mapped not in PERSONAS: 
        key_mapped = "sweet"
    user_persona[chat_id] = key_mapped
    logger.info(f"人設切換：{chat_id[:20]}... -> {PERSONAS[key_mapped]['title']}")
    return key_mapped

def build_persona_prompt(chat_id: str, sentiment: str) -> str:
    """建構人設 Prompt"""
    key = user_persona.get(chat_id, "sweet")
    p = PERSONAS[key]
    prompt = (f"你是一位「{p['title']}」。風格：{p['style']}。\n"
              f"使用者情緒：{sentiment}。\n"
              f"回覆請精煉自然，使用繁體中文，帶少量表情 {p['emoji']}。")
    logger.debug(f"建構人設 Prompt (key={key}): {prompt[:50]}...")
    return prompt

# ── 金價 / 外匯 / 股票 ──────────────────────────────────────────────────────
BOT_GOLD_URL = "https://rate.bot.com.tw/gold?Lang=zh-TW"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def get_bot_gold_quote() -> dict:
    """獲取台灣銀行金價"""
    logger.debug(f"呼叫：get_bot_gold_quote() (URL: {BOT_GOLD_URL})")
    try:
        r = requests.get(BOT_GOLD_URL, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        
        m_time = re.search(r"掛牌時間[:：]\s*([0-9]{4}/[0-9]{2}/[0-9]{2}\s+[0-9]{2}:[0-9]{2})", text)
        listed_at = m_time.group(1) if m_time else "未知"
        
        m_sell = re.search(r"本行賣出\s*([0-9,]+(?:\.[0-9]+)?)", text)
        m_buy = re.search(r"本行買進\s*([0-9,]+(?:\.[0-9]+)?)", text)
        
        if not (m_sell and m_buy): 
            raise RuntimeError("找不到『本行賣出/本行買進』欄位")
        
        sell = float(m_sell.group(1).replace(",", ""))
        buy = float(m_buy.group(1).replace(",", ""))
        
        logger.debug(f"金價資料：賣出={sell}, 買進={buy}, 時間={listed_at}")
        return {"listed_at": listed_at, "sell_twd_per_g": sell, "buy_twd_per_g": buy}
    except Exception as e:
        logger.error(f"金價獲取失敗：{e}")
        return {"listed_at": "錯誤", "sell_twd_per_g": 0, "buy_twd_per_g": 0}

FX_CODES = {"USD","TWD","JPY","EUR","GBP","CNY","HKD","AUD","CAD","CHF","SGD","KRW","NZD","THB","MYR","IDR","PHP","INR","ZAR"}
FX_ALIAS = {"日圓":"JPY","日元":"JPY","美元":"USD","台幣":"TWD","新台幣":"TWD","人民幣":"CNY","港幣":"HKD","韓元":"KRW","歐元":"EUR","英鎊":"GBP"}

def _is_fx_query(text: str) -> bool:
    """判斷是否為外匯查詢"""
    t = text.strip().upper()
    if t in FX_CODES or t in set(FX_ALIAS.values()): 
        return True
    is_match = bool(re.match(r"^[A-Za-z]{3}[\s/\-_]?([A-Za-z]{3})?$", t))
    logger.debug(f"_is_fx_query({text}) -> {is_match}")
    return is_match

def _normalize_fx_token(tok: str) -> str: 
    """標準化外匯代碼"""
    return FX_ALIAS.get(tok.strip().upper(), tok.strip().upper())

def parse_fx_pair(user_text: str) -> Tuple[str,str,str]:
    """解析外匯貨幣對"""
    logger.debug(f"呼叫：parse_fx_pair(user_text={user_text})")
    raw = user_text.strip()
    m = re.findall(r"[A-Za-z\u4e00-\u9fa5]{2,5}", raw)
    toks = [_normalize_fx_token(x) for x in m]
    toks = [x for x in toks if x in FX_CODES]
    
    if not toks:
        t = _normalize_fx_token(raw)
        if len(t) == 3 and t in FX_CODES: 
            base, quote = t, "TWD"
        else: 
            base, quote = "USD", "JPY"
    elif len(toks) == 1: 
        base, quote = toks[0], "TWD"
    else:
        base, quote = toks[0], toks[1]
    
    symbol = f"{base}{quote}=X"
    link = f"https://finance.yahoo.com/quote/{symbol}/"
    logger.debug(f"解析結果：{base}/{quote}, symbol={symbol}")
    return base, quote, link

def fetch_fx_quote_yf(symbol: str):
    """從 Yahoo Finance 獲取外匯報價"""
    logger.debug(f"呼叫：fetch_fx_quote_yf(symbol={symbol})")
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(period="5d", interval="1d")
        if df is None or df.empty: 
            logger.warning(f"yf.history 查無資料 for {symbol}")
            return None, None, None, None
        
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df)>=2 else None
        last_price = float(last_row["Close"])
        change_pct = None if prev_row is None else (last_price/float(prev_row["Close"]) - 1.0)*100.0
        
        ts = last_row.name
        if hasattr(ts, "tz_convert"):
            ts_iso = ts.tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M %Z")
        else:
            ts_iso = str(ts)
        
        logger.debug(f"外匯 {symbol}：價格={last_price}, 變動={change_pct}, 時間={ts_iso}")
        return last_price, change_pct, ts_iso, df
    except Exception as e:
        logger.error(f"fetch_fx_quote_yf error for {symbol}: {e}")
        return None, None, None, None

def render_fx_report(base, quote, link, last, chg, ts, df) -> str:
    """渲染外匯報表"""
    logger.debug("呼叫：render_fx_report()")
    trend = ""
    if df is not None and not df.empty:
        diff = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[0])
        trend = "上升" if diff>0 else ("下跌" if diff<0 else "持平")
    
    lines = [f"#### 外匯報告（查匯優先）\n- 幣別對：**{base}/{quote}**\n- 來源：Yahoo Finance\n- 連結：{link}"]
    if last is not None: 
        lines.append(f"- 目前匯率：**{last:.6f}**（{base}/{quote}）")
    if chg is not None: 
        lines.append(f"- 日變動：**{chg:+.2f}%**")
    if ts: 
        lines.append(f"- 資料時間：{ts}")
    if trend: 
        lines.append(f"- 近 5 日趨勢：{trend}")
    lines.append(f"\n[外匯連結（Yahoo）]({link})")
    return "\n".join(lines)

TW_TICKER_RE = re.compile(r"^\d{4,6}[A-Za-z]?$")
US_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}$")

def _is_stock_query(text: str) -> bool:
    """判斷是否為股票查詢"""
    t = text.strip()
    if t in ("大盤","台股大盤","台灣大盤","美盤","美股大盤","美股"): 
        return True
    if TW_TICKER_RE.match(t): 
        return True
    if US_TICKER_RE.match(t) and t.upper() in {"NVDA","AAPL","TSLA","MSFT"}: 
        return True
    logger.debug(f"_is_stock_query({text}) -> False")
    return False

def _normalize_ticker_and_name(user_text: str) -> Tuple[str,str,str]:
    """標準化股票代碼和名稱"""
    logger.debug(f"呼叫：_normalize_ticker_and_name(user_text={user_text})")
    raw = user_text.strip()
    if raw in ("大盤","台股大盤","台灣大盤"): 
        return "^TWII","台灣大盤","https://tw.finance.yahoo.com/quote/%5ETWII/"
    if raw in ("美盤","美股大盤","美股"):     
        return "^GSPC","美國大盤","https://tw.finance.yahoo.com/quote/%5EGSPC/"
    
    ticker = raw.upper()
    link = f"https://tw.stock.yahoo.com/quote/{ticker}" if TW_TICKER_RE.match(ticker) else f"https://tw.finance.yahoo.com/quote/{ticker}"
    logger.debug(f"解析結果：ticker={ticker}, link={link}")
    return ticker, ticker, link

def _safe_to_str(x) -> str:
    """安全轉換為字串"""
    try: 
        return str(x)
    except Exception: 
        return repr(x)

def _remove_full_width_spaces(data):
    """移除全形空格"""
    if isinstance(data, list): 
        return [_remove_full_width_spaces(i) for i in data]
    if isinstance(data, str):  
        return data.replace('\u3000',' ')
    return data

def _truncate_text(data, max_length=1024):
    """截斷文字"""
    if isinstance(data, list): 
        return [_truncate_text(i, max_length) for i in data]
    if isinstance(data, str):  
        return data[:max_length]
    return data

def build_stock_prompt_block(stock_id: str, stock_name_hint: str) -> Tuple[str, dict]:
    """建構股票分析 Prompt（stub 版使用 yfinance）"""
    logger.debug(f"呼叫：build_stock_prompt_block(stock_id={stock_id})")
    try:
        ys = YahooStock(stock_id)
        price_df = stock_price(stock_id)
        news = _remove_full_width_spaces(stock_news(stock_name_hint))
        news = _truncate_text(news, 1024)
        
        fund_text = div_text = None
        if stock_id not in ["^TWII","^GSPC"]:
            fund_text = stock_fundamental(stock_id)
            div_text = stock_dividend(stock_id)
        
        blk = [
            f"**股票代碼:** {stock_id}, **股票名稱:** {ys.name}",
            f"**即時資訊:** 使用 yfinance 獲取",
            f"近期價格資訊:\n{price_df if not price_df.empty else '無法取得'}"
        ]
        if stock_id not in ["^TWII","^GSPC"]:
            blk += [f"每季營收資訊:\n{fund_text}", f"配息資料:\n{div_text}"]
        blk.append(f"近期新聞資訊:\n{news}")
        
        result = "\n".join(_safe_to_str(x) for x in blk)
        logger.debug(f"股票 Prompt 建構完成，長度：{len(result)}")
        return result, {}
    except Exception as e:
        logger.error(f"股票資料建構失敗：{e}")
        return f"股票資料獲取錯誤：{e}", {}

def render_stock_report(stock_id: str, stock_link: str, content_block: str) -> str:
    """渲染股票分析報告"""
    logger.debug(f"呼叫：render_stock_report(stock_id={stock_id})")
    sys_prompt = ("你現在是一位專業的證券分析師。請基於近期走勢、基本面、新聞與籌碼概念進行綜合分析，"
                  "條列清楚、數字精確、可讀性高。\n"
                  "- 股名(股號)/現價(與漲跌幅)/資料時間\n"
                  "- 走勢\n"
                  "- 基本面\n"
                  "- 技術面\n"
                  "- 消息面\n"
                  "- 籌碼面\n"
                  "- 建議買進區間\n"
                  "- 停利點\n"
                  "- 建議部位\n"
                  "- 總結\n"
                  f"最後附上正確連結：[股票資訊連結]({stock_link})。\n"
                  "使用台灣繁體中文，回覆精簡有力。")
    try:
        result = get_analysis_reply([{"role":"system","content":sys_prompt},{"role":"user","content":content_block}])
        logger.debug(f"股票分析完成，長度：{len(result)}")
        return result
    except Exception as e:
        logger.error(f"股票分析失敗：{e}")
        return f"（分析模型不可用）原始資料：\n{content_block[:500]}...\n\n連結：{stock_link}"

# ── 事件處理 ─────────────────────────────────────────────────────────────────
async def handle_text_message(event: MessageEvent):
    """處理文字訊息（所有分支均走統一回覆，確保 Quick Reply 底部顯示）"""
    chat_id = get_chat_id(event)
    msg_raw = (event.message.text or "").strip()
    reply_tok = event.reply_token
    
    logger.info(f"收到文字訊息：{msg_raw[:50]}... (chat_id: {chat_id[:20]}..., reply_token: {reply_tok[:10]}...)")
    
    if not msg_raw: 
        logger.debug("空訊息，忽略")
        return
    
    try:
        bot_name = "AI 助手"
        # --- 繁體中文解：[修正] 檢查 V3 configuration 物件 ---
        if configuration is not None:
            # --- 繁體中文解：[修正] 使用 with...as... 語法動態建立 V3 ApiClient 和 MessagingApi ---
            async with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                bot_info = await line_bot_api.get_bot_info()
                bot_name = bot_info.display_name
        else:
            bot_name = "AI 助手 (MOCK)"
        logger.debug(f"Bot 名稱：{bot_name}")
    except Exception as e:
        logger.warning(f"獲取 Bot 資訊失敗：{e}")
        bot_name = "AI 助手"

    msg = msg_raw
    if msg_raw.startswith(f"@{bot_name}"):
        msg = re.sub(f'^@{re.escape(bot_name)}\\s*','', msg_raw).strip()
        logger.debug(f"提及 Bot，清理後訊息：{msg[:30]}...")
    
    if not msg: 
        logger.debug("清理後訊息為空，忽略")
        return

    m = TRANSLATE_CMD.match(msg)
    if m:
        lang_token = m.group(1)
        logger.info(f"分支：匹配到翻譯指令 (TRANSLATE_CMD): {lang_token}")
        rev = {"english":"英文","japanese":"日文","korean":"韓文","vietnamese":"越南文","繁體中文":"繁體中文","中文":"繁體中文"}
        lang_display = rev.get(lang_token.lower(), lang_token)
        await _tstate_set(chat_id, lang_display)
        await reply_text_with_tts_and_extras(reply_tok, f"🌐 已開啟翻譯 → {lang_display}，請直接輸入要翻的內容。", event=event)
        logger.info(f"開啟翻譯模式：{lang_display}")
        return
    
    if msg.startswith("翻譯->"):
        lang = msg.split("->",1)[1].strip()
        logger.info(f"分支：匹配到翻譯切換 (翻譯->): {lang}")
        if lang=="結束":
            await _tstate_clear(chat_id)
            await reply_text_with_tts_and_extras(reply_tok, "✅ 已結束翻譯模式", event=event)
            logger.info(f"結束翻譯模式：{chat_id}")
        else:
            await _tstate_set(chat_id, lang)
            await reply_text_with_tts_and_extras(reply_tok, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。", event=event)
            logger.info(f"開啟翻譯模式：{lang}")
        return
    
    im = INLINE_TRANSLATE.match(msg)
    if im:
        lang_key, text_to_translate = im.group(1).lower(), im.group(2)
        logger.info(f"分支：匹配到內聯翻譯 (INLINE_TRANSLATE): {lang_key} <- {text_to_translate[:30]}...")
        lang_display = {"en":"英文","eng":"英文","英文":"英文","ja":"日文","jp":"日文","日文":"日文","zh":"繁體中文","繁中":"繁體中文","中文":"繁體中文"}.get(lang_key,"英文")
        out = await translate_text(text_to_translate, lang_display)
        await reply_text_with_tts_and_extras(reply_tok, out, event=event)
        return

    current_lang = _tstate_get(chat_id)
    if current_lang:
        logger.info(f"分支：處於翻譯模式 ({current_lang})")
        out = await translate_text(msg, current_lang)
        await reply_text_with_tts_and_extras(reply_tok, out, event=event)
        return

    low = msg.lower()
    if low in ("menu","選單","主選單"):
        logger.info("分支：匹配到關鍵字 (menu/選單)")
        await reply_menu_with_hint(reply_tok, build_main_menu())
        return
    
    if msg in PERSONA_ALIAS:
        logger.info(f"分支：匹配到人設切換 ({msg})")
        key = set_user_persona(chat_id, msg)
        p = PERSONAS[key]
        await reply_text_with_tts_and_extras(reply_tok, f"已切換為「{p['title']}」模式～{p['emoji']}", event=event)
        return

    if msg in ("金價","黃金"):
        logger.info("分支：匹配到金價查詢")
        try:
            logger.debug("呼叫：get_bot_gold_quote()")
            d = get_bot_gold_quote()
            ts, sell, buy = d.get("listed_at") or "（未標示）", d["sell_twd_per_g"], d["buy_twd_per_g"]
            spread = sell - buy
            txt = (f"**金價（台灣銀行）**\n"
                   f"- 掛牌時間：{ts}\n"
                   f"- 賣出(1g)：{sell:,.0f} 元\n"
                   f"- 買進(1g)：{buy:,.0f} 元\n"
                   f"- 價差：{spread:,.0f} 元\n"
                   f"來源：{BOT_GOLD_URL}")
            await reply_text_with_tts_and_extras(reply_tok, txt, event=event)
        except Exception as e:
            logger.error(f"金價查詢失敗：{e}")
            await reply_text_with_tts_and_extras(reply_tok, "抱歉，目前無法取得金價，請稍後再試。", event=event)
        return

    if msg in ("大樂透","威力彩","539","今彩539","雙贏彩","3星彩","三星彩","4星彩","38樂合彩","39樂合彩","49樂合彩","運彩"):
        logger.info(f"分支：匹配到彩票查詢 ({msg})，模組狀態：LOTTERY_OK={LOTTERY_OK}")
        try:
            logger.debug(f"呼叫：run_in_threadpool(run_lottery_analysis, {msg})")
            report = await run_in_threadpool(run_lottery_analysis, msg)
            await reply_text_with_tts_and_extras(reply_tok, report, event=event)
            logger.info(f"彩票回覆成功：{msg}")
        except Exception as e:
            logger.error(f"彩票分析失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_tok, f"抱歉，分析 {msg} 時發生錯誤：{e}", event=event)
        return

    if _is_fx_query(msg):
        logger.info(f"分支：匹配到外匯查詢 (_is_fx_query): {msg}")
        try:
            logger.debug(f"呼叫：parse_fx_pair({msg})")
            base, quote, link = parse_fx_pair(msg)
            symbol = f"{base}{quote}=X"
            logger.debug(f"呼叫：fetch_fx_quote_yf({symbol})")
            last, chg, ts, df = fetch_fx_quote_yf(symbol)
            logger.debug("呼叫：render_fx_report()")
            report = render_fx_report(base, quote, link, last, chg, ts, df)
            await reply_text_with_tts_and_extras(reply_tok, report, event=event)
        except Exception as e:
            logger.error(f"外匯查詢失敗：{e}")
            await reply_text_with_tts_and_extras(reply_tok, f"抱歉，取得 {msg} 匯率時發生錯誤：{e}", event=event)
        return

    if _is_stock_query(msg):
        logger.info(f"分支：匹配到股票查詢 (_is_stock_query): {msg}")
        try:
            logger.debug(f"呼叫：_normalize_ticker_and_name({msg})")
            ticker, name_hint, link = _normalize_ticker_and_name(msg)
            logger.debug(f"呼叫：run_in_threadpool(build_stock_prompt_block, {ticker}, ...)")
            content_block, _ = await run_in_threadpool(build_stock_prompt_block, ticker, name_hint)
            logger.debug(f"呼叫：run_in_threadpool(render_stock_report, {ticker}, ...)")
            report = await run_in_threadpool(render_stock_report, ticker, link, content_block)
            await reply_text_with_tts_and_extras(reply_tok, report, event=event)
        except Exception as e:
            logger.error(f"股票查詢失敗：{e}")
            await reply_text_with_tts_and_extras(reply_tok, f"抱歉，取得 {msg} 分析時發生錯誤：{e}\n請稍後再試或換個代碼。", event=event)
        return

    logger.info(f"分支：進入一般聊天模式 (Groq/OpenAI)")
    try:
        history = conversation_history.get(chat_id, [])
        logger.debug("呼叫：analyze_sentiment()")
        sentiment = await analyze_sentiment(msg)
        logger.debug(f"呼叫：build_persona_prompt() (chat_id: {chat_id[:10]}..., sentiment: {sentiment})")
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        
        messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]
        logger.debug(f"呼叫：groq_chat_async() (History len: {len(history)})")
        final_reply = await groq_chat_async(messages)
        
        history.extend([{"role":"user","content":msg},{"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
        logger.debug("對話歷史已儲存")
        
        await reply_text_with_tts_and_extras(reply_tok, final_reply, event=event)
    except Exception as e:
        logger.error(f"一般聊天失敗：{e}")
        await reply_text_with_tts_and_extras(reply_tok, "抱歉我剛剛走神了 😅 再說一次讓我補上！", event=event)

async def handle_audio_message(event: MessageEvent):
    """處理語音訊息（統一走帶 Quick Reply 的回覆，底部顯示）"""
    reply_tok = event.reply_token
    logger.info(f"收到語音訊息：{event.message.id} (reply_token: {reply_tok[:10]}...)")
    
    try:
        # --- 繁體中文解：[修正] 檢查 V3 configuration 物件 ---
        if configuration is None:
            logger.warning("語音處理：處於 Mock 模式")
            await reply_text_with_tts_and_extras(reply_tok, "🎧 [MOCK] 語音收到！目前語音轉文字失敗，請稍後再試。", event=event)
            return
        
        audio_in = None
        # --- 繁體中文解：[修正] 使用 with...as... 語法動態建立 V3 ApiClient 和 MessagingApi ---
        async with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            logger.debug(f"呼叫：line_bot_api.get_message_content({event.message.id})")
            response = await line_bot_api.get_message_content(event.message.id)
            audio_in = await response.content.read()
            logger.debug(f"取得語音資料，長度：{len(audio_in)}")
        
        logger.debug("呼叫：speech_to_text_async()")
        text = await speech_to_text_async(audio_in)
        
        if not text:
            logger.warning("語音轉文字失敗 (回傳空值)")
            await reply_text_with_tts_and_extras(reply_tok, "🎧 語音收到！目前語音轉文字失敗，請稍後再試。", event=event)
            return
        
        logger.info(f"語音轉文字成功：{text[:50]}...")
        await reply_text_with_tts_and_extras(reply_tok, f"🎧 我聽到了：\n{text}", event=event)
    except Exception as e:
        logger.error(f"語音處理失敗: {e}", exc_info=True)
        await reply_text_with_tts_and_extras(reply_tok, "抱歉，語音處理失敗，請稍後再試。", event=event)

async def handle_postback(event: PostbackEvent):
    """處理 Postback 事件（走選單回覆，確保 Quick Reply 底部）"""
    data = event.postback.data or ""
    reply_tok = event.reply_token
    logger.info(f"收到 Postback 事件：{data} (reply_token: {reply_tok[:10]}...)")
    
    if data.startswith("menu:"):
        kind = data.split(":",1)[-1]
        logger.info(f"分支：Postback 匹配 (menu:): {kind}")
        logger.debug(f"呼叫：build_submenu(kind={kind})")
        await reply_menu_with_hint(reply_tok, build_submenu(kind), hint="👇 子選單")
    else:
        logger.warning(f"未處理的 Postback data: {data}")

async def handle_events(events):
    """處理事件列表"""
    logger.info(f"準備處理 {len(events)} 個事件...")
    for i, event in enumerate(events):
        logger.debug(f"--- 正在處理事件 {i+1}/{len(events)} (類型: {type(event)}) ---")
        try:
            if isinstance(event, MessageEvent):
                if isinstance(event.message, TextMessageContent):
                    logger.debug("事件類型：TextMessage")
                    await handle_text_message(event)
                elif isinstance(event.message, AudioMessageContent):
                    logger.debug("事件類型：AudioMessage")
                    await handle_audio_message(event)
                else:
                    logger.debug(f"事件類型：未處理的 MessageEvent ({type(event.message)})")
            elif isinstance(event, PostbackEvent):
                logger.debug("事件類型：PostbackEvent")
                await handle_postback(event)
            else:
                logger.debug(f"事件類型：未處理的 WebhookEvent ({type(event)})")
        except Exception as e:
            logger.error(f"處理事件 {i+1} (類型: {type(event)}) 時發生頂層錯誤：{e}", exc_info=True)
            pass
    logger.info(f"--- 所有 {len(events)} 個事件處理完畢 ---")


# ── FastAPI ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    [CHANGED]
    1) 只呼叫 api.line.me 設定 Webhook
    2) 啟動健康檢查：OpenAI / Groq
    """
    global OPENAI_OK, GROQ_OK, OPENAI_LAST_REASON, GROQ_LAST_REASON
    logger.info("應用程式啟動 (lifespan)...")

    # 1) LINE Webhook（官方域名）
    # --- 繁體中文解：[修正] 檢查 BASE_URL 是否存在，以及 CHANNEL_TOKEN 是否有設定 (不再檢查 line_bot_api) ---
    if BASE_URL and CHANNEL_TOKEN != "dummy":
        logger.info(f"準備更新 Webhook 至：{BASE_URL}/callback")
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                headers={"Authorization":f"Bearer {CHANNEL_TOKEN}","Content-Type":"application/json"}
                payload={"endpoint":f"{BASE_URL}/callback"}
                r = await c.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=payload)
                r.raise_for_status()
                logger.info("✅ Webhook 更新成功（api.line.me=%s）", r.status_code)
        except Exception as e:
            logger.warning(f"⚠️ Webhook 更新失敗（api.line.me）：{e}")
    else:
        # --- 繁體中文解：[修正] 更新 Mock 模式的警告訊息 ---
        logger.warning("Webhook 未更新：未設定 BASE_URL 或 CHANNEL_TOKEN (Mock 模式)")


    # 2) OpenAI 健檢
    logger.info("執行 OpenAI 健檢...")
    if openai_client and OPENAI_API_KEY:
        try:
            _ = openai_client.models.list()
            OPENAI_OK = True
            OPENAI_LAST_REASON = ""
            logger.info("✅ OpenAI 健檢通過")
        except Exception as e:
            OPENAI_OK = False
            OPENAI_LAST_REASON = f"startup_check_failed: {e}"
            logger.error("❌ OpenAI 健檢失敗：%s", e)
    else:
        OPENAI_OK = False
        if not OPENAI_API_KEY:
            OPENAI_LAST_REASON = "missing_api_key"
            logger.info("ℹ️ OpenAI 未啟用（缺少 API Key）")
        else:
            logger.info("ℹ️ OpenAI 未啟用 (openai_client 為 None)")

    # 3) Groq 健檢（可用且未手動停用）
    logger.info("執行 Groq 健檢...")
    if not DISABLE_GROQ and async_groq_client and GROQ_API_KEY:
        try:
            resp = await async_groq_client.chat.completions.create(
                model=GROQ_MODEL_FALLBACK,
                messages=[{"role":"system","content":"ping"},{"role":"user","content":"pong"}],
                max_tokens=1, temperature=0
            )
            GROQ_OK = True
            GROQ_LAST_REASON = ""
            logger.info("✅ Groq 健檢通過")
        except Exception as e:
            GROQ_OK = False
            GROQ_LAST_REASON = f"startup_check_failed: {e}"
            if "organization_restricted" in str(e):
                logger.error("❌ Groq 組織受限（organization_restricted），已停用 Groq。")
            else:
                logger.error("❌ Groq 健檢失敗：%s", e)
    else:
        GROQ_OK = False
        if DISABLE_GROQ:
            GROQ_LAST_REASON = "manually_disabled"
            logger.info("ℹ️ Groq 被手動停用 (DISABLE_GROQ=true)")
        elif not GROQ_API_KEY:
            GROQ_LAST_REASON = "missing_api_key"
            logger.info("ℹ️ Groq 未啟用（缺少 API Key）")
        else:
             logger.info("ℹ️ Groq 未啟用 (async_groq_client 為 None)")

    # 摘要
    log_provider_status()
    logger.info("Lifespan 啟動程序完成，應用程式準備就緒。")
    yield
    # (關閉時執行的程式碼)
    logger.info("應用程式關閉 (lifespan)...")

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.5.15-v3-fix") # --- 繁體中文解：更新版本號 ---
router = APIRouter()

@router.post("/callback")
async def callback(request: Request):
    logger.info("收到 /callback 請求")
    # --- 繁體中文解：[修正] 檢查 parser 物件是否存在 ---
    if parser is None:
        logger.error("Callback：處於 Mock 模式 (parser 為 None)，請檢查 CHANNEL_SECRET 環境變數。")
        return JSONResponse({"status": "mock mode, no parser"}, status_code=200)
    
    # 獲取簽章和內容
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    logger.debug(f"Callback：Signature: {signature[:10]}..., Body size: {len(body)}")

    try:
        logger.debug("解析 Webhook 事件...")
        events = parser.parse(body.decode("utf-8"), signature)
        logger.debug(f"共 {len(events)} 個事件，準備進入 handle_events")
        await handle_events(events)
        logger.info(f"成功處理完 {len(events)} 個事件")
    except InvalidSignatureError:
        logger.error(f"Invalid signature 驗證失敗 (Signature: {signature})，請檢查 CHANNEL_SECRET 是否正確。")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Callback 處理事件時發生未預期錯誤：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
    
    return JSONResponse({"status":"ok"})

@router.get("/")
async def root(): 
    logger.debug("收到 / (root) 請求")
    return PlainTextResponse("LINE Bot is running.", status_code=200)

@router.get("/healthz")
async def healthz(): 
    return PlainTextResponse("ok", status_code=200)

# === [ADDED] 供應商健康檢視 API ===
@router.get("/health/providers")
async def providers_health():
    logger.info("收到 /health/providers 請求")
    return {
        "openai": {"ok": OPENAI_OK, "reason": OPENAI_LAST_REASON},
        "groq": {"ok": GROQ_OK, "reason": GROQ_LAST_REASON},
        "ts": datetime.utcnow().isoformat() + "Z",
    }

app.include_router(router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"準備啟動 Uvicorn (app_fastapi:app) 於 0.0.0.0:{port}")
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)