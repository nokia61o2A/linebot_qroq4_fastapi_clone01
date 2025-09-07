# ========== app_fastapi.py ==========
# -*- coding: utf-8 -*-
# 功能：LINE Bot（FastAPI）整合 金價 / 匯率 / 股票(含台股ETF英字尾) / 彩票 / 翻譯 / 人設
# 特色：
# 1) 金價：台銀黃金牌價 → 動態解析表頭找「賣出」，失敗自動備援 XAUUSD+USDTWD 換算（元/公克）
# 2) 股票：Yahoo quote API 快照（自動將台股數字代碼補 .TW；亦支援 00937B 這類英字尾）
# 3) 翻譯：異步 Groq，指令「翻譯->英文」等；「翻譯->結束」關閉
# 4) 人設：甜/鹹/萌/酷 + 隨機
# 5) UI：主選單 / 子選單 / 快速回覆列
# 6) Webhook：開機自動綁定

# ========== 1) Imports ==========
import os
import re
import random
import logging
import asyncio
from typing import Dict, List
from contextlib import asynccontextmanager

# --- 數據處理與爬蟲 ---
import math
import requests
from bs4 import BeautifulSoup
import httpx
import pandas as pd
import yfinance as yf

# --- FastAPI 與 LINE Bot SDK ---
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
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

# --- 擴充：自家模組（非必要則關閉） ---
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    LOTTERY_ENABLED = True
except Exception as e:
    logging.warning(f"彩票模組停用：{e}")
    LOTTERY_ENABLED = False

try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
    STOCK_ENABLED = True
except Exception as e:
    logging.warning(f"股票模組停用：{e}")
    STOCK_ENABLED = False

# ========== 2) Setup ==========
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# --- 環境變數 ---
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 可留空

if not all([BASE_URL, CHANNEL_TOKEN, CHANNEL_SECRET, GROQ_API_KEY]):
    raise RuntimeError("缺少必要環境變數：BASE_URL / CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET / GROQ_API_KEY")

# --- API 客戶端 ---
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
sync_groq_client = Groq(api_key=GROQ_API_KEY)

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.warning(f"OpenAI 初始化失敗：{e}，將僅使用 Groq。")

# Groq 模型（現行可用）
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-70b-versatile")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

# 彩票
if LOTTERY_ENABLED:
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()

# 狀態
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}
auto_reply_status: Dict[str, bool] = {}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji": "🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji": "😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji": "✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議", "greetings": "我在。說重點。", "emoji": "🧊⚡️"}
}
LANGUAGE_MAP = {
    "英文": "English", "日文": "Japanese", "韓文": "Korean",
    "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"
}

# ========== 3) FastAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時綁定 LINE Webhook
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            headers = {
                "Authorization": f"Bearer {CHANNEL_TOKEN}",
                "Content-Type": "application/json"
            }
            payload = {"endpoint": f"{BASE_URL}/callback"}
            r = await c.put(
                "https://api.line.me/v2/bot/channel/webhook/endpoint",
                headers=headers, json=payload
            )
            r.raise_for_status()
            logger.info(f"✅ Webhook 更新成功: {r.status_code}")
    except Exception as e:
        logger.error(f"Webhook 更新失敗: {e}", exc_info=True)
    yield

app = FastAPI(lifespan=lifespan, title="LINE Bot", version="1.0.0")
router = APIRouter()

# ========== 4) 共用 Helper ==========
def get_chat_id(event: MessageEvent) -> str:
    if isinstance(event.source, SourceGroup): return event.source.group_id
    if isinstance(event.source, SourceRoom):  return event.source.room_id
    return event.source.user_id

def remove_full_width_spaces(text: str) -> str:
    return text.replace("\u3000", " ") if isinstance(text, str) else text

# ========== 5) AI 封裝 ==========
def get_analysis_reply(messages: List[dict]) -> str:
    # 優先 OpenAI（若有設），失敗或未設則 Groq 主要→備援
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI 失敗：{e}")

    try:
        resp = sync_groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY, messages=messages,
            max_tokens=2000, temperature=0.8
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq 主要模型失敗：{e}")
        try:
            resp = sync_groq_client.chat.completions.create(
                model=GROQ_MODEL_FALLBACK, messages=messages,
                max_tokens=1500, temperature=1.0
            )
            return resp.choices[0].message.content
        except Exception as ee:
            logger.error(f"Groq 備援也失敗：{ee}", exc_info=True)
            return "抱歉，AI 分析師目前連線不穩定，請稍後再試。"

async def groq_chat_async(messages, max_tokens=600, temperature=0.7):
    resp = await async_groq_client.chat.completions.create(
        model=GROQ_MODEL_FALLBACK,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return resp.choices[0].message.content.strip()

# ========== 6) 金價（穩健版 + 備援） ==========
def get_bot_gold_quote() -> dict:
    """
    回傳 {'twd_per_gram_sell': float, 'row_name': str, 'asof': str}
    來源：台灣銀行黃金牌價頁（動態找「賣出」欄與「黃金（公克）」列）
    """
    url = "https://rate.bot.com.tw/gold?Lang=zh-TW"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=12)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # 取頁面時間（若有）
    asof = None
    for el in soup.find_all(text=True):
        t = (el or "").strip()
        if any(k in t for k in ("資料時間", "更新時間", "最後更新")):
            asof = t
            break

    tables = soup.find_all("table")
    if not tables:
        raise RuntimeError("找不到任何表格")

    candidate = []
    for tb in tables:
        thead = tb.find("thead")
        if not thead: 
            continue
        heads = [th.get_text(strip=True) for th in thead.find_all("th")]
        if not heads:
            continue
        has_sell = any("賣出" in h or "本行賣出" in h or "賣出價" in h for h in heads)
        if has_sell:
            candidate.append((tb, heads))

    if not candidate:
        raise RuntimeError("找不到含『賣出』欄位的表格")

    KEYWORDS = ("黃金牌價", "黃金（公克）", "黃金(公克)", "1 公克", "1公克", "黃金")
    for tb, heads in candidate:
        sell_idx = None
        for i, h in enumerate(heads):
            if any(k in h for k in ("賣出", "本行賣出", "賣出價")):
                sell_idx = i
                break
        if sell_idx is None:
            continue

        tbody = tb.find("tbody") or tb
        for tr in tbody.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if not tds:
                continue
            row_text = " ".join(td.get_text(" ", strip=True) for td in tds)
            if any(k in row_text for k in KEYWORDS):
                if sell_idx < len(tds):
                    cell = tds[sell_idx].get_text(strip=True).replace(",", "")
                    cell = re.sub(r"[^\d\.]", "", cell)
                    if cell:
                        val = float(cell)
                        return {
                            "twd_per_gram_sell": val,
                            "row_name": tds[0].get_text(strip=True),
                            "asof": asof or "（頁面未標示）"
                        }

    raise RuntimeError("未能在表格中定位『黃金（公克）』的賣出價欄位")

def get_gold_analysis() -> str:
    logger.info("開始執行黃金價格分析...")

    def xauusd_fallback() -> dict:
        # 以 Yahoo Finance 參考：XAUUSD=X（每盎司美元），USDTWD=X（匯率）
        px = yf.Ticker("XAUUSD=X").history(period="1d")
        if px.empty:
            raise RuntimeError("XAUUSD 抓不到價格")
        usd_per_oz = float(px["Close"].iloc[-1])

        fx = yf.Ticker("USDTWD=X").history(period="1d")
        if fx.empty:
            raise RuntimeError("USDTWD 抓不到匯率")
        twd_per_usd = float(fx["Close"].iloc[-1])

        twd_per_gram = usd_per_oz * twd_per_usd / 31.1034768
        return {
            "twd_per_gram_sell": round(twd_per_gram, 2),
            "row_name": "估算：國際金價換算（每公克台幣）",
            "asof": "XAUUSD / USDTWD 當日參考"
        }

    try:
        q = get_bot_gold_quote()
        price = q["twd_per_gram_sell"]
        row = q["row_name"]
        asof = q["asof"]
        return (
            f"最新台銀黃金牌價（{row}，賣出）：**{price:,.2f} 元/公克**\n"
            f"資料時間：{asof}\n\n"
            f"簡評：\n"
            f"- 以台銀實際牌價為準，適合臨櫃買賣參考。\n"
            f"- 若官網暫不可用，系統會切換國際金價推估作為備援。"
        )
    except Exception as e:
        logger.warning(f"台銀金價抓取失敗，改用備援：{e}")
        fb = xauusd_fallback()
        price = fb["twd_per_gram_sell"]
        row = fb["row_name"]
        asof = fb["asof"]
        return (
            f"（備援）{row}：**約 {price:,.2f} 元/公克**\n"
            f"資料來源：{asof}\n\n"
            f"說明：此為 XAU/USD 與 USD/TWD 換算之參考價，非台銀櫃檯實際牌價。"
        )

# ========== 7) 外匯 ==========
def get_currency_analysis(target_currency: str) -> str:
    logger.info(f"開始執行 {target_currency} 匯率分析...")
    try:
        base_currency = 'TWD'
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("result") == "success":
            rate = data["rates"].get(base_currency)
            if rate is None:
                return f"抱歉，API 中找不到 {base_currency} 匯率。"
            content_msg = (
                "你是一位外匯分析師，請根據最新即時匯率撰寫一則日圓(JPY)匯率快訊。\n"
                f"最新：1 日圓 (JPY) ≈ {rate:.5f} 新台幣 (TWD)。\n"
                "要求：\n1) 直接報價\n2) 旅遊/換匯簡評\n3) 1 句實用建議\n4) 使用繁體中文"
            )
            msgs = [
                {"role": "system", "content": "你是一位專業的外匯分析師。"},
                {"role": "user", "content": content_msg},
            ]
            return get_analysis_reply(msgs)
        return f"抱歉，匯率服務回應：{data.get('error-type','未知錯誤')}"
    except Exception as e:
        logger.error(f"外匯分析錯誤：{e}", exc_info=True)
        return "抱歉，外匯資料暫時無法取得。"

# ========== 8) 彩票 ==========
def get_lottery_analysis(lottery_type_input: str):
    if not LOTTERY_ENABLED:
        return "抱歉，彩票分析功能尚未啟用。"
    logger.info(f"開始執行 {lottery_type_input} 彩票分析...")
    typ = lottery_type_input
    if "威力" in typ: last = lottery_crawler.super_lotto()
    elif "大樂" in typ: last = lottery_crawler.lotto649()
    elif "539" in typ:  last = lottery_crawler.daily_cash()
    else: return f"抱歉，暫不支援 {typ}。"

    try:
        info = caiyunfangwei_crawler.get_caiyunfangwei()
        content_msg = (
            f'你是專業樂透分析師，使用 {typ} 的資料撰寫報告：\n'
            f'近幾期號碼：\n{last}\n'
            f'今天日期：{info.get("今天日期","未知")}；歲次：{info.get("今日歲次","未知")}；財神方位：{info.get("財神方位","未知")}\n'
            '請產出趨勢分析＋冷/熱號＋三組推薦號（數字由小到大；威力彩含二區），繁體中文。'
        )
    except Exception:
        content_msg = (
            f'你是專業樂透分析師，使用 {typ} 的資料撰寫報告：\n'
            f'近幾期號碼：\n{last}\n'
            '財神方位暫缺；請仍完成趨勢分析與三組號碼（繁中）。'
        )
    msgs = [
        {"role": "system", "content": f"你現在是一位專業的彩券分析師。"},
        {"role": "user", "content": content_msg},
    ]
    return get_analysis_reply(msgs)

# ========== 9) 股票 ==========
# 讀取台股代碼對照表（若無則空表）
_stock_df_cache = None
def load_stock_data():
    global _stock_df_cache
    if _stock_df_cache is None:
        try:
            _stock_df_cache = pd.read_csv("name_df.csv")
        except Exception:
            _stock_df_cache = pd.DataFrame(columns=["股號", "股名"])
    return _stock_df_cache

def get_stock_name(twid: str):
    df = load_stock_data()
    out = df[df["股號"] == twid]
    return out.iloc[0]["股名"] if not out.empty else None

def normalize_symbol(user_input: str) -> str:
    """
    - 台股：4~6碼數字，可含1碼英字尾 → 補 .TW（支援 00937B）
    - 大盤：台股大盤/大盤 → ^TWII；美股大盤/美盤/美股 → ^GSPC
    - 其他：轉大寫回傳（NVDA/AAPL…）
    """
    s = user_input.strip()
    u = s.upper()
    if u in ("台股大盤", "大盤"): return "^TWII"
    if u in ("美股大盤", "美盤", "美股"): return "^GSPC"
    if re.fullmatch(r"\d{4,6}[A-Z]?", u):
        return f"{u}.TW"
    return u

def get_stock_analysis(stock_id_input: str) -> str:
    logger.info(f"開始執行 {stock_id_input} 股票分析...")
    norm = normalize_symbol(stock_id_input)

    # 顯示名稱
    disp_name = stock_id_input
    if norm.endswith(".TW"):
        num = stock_id_input.upper().rstrip(".TW")
        nm = get_stock_name(num)
        disp_name = nm if nm else stock_id_input

    try:
        # 快照（穩定，不受頁面 class 影響）
        snap = YahooStock(norm)

        # 價格時間序列（若取不到，讓子函式自己處理例外）
        price_data = stock_price(norm) if STOCK_ENABLED else "（價格序列模組未載入）"

        # 新聞
        news_raw = str(stock_news(disp_name)) if STOCK_ENABLED else ""
        news_data = remove_full_width_spaces(news_raw)[:1024]

        content = (
            "你現在是一位專業的證券分析師，請依據以下資料撰寫完整分析報告：\n"
            f"**股票代碼**：{norm}；**股票名稱**：{snap.name}\n"
            f"**即時報價**：{vars(snap)}\n"
            f"**近期價格資訊**：\n{price_data}\n"
        )

        if norm not in ("^TWII", "^GSPC"):
            try:
                value_data = stock_fundamental(norm)
            except Exception:
                value_data = None
            try:
                div_data = stock_dividend(norm)
            except Exception:
                div_data = None
            content += f"**每季營收/財報**：\n{value_data if value_data is not None else '無法取得'}\n"
            content += f"**配息資料**：\n{div_data if div_data is not None else '無法取得'}\n"

        content += f"**近期新聞**：\n{news_data}\n"
        content += f"請以嚴謹專業、繁體中文，條列並給出策略建議與風險提醒。"

        link = f"https://finance.yahoo.com/quote/{norm}"
        sys = (
            "你是專業證券分析師。請包含：\n"
            "- 股名(股號)、現價(漲跌幅)、現價的時間\n"
            "- 股價走勢 / 基本面 / 技術面 / 消息面 / 籌碼面\n"
            "- 推薦買進區間、停利點(%)、建議買入張數\n"
            "- 市場趨勢判讀、配息分析、綜合結論\n"
            f"最後附上正確連結：[股票資訊連結]({link})。\n"
            "使用 Markdown 與繁體中文。"
        )
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": content}]
        return get_analysis_reply(msgs)

    except Exception as e:
        logger.error(f"股票分析流程失敗：{e}", exc_info=True)
        return f"抱歉，分析「{stock_id_input}」時發生錯誤或該代碼暫無資料。"

# ========== 10) 介面：快速回覆 / 選單 ==========
def build_quick_reply() -> QuickReply:
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
    line_bot_api.reply_message(
        reply_token, TextSendMessage(text=text, quick_reply=build_quick_reply())
    )

def build_main_menu_flex() -> FlexSendMessage:
    bubble = BubbleContainer(
        direction="ltr",
        header=BoxComponent(layout="vertical", contents=[
            TextComponent(text="AI 助理主選單", weight="bold", size="lg")
        ]),
        body=BoxComponent(
            layout="vertical", spacing="md",
            contents=[
                TextComponent(text="請選擇功能：", size="sm"),
                SeparatorComponent(margin="md"),
                ButtonComponent(action=PostbackAction(label="💹 金融查詢", data="menu:finance"),
                                style="primary", color="#5E86C1"),
                ButtonComponent(action=PostbackAction(label="🎰 彩票分析", data="menu:lottery"),
                                style="primary", color="#5EC186"),
                ButtonComponent(action=PostbackAction(label="💖 AI 角色扮演", data="menu:persona"),
                                style="secondary"),
                ButtonComponent(action=PostbackAction(label="🌐 翻譯工具", data="menu:translate"),
                                style="secondary"),
                ButtonComponent(action=PostbackAction(label="⚙️ 系統設定", data="menu:settings"),
                                style="secondary"),
            ]
        )
    )
    return FlexSendMessage(alt_text="主選單", contents=bubble)

def build_submenu_flex(kind: str) -> FlexSendMessage:
    title, buttons = "子選單", []
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
        title = "🎰 彩票分析"
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
        header=BoxComponent(layout="vertical", contents=[
            TextComponent(text=title, weight="bold", size="lg")
        ]),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm")
    )
    return FlexSendMessage(alt_text=title, contents=bubble)

# ========== 11) LINE Handlers ==========
@handler.add(MessageEvent, message=TextMessage)
def on_message_text(event: MessageEvent):
    try:
        asyncio.run(handle_message_async(event))
    except Exception as e:
        logger.error(f"Handle message failed: {e}", exc_info=True)

@handler.add(PostbackEvent)
def on_postback(event: PostbackEvent):
    data = (event.postback.data or "").strip()
    if data.startswith("menu:"):
        kind = data.split(":", 1)[-1]
        line_bot_api.reply_message(
            event.reply_token,
            [build_submenu_flex(kind),
             TextSendMessage(text="請選擇一項服務", quick_reply=build_quick_reply())]
        )

def _is_stock_query(text: str) -> bool:
    u = text.upper().strip()
    if u in ["台股大盤", "大盤", "美股大盤", "美盤", "美股"]:
        return True
    # 台股代碼：4~6位數字，末可接 1 英字（支援 00937B）
    if re.fullmatch(r"^\d{4,6}[A-Z]?$", u):
        return True
    # 美股代碼：1~5 位英字（排除 JPY 這類匯率關鍵詞）
    if re.fullmatch(r"^[A-Z]{1,5}$", u) and u not in {"JPY"}:
        return True
    return False

async def analyze_sentiment(text: str) -> str:
    msgs = [
        {"role": "system", "content": "Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
        {"role": "user", "content": text},
    ]
    out = await groq_chat_async(msgs, max_tokens=10, temperature=0)
    return (out or "neutral").strip().lower()

async def translate_text(text: str, target_lang_display: str) -> str:
    target = LANGUAGE_MAP.get(target_lang_display, target_lang_display)
    sys = "You are a precise translation engine. Output ONLY the translated text."
    usr = f'{{"source_language":"auto","target_language":"{target}","text_to_translate":"{text}"}}'
    return await groq_chat_async(
        [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        max_tokens=800, temperature=0.2
    )

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
        f"使用者情緒：{sentiment}（開心→一起開心；難過/生氣→先共情安撫再建議；中性→自然）。\n"
        f"回覆使用繁體中文，精煉自然，帶少量表情 {p['emoji']}。"
    )

# 主處理
async def handle_message_async(event: MessageEvent):
    chat_id = get_chat_id(event)
    msg_raw = event.message.text.strip()
    reply_token = event.reply_token
    is_group = not isinstance(event.source, SourceUser)

    try:
        bot_info = await run_in_threadpool(line_bot_api.get_bot_info)
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if not msg_raw:
        return

    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True

    # 群組中若關閉自動，需 @ 機器人
    if is_group and not auto_reply_status.get(chat_id, True) and not msg_raw.startswith(f"@{bot_name}"):
        return
    msg = msg_raw[len(f"@{bot_name}"):].strip() if msg_raw.startswith(f"@{bot_name}") else msg_raw
    if not msg:
        return

    low = msg.lower()

    # --- 選單 ---
    if low in ("menu", "選單", "主選單"):
        return line_bot_api.reply_message(reply_token, build_main_menu_flex())

    # --- 彩票 ---
    if msg in ("大樂透", "威力彩", "539"):
        try:
            analysis_report = await run_in_threadpool(get_lottery_analysis, msg)
            return reply_with_quick_bar(reply_token, analysis_report)
        except Exception as e:
            logger.error(f"彩票分析失敗：{e}", exc_info=True)
            return reply_with_quick_bar(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")

    # --- 股票 ---
    if _is_stock_query(msg):
        if not STOCK_ENABLED:
            return reply_with_quick_bar(reply_token, "抱歉，股票分析模組未啟用或載入失敗。")
        try:
            analysis_report = await run_in_threadpool(get_stock_analysis, msg)
            return reply_with_quick_bar(reply_token, analysis_report)
        except Exception as e:
            logger.error(f"股票分析失敗：{e}", exc_info=True)
            return reply_with_quick_bar(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")

    # --- 金價 ---
    if low in ("金價", "黃金"):
        try:
            analysis_report = await run_in_threadpool(get_gold_analysis)
            return reply_with_quick_bar(reply_token, analysis_report)
        except Exception as e:
            logger.error(f"金價流程失敗：{e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "抱歉，金價服務暫時無法使用。")

    # --- 匯率（日圓）---
    if low == "jpy":
        try:
            analysis_report = await run_in_threadpool(get_currency_analysis, "JPY")
            return reply_with_quick_bar(reply_token, analysis_report)
        except Exception as e:
            logger.error(f"日圓匯率流程失敗：{e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "抱歉，日圓匯率服務暫時無法使用。")

    # --- 自動回覆設定（群組）---
    if low in ("開啟自動回答", "關閉自動回答"):
        is_on = (low == "開啟自動回答")
        auto_reply_status[chat_id] = is_on
        text = "✅ 已開啟自動回答" if is_on else "❌ 已關閉自動回答（群組需 @我 才回）"
        return reply_with_quick_bar(reply_token, text)

    # --- 翻譯模式 ---
    if low.startswith("翻譯->"):
        lang = msg.split("->", 1)[1].strip()
        if lang == "結束":
            translation_states.pop(chat_id, None)
            return reply_with_quick_bar(reply_token, "✅ 已結束翻譯模式")
        translation_states[chat_id] = lang
        return reply_with_quick_bar(reply_token, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")

    # --- 人設切換 ---
    persona_keys = {"甜": "sweet", "鹹": "salty", "萌": "moe", "酷": "cool", "random": "random"}
    if low in persona_keys:
        key = set_user_persona(chat_id, persona_keys[low])
        p = PERSONAS[key]
        return reply_with_quick_bar(
            reply_token, f"💖 已切換人設：{p['title']}\n\n{p['greetings']}"
        )

    # --- 翻譯狀態處理 ---
    if chat_id in translation_states:
        try:
            out = await translate_text(msg, translation_states[chat_id])
            return reply_with_quick_bar(reply_token, f"🌐 ({translation_states[chat_id]})\n{out}")
        except Exception as e:
            logger.error(f"翻譯失敗：{e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "翻譯暫時失效，等我回神再來一次 🙏")

    # --- 一般聊天（帶人設、情緒）---
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": msg}]
        final_reply = await groq_chat_async(messages)
        history.extend([{"role": "user", "content": msg}, {"role": "assistant", "content": final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN * 2:]
        return reply_with_quick_bar(reply_token, final_reply)
    except Exception as e:
        logger.error(f"AI 回覆失敗：{e}", exc_info=True)
        return reply_with_quick_bar(reply_token, "抱歉我剛剛走神了 😅 再說一次讓我補上！")

# ========== 12) FastAPI Routes ==========
@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        await run_in_threadpool(handler.handle, body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Callback 處理失敗：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse({"status": "ok"})

@router.get("/")
async def root():
    return PlainTextResponse("LINE Bot is running.", status_code=200)

@router.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

app.include_router(router)

# ========== 13) Local run ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)