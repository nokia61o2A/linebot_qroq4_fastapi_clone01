# ========== 1) Imports ==========
import os
import re
import random
import logging
import asyncio
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

# --- 數據處理與爬蟲 ---
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

# --- 【靈活載入】自訂模組（可缺省） ---
try:
    from TaiwanLottery import TaiwanLotteryCrawler
    from my_commands.CaiyunfangweiCrawler import CaiyunfangweiCrawler
    LOTTERY_ENABLED = True
except Exception as e:
    logging.warning(f"無法載入彩票模組，彩票功能將停用。原因: {e}")
    LOTTERY_ENABLED = False

try:
    from my_commands.stock.stock_price import stock_price
    from my_commands.stock.stock_news import stock_news
    from my_commands.stock.stock_value import stock_fundamental
    from my_commands.stock.stock_rate import stock_dividend
    from my_commands.stock.YahooStock import YahooStock
    STOCK_ENABLED = True
except Exception as e:
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
    raise RuntimeError("缺少必要環境變數：請設定 BASE_URL / CHANNEL_ACCESS_TOKEN / CHANNEL_SECRET / GROQ_API_KEY")

# --- API 用戶端初始化 ---
line_bot_api = LineBotApi(CHANNEL_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)
sync_groq_client = Groq(api_key=GROQ_API_KEY)

openai_client: Optional[openai.OpenAI] = None
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.warning(f"初始化 OpenAI 失敗，將僅使用 Groq。原因：{e}")
else:
    logger.warning("未設定 OPENAI_API_KEY，分析功能將僅使用 Groq。")

# Groq 模型名稱可由環境覆寫，避免使用已退役型號
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-8b-instant")
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

# 彩票
if LOTTERY_ENABLED:
    lottery_crawler = TaiwanLotteryCrawler()
    caiyunfangwei_crawler = CaiyunfangweiCrawler()

# --- 狀態字典與常數 ---
conversation_history: Dict[str, List[dict]] = {}
MAX_HISTORY_LEN = 10
user_persona: Dict[str, str] = {}
translation_states: Dict[str, str] = {}   # chat_id -> 目標語言（顯示名稱）
auto_reply_status: Dict[str, bool] = {}

PERSONAS = {
    "sweet": {"title": "甜美女友", "style": "溫柔體貼，鼓勵安慰", "greetings": "親愛的～我在這裡聽你說 🌸", "emoji":"🌸💕😊"},
    "salty": {"title": "傲嬌女友", "style": "機智吐槽，壞壞但有溫度", "greetings": "你又來啦？說吧，哪裡卡住了。😏", "emoji":"😏🙄"},
    "moe":   {"title": "萌系女友", "style": "動漫語氣＋可愛顏文字", "greetings": "呀呼～今天也被我治癒一下嗎？(ﾉ>ω<)ﾉ", "emoji":"✨🎀"},
    "cool":  {"title": "酷系御姐", "style": "冷靜精煉，關鍵建議", "greetings": "我在。說重點。", "emoji":"🧊⚡️"}
}
LANGUAGE_MAP = {"英文": "English", "日文": "Japanese", "韓文": "Korean", "越南文": "Vietnamese", "繁體中文": "Traditional Chinese"}


# ========== 3) FastAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時更新 LINE Webhook
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

# --- AI & 分析 ---
def get_analysis_reply(messages: List[dict]) -> str:
    """優先使用 OpenAI；失敗改用 Groq；再失敗回覆友善訊息。"""
    try:
        if not openai_client:
            raise Exception("OpenAI client not initialized.")
        resp = openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        return resp.choices[0].message.content
    except Exception as openai_err:
        logger.warning(f"OpenAI API 失敗: {openai_err}")
        try:
            resp = sync_groq_client.chat.completions.create(
                model=GROQ_MODEL_PRIMARY, messages=messages, max_tokens=2000, temperature=0.8
            )
            return resp.choices[0].message.content
        except Exception as groq_err:
            logger.warning(f"Groq 主要模型失敗: {groq_err}")
            try:
                resp = sync_groq_client.chat.completions.create(
                    model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=1500, temperature=1.0
                )
                return resp.choices[0].message.content
            except Exception as fallback_err:
                logger.error(f"所有 AI API 都失敗: {fallback_err}")
                return "抱歉，AI 分析師目前連線不穩定，請稍後再試。"

async def groq_chat_async(messages: List[dict], max_tokens=600, temperature=0.7) -> str:
    resp = await async_groq_client.chat.completions.create(
        model=GROQ_MODEL_FALLBACK, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    return resp.choices[0].message.content.strip()

# --- 金融 & 彩票 ---
def get_gold_analysis():
    logger.info("開始執行黃金價格分析...")
    try:
        url = "https://rate.bot.com.tw/gold?Lang=zh-TW"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        table = soup.find("table", {"class": "table-striped"})
        rows = table.find("tbody").find_all("tr")
        gold_price = None
        for row in rows:
            tds = row.find_all("td")
            if len(tds) > 1 and "黃金牌價" in tds[0].get_text():
                gold_price = tds[4].get_text(strip=True)
                break
        if not gold_price:
            raise ValueError("找不到黃金牌價欄位")

        msg = [
            {"role": "system", "content": "你是一位專業的金融記者。"},
            {"role": "user", "content": (
                f"請根據台銀黃金牌價撰寫快訊：黃金（1公克）賣出價 {gold_price} 元。"
                "開頭直接點出價格；簡述此價位在近期所處區間；提及影響因子（通膨、美元、避險）；用繁中。"
            )}
        ]
        return get_analysis_reply(msg)
    except Exception as e:
        logger.error(f"黃金價格分析失敗: {e}", exc_info=True)
        return "抱歉，目前無法獲取黃金價格，可能是網站結構已變更，請稍後再試。"

def get_currency_analysis(target_currency: str):
    logger.info(f"開始執行 {target_currency} 匯率分析...")
    try:
        base_currency = 'TWD'
        url = f"https://open.er-api.com/v6/latest/{target_currency.upper()}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("result") != "success":
            return f"抱歉，匯率 API 回應失敗：{data.get('error-type', '未知錯誤')}"
        rate = data["rates"].get(base_currency)
        if rate is None:
            return f"抱歉，API 中找不到 {base_currency} 匯率。"

        msg = [
            {"role": "system", "content": "你是一位專業的外匯分析師。"},
            {"role": "user", "content": (
                f"撰寫 JPY 快訊：1 JPY = {rate:.5f} TWD。\n"
                "請：1) 直述當前匯率；2) 說明對旅遊/換匯的相對划算度；3) 給一則換匯族實用建議；"
                "4) 用繁體中文、口吻輕鬆。")}
        ]
        return get_analysis_reply(msg)
    except Exception as e:
        logger.error(f"外匯分析失敗: {e}", exc_info=True)
        return "抱歉，日圓匯率分析服務暫時無法使用。"

# ---------- 股票輔助 ----------
stock_data_df: Optional[pd.DataFrame] = None

def load_stock_data() -> pd.DataFrame:
    global stock_data_df
    if stock_data_df is None:
        try:
            stock_data_df = pd.read_csv('name_df.csv')
        except FileNotFoundError:
            logger.error("找不到 name_df.csv，台股代碼→名稱對照將停用。")
            stock_data_df = pd.DataFrame(columns=['股號', '股名'])
    return stock_data_df

def get_stock_name(stock_id: str) -> Optional[str]:
    df = load_stock_data()
    row = df[df['股號'] == stock_id]
    return None if row.empty else row.iloc[0]['股名']

def remove_full_width_spaces(s: str) -> str:
    return s.replace('\u3000', ' ') if isinstance(s, str) else s

def normalize_stock_input(user_input: str) -> (str, str):
    """
    回傳 (yfinance/查價用代碼, 顯示名稱建議)。
    規則：
    - 台股大盤/美股大盤 特例 -> ^TWII / ^GSPC
    - 純數字或「數字+一個大寫字母」（ETF/權證等） -> 補 .TW（大小寫不敏感）
      例：2330 -> 2330.TW, 00937b -> 00937B.TW
    - 純字母 1~5 碼 -> 視為美股代碼（NVDA、QQQ...）
    - 其餘維持原樣（例如已含 .TW 或指數符號）
    """
    s = user_input.strip()
    s_upper = s.upper()

    # 指數簡稱
    if s_upper in ["台股大盤", "大盤"]:
        return "^TWII", "台灣加權指數"
    if s_upper in ["美股大盤", "美盤", "美股"]:
        return "^GSPC", "S&P 500 指數"

    # 已含 .TW 直接回傳
    if s_upper.endswith(".TW") or s_upper.startswith("^"):
        return s_upper, s_upper

    # 台股格式：4~6位數字 + 可選 1 位英文字母（不分大小寫）
    if re.fullmatch(r'\d{4,6}[A-Z]?', s_upper):
        symbol = f"{s_upper}.TW"
        # 顯示名稱：若有本地對照名稱就用
        base_code = re.match(r'(\d{4,6}[A-Z]?)', s_upper).group(1)
        name = get_stock_name(base_code) or base_code
        return symbol, name

    # 美股：1~5位字母
    if re.fullmatch(r'[A-Z]{1,5}', s_upper) and s_upper not in ["JPY"]:
        return s_upper, s_upper

    # 其他：原樣
    return s_upper, s


def get_stock_analysis(stock_id_input: str) -> str:
    """整合：YahooStock 即時、stock_price 歷史、news、基本面/配息，交給 LLM 生成報告。"""
    logger.info(f"開始執行 {stock_id_input} 股票分析...")
    norm_code, display_name = normalize_stock_input(stock_id_input)

    try:
        # 即時報價（使用你給的 YahooStock，避免直接打 Yahoo API 被 401）
        newprice_stock = YahooStock(norm_code)

        # 歷史價格
        price_data = stock_price(norm_code)

        # 新聞（使用顯示名稱關鍵字）
        news_raw = str(stock_news(display_name))
        news_data = remove_full_width_spaces(news_raw)[:1024]

        content = [
            "你現在是一位專業的證券分析師，依據下列資料撰寫一份完整的分析報告：",
            f"**股票代碼:** {norm_code} ；**股票名稱:** {newprice_stock.name or display_name}",
            f"**即時報價物件:** {vars(newprice_stock)}",
            f"**近期價格資訊:**\n{price_data}"
        ]

        # 基本面/配息：指數不需要
        if norm_code not in ["^TWII", "^GSPC"]:
            try:
                val = stock_fundamental(norm_code)
            except Exception as e:
                logger.warning(f"基本面抓取失敗: {e}")
                val = None
            try:
                div = stock_dividend(norm_code)
            except Exception as e:
                logger.warning(f"配息抓取失敗: {e}")
                div = None
            content.append(f"**每季營收資訊：**\n{val if val is not None else '無法取得'}")
            content.append(f"**配息資料：**\n{div if div is not None else '無法取得'}")

        content.append(f"**近期新聞資訊：**\n{news_data}")

        content_msg = "\n".join(content)

        stock_link = f"https://finance.yahoo.com/quote/{norm_code}"
        system_prompt = (
            "你現在是一位專業的證券分析師。請綜合最近股價、基本面、技術面、消息面與籌碼面，"
            "以繁體中文、Markdown 輸出：\n"
            "- **股名(股號)、現價、漲跌、報價時間**\n"
            "- 股價走勢\n- 基本面分析\n- 技術面分析\n- 消息面\n- 籌碼面\n"
            "- 推薦買進區間\n- 預計停利點(%)\n- 建議買入張數\n- 市場趨勢（偏多/偏空）\n- 配息分析\n- 綜合結論\n\n"
            f"最後附上正確連結：[股票資訊連結]({stock_link})"
        )

        msg = [{"role": "system", "content": system_prompt},
               {"role": "user", "content": content_msg}]
        return get_analysis_reply(msg)

    except Exception as e:
        logger.error(f"股票分析流程失敗: {e}", exc_info=True)
        return f"抱歉，分析「{stock_id_input}」時發生錯誤，請確認代碼是否正確。"


# --- UI & 對話 Helpers ---
async def analyze_sentiment(text: str) -> str:
    msgs = [
        {"role": "system", "content": "Analyze sentiment; respond ONLY one of: positive, neutral, negative, angry."},
        {"role": "user", "content": text}
    ]
    try:
        out = await groq_chat_async(msgs, max_tokens=10, temperature=0)
        return (out or "neutral").strip().lower()
    except Exception as e:
        logger.warning(f"情緒判定失敗，預設 neutral。原因: {e}")
        return "neutral"

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
            f"使用者情緒：{sentiment}；請調整語氣（開心→一起開心；難過/生氣→先共情再建議；中性→自然聊天）。\n"
            f"回覆使用繁體中文，精煉自然，帶少量表情 {p['emoji']}。")

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
        reply_token,
        TextSendMessage(text=text, quick_reply=build_quick_reply())
    )

def build_main_menu_flex() -> FlexSendMessage:
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
        header=BoxComponent(layout="vertical", contents=[TextComponent(text=title, weight="bold", size="lg")]),
        body=BoxComponent(layout="vertical", contents=buttons, spacing="sm")
    )
    return FlexSendMessage(alt_text=title, contents=bubble)


# ========== 5) LINE Handlers ==========
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
            [build_submenu_flex(kind), TextSendMessage(text="請選擇一項服務", quick_reply=build_quick_reply())]
        )
        return


# ---------- 主訊息流程 ----------
async def handle_message_async(event: MessageEvent):
    chat_id = get_chat_id(event)
    msg_raw = event.message.text.strip()
    reply_token = event.reply_token
    is_group = not isinstance(event.source, SourceUser)

    # 取得 Bot 顯示名稱（群組 @判斷）
    try:
        bot_info = await run_in_threadpool(line_bot_api.get_bot_info)
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if not msg_raw:
        return
    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True

    # 群組中關閉自動回答且未 @Bot 時，不回覆
    if is_group and not auto_reply_status.get(chat_id, True) and not msg_raw.startswith(f"@{bot_name}"):
        return

    msg = msg_raw[len(f"@{bot_name}"):].strip() if msg_raw.startswith(f"@{bot_name}") else msg_raw
    if not msg:
        return

    low = msg.lower()

    def is_stock_query(text: str) -> bool:
        t = text.upper()
        if t in ["台股大盤", "大盤", "美股大盤", "美盤", "美股"]:
            return True
        # 台股：4~6位數＋可選一位英文字母（大小寫都行）
        if re.fullmatch(r'\d{4,6}[A-Za-z]?', t):
            return True
        # 美股：1~5 位字母（排除 JPY）
        if re.fullmatch(r'[A-Z]{1,5}', t) and t not in ["JPY"]:
            return True
        return False

    # --- 菜單 ---
    if low in ("menu", "選單", "主選單"):
        return line_bot_api.reply_message(reply_token, build_main_menu_flex())

    # --- 子選單（postback 已處理） ---

    # --- 彩票 ---
    if msg in ["大樂透", "威力彩", "539"]:
        if not LOTTERY_ENABLED:
            return reply_with_quick_bar(reply_token, "抱歉，彩票分析功能目前設定不完整。")
        try:
            report = await run_in_threadpool(get_lottery_analysis, msg)
            return reply_with_quick_bar(reply_token, report)
        except Exception as e:
            logger.error(f"彩票分析流程失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")

    # --- 股票 ---
    if is_stock_query(msg):
        if not STOCK_ENABLED:
            return reply_with_quick_bar(reply_token, "抱歉，股票分析模組目前設定不完整或載入失敗。")
        try:
            report = await run_in_threadpool(get_stock_analysis, msg)
            return reply_with_quick_bar(reply_token, report)
        except Exception as e:
            logger.error(f"股票分析流程失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")

    # --- 金價 / 匯率 ---
    if low in ("金價", "黃金"):
        try:
            report = await run_in_threadpool(get_gold_analysis)
            return reply_with_quick_bar(reply_token, report)
        except Exception as e:
            logger.error(f"黃金分析流程失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "抱歉，金價分析服務暫時無法使用。")

    if low == "jpy":
        try:
            report = await run_in_threadpool(get_currency_analysis, "JPY")
            return reply_with_quick_bar(reply_token, report)
        except Exception as e:
            logger.error(f"日圓分析流程失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "抱歉，日圓匯率分析服務暫時無法使用。")

    # --- 系統設定 ---
    if low in ("開啟自動回答", "關閉自動回答"):
        is_on = (low == "開啟自動回答")
        auto_reply_status[chat_id] = is_on
        text = "✅ 已開啟自動回答" if is_on else "❌ 已關閉自動回答（群組需 @我 才回）"
        return reply_with_quick_bar(reply_token, text)

    # --- 翻譯模式開關 ---
    if low.startswith("翻譯->"):
        lang = msg.split("->", 1)[1].strip()
        if lang == "結束":
            translation_states.pop(chat_id, None)
            return reply_with_quick_bar(reply_token, "✅ 已結束翻譯模式")
        translation_states[chat_id] = lang
        return reply_with_quick_bar(reply_token, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")

    # ⭐ 翻譯模式直通（修復點）：只要模式開著，就攔截並翻譯，不讓它走到一般聊天
    if chat_id in translation_states:
        try:
            out = await translate_text(msg, translation_states[chat_id])
            return reply_with_quick_bar(reply_token, f"🌐 ({translation_states[chat_id]})\n{out}")
        except Exception as e:
            logger.error(f"翻譯失敗: {e}", exc_info=True)
            return reply_with_quick_bar(reply_token, "翻譯暫時失效，等我回神再來一次 🙏")

    # --- 人設切換 ---
    persona_keys = {"甜":"sweet", "鹹":"salty", "萌":"moe", "酷":"cool", "random":"random"}
    if low in persona_keys:
        key = set_user_persona(chat_id, persona_keys[low])
        p = PERSONAS[key]
        return reply_with_quick_bar(reply_token, f"💖 已切換人設：{p['title']}\n\n{p['greetings']}")

    # --- 一般對話 ---
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]
        final_reply = await groq_chat_async(messages)
        # 紀錄歷史
        history.extend([{"role":"user","content":msg}, {"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
        return reply_with_quick_bar(reply_token, final_reply)
    except Exception as e:
        logger.error(f"AI 回覆失敗: {e}", exc_info=True)
        return reply_with_quick_bar(reply_token, "抱歉我剛剛走神了 😅 再說一次讓我補上！")


# ---------- 彩票分析（封裝；供上面呼叫） ----------
def get_lottery_analysis(lottery_type_input: str) -> str:
    if not LOTTERY_ENABLED:
        return "抱歉，彩票分析功能尚未啟用。"
    logger.info(f"開始執行 {lottery_type_input} 彩票分析...")

    t = lottery_type_input
    if "威力" in t: last_lotto = lottery_crawler.super_lotto()
    elif "大樂" in t: last_lotto = lottery_crawler.lotto649()
    elif "539" in t: last_lotto = lottery_crawler.daily_cash()
    else: return f"抱歉，暫不支援 {lottery_type_input} 類型的分析。"

    try:
        info = caiyunfangwei_crawler.get_caiyunfangwei()
        content_msg = (
            f'你現在是一位專業的樂透彩分析師, 使用{t}的資料來撰寫分析報告:\n'
            f'近幾期號碼資訊:\n{last_lotto}\n'
            f'顯示今天國歷/農歷日期：{info.get("今天日期", "未知")}\n'
            f'今日歲次：{info.get("今日歲次", "未知")}\n'
            f'財神方位：{info.get("財神方位", "未知")}\n'
            '最冷號碼，最熱號碼\n請給出完整的趨勢分析報告，最近所有每次開號碼,'
            '並給3組與彩類同數位數字隨機號和不含特別號(如有)\n'
            '第1組最冷組合：同彩種位數，數字小到大；威力彩需分二區，其他不分\n'
            '第2組最熱組合：同彩種位數，數字小到大；威力彩需分二區，其他不分\n'
            '第3組隨機組合：同彩種位數，數字小到大；威力彩需分二區，其他不分\n'
            '給 20 字內勵志吉祥句。\n'
            '使用台灣繁體中文。'
        )
    except Exception as e:
        logger.error(f"財神方位取得失敗: {e}")
        content_msg = (
            f'你現在是一位專業的樂透彩分析師, 使用{t}的資料來撰寫分析報告:\n'
            f'近幾期號碼資訊:\n{last_lotto}\n'
            '（財神方位資訊暫時無法獲取）\n'
            '請給出完整的趨勢分析報告，並提供 3 組隨機號碼組合；使用繁體中文。'
        )

    msg = [
        {"role": "system", "content": f"你是{t}的專業彩券分析師，輸出精簡且有條理的趨勢報告。"},
        {"role": "user", "content": content_msg}
    ]
    return get_analysis_reply(msg)


# ========== 6) FastAPI Routes ==========
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


# ========== 7) Local run ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)