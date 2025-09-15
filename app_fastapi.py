import os
import re
import io
import random
import logging
import pkg_resources
from typing import Dict, List, Tuple, Optional
from contextlib import asynccontextmanager
from datetime import datetime

# --- 數據處理與爬蟲 ---
import requests
from bs4 import BeautifulSoup
import httpx
import pandas as pd
import yfinance as yf

# --- FastAPI 與 LINE Bot SDK v3 ---
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from linebot.v3.webhook import WebhookHandler  # 修正匯入路徑
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    AsyncMessagingApi,
    ReplyMessageRequest,
    TextMessage,
    AudioMessage,
    ImageMessage,
    FlexMessage,
    FlexBubble,
    FlexBox,
    FlexText,
    FlexButton,
    QuickReply,
    QuickReplyItem,
    MessageAction,
    PostbackAction,
    BotInfoResponse,
)

# --- Cloudinary（上傳音訊/圖片） ---
import cloudinary
import cloudinary.uploader

# --- gTTS（免費 TTS 後備） ---
from gtts import gTTS

# --- AI 相關 ---
from groq import AsyncGroq, Groq
import openai

# --- 圖表（可選，無則自動跳過） ---
try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import mplfinance as mpf
    HAS_MPLFIN = True
except Exception:
    HAS_MPLFIN = False

# ...（後續程式碼保持不變，直到相關部分）

# ========== 10) LINE Event Handlers ==========
@handler.add(MessageEvent, message=TextMessageContent)
async def handle_text_message(event: MessageEvent):
    chat_id, msg_raw, reply_token = get_chat_id(event), event.message.text.strip(), event.reply_token

    # 取得 bot 顯示名稱（供 @bot 判斷）
    try:
        bot_info: BotInfoResponse = await line_bot_api.get_bot_info()
        bot_name = bot_info.display_name
    except Exception:
        bot_name = "AI 助手"

    if not msg_raw:
        return

    # 預設群組自動回覆開啟
    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True

    # 群組/聊天室：若關閉自動回覆，必須 @bot 才回
    is_group_or_room = getattr(event.source, "type", "") in ("group", "room")
    if is_group_or_room and not auto_reply_status.get(chat_id, True) and not msg_raw.startswith(f"@{bot_name}"):
        return

    # 去除 @botname 前綴
    msg = msg_raw
    if msg_raw.startswith(f"@{bot_name}"):
        msg = re.sub(f'^@{re.escape(bot_name)}\\s*', '', msg_raw).strip()
    if not msg:
        return

    low = msg.lower()

    # === 路由 ===
    # 主選單
    if low in ("menu", "選單", "主選單"):
        await line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=[build_main_menu()]))
        return

    # 彩票
    if msg in ("大樂透", "威力彩", "539"):
        try:
            report = await run_in_threadpool(get_lottery_analysis, msg)
            await reply_text_with_tts_and_extras(reply_token, report)
        except Exception as e:
            logger.error(f"彩票分析流程失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")
        return

    # 金價
    if low in ("金價", "黃金"):
        try:
            out = await run_in_threadpool(get_gold_analysis)
            await reply_text_with_tts_and_extras(reply_token, out)
        except Exception as e:
            logger.error(f"金價分析流程失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_token, "抱歉，金價分析服務暫時無法使用。")
        return

    # 匯率（簡化：僅 JPY；你可自行擴充 USD/EUR）
    if low == "jpy":
        try:
            out = await run_in_threadpool(get_currency_analysis, "JPY")
            await reply_text_with_tts_and_extras(reply_token, out)
        except Exception as e:
            logger.error(f"日圓分析流程失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_token, "抱歉，日圓匯率分析服務暫時無法使用。")
        return

    # 翻譯模式切換（開/關）
    if low.startswith("翻譯->"):
        lang = msg.split("->", 1)[1].strip()
        if lang == "結束":
            translation_states.pop(chat_id, None)
            await reply_text_with_tts_and_extras(reply_token, "✅ 已結束翻譯模式")
        else:
            translation_states[chat_id] = lang
            await reply_text_with_tts_and_extras(reply_token, f"🌐 已開啟翻譯 → {lang}，請直接輸入要翻的內容。")
        return

    # ✅ 只要翻譯模式開著，且有輸入訊息，就優先翻譯
    if chat_id in translation_states and msg:
        try:
            out = await translate_text(msg, translation_states[chat_id])
            if not out:  # 確保有回應，避免空值
                out = "抱歉，翻譯失敗，請稍後再試。"
            await reply_text_with_tts_and_extras(reply_token, out)
        except Exception as e:
            logger.error(f"翻譯失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_token, "抱歉，翻譯目前不可用。")
        return

    # 股票/指數
    if re.fullmatch(r"\^?[A-Z0-9.]{2,10}", msg) or msg.isdigit() or msg in ("台股大盤", "美股大盤", "大盤", "美股"):
        try:
            text = await run_in_threadpool(get_stock_analysis, msg)
            extras = []
            try:
                chart_url = await get_stock_chart_url_async(msg)
                if chart_url:
                    extras.append(ImageMessage(original_content_url=chart_url, preview_image_url=chart_url))
            except Exception as ce:
                logger.warning(f"附圖失敗（忽略）：{ce}")
            await reply_text_with_tts_and_extras(reply_token, text, extras=extras)
        except Exception as e:
            logger.error(f"股票分析流程失敗: {e}", exc_info=True)
            await reply_text_with_tts_and_extras(reply_token, f"抱歉，分析 {msg} 時發生錯誤。")
        return

    # 自動回覆設定（僅群組/聊天室有意義）
    if low in ("開啟自動回答", "關閉自動回答"):
        is_on = low == "開啟自動回答"
        auto_reply_status[chat_id] = is_on
        text = "✅ 已開啟自動回答" if is_on else "❌ 已關閉自動回答（群組需 @我 才回）"
        await reply_text_with_tts_and_extras(reply_token, text)
        return

    # 人設切換（注意：因為翻譯模式分支已提前處理，不會誤觸）
    if msg in PERSONA_ALIAS or low in PERSONA_ALIAS:
        key = set_user_persona(chat_id, PERSONA_ALIAS.get(msg, PERSONA_ALIAS.get(low, "sweet")))
        p = PERSONAS[user_persona[chat_id]]
        txt = f"💖 已切換人設：{p['title']}\n\n{p['greetings']}"
        await reply_text_with_tts_and_extras(reply_token, txt)
        return

    # 一般聊天（人設 + 情緒）
    try:
        history = conversation_history.get(chat_id, [])
        sentiment = await analyze_sentiment(msg)
        sys_prompt = build_persona_prompt(chat_id, sentiment)
        messages = [{"role":"system","content":sys_prompt}] + history + [{"role":"user","content":msg}]
        final_reply = await groq_chat_async(messages)
        history.extend([{"role":"user","content":msg}, {"role":"assistant","content":final_reply}])
        conversation_history[chat_id] = history[-MAX_HISTORY_LEN*2:]
        await reply_text_with_tts_and_extras(reply_token, final_reply)
    except Exception as e:
        logger.error(f"AI 回覆失敗: {e}", exc_info=True)
        await reply_text_with_tts_and_extras(reply_token, "抱歉我剛剛走神了 😅 再說一次讓我補上！")

# ...（後續程式碼保持不變）
@handler.add(MessageEvent, message=AudioMessageContent)
async def handle_audio_message(event: MessageEvent):
    reply_token = event.reply_token
    try:
        content_stream = await line_bot_api.get_message_content(event.message.id)
        audio_in = await content_stream.read()

        text = await speech_to_text_async(audio_in)
        if not text:
            raise RuntimeError("語音轉文字失敗")

        sentiment = await analyze_sentiment(text)
        sys_prompt = build_persona_prompt(get_chat_id(event), sentiment)
        final_reply_text = await groq_chat_async(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}]
        )

        await reply_text_with_tts_and_extras(
            reply_token,
            f"🎧 我聽到了：\n{text}\n\n—\n{final_reply_text}"
        )
    except Exception as e:
        logger.error(f"處理語音訊息失敗: {e}", exc_info=True)
        await reply_text_with_tts_and_extras(reply_token, "抱歉，我沒聽清楚，可以再說一次嗎？")


@handler.add(PostbackEvent)
async def handle_postback(event: PostbackEvent):
    data = event.postback.data or ""
    if data.startswith("menu:"):
        kind = data.split(":", 1)[-1]
        await line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=event.reply_token, messages=[build_submenu(kind)])
        )


# ========== 11) FastAPI Routes ==========
@router.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        await handler.handle(body.decode("utf-8"), signature)
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
    return PlainTextResponse("ok", status_code=200)

app.include_router(router)


# ========== 12) Local run ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, log_level="info", reload=True)