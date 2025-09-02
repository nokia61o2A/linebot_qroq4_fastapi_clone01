"""
aibot FastAPI 應用程序初始化 (v4 - 修正 Pykakasi 錯誤與 Health Check)
"""
import os
import re
import asyncio
import logging
import random
from contextlib import asynccontextmanager
from typing import Dict, List

import httpx
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool

from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    QuickReply, QuickReplyButton, MessageAction,
    SourceGroup, SourceRoom, PostbackEvent,
    FlexSendMessage, BubbleContainer, BoxComponent, TextComponent,
    ButtonComponent
)
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from groq import AsyncGroq

# --- Logger, 函式庫檢查, 基礎設定 (與前版相同) ---
import logging
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# (此處省略了與前一版完全相同的函式庫檢查、基礎設定、自訂模組匯入等程式碼，以保持簡潔)
# ...

# ============================================
# 發音標註功能 (修正版)
# ============================================
# (korean_to_bopomofo 函式與前版相同)

def get_phonetic_transcription(text: str, target_language: str) -> str:
    """根據目標語言生成發音標註"""
    phonetics = []
    
    if target_language in ["繁體中文", "簡體中文"] and PINYIN_ENABLED:
        # ... (中文處理邏輯不變)
        pass

    elif target_language == "日文" and KAKASI_ENABLED:
        try:
            kks = pykakasi.kakasi()
            result = kks.convert(text)
            # 🔥 核心修正：使用 .get() 安全地處理標點符號等非日文字元
            romaji = ''.join([item.get('romaji', item['orig']) for item in result])
            phonetics.append(f"羅馬拼音: {romaji}")
        except Exception as e:
            logger.error(f"Pykakasi 處理失敗: {e}")

    elif target_language == "韓文":
        # ... (韓文處理邏輯不變)
        pass
            
    return "\n".join(phonetics)

# ============================================
# Groq & 人設 & 主邏輯 (與前版相同)
# ============================================
# (所有相關函式，包括 groq_chat_completion, translate_text, handle_message, reply_simple 等，都與前一版完全相同)
# (您無需修改這些函式)

# ============================================
# FastAPI 路由
# ============================================
@router.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    try:
        await run_in_threadpool(handler.handle, body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(400, "Invalid signature")
    return JSONResponse({"message": "ok"})

# 🔥 核心修正：新增 /healthz 路由給 Render.com 使用
@router.get("/healthz")
async def health_check():
    """健康檢查端點"""
    return {"status": "ok"}

@router.get("/")
async def root():
    return {"message": "Line Bot Service is live.", "version": "1.0.0"}

app.include_router(router)