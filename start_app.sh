#!/bin/bash

# 啟動 Line Bot 應用程式
echo "🚀 啟動 Line Bot 應用程式..."

# 激活虛擬環境
source venv/bin/activate

# 檢查必要的環境變數
if [ -z "$LINE_CHANNEL_ACCESS_TOKEN" ]; then
    echo "❌ 錯誤: LINE_CHANNEL_ACCESS_TOKEN 環境變數未設定"
    exit 1
fi

if [ -z "$LINE_CHANNEL_SECRET" ]; then
    echo "❌ 錯誤: LINE_CHANNEL_SECRET 環境變數未設定"
    exit 1
fi

if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ 錯誤: GROQ_API_KEY 環境變數未設定"
    exit 1
fi

echo "✅ 環境變數檢查完成"
echo "✅ 虛擬環境已激活"
echo "🌐 啟動 FastAPI 服務器..."

# 啟動應用程式
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload