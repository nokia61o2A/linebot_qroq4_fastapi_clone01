# Line Bot + FastAPI (Groq/OpenAI)qroq4_clon
# 小豬助理

這是一個基於 FastAPI 的 Line Bot 範本，整合了 Groq (Llama 3) 與 OpenAI，並包含彩票分析、股票查詢、匯率查詢等功能。

## 功能特色

- **多模態 AI 對話**：支援 Groq 與 OpenAI。
- **彩票分析**：整合大樂透、威力彩等台灣彩券分析。
- **財經資訊**：即時金價、股票查詢。
- **自動應答模式**：可切換自動回覆或安靜模式。
- **翻譯與語音**：支援多國語言翻譯與 TTS。

## 快速部署 (Deploy to Render)

你可以直接點擊下方按鈕，將此專案部署到 Render：

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### 部署步驟

1. 點擊上方的 **Deploy to Render** 按鈕。
2. 在 Render 頁面中，輸入 `Service Group Name` (例如 `my-line-bot`)。
3. 填寫必要的環境變數 (Environment Variables)：
   - `CHANNEL_ACCESS_TOKEN`: Line Messaging API 的 Access Token。
   - `CHANNEL_SECRET`: Line Messaging API 的 Channel Secret。
   - `GROQ_API_KEY`: Groq 的 API Key。
   - `BASE_URL`: 部署後的網址 (例如 `https://your-app.onrender.com`)。
     - *注意：初次部署時可能還不知道網址，可先填暫定值，部署完成後再回來修改並重啟。*
4. 點擊 **Apply** 開始部署。
5. 部署成功後，將取得的網址 (加上 `/callback`) 填回 Line Developers Console 的 Webhook URL。
   - 例如：`https://your-app.onrender.com/callback`

## 本地開發

1. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```
2. 設定環境變數 (參考 `start_app.sh` 或直接設定 .env)。
3. 啟動伺服器：
   ```bash
   uvicorn app_fastapi:app --reload
   ```
