"""
低配ai醬，代班Ai醬
"""
import re
import time
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import PostbackEvent, TextSendMessage, MessageEvent, TextMessage
from linebot.models import *
import os
from openai import OpenAI
import requests
from groq import Groq
from my_commands.lottery_gpt import lottery_gpt
from my_commands.gold_gpt import gold_gpt
from my_commands.platinum_gpt import platinum_gpt
from my_commands.money_gpt import money_gpt
from my_commands.one04_gpt import one04_gpt
from my_commands.partjob_gpt import partjob_gpt
from my_commands.crypto_coin_gpt import crypto_gpt
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from my_commands.stock.stock_gpt import stock_gpt


app = Flask(__name__)

# SET BASE URL
base_url = os.getenv("BASE_URL")
# Channel Access Token
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
# Channel Secret
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))
# 初始化 Groq API client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# 初始化對話歷史
conversation_history = {}
MAX_HISTORY_LEN = 10  # 設定最大對話記憶長度


# 初始化 OpenAI 客戶端
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = "https://free.v36.cm/v1"
)

# 初始化對話歷史
conversation_history = {}
MAX_HISTORY_LEN = 10  # 設定最大對話記憶長度

# 建立 GPT 模型
"""
 gpt-4o-mini（速度一般，若要体验极速回复，可购买付费API）
 gpt-3.5-turbo-0125
 gpt-3.5-turbo-1106
 gpt-3.5-turbo
 gpt-3.5-turbo-16k
 net-gpt-3.5-turbo (可联网搜索模型-稳定性稍差)
 whisper-1
 dall-e-2
 """
def start_loading_animation(chat_id, loading_seconds=5):
    url = 'https://api.line.me/v2/bot/chat/loading/start'
    headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {os.getenv('CHANNEL_ACCESS_TOKEN')}",
    }
    data = {
        "chatId": chat_id,
        "loadingSeconds": loading_seconds
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.status_code, response.json()  # 回傳JSON格式
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return response.status_code, response.text
    except requests.exceptions.RequestException as e:
        print(f"Exception during API request: {e}")
        return None, str(e)

def get_reply(messages):
    select_model = "gpt-4o-mini"
    print (f"free gpt:{select_model}")
    try:
        completion = client.chat.completions.create(
            model=select_model,
            messages=messages,
            max_tokens=800  
        )
        # 獲取回覆的內容
        reply = completion.choices[0].message.content
    except Exception as e:
        print ("Groq: partjob:")
        try:
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                max_tokens=2000,
                temperature=1.2
            )
            reply = response.choices[0].message.content
        except GroqError as groq_err:
            reply = f"OpenAI API 發生錯誤: {openai_err.error.message}，GROQ API 發生錯誤: {groq_err.message}"
    return reply

# 要檢查 LINE Webhook URL 的函數
def check_line_webhook():
    url = "https://api.line.me/v2/bot/channel/webhook/endpoint"
    headers = {
        "Authorization": f"Bearer {os.getenv('CHANNEL_ACCESS_TOKEN')}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        current_webhook = response.json().get("endpoint", "無法取得 Webhook URL")
        print(f"當前 Webhook URL: {current_webhook}")
        return current_webhook
    else:
        print(f"檢查 Webhook URL 失敗，狀態碼: {response.status_code}, 原因: {response.text}")
        return None

# 更新 LINE Webhook URL 的函數
def update_line_webhook():
    new_webhook_url = base_url + "/callback"
    current_webhook_url = check_line_webhook()  # 檢查當前的 Webhook URL

    if current_webhook_url != new_webhook_url:
        url = "https://api.line.me/v2/bot/channel/webhook/endpoint"
        headers = {
            "Authorization": f"Bearer {os.getenv('CHANNEL_ACCESS_TOKEN')}",
            "Content-Type": "application/json"
        }
        payload = {
            "endpoint": new_webhook_url
        }

        response = requests.put(url, headers=headers, json=payload)
        if response.status_code == 200:
            print(f"Webhook URL 更新成功: {new_webhook_url}")
        else:
            print(f"更新失敗，狀態碼: {response.status_code}, 原因: {response.text}")
    else:
        print("當前的 Webhook URL 已是最新，無需更新。")


# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# 處理訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    global conversation_history
    user_id = event.source.user_id
    msg = event.message.text

    # 檢查是否為群組或聊天室訊息
    is_group_or_room = isinstance(event.source, (SourceGroup, SourceRoom))

    if is_group_or_room:
        # 獲取 bot 的資訊
        bot_info = line_bot_api.get_bot_info()
        bot_name = bot_info.display_name

        # 檢查訊息是否包含@標記
        if '@' in msg:
            # 提取@後的文字
            at_text = msg.split('@')[1].split()[0] if len(msg.split('@')) > 1 else ''
            # 模糊匹配檢查機器人名稱
            if at_text.lower() not in bot_name.lower():
                return  # 如果@後的文字不匹配機器人名稱，直接返回不處理
            # 移除@和機器人名稱部分，只保留實際訊息內容
            msg = msg.replace(f'@{at_text}', '').strip()
        else:
            return  # 如果沒有@標記，直接返回不處理
        
        # 如果移除後訊息為空，則不處理
        if not msg:
            return

    # 初始化使用者的對話歷史
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # 將訊息加入對話歷史
    conversation_history[user_id].append({"role": "user", "content": msg + ", 請以繁體中文回答我問題"})

     # 台股代碼邏輯：必須以 4-6 個數字開頭，後面可選擇性有一個英文字母
    stock_code = re.search(r'^\d{4,6}[A-Za-z]?\b', msg)
    # 美股代碼邏輯：必須以 1-5 個字母開頭
    stock_symbol = re.search(r'^[A-Za-z]{1,5}\b', msg)

    # 限制對話歷史長度
    if len(conversation_history[user_id]) > MAX_HISTORY_LEN * 2:
        conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY_LEN * 2:]

    # 定義彩種關鍵字列表
    lottery_keywords = ["威力彩", "大樂透", "539", "雙贏彩", "3星彩", "三星彩", "4星彩", "四星彩", "38樂合彩", "39樂合彩", "49樂合彩", "運彩"]

    # 判斷是否為彩種相關查詢
    if any(keyword in msg for keyword in lottery_keywords):
        reply_text = lottery_gpt(msg)  # 呼叫對應的彩種處理函數
    elif msg.lower().startswith("大盤") or msg.lower().startswith("台股"):
        reply_text = stock_gpt("大盤")
    elif msg.lower().startswith("美盤") or msg.lower().startswith("美股"): 
        reply_text = stock_gpt("美盤")
    elif stock_code:
        stock_id = stock_code.group()
        reply_text = stock_gpt(stock_id)
    elif stock_symbol:
        stock_id = stock_symbol.group()
        reply_text = stock_gpt(stock_id)
    elif any(msg.lower().startswith(currency.lower()) for currency in ["金價", "金", "黃金", "gold"]):
        reply_text = gold_gpt()
    elif any(msg.lower().startswith(currency.lower()) for currency in ["鉑", "鉑金", "platinum", "白金"]):
        reply_text = platinum_gpt()
    elif msg.lower().startswith(tuple(["日幣", "日元", "jpy", "換日幣"])):
        reply_text = money_gpt("JPY")
    elif any(msg.lower().startswith(currency.lower()) for currency in ["美金", "usd", "美元", "換美金"]):
        reply_text = money_gpt("USD")
    elif msg.startswith("104:"):
        reply_text = one04_gpt(msg[4:])
    elif msg.startswith("pt:"):
        reply_text = partjob_gpt(msg[3:])
    elif msg.startswith("cb:"):  # 新增這個條件來處理加密貨幣查詢
        coin_id = msg[3:].strip()
        reply_text = crypto_gpt(coin_id)
    elif msg.startswith("$:"):  # 新增這個條件來處理加密貨幣查詢
        coin_id = msg[2:].strip()
        reply_text = crypto_gpt(coin_id)
    else:
        # 傳送最新對話歷史給 Groq
        messages = conversation_history[user_id][-MAX_HISTORY_LEN:]
        try:
            reply_text = get_reply(messages)  # 呼叫 Groq API 取得回應
        except Exception as e:
            reply_text = f" API 發生錯誤: {str(e)}"

    # 如果 `reply_text` 為空，設定一個預設回應
    if not reply_text:
        reply_text = "抱歉，目前無法提供回應，請稍後再試。"

    # 回應使用者
    try:
        #單人才會顯示 (...)
        if event.source.type == 'user':
            start_loading_animation(chat_id=chat_id, loading_seconds=5)
            
        line_bot_api.reply_message(event.reply_token, TextSendMessage(reply_text))
    except LineBotApiError as e:
        print(f"LINE 回覆失敗: {e}")
        # 這裡可以根據需要增加錯誤處理

    # 將 GPT 的回應加入對話歷史
    conversation_history[user_id].append({"role": "assistant", "content": reply_text})


@handler.add(PostbackEvent)
def handle_postback(event):
    print(event.postback.data)

#觀迎剛剛加入的人
@handler.add(MemberJoinedEvent)
def welcome(event):
    uid = event.joined.members[0].user_id
    if isinstance(event.source, SourceGroup):
        gid = event.source.group_id
        profile = line_bot_api.get_group_member_profile(gid, uid)
    elif isinstance(event.source, SourceRoom):
        rid = event.source.room_id
        profile = line_bot_api.get_room_member_profile(rid, uid)
    else:
        profile = line_bot_api.get_profile(uid)
    name = profile.display_name
    message = TextSendMessage(text=f'{name} 歡迎加入')
    line_bot_api.reply_message(event.reply_token, message)

# 健康檢查端點
@app.route('/healthz', methods=['GET'])
def health_check():
    return 'OK', 200


# 啟動應用
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    try:
        update_line_webhook()  # 啟動時自動更新 Webhook URL
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f"伺服器啟動失敗: {e}")
