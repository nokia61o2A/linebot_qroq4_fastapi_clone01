import os
import re
import pandas as pd
import yfinance as yf
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import PostbackEvent, TextSendMessage, MessageEvent, TextMessage
from linebot.models import *
from groq import Groq, GroqError
import requests
from my_commands.stock.stock_price import stock_price
from my_commands.stock.stock_news import stock_news
from my_commands.stock.stock_value import stock_fundamental
from my_commands.stock.stock_rate import stock_dividend
from my_commands.stock.YahooStock import YahooStock
from openai import OpenAI

# 設定 API 金鑰
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# 初始化 OpenAI 客戶端
client = OpenAI( 
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = "https://free.v36.cm/v1"
)

# 初始化全局變數以存儲股票資料
stock_data_df = None

# 讀取 CSV 檔案並將其轉換為 DataFrame，只在首次調用時讀取
def load_stock_data():
    global stock_data_df
    if stock_data_df is None:
        stock_data_df = pd.read_csv('name_df.csv')
    return stock_data_df

# 根據股號查找對應的股名
def get_stock_name(stock_id):
    stock_data_df = load_stock_data()  # 加載股票資料
    # 確保股號欄位為字串型態以便比較
    stock_data_df['股號'] = stock_data_df['股號'].astype(str)
    result = stock_data_df[stock_data_df['股號'] == str(stock_id)]
    if not result.empty:
        return result.iloc[0]['股名']
    return None

# 移除全形空格的函數
def remove_full_width_spaces(data):
    if isinstance(data, list):
        return [remove_full_width_spaces(item) for item in data]
    if isinstance(data, str):
        return data.replace('\u3000', ' ')
    return data

# 截取前1024個字的函數
def truncate_text(data, max_length=1024):
    if isinstance(data, list):
        return [truncate_text(item, max_length) for item in data]
    if isinstance(data, str):
        return data[:max_length]
    return data

def get_reply(messages):
    select_model = "gpt-4o-mini"
    print (f"[stock_gpt] free gpt:{select_model}")
    try:
        completion = client.chat.completions.create(
            model=select_model,
            messages=messages,
            max_tokens=1000,
            temperature=1.2
        )
        # 獲取回覆的內容
        reply = completion.choices[0].message.content
    except Exception as e:
        print ("Groq: partjob:")
        try:
            response = groq_client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=messages,
                max_tokens=2000,
                temperature=1.2
            )
            reply = response.choices[0].message.content
        except GroqError as groq_err:
            reply = f"OpenAI API 發生錯誤: {openai_err.error.message}，GROQ API 發生錯誤: {groq_err.message}"
    return reply

# # 建立 GPT 模型
# def get_reply(messages):
#     print("* stock_gpt ")
#     try:
#         # 預設使用大模型 llama3-70b-8192
#         response = groq_client.chat.completions.create(
#             model="llama3-groq-70b-8192-tool-use-preview",
#             # model="llama3-70b-8192",
#             messages=messages,
#             max_tokens=2000,
#             temperature=1.2
#         )
#         reply = response.choices[0].message.content
#         return reply
#     except GroqError as groq_err:
#         # 如果遇到流量限制錯誤，切換到較小模型 llama3-8b-8192
#         if 'rate_limit_exceeded' in str(groq_err):
#             print("GROQ API 達到流量上限，切換到較小模型分析...")
#             try:
#                 response = groq_client.chat.completions.create(
#                     model="llama3-groq-8b-8192-tool-use-preview",
#                     messages=messages,
#                     max_tokens=1200,  # 減少使用量
#                     temperature=1.2
#                 )
#                 reply = response.choices[0].message.content
#                 return reply
#             except GroqError as fallback_err:
#                 return f"切換較小模型時發生錯誤: {fallback_err.message}"
#         else:
#             return f"GROQ API 發生錯誤: {groq_err.message}"


# 建立訊息指令(Prompt)
def generate_content_msg(stock_id):
    # 檢查是否為美盤或台灣大盤
    if stock_id == "美盤" or stock_id == "美股":
        stock_id = "^GSPC"  # 標普500指數的代碼
        stock_name = "美國大盤" 
    elif stock_id == "大盤":
        stock_id = "^TWII"  # 台灣加權股價指數的代碼
        stock_name = "台灣大盤"
    else:
        # 使用正則表達式判斷台股（4-6位數字，可帶字母）和美股（1-5位字母）
        if re.match(r'^\d{4,6}[A-Za-z]?$', stock_id):  # 台股代碼格式 
            stock_name = get_stock_name(stock_id)  # 查找台股代碼對應的股名
            if stock_name is None:
                stock_name = stock_id  # 如果股名未找到，使用代碼
        else:
            stock_name = stock_id  # 將美股代碼或無法匹配的代碼當作股名

    # 取得即時價格資訊
    newprice_stock = YahooStock(stock_id) 

    # 取得價格資訊by日
    price_data = stock_price(stock_id)
    # 取得新聞資料並移除全形空格字符及截取
    news_data = remove_full_width_spaces(stock_news(stock_name))
    news_data = truncate_text(remove_full_width_spaces(news_data), 1024)

    # 組合訊息，加入股名和股號
    content_msg = f'你現在是一位專業的證券分析師, 你會依據以下資料來進行分析並給出一份完整的分析報告:\n'
    content_msg += f'**股票代碼:** {stock_id}, **股票名稱:** {newprice_stock.name}\n'
    # content_msg += f'**即時現價 {newprice_stock.now_price},漲跌{newprice_stock.change},更新:{newprice_stock.close_time}\n'
    content_msg += f'**即時現價 {vars(newprice_stock)}\n'
    content_msg += f'近期價格資訊:\n {price_data}\n'

    if stock_id not in ["^TWII", "^GSPC"]:
        stock_value_data = stock_fundamental(stock_id)
        stock_vividend_data = stock_dividend(stock_id)      #配息資料
        if stock_value_data:
            content_msg += f'每季營收資訊：\n {stock_value_data}\n'
        else:
            content_msg += '每季營收資訊無法取得。\n'

        if stock_vividend_data:
            content_msg += f'配息資料：\n {stock_vividend_data}\n'
        else:
            content_msg += '配息資料資訊無法取得。\n'

    content_msg += f'近期新聞資訊: \n {news_data}\n'
    content_msg += f'請給我{stock_name}近期的趨勢報告。請以詳細、嚴謹及專業的角度撰寫此報告，並提及重要的數字，請使用台灣地區的繁體中文回答。'

    return content_msg

# StockGPT 主程式
def stock_gpt(stock_id):
    # 生成內容訊息
    content_msg = generate_content_msg(stock_id)

    # 根據股票代號判斷是否為台灣大盤或美股，並生成相應的連結
    if stock_id == "大盤":
        stock_link = "https://tw.finance.yahoo.com/quote/%5ETWII/"
    elif stock_id == "美盤" or stock_id == "美股":
        stock_link = "https://tw.finance.yahoo.com/quote/%5EGSPC/"
    else:
        stock_link = f"https://tw.stock.yahoo.com/quote/{stock_id}"

    # 設置訊息指令
    msg = [{
        "role": "system",
        "content": f"你現在是一位專業的證券分析師。請基於近期的股價走勢、基本面分析、新聞資訊等進行綜合分析。\
                    請提供以下內容：\
                    ** 股名(股號) ,現價(現漲跌幅),現價的資料的取得時間\
                    - 股價走勢\
                    - 基本面分析\
                    - 技術面分析\
                    - 消息面\
                    - 籌碼面\
                    - 推薦購買區間 (例: 100-110元)\
                    - 預計停利點：百分比 (例: ?%)\
                    - 建議買入張數 (例: ?張)\
                    - 市場趨勢：請分析目前是適合做多還是空頭操作\
                    - 配息分析\
                    - 綜合分析\
                    然後生成一份專業的趨勢分析報告 \
                    最後，請提供一個正確的股票連結：[股票資訊連結]({stock_link})。\
                    回應請使用繁體中文並格式化為 Markdown。 ,reply in 繁體中文(zh-TW)回覆 "
    }, {
        "role": "user",
        "content": content_msg
    }]

    # 調用 GPT 模型進行回應生成
    reply_data = get_reply(msg)

    return reply_data

# stock_value.py 內的 stock_fundamental 函數增加錯誤處理
def stock_fundamental(stock_id):
    if stock_id == "^GSPC":
        print("美國大盤無需營收資料分析")
        return None  # 大盤無需營收資料

    stock = yf.Ticker(stock_id)
    try:
        earnings_dates = stock.get_earnings_dates()
    except Exception as e:
        print(f"Error fetching earnings dates: {e}")
        return None

    if earnings_dates is None:
        print("No earnings dates found for the symbol.")
        return None

    # 確認 'Earnings Date' 列是否存在，避免 KeyError
    if "Earnings Date" not in earnings_dates.columns:
        print("Column 'Earnings Date' not found in the data")
        return None

    if "Reported EPS" in earnings_dates.columns:
        reported_eps = earnings_dates["Reported EPS"]
        return reported_eps
    else:
        print("Column 'Reported EPS' not found in the data")
        return None
