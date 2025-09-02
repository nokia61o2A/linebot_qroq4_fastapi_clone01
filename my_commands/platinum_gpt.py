import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import openai
from groq import Groq

# 初始化 GROQ API 客戶端
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
groq_tokens_used = 0
groq_last_request_time = 0
GROQ_RATE_LIMIT = 10000  # 假設的速率限制，請根據實際情況調整
GROQ_REQUEST_INTERVAL = 60  # 請根據實際需求調整

# 定義 get_reply 函數
def get_reply(messages):
    global groq_tokens_used, groq_last_request_time

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=messages
        )
        reply = response["choices"][0]["message"]["content"]
    except openai.OpenAIError as openai_err:
        try:
            request_tokens = sum(len(message['content']) for message in messages)

            current_time = time.time()
            if groq_tokens_used + request_tokens > GROQ_RATE_LIMIT:
                wait_time = GROQ_REQUEST_INTERVAL - (current_time - groq_last_request_time)
                if wait_time > 0:
                    print(f"超過速率限制，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                groq_tokens_used = 0
                groq_last_request_time = time.time()

            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=1500,
                temperature=1.2
            )
            reply = response.choices[0].message.content
            groq_tokens_used += request_tokens
        except Groq.RateLimitError as rate_err:
            print("遇到 RateLimitError，等待 15 秒...")
            time.sleep(15)
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=1500,
                temperature=1.2
            )
            reply = response.choices[0].message.content
        except Groq.GroqError as groq_err:
            reply = f"OpenAI API 發生錯誤: {openai_err.error.message}, GROQ API 發生錯誤: {groq_err.message}"
    return reply

# 獲取並處理鉑金數據
def fetch_and_process_platinum_data():
    # 發送HTTP請求以獲取網頁內容
    url = "https://tw.bullion-rates.com/platinum/TWD-history.htm"
    response = requests.get(url)
    response.encoding = 'utf-8'

    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # 找到包含歷史數據的表格
    table = soup.find('table', {'id': 'dtDGrid'})

    # 確保找到了目標表格
    if table is None:
        raise ValueError("未找到包含歷史數據的表格")

    # 解析表格中的行，忽略廣告行
    rows = table.find_all('tr', {'class': ['DataRow', 'AltDataRow']})

    # 建立用於存儲日期和價格數據的列表
    data = []

    for row in rows:
        cells = row.find_all('td')
        if len(cells) == 3:  # 確保每行有三個單元格
            date = cells[0].get_text(strip=True)
            price_per_gram = cells[1].get_text(strip=True)
            data.append([date, price_per_gram])

    # 將數據轉換為DataFrame
    df = pd.DataFrame(data, columns=['日期', '鉑金價格/公克'])

    # 處理數據：將日期設置為索引並轉換價格為浮點數
    df['日期'] = pd.to_datetime(df['日期'], format='%Y/%m/%d')
    df['鉑金價格/公克'] = df['鉑金價格/公克'].str.replace(',', '').astype(float)
    df.set_index('日期', inplace=True)
    df.sort_index(inplace=True)

    return df

# 生成鉑金價格分析報告的消息
def generate_platinum_content_msg():
    # 获取和处理数据
    platinum_prices_df = fetch_and_process_platinum_data()

    # 从数据中获取需要的最高价和最低价信息
    max_price = platinum_prices_df['鉑金價格/公克'].max()
    min_price = platinum_prices_df['鉑金價格/公克'].min()
    last_date = platinum_prices_df.index[-1].strftime("%Y-%m-%d")

    # 构造专业分析报告的内容
    content_msg = f'你現在是一位專業的鉑金價格分析師, 使用以下数据来撰写分析报告:\n'
    content_msg += f'{platinum_prices_df}\n'
    content_msg += f'最新日期: {last_date}, 最高鉑金價格: {max_price} (日期), 最低鉑金價格: {min_price} (日期)。\n'
    content_msg += '請給出完整的趨勢分析報告，顯示每日鉑金價格（日期：價格）,台幣/每克，使用繁體中文。'

    return content_msg

# 主函數，調用 GPT 生成分析報告
def platinum_gpt():
    content_msg = generate_platinum_content_msg()
    print(content_msg)  # 調試輸出

    msg = [{
        "role": "system",
        "content": "你現在是一位專業的鉑金價格分析師, 使用以下数据来撰写分析报告。"
    }, {
        "role": "user",
        "content": content_msg
    }]

    reply_data = get_reply(msg)
    return reply_data

# # 測試時，您可以調用 fetch_and_process_platinum_data() 來查看抓取的數據結構
# if __name__ == "__main__":
#     try:
#         print(fetch_and_process_platinum_data())
#     except Exception as e:
#         print(f"錯誤: {e}")
