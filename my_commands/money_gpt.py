import os
import openai
from groq import Groq
from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup

# 設定 API 金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 建立 GPT 模型
def get_reply(messages):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=messages)
        reply = response["choices"][0]["message"]["content"]
    except openai.OpenAIError as openai_err:
        try:
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                max_tokens=1000,
                temperature=1.2
            )
            reply = response.choices[0].message.content
        except groq.GroqError as groq_err:
            reply = f"OpenAI API 發生錯誤: {openai_err.error.message}，GROQ API 發生錯誤: {groq_err.message}"
    return reply

# 擷取匯率資料
def fetch_jpy_rates(kind):
    # 目標網址
    url = f"https://rate.bot.com.tw/xrt/quote/day/{kind}"

    # 發送HTTP請求
    response = requests.get(url)

    # 確定HTTP請求成功
    if response.status_code == 200:
        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 在網頁中找到表格並提取即期匯率和本行賣出價格
        table = soup.find('table', class_='table table-striped table-bordered table-condensed table-hover')
        if not table:
            print("找不到匯率資料的表格。")
            return None

        rows = table.find('tbody').find_all('tr')

        time_rates = []  # 日期時間
        spot_rates = []  # 即期匯率
        selling_rates = []  # 本行賣出價格

        for row in rows:
            columns = row.find_all('td')
            if len(columns) >= 5:
                time_rate = columns[0].text.strip()
                spot_rate = columns[2].text.strip()
                selling_rate = columns[3].text.strip()

                print(f"日期時間: {time_rate}, 即期匯率: {spot_rate}, 本行賣出價格: {selling_rate}")

                time_rates.append(time_rate)
                spot_rates.append(spot_rate)
                selling_rates.append(selling_rate)

        # 建立 DataFrame
        df = pd.DataFrame({
            '日期時間': time_rates,
            '即期匯率': spot_rates,
            '本行賣出價格': selling_rates
        })

        return df
    else:
        print(f"HTTP 請求失敗，狀態碼: {response.status_code}")
        return None

# 生成分析報告內容
def generate_content_msg(kind):
    # 獲取和處理資料
    money_prices_df = fetch_jpy_rates(kind)
    if money_prices_df is None or money_prices_df.empty:
        return "無法獲取匯率資料。"

    # 從資料中獲取需要的最高價和最低價資訊
    money_prices_df['本行賣出價格'] = pd.to_numeric(money_prices_df['本行賣出價格'], errors='coerce')
    max_price = money_prices_df['本行賣出價格'].max()  # 最高本行賣出價格
    min_price = money_prices_df['本行賣出價格'].min()  # 最低本行賣出價格
    last_date = money_prices_df['日期時間'].iloc[-1]  # 假設最後一行是最新日期資料

    # 構造專業分析報告的內容
    content_msg = f'你現在是一位專業的{kind}幣種分析師，使用以下資料來撰寫分析報告：\n'
    content_msg += f'{money_prices_df.tail(30)} 顯示最近的30筆資料，\n'
    content_msg += f'最新日期時間: {last_date}，最高價: {max_price}，最低價: {min_price}。\n'
    content_msg += '請給出完整的趨勢分析報告，顯示每日匯率（日期時間、匯率）（幣種/台幣），使用繁體中文。'

    return content_msg

# 主函式
def money_gpt(kind):
    content_msg = generate_content_msg(kind)
    if content_msg == "無法獲取匯率資料。":
        return content_msg

    print(content_msg)  # 調試輸出

    msg = [{
        "role": "system",
        "content": f"你現在是一位專業的{kind}幣種分析師，使用以下資料來撰寫分析報告。"
    }, {
        "role": "user",
        "content": content_msg
    }]

    reply_data = get_reply(msg)
    return reply_data