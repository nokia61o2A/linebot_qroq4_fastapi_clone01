import os
import openai
from groq import Groq
from datetime import datetime
import pandas as pd

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
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=1000,
                temperature=1.2
            )
            reply = response.choices[0].message.content
        except Exception as groq_err:
            reply = f"OpenAI API 發生錯誤: {openai_err.error.message}，GROQ API 發生錯誤: {groq_err.message}"
    return reply

def fetch_and_process_data():
    # 获取和处理数据
    df_list = pd.read_html("https://rate.bot.com.tw/gold/chart/year/TWD")
    df = df_list[0]

    # 数据处理
    df = df[["日期", "本行賣出價格"]].copy()  # 复制以避免 SettingWithCopyWarning
    df.index = pd.to_datetime(df["日期"], format="%Y/%m/%d")
    df.sort_index(inplace=True)

    return df

def generate_content_msg():
    # 获取和处理数据
    gold_prices_df = fetch_and_process_data()

    # 从数据中获取需要的最高价和最低价信息
    max_price = gold_prices_df['本行賣出價格'].max()
    min_price = gold_prices_df['本行賣出價格'].min()
    last_date = gold_prices_df.index[-1].strftime("%Y-%m-%d")  # 假设最后一行是最新日期数据

    # 构造专业分析报告的内容
    content_msg = f'你現在是一位專業的金價分析師, 使用以下数据来撰写分析报告:\n'
    content_msg += f'{gold_prices_df}\n'
    content_msg += f'最新日期: {last_date}, 最高金價: {max_price} {{日期}}, 最低金價: {min_price}{{日期}}。\n'
    content_msg += '請給出完整的趨勢分析報告，顯示每日金價{日期}{金價}(台幣)，'
    content_msg += '，使用繁體中文。'

    return content_msg

def gold_gpt():
    content_msg = generate_content_msg()
    print(content_msg)  # 调试输出

    msg = [{
        "role": "system",
        "content": "你現在是一位專業的金價分析師, 使用以下数据来撰写分析报告。"
    }, {
        "role": "user",
        "content": content_msg
    }]

    reply_data = get_reply(msg)
    return reply_data
