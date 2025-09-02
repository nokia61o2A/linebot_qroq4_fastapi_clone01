import os
import openai
from groq import Groq
import requests  # 確保引入 requests 模組
import time
import json

# 設定 API 金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 設定 groq API 的速率限制參數
GROQ_RATE_LIMIT = 6000  # 每分鐘允許的 token 數量
GROQ_REQUEST_INTERVAL = 60  # 速率限制的時間間隔，單位：秒

# 初始化請求計數器和時間戳
groq_tokens_used = 0
groq_last_request_time = time.time()

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

class CryptoAnalyzer:
    def fetch_crypto_data(self, coin_id, vs_currency='twd', days='30'):
        """抓取特定加密貨幣的市場數據"""
        url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
        params = {
            'vs_currency': vs_currency,
            'days': days
        }
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f'無法取得加密貨幣數據，錯誤: {response.status_code}')
            return None

    def fetch_current_price(self, coin_id):
        """抓取加密貨幣的現價（TWD 和 USD）"""
        url = 'https://api.coingecko.com/api/v3/simple/price'
        params = {
            'ids': coin_id,
            'vs_currencies': 'twd,usd'
        }
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            return data[coin_id]
        else:
            print(f'無法取得現價，錯誤: {response.status_code}')
            return None

    def analyze_data(self, data):
        """分析抓取的市場數據，找出最高和最低價"""
        if data:
            prices = data['prices']
            highest = max(prices, key=lambda x: x[1])
            lowest = min(prices, key=lambda x: x[1])

            highest_price_date = time.strftime('%Y-%m-%d', time.gmtime(highest[0] / 1000))
            lowest_price_date = time.strftime('%Y-%m-%d', time.gmtime(lowest[0] / 1000))

            return {
                'highest_price': highest[1],
                'highest_price_date': highest_price_date,
                'lowest_price': lowest[1],
                'lowest_price_date': lowest_price_date,
                'price_history': prices
            }
        return None

def generate_crypto_report(coin_id):
    analyzer = CryptoAnalyzer()

    # 抓取現價
    current_price = analyzer.fetch_current_price(coin_id)
    if not current_price:
        return "無法取得現價，請檢查數據源。"

    # 抓取數據
    data = analyzer.fetch_crypto_data(coin_id)

    # 分析數據
    analysis = analyzer.analyze_data(data)

    if not analysis:
        return "無法生成分析報告，請檢查數據源。"

    # 構造報告內容
    report_content = (
        f"加密貨幣: {coin_id}, 現價: {current_price['twd']} TWD / {current_price['usd']} USD\n"
        f"你現在是一位專業的加密貨幣分析師, 使用以下數據來撰寫分析報告：\n"
    )
    report_content += f"最高價: {analysis['highest_price']} TWD ({analysis['highest_price_date']})\n"
    report_content += f"最低價: {analysis['lowest_price']} TWD ({analysis['lowest_price_date']})\n"
    report_content += '價格趨勢:\n'

    for price in analysis['price_history'][:10]:  # 只顯示前10筆
        date = time.strftime('%Y-%m-%d', time.gmtime(price[0] / 1000))
        report_content += f"{date}: {price[1]:.2f} TWD\n"

    report_content += '請提供詳細的趨勢分析，並以台灣繁體中文撰寫。'

    return report_content

def crypto_gpt(coin_id):
    content_msg = generate_crypto_report(coin_id)
    print(content_msg)  # 調試輸出

    msg = [{
        "role": "system",
        "content": "你現在是一位專業的加密貨幣分析師, 使用以下數據來撰寫分析報告。(回答時 length < 2000 tokens)"
    }, {
        "role": "user",
        "content": content_msg
    }]

    reply_data = get_reply(msg)
    return reply_data

# # 範例呼叫
# result = crypto_gpt('bitcoin')
# print(result)
