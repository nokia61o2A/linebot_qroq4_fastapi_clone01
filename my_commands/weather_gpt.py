import os
import pandas as pd
import requests
from datetime import datetime
import openai
from groq import Groq

# 設定 API 金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 中央氣象局授權金鑰，請在環境變數設定 CWB_AUTHORIZATION
CWB_AUTHORIZATION = os.getenv("CWB_AUTHORIZATION", "CWA-272CECC4-4454-459C-A846-12E5EB4ABF74")


def fetch_and_process_weather(location: str = "臺北市") -> pd.DataFrame:
    """
    從中央氣象局開放資料取得未來 36 小時的天氣預報
    資料來源: 中央氣象局資料集 F-C0032-001 (天氣預報36小時)
    """
    api_url = (
        "https://opendata.cwb.gov.tw/api/v1/rest/datastore/F-C0032-001"
        f"?Authorization={CWB_AUTHORIZATION}&format=JSON&locationName={location}"
    )
    resp = requests.get(api_url)
    resp.raise_for_status()
    data = resp.json()

    # 解析指定地區
    locations = data["records"]["location"]
    target = next(loc for loc in locations if loc["locationName"] == location)
    weather_elements = target["weatherElement"]

    # 建立 DataFrame
    records = []
    for element in weather_elements:
        name = element["elementName"]
        for period in element["time"]:
            start = pd.to_datetime(period["startTime"])
            end = pd.to_datetime(period["endTime"])
            value = period["parameter"].get("parameterName", "")
            records.append({
                "起始時間": start,
                "結束時間": end,
                "天氣參數": name,
                "參數值": value
            })
    df = pd.DataFrame(records)
    df.sort_values(by="起始時間", inplace=True)
    return df


def generate_content_msg(location: str = "臺北市") -> str:
    df = fetch_and_process_weather(location)
    latest = df.iloc[-1]

    content = f"### {location} 未來 36 小時天氣資料\n"
    content += df.to_string(index=False)
    content += ("\n" +
        f"最新時段：{latest['起始時間']} ~ {latest['結束時間']}，"
        f"參數：{latest['天氣參數']} = {latest['參數值']}\n"
    )
    content += "請根據以上資料，以繁體中文撰寫專業台灣氣象分析報告，並提供趨勢與建議。"
    return content


def weather_gpt(location: str = "臺北市") -> str:
    """
    主函式：呼叫 LLM 生成天氣分析報告
    """
    content_msg = generate_content_msg(location)
    messages = [
        {"role": "system", "content": "你現在是一位專業台灣氣象分析師，使用以下資料來撰寫分析報告。"},
        {"role": "user", "content": content_msg}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=messages
        )
        return response["choices"][0]["message"]["content"]
    except openai.OpenAIError:
        resp = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            max_tokens=1000,
            temperature=1.2
        )
        return resp.choices[0].message.content
