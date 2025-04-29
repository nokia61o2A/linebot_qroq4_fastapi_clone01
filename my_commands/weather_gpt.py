import os
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import openai
from groq import Groq

# 設定 API 金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 中央氣象局授權金鑰，請在環境變數設定 CWB_AUTHORIZATION
CWB_AUTHORIZATION = os.getenv(
    "CWB_AUTHORIZATION", 
    "CWA-272CECC4-4454-459C-A846-12E5EB4ABF74"
)


def get_cwb_session() -> requests.Session:
    """
    建立具有重試機制的 Session，以提高對 CWB API 的連線穩定度
    """
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,  # 增加退避時間
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        connect=5,  # 連接重試次數
        read=5,    # 讀取重試次數
        redirect=5  # 重定向重試次數
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_and_process_weather(location: str = "臺北市") -> pd.DataFrame:
    """
    從中央氣象局開放資料取得未來 36 小時的天氣預報，並處理為 DataFrame。
    使用重試機制以避免暫時性解析或連線錯誤。
    """
    session = get_cwb_session()
    api_url = (
        "https://opendata.cwb.gov.tw/api/v1/rest/datastore/F-C0032-001"
        f"?Authorization={CWB_AUTHORIZATION}&format=JSON&locationName={location}"
    )

    print(f"正在連線至中央氣象局 API：{api_url}")
    
    try:
        resp = session.get(api_url, timeout=(10, 30))  # (連接超時, 讀取超時)
        resp.raise_for_status()
    except requests.RequestException as e:
        # 捕捉所有與請求相關的錯誤
        raise RuntimeError(f"無法連線至中央氣象局 API：{e}")

    data = resp.json()
    locations = data.get("records", {}).get("location", [])
    if not locations:
        raise RuntimeError(f"中央氣象局回傳資料格式異常，無法找到 location 資訊。")

    # 找出對應地區
    target = next((loc for loc in locations if loc.get("locationName") == location), None)
    if not target:
        raise RuntimeError(f"中央氣象局資料中不包含地區：{location}")

    weather_elements = target.get("weatherElement", [])
    records = []
    for element in weather_elements:
        name = element.get("elementName")
        for period in element.get("time", []):
            start = pd.to_datetime(period.get("startTime"))
            end = pd.to_datetime(period.get("endTime"))
            value = period.get("parameter", {}).get("parameterName", "")
            records.append({
                "起始時間": start,
                "結束時間": end,
                "天氣參數": name,
                "參數值": value
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("取得的天氣資料為空，請確認參數是否正確。")
    df.sort_values(by="起始時間", inplace=True)
    return df


def generate_content_msg(location: str = "臺北市") -> str:
    """
    組成傳給 LLM 的使用者訊息，包含完整 DataFrame 與最新時段資訊。
    """
    df = fetch_and_process_weather(location)
    latest = df.iloc[-1]

    content = f"### {location} 未來 36 小時天氣資料\n"
    content += df.to_string(index=False)
    content += (
        "\n"
        f"最新時段：{latest['起始時間']} ~ {latest['結束時間']}，"
        f"參數：{latest['天氣參數']} = {latest['參數值']}\n"
    )
    content += "請根據以上資料，以繁體中文撰寫專業台灣氣象分析報告，並提供趨勢與建議。"
    return content


def weather_gpt(location: str = "臺北市") -> str:
    """
    主函式：呼叫 LLM 生成天氣分析報告，並處理例外情況。
    """
    try:
        content_msg = generate_content_msg(location)
    except RuntimeError as e:
        # 回傳錯誤訊息給呼叫者
        return str(e)

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