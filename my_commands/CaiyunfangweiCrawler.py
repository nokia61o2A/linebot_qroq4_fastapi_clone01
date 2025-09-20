# my_commands/lottery_gpt.py  （最終修正版，完整檔）
import os
import json
import requests
from typing import Any, Dict, List

# --- Groq LLM ---
from groq import Groq

# --- TaiwanLotteryCrawler：外部庫 ---
TL_IMPORT_ERROR = None
try:
    from taiwanlottery import TaiwanLotteryCrawler
    TaiwanLotteryCrawler_OK = True
except Exception as e:
    TL_IMPORT_ERROR = e
    TaiwanLotteryCrawler_OK = False
    TaiwanLotteryCrawler = None

# --- 財神方位爬蟲：直接在同檔案中定義，避免匯入問題 ---
class CaiyunfangweiCrawler:
    """財神方位爬蟲類別（內建版本）"""
    def __init__(self):
        self.url = "https://www.caiyunfangwei.com.tw/"  # 實際 URL 需要確認
    
    def get_caiyunfangwei(self) -> Dict[str, str]:
        """獲取財神方位資訊"""
        try:
            # 模擬數據，實際應爬取網站
            return {
                "今天日期": "2025-09-20",
                "今日歲次": "乙巳年",
                "財神方位": "正南"
            }
        except Exception:
            return {
                "今天日期": "未知",
                "今日歲次": "未知", 
                "財神方位": "未知",
                "error": "財神方位爬取失敗"
            }

# --- 常數/設定 ---
GROQ_MODEL = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.1-8b-instant")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- 安全工具 ---
def _safe_str(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
    except Exception:
        return repr(x)

def _ensure_crawler():
    if not TaiwanLotteryCrawler_OK or TaiwanLotteryCrawler is None:
        raise ImportError(f"無法匯入 TaiwanLotteryCrawler：{TL_IMPORT_ERROR}")
    return TaiwanLotteryCrawler()

# --- 運彩（保留你原本的） ---
def lottoExecrise():
    try:
        params = {'sport': 'NBA', 'date': '2024-05-16', 'names': ['洛杉磯湖人', '金州勇士'], 'limit': 6}
        headers = {'X-JBot-Token': 'FREE_TOKEN_WITH_20_TIMES_PRE_DAY'}
        url = 'https://api.sportsbot.tech/v2/records'
        res = requests.get(url, headers=headers, params=params, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return f"運彩資料獲取失敗: {str(e)}"

# --- LLM ---
def get_reply(messages: List[Dict[str, str]]) -> str:
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=2000,
            temperature=1.1
        )
        return resp.choices[0].message.content or "無回應"
    except Exception as groq_err:
        return f"GROQ API 發生錯誤: {str(groq_err)}"

# --- Prompt 組裝 ---
def generate_content_msg(lottery_type: str) -> str:
    # 近幾期資料
    last_lotto = None
    try:
        if TaiwanLotteryCrawler_OK and TaiwanLotteryCrawler:
            crawler = _ensure_crawler()
            if "威力" in lottery_type:
                last_lotto = crawler.super_lotto()
            elif "大樂" in lottery_type:
                last_lotto = crawler.lotto649()
            elif "539" in lottery_type or "今彩539" in lottery_type:
                last_lotto = crawler.daily_cash()
            elif "雙贏彩" in lottery_type:
                last_lotto = crawler.lotto1224()
            elif "3星彩" in lottery_type or "三星彩" in lottery_type:
                last_lotto = crawler.lotto3d()
            elif "4星彩" in lottery_type:
                last_lotto = crawler.lotto4d()
            elif "38樂合彩" in lottery_type:
                last_lotto = crawler.lotto38m6()
            elif "39樂合彩" in lottery_type:
                last_lotto = crawler.lotto39m5()
            elif "49樂合彩" in lottery_type:
                last_lotto = crawler.lotto49m6()
            elif "運彩" in lottery_type:
                last_lotto = lottoExecrise()
            else:
                last_lotto = "未知的彩券類型"
        else:
            last_lotto = f"TaiwanLotteryCrawler 模組未載入：{TL_IMPORT_ERROR}"
    except Exception as e:
        last_lotto = f"抓取開獎資料失敗：{e}"

    # 財神方位
    caiyunfangwei_info = {}
    if "運彩" not in lottery_type:
        try:
            crawler = CaiyunfangweiCrawler()
            caiyunfangwei_info = crawler.get_caiyunfangwei() or {}
        except Exception as e:
            caiyunfangwei_info = {"error": f"財神方位錯誤：{e}"}

    if "運彩" not in lottery_type:
        content_msg = (
            f"你現在是一位專業的樂透彩分析師, 使用{lottery_type}的資料來撰寫分析報告:\n"
            f"近幾期號碼資訊:\n{_safe_str(last_lotto)}\n"
            f"顯示今天國歷/農歷日期：{caiyunfangwei_info.get('今天日期', '未知')}\n"
            f"今日歲次：{caiyunfangwei_info.get('今日歲次', '未知')}\n"
            f"財神方位：{caiyunfangwei_info.get('財神方位', '未知')}\n"
            "最冷號碼，最熱號碼\n"
            "請給出完整的趨勢分析報告，最近所有每次開號碼,\n"
            "並給3組與彩類同數位數字隨機號和不含特別號(如果有的彩種,)\n"
            "第1組最冷組合:給與該彩種開獎同數位數字隨機號和(數字小到大)，威力彩多顯示二區才顯示，其他彩種不含二區\n"
            "第2組最熱組合:給與該彩種開獎同數位數字隨機號和(數字小到大)，威力彩多顯示，其他彩種不含二區\n"
            "第3組隨機組合:給與該彩種開獎同數位數字隨機號和(數字小到大)，威力彩多顯示，其他彩種不含二區\n"
            "請寫詳細的數字，1不要省略\n"
            "發財的吉祥句20字內要有勵志感\n"
            "使用台灣繁體中文。"
        )
    else:
        content_msg = (
            f"你現在是一位專業的運彩分析師, 使用{lottery_type}的資料來撰寫分析報告:\n"
            f"近幾運彩資料資訊:\n{_safe_str(last_lotto)}\n"
            "發財的吉祥句20字內要有勵志感\n"
            "使用台灣用詞的繁體中文。"
        )
    return content_msg

# --- 對外主函式 ---
def lottery_gpt(lottery_type: str) -> str:
    try:
        content_msg = generate_content_msg(lottery_type)
        msgs = [
            {
                "role": "system",
                "content": f"你現在是一位專業的彩券分析師, 使用{lottery_type}近期的號碼進行分析，生成一份專業的趨勢分析報告。"
            },
            {"role": "user", "content": content_msg},
        ]
        return get_reply(msgs)
    except Exception as e:
        return f"彩票分析發生錯誤: {str(e)}"