# my_commands/__init__.py
# 說明：
# - 讓 my_commands 成為可匯入套件
# - 「便利匯出」：直接 from my_commands import lottery_gpt 也可拿到函式
# - 若子模組缺失，提供友善 ImportError 訊息而非讓整個 import 崩潰
# - 如不想便利匯出，可使用「最小版」(見下一段)

from importlib import import_module

__all__ = [
    "lottery_gpt",            # 直接匯出函式名（便利）
    "CaiyunfangweiCrawler",   # 直接匯出爬蟲類別（便利）
    "stock",                  # 讓 my_commands.stock.* 保持可被尋址
]

# 便利匯出：函式 lottery_gpt（對應 my_commands/lottery_gpt.py）
try:
    # 這行讓你可以：from my_commands import lottery_gpt  ← 取得同名函式
    from .lottery_gpt import lottery_gpt  # type: ignore
except Exception as e:
    # 若你的 lottery_gpt.py 有語法/相依錯誤，這裡會保一個同名函式，
    # 呼叫時才丟出更友善的 ImportError 提示。
    def lottery_gpt(*args, **kwargs):  # type: ignore
        raise ImportError(f"無法載入 my_commands.lottery_gpt：{e}")  # 清楚的錯誤訊息

# 便利匯出：財神方位抓取（對應 my_commands/CaiyunfangweiCrawler.py）
try:
    from .CaiyunfangweiCrawler import CaiyunfangweiCrawler  # type: ignore
except Exception:
    # 不強制；缺了就算了，不影響其他功能
    pass

# 可選：版本資訊
__version__ = "0.1.0"