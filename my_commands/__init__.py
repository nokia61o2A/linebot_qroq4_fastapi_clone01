# my_commands/__init__.py
"""
我的指令模組套件
包含各種 AI 分析功能，如彩票分析、財神方位等
"""

# 版本資訊
__version__ = "1.0.0"
__author__ = "AI Bot Developer"

# 重新導出常用模組，讓使用者可以直接匯入
from .lottery_gpt import lottery_gpt
try:
    from .CaiyunfangweiCrawler import CaiyunfangweiCrawler
except Exception:
    # 後備：若未提供方位爬蟲，給一個安全占位，避免套件初始化失敗
    class CaiyunfangweiCrawler:  # type: ignore
        def get_caiyunfangwei(self):
            return {
                "今天日期": "未知",
                "今日歲次": "未知",
                "財神方位": "未知",
                "error": "缺少 CaiyunfangweiCrawler，已使用後備。"
            }

# 定義公開 API
__all__ = [
    'lottery_gpt',
    'CaiyunfangweiCrawler',
    '__version__',
    '__author__'
]
