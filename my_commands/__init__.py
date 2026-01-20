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
from .CaiyunfangweiCrawler import CaiyunfangweiCrawler

# 定義公開 API
__all__ = [
    'lottery_gpt', 
    'CaiyunfangweiCrawler',
    '__version__',
    '__author__'
]
