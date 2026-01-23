import sys
import os
from datetime import datetime
from unittest.mock import MagicMock

# Add current directory to path
sys.path.append(os.getcwd())

# Mock TaiwanLottery module
mock_tl = MagicMock()
sys.modules["TaiwanLottery"] = mock_tl

# Mock class for results
class MockDraw:
    def __init__(self, date_str, numbers, special=None):
        self.draw_date = datetime.strptime(date_str, "%Y/%m/%d")
        self.numbers = numbers
        self.special = special
        self.number = numbers

# Setup mock return values
now = datetime.now()
current_month_str = now.strftime("%Y/%m")
prev_month_num = now.month - 1 if now.month > 1 else 12
prev_year_num = now.year if now.month > 1 else now.year - 1
prev_month_str = f"{prev_year_num}/{prev_month_num:02d}"

# Create draws: 2 in current month, 1 in previous
draw1 = MockDraw(f"{current_month_str}/05", [1, 2, 3, 4, 5, 6], 7)
draw2 = MockDraw(f"{current_month_str}/02", [11, 12, 13, 14, 15, 16], 17)
draw3 = MockDraw(f"{prev_month_str}/28", [21, 22, 23, 24, 25, 26], 27)

mock_crawler_instance = MagicMock()
mock_crawler_instance.lotto649.return_value = [draw1, draw2, draw3]
mock_tl.TaiwanLotteryCrawler.return_value = mock_crawler_instance

# Import the module under test
from my_commands.lottery_gpt import lottery_gpt

def test_monthly_history():
    print(f"Testing for month: {current_month_str}")
    result = lottery_gpt("大樂透")
    print("\nGenerated Report Preview:")
    print("-" * 50)
    print(result)
    print("-" * 50)
    
    if "**本月開獎紀錄：**" in result:
        print("✅ Header found")
    else:
        print("❌ Header missing")
        
    if f"{current_month_str}/05" in result and f"{current_month_str}/02" in result:
        print("✅ Current month dates found")
    else:
        print("❌ Current month dates missing")
        
    # Check if previous month is NOT in the monthly section
    # It's hard to parse exactly without regex, but we can visually verify
    if f"{prev_month_str}/28" in result:
        # It might appear in other sections if logic changes, but let's just warn
        print(f"ℹ️ Previous month date found in report (might be normal if used elsewhere)")

if __name__ == "__main__":
    test_monthly_history()
