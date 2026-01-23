
import sys
import os

# 將當前目錄加入 path 以便 import my_commands
sys.path.append(os.getcwd())

try:
    print("Attempting to import my_commands.lottery_gpt...")
    from my_commands.lottery_gpt import lottery_gpt
    print("Import successful.")
    
    print("Attempting to run lottery_gpt('大樂透')...")
    result = lottery_gpt('大樂透')
    print("Execution successful.")
    print("Result preview:")
    print(result[:200] + "...")
    
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"RuntimeError: {e}")
    import traceback
    traceback.print_exc()
