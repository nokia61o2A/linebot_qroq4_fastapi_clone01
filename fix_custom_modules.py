#!/usr/bin/env python3
"""
修復所有自訂模組中的模型設定
"""
import os
import importlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 設置正確的模型
GROQ_MODEL_CORRECT = "llama-3.1-8b-instant"

def fix_module_errors(module_path):
    """修復單個模組的錯誤"""
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替換已停用的模型
        content = content.replace('"llama3-70b-8192"', f'"{GROQ_MODEL_CORRECT}"')
        content = content.replace("'llama3-70b-8192'", f"'{GROQ_MODEL_CORRECT}'")
        
        # 修復錯誤處理
        content = content.replace('except groq.GroqError as groq_err:', 'except Exception as groq_err:')
        content = content.replace('except GroqError as groq_err:', 'except Exception as groq_err:')
        
        # 寫回文件
        with open(module_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"已修復: {module_path}")
        
    except Exception as e:
        logger.error(f"修復 {module_path} 時發生錯誤: {e}")

# 要修復的模組列表
modules_to_fix = [
    'my_commands/lottery_gpt.py',
    'my_commands/gold_gpt.py',
    'my_commands/platinum_gpt.py',
    'my_commands/money_gpt.py',
    'my_commands/one04_gpt.py',
    'my_commands/partjob_gpt.py',
    'my_commands/crypto_coin_gpt.py',
    'my_commands/weather_gpt.py'
]

if __name__ == "__main__":
    for module_path in modules_to_fix:
        if os.path.exists(module_path):
            fix_module_errors(module_path)
        else:
            logger.warning(f"模組不存在: {module_path}")
    
    logger.info("修復完成！")