#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試翻譯功能和羅馬拼音標註
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 測試日文羅馬拼音功能
try:
    import pykakasi
    print("✅ pykakasi 已安裝")
    
    # 測試日文轉羅馬拼音
    kks = pykakasi.kakasi()
    test_japanese = "血圧が高い"
    result = kks.convert(test_japanese)
    romaji = ''.join([item['hepburn'] for item in result])
    print(f"日文: {test_japanese}")
    print(f"羅馬拼音: {romaji}")
    
except ImportError:
    print("❌ pykakasi 未安裝")

# 測試韓文羅馬拼音功能
try:
    from korean_romanizer.romanizer import Romanizer
    print("\n✅ korean-romanizer 已安裝")
    
    # 測試韓文轉羅馬拼音
    test_korean = "안녕하세요"
    romanizer = Romanizer(test_korean)
    romaji = romanizer.romanize()
    print(f"韓文: {test_korean}")
    print(f"羅馬拼音: {romaji}")
    
except ImportError:
    print("❌ korean-romanizer 未安裝")

# 測試韓文注音功能
try:
    from hangul_jamo import decompose
    print("\n✅ hangul-jamo 已安裝")
    
    # 測試韓文分解
    test_korean = "안녕"
    decomposed = decompose(test_korean)
    print(f"韓文: {test_korean}")
    print(f"分解結果: {decomposed}")
    
except ImportError:
    print("❌ hangul-jamo 未安裝")

print("\n🎉 測試完成！")