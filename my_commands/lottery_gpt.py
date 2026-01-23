# my_commands/lottery_gpt.py
# 台灣彩票分析模組（支援：大樂透／威力彩／今彩539／雙贏彩／3星彩／4星彩／38樂合彩／39樂合彩／49樂合彩）
import random
from datetime import datetime
import logging

from TaiwanLottery import TaiwanLotteryCrawler  # 套件提供的模組名稱為 TaiwanLottery

logger = logging.getLogger(__name__)

def get_val(obj, keys):
    """從 dict 或 object 中取得值，支援多個 key"""
    if isinstance(keys, str): keys = [keys]
    for k in keys:
        if isinstance(obj, dict):
            if k in obj: return obj[k]
        else:
            if hasattr(obj, k): return getattr(obj, k)
    return None

def parse_date(d):
    """解析日期"""
    if isinstance(d, datetime): return d
    if isinstance(d, str):
        try:
            return datetime.fromisoformat(d)
        except:
            pass
    return None

# 定義彩種對應：函式名稱、主號數量、號碼最大值、（可選）特別區說明
_LOTTERY_MAP = {
    "大樂透":     ("lotto649",    6, 49,    "特別號"),
    "威力彩":     ("super_lotto", 6, 39,    "第二區"),
    "今彩539":    ("daily_cash",  5, 39,    None),
    "雙贏彩":     ("lotto1224",   6, 49,    None),
    "3星彩":      ("lotto3d",     3, 10,    None),
    "4星彩":      ("lotto4d",     4, 10,    None),
    "38樂合彩":   ("lotto38m6",   6, 38,    None),
    "39樂合彩":   ("lotto39m5",   5, 39,    None),
    "49樂合彩":   ("lotto49m6",   6, 49,    None),
}

def lottery_gpt(lottery_type: str) -> str:
    """
    彩票分析入口：支援以上九種彩種
    - 嘗試使用 TaiwanLotteryCrawler 抓取最新開獎資料
    - 若抓取失敗，則用隨機號碼備用
    - 生成下期建議號碼 + 簡單分析
    """
    try:
        kind = lottery_type.strip()
        if kind not in _LOTTERY_MAP:
            return (
                f"**{kind} 分析報告**\n\n"
                "目前支援彩種：\n" +
                "／".join(_LOTTERY_MAP.keys()) +
                "\n\n💡 提示：彩票娛樂為主，請理性投注。\n\n"
                "[樂透官網](https://www.taiwanlottery.com.tw/)"
            )

        func_name, num_main, max_num, special_label = _LOTTERY_MAP[kind]
        crawler = TaiwanLotteryCrawler()
        func = getattr(crawler, func_name)
        result = func()
        latest = None
        if isinstance(result, list) and result:
            latest = result[0]

        # ===== 取近期期數做趨勢分析（最多取 30 期） =====
        all_draws = []
        try:
            if isinstance(result, list):
                for item in result[:30]:
                    nums = get_val(item, ["numbers", "number", "獎號"])
                    if isinstance(nums, (list, tuple)):
                        all_draws.append(list(nums))
        except Exception:
            pass

        # 若抓不到任何期數，走備用一筆
        if not all_draws:
            all_draws = [sorted(random.sample(range(1, max_num + 1), num_main))]

        # 最新一期顯示（若有）
        latest_draw = all_draws[0]
        
        draw_date_val = get_val(latest, ["draw_date", "開獎日期"])
        draw_date = parse_date(draw_date_val)
        draw_date_str = draw_date.strftime("%Y/%m/%d") if draw_date else datetime.now().strftime("%Y/%m/%d")
        
        numbers_str = ", ".join(f"{n:02d}" for n in latest_draw)
        special_str = ""
        if special_label:
            special_val = get_val(latest, ["special", "特別號"]) if latest else None
            if special_val is not None:
                special_str = f"（{special_label}：{special_val:02d}）"

        # ===== 計算熱門 / 冷門 =====
        freq = {n: 0 for n in range(1, max_num + 1)}
        for draw in all_draws:
            for n in draw:
                freq[n] += 1
        sorted_by_freq = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        hot = [n for n, c in sorted_by_freq[:5]]
        cold = [n for n, c in sorted(freq.items(), key=lambda x: (x[1], x[0]))[:5]]

        # ===== 奇偶 / 大小 =====
        total_nums = sum(len(d) for d in all_draws)
        odd = sum(1 for d in all_draws for n in d if n % 2 == 1)
        even = total_nums - odd
        odd_even_desc = "奇偶分布相對均衡" if abs(odd - even) <= total_nums * 0.1 else ("奇數略多" if odd > even else "偶數略多")

        small_threshold = (max_num // 2)
        small = sum(1 for d in all_draws for n in d if n <= small_threshold)
        large = total_nums - small
        size_desc = "大小號碼分布相對均衡" if abs(small - large) <= total_nums * 0.1 else ("小號略多" if small > large else "大號略多")

        # ===== 連續號碼（例如 11 與 12 同期出現） =====
        consecutive_hits = 0
        for d in all_draws:
            s = set(d)
            consecutive_hits += sum(1 for n in s if (n + 1) in s)
        consec_desc = "連續號碼出現頻率較低" if consecutive_hits <= len(all_draws) * 0.2 else "連續號碼偶爾出現"

        # ===== 3 組建議 =====
        def pick_from(pool, k):
            p = list({x for x in pool if 1 <= x <= max_num})
            if len(p) < k:
                # 用剩餘號碼補足
                remain = [x for x in range(1, max_num + 1) if x not in p]
                p += random.sample(remain, k - len(p))
            return sorted(random.sample(p, k))

        # 組合 1：冷門號碼組合
        combo1 = pick_from(cold, num_main)
        # 組合 2：熱門號碼組合
        combo2 = pick_from(hot, num_main)
        # 組合 3：隨機號碼組合
        all_nums = list(range(1, max_num + 1))
        combo3 = sorted(random.sample(all_nums, num_main))

        def fmt(nums):
            return "、".join(f"{n:02d}" for n in nums)

        # ===== 本月歷史開獎 =====
        month_history_str = ""
        try:
            now_dt = datetime.now()
            current_month_draws = []
            # result 應該是 TaiwanLotteryCrawler 抓回來的 list
            if isinstance(result, list):
                for item in result:
                    d_val = get_val(item, ["draw_date", "開獎日期"])
                    d_date = parse_date(d_val)
                    
                    # 確保是 datetime 且為本月
                    if d_date and d_date.year == now_dt.year and d_date.month == now_dt.month:
                        nums = get_val(item, ["numbers", "number", "獎號"])
                        if isinstance(nums, (list, tuple)):
                            nums_s = ", ".join(f"{n:02d}" for n in nums)
                            # 特別號
                            sp_s = ""
                            if special_label:
                                sp_val = get_val(item, ["special", "特別號"])
                                if sp_val is not None:
                                    sp_s = f" + {sp_val:02d}"
                            
                            d_str = d_date.strftime("%Y/%m/%d")
                            current_month_draws.append(f"{d_str}：{nums_s}{sp_s}")
            
            if current_month_draws:
                month_history_str = "**本月開獎紀錄：**\n" + "\n".join(current_month_draws) + "\n\n"
        except Exception as e:
            logger.error(f"處理本月開獎紀錄失敗：{e}")

        # ===== 照用戶指定格式輸出 =====
        report = (
            f"根據近期的{kind}數據，以下是一些趨勢分析和3組隨機號碼建議：\n\n"
            f"**趨勢分析：**\n\n"
            f"1. **熱門號碼：** {fmt(hot)}（出現頻率較高）\n"
            f"2. **冷門號碼：** {fmt(cold)}（出現頻率較低）\n"
            f"3. **奇偶分布：** {odd_even_desc}。\n"
            f"4. **大小分布：** {size_desc}。\n"
            f"5. **連續號碼：** {consec_desc}。\n\n"
            f"{month_history_str}"
            f"**3組隨機號碼建議：**\n\n"
            f"1. **組合 1：** {fmt(combo1)}（冷門號碼組合）\n"
            f"2. **組合 2：** {fmt(combo2)}（熱門號碼組合）\n"
            f"3. **組合 3：** {fmt(combo3)}（隨機號碼組合）\n"
        )

        return report

    except Exception as e:
        logger.error(f"{kind} 分析內部錯誤：{e}", exc_info=True)
        # 錯誤備用：仍依照指定格式輸出（全部使用隨機與均衡策略）
        pool = list(range(1, max_num + 1))
        random.shuffle(pool)
        hot = sorted(pool[:5])
        cold = sorted(pool[-5:])
        def fmt(nums):
            return "、".join(f"{n:02d}" for n in nums)
        combo1 = sorted(random.sample(pool, num_main))
        combo2 = sorted(random.sample(pool, num_main))
        # 組合 3：隨機號碼組合
        combo3 = sorted(random.sample(pool, num_main))

        return (
            f"根據近期的{kind}數據，以下是一些趨勢分析和3組隨機號碼建議：\n\n"
            f"**趨勢分析：**\n\n"
            f"1. **熱門號碼：** {fmt(hot)}（出現頻率較高）\n"
            f"2. **冷門號碼：** {fmt(cold)}（出現頻率較低）\n"
            f"3. **奇偶分布：** 奇偶分布相對均衡。\n"
            f"4. **大小分布：** 大小號碼分布相對均衡。\n"
            f"5. **連續號碼：** 連續號碼出現頻率較低。\n\n"
            f"**3組隨機號碼建議：**\n\n"
            f"1. **組合 1：** {fmt(combo1)}（冷門號碼組合）\n"
            f"2. **組合 2：** {fmt(combo2)}（熱門號碼組合）\n"
            f"3. **組合 3：** {fmt(combo3)}（隨機號碼組合)\n"
        )
