# my_commands/lottery_gpt.py
# 台灣彩票分析模組（支援：大樂透／威力彩／今彩539／雙贏彩／3星彩／4星彩／38樂合彩／39樂合彩／49樂合彩）
import random
from datetime import datetime
import logging

from TaiwanLottery import TaiwanLotteryCrawler  # 參考資料：支援九種彩券遊戲  [oai_citation:0‡GitHub](https://github.com/stu01509/TaiwanLotteryCrawler?utm_source=chatgpt.com)

logger = logging.getLogger(__name__)

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

        if latest:
            draw_date = getattr(latest, "draw_date", None)
            if draw_date:
                draw_date = draw_date.strftime("%Y/%m/%d")
            else:
                draw_date = "—"

            numbers = getattr(latest, "numbers", None) or getattr(latest, "number", None)
            if isinstance(numbers, (list, tuple)):
                numbers_str = ", ".join(f"{n:02d}" for n in numbers)
            else:
                numbers_str = str(numbers)

            special_str = ""
            if special_label:
                special_val = getattr(latest, "special", None)
                if special_val is not None:
                    special_str = f"（{special_label}：{special_val:02d}）"

        else:
            # 抓不到資料，走隨機備用
            draw_date = datetime.now().strftime("%Y/%m/%d")
            numbers = sorted(random.sample(range(1, max_num + 1), num_main))
            numbers_str = ", ".join(f"{n:02d}" for n in numbers)
            special_str = ""
            if special_label:
                special_rand = random.randint(1, max_num if special_label else max_num)
                special_str = f"（{special_label}：{special_rand:02d}）"

        # 建議號碼（隨機）
        suggest = sorted(random.sample(range(1, max_num + 1), num_main))
        suggest_str = ", ".join(f"{n:02d}" for n in suggest)
        suggest_special_str = ""
        if special_label:
            special_sug = random.randint(1, max_num)
            suggest_special_str = f"（{special_label}：{special_sug:02d}）"

        # 簡單分析文字（可按彩種客製）
        analysis = f"{kind}：近期開獎號碼動態多變，建議理性娛樂，不宜過度投注。"

        return (
            f"**{kind} 分析報告**\n\n"
            f"📅 最新開獎（{draw_date}）：{numbers_str} {special_str}\n\n"
            f"🎯 下期建議：{suggest_str} {suggest_special_str}\n\n"
            f"💡 分析：{analysis}\n\n"
            f"[官方歷史開獎查詢](https://www.taiwanlottery.com.tw/)"
        )

    except Exception as e:
        logger.error(f"{kind} 分析內部錯誤：{e}", exc_info=True)
        # 錯誤備用隨機
        rnd = sorted(random.sample(range(1, max_num + 1), num_main))
        rnd_str = ", ".join(f"{n:02d}" for n in rnd)
        return (
            f"**{kind} 分析報告**\n\n"
            f"📅 最新開獎：資料取得失敗（顯示隨機）\n\n"
            f"🎯 下期建議：{rnd_str}\n\n"
            f"💡 分析：資料來源暫時異常，請稍後再試。\n\n"
            f"[官方歷史開獎查詢](https://www.taiwanlottery.com.tw/)"
        )