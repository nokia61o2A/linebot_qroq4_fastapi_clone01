# my_commands/stock/stock_news.py
import time
import random
import requests
from typing import List, Dict

_CNYES_API = "https://ess.api.cnyes.com/ess/api/v1/news/keyword"
_HEADERS = {
    # 模擬瀏覽器，避免被擋
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cnyes.com/",
    "Origin": "https://www.cnyes.com",
}

def _sleep_with_jitter(base: float = 0.6, max_extra: float = 0.6):
    time.sleep(base + random.random() * max_extra)

def _fetch_from_cnyes(keyword: str, limit: int = 5, timeout: float = 6.0) -> List[Dict]:
    """
    嘗試從鉅亨關鍵字 API 抓新聞；抓不到就丟出例外給上層處理。
    這裡做了 2 次簡單重試（含抖動），避免偶發 403/5xx。
    """
    params = {"q": keyword, "limit": limit, "page": 1}
    last_err = None
    for _ in range(2):
        try:
            r = requests.get(_CNYES_API, headers=_HEADERS, params=params, timeout=timeout)
            # 429/503 之類先稍等再重試
            if r.status_code in (429, 503):
                _sleep_with_jitter(1.0, 1.0)
                continue
            r.raise_for_status()

            # 有些情況會回 HTML（被擋），這時 .json() 會炸；先檢查 content-type
            ctype = r.headers.get("Content-Type", "")
            if "json" not in ctype.lower():
                raise ValueError(f"unexpected content-type: {ctype}")

            data = r.json()
            items = (data or {}).get("data", {}).get("items", [])
            out = []
            for it in items:
                title = (it or {}).get("title", "").strip()
                link = (it or {}).get("url", "").strip()
                if title:
                    out.append({"title": title, "url": link})
            return out
        except Exception as e:
            last_err = e
            _sleep_with_jitter()
    raise RuntimeError(f"cnyes fetch failed: {last_err}")

def stock_news(keyword: str, limit: int = 5) -> str:
    """
    回傳「純文字摘要」，一行一則：『• 標題｜URL』
    抓不到就回空字串（讓上層自己決定如何顯示）。
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return ""

    try:
        items = _fetch_from_cnyes(keyword, limit=limit)
        if not items:
            return ""
        lines = [f"• {it['title']}｜{it.get('url','')}".strip() for it in items]
        return "\n".join(lines)
    except Exception:
        # 這裡先不做第二來源，以穩定為主；要加 RSS/其他 API 可在這裡擴充
        return ""