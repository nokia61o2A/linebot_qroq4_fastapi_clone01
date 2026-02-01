
import requests
from bs4 import BeautifulSoup
import re

_TWSE_ETF_URLS = [
    "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
    "https://www.twse.com.tw/zh/products/securities/etf/products/domestic.html",
]

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def test_fetch():
    print("Starting fetch test...")
    for url in _TWSE_ETF_URLS:
        print(f"\nScanning: {url}")
        try:
            r = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
            print(f"Status: {r.status_code}")
            
            found_count = 0
            samples = []
            
            if "isin.twse.com.tw" in url:
                r.encoding = 'big5'
                soup = BeautifulSoup(r.text, "html.parser")
                rows = soup.find_all("tr")
                print(f"Total rows: {len(rows)}")
                for tr in rows:
                    tds = tr.find_all("td")
                    if not tds: continue
                    raw_txt = tds[0].get_text(strip=True)
                    parts = raw_txt.split()
                    if len(parts) >= 2:
                        code = parts[0]
                        name = parts[1]
                        if re.match(r"^[A-Za-z0-9]{4,6}$", code):
                            found_count += 1
                            if found_count <= 5 or code == "0050":
                                samples.append(f"{code}: {name}")
            else:
                r.encoding = 'utf-8'
                s = " ".join(BeautifulSoup(r.text, "html.parser").stripped_strings)
                for mt in re.finditer(r"([A-Za-z0-9]{4,6})\s+([^\s]+)", s):
                    code = mt.group(1).strip()
                    name = mt.group(2).strip()
                    if len(name) > 1:
                        if code.isdigit() and (code.startswith("202") or code.startswith("199")) and "年" in name:
                            continue
                        found_count += 1
                        if found_count <= 5 or code == "0050":
                             samples.append(f"{code}: {name}")

            print(f"Found {found_count} items.")
            print("Samples:", samples)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_fetch()
