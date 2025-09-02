# ✅ 修正 GROQ RateLimitError 與錯誤處理的完整版本
import os
import openai
from groq import Groq
import requests
import time
import random
from bs4 import BeautifulSoup
import json

# 設定 API 金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 設定 groq API 的速率限制參數
GROQ_RATE_LIMIT = 6000
GROQ_REQUEST_INTERVAL = 60

groq_tokens_used = 0
groq_last_request_time = time.time()

def get_reply(messages):
    global groq_tokens_used, groq_last_request_time

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=messages
        )
        reply = response["choices"][0]["message"]["content"]

    except openai.OpenAIError as openai_err:
        try:
            request_tokens = sum(len(message['content']) for message in messages)
            current_time = time.time()

            if groq_tokens_used + request_tokens > GROQ_RATE_LIMIT:
                wait_time = GROQ_REQUEST_INTERVAL - (current_time - groq_last_request_time)
                if wait_time > 0:
                    print(f"超過速率限制，等待 {wait_time:.2f} 秒...")
                    time.sleep(wait_time)

                groq_tokens_used = 0
                groq_last_request_time = time.time()

            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=1500,
                temperature=1.2
            )
            reply = response.choices[0].message.content
            groq_tokens_used += request_tokens

        except Exception as groq_err:
            # ✅ 用字串偵測 RateLimit（429）或 fallback 錯誤訊息
            if "429" in str(groq_err) or "rate limit" in str(groq_err).lower():
                print("GROQ API 速率限制，等待 15 秒再試...")
                time.sleep(15)
                try:
                    response = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages,
                        max_tokens=1500,
                        temperature=1.2
                    )
                    reply = response.choices[0].message.content
                except Exception as second_err:
                    reply = f"重試 GROQ 仍失敗: {str(second_err)}"
            else:
                reply = f"OpenAI API 發生錯誤: {str(openai_err)}，GROQ API 發生錯誤: {str(groq_err)}"

    return reply

class Job104Spider:
    def search(self, keyword, max_num=10, filter_params=None, sort_type='符合度', is_sort_asc=False):
        jobs = []
        total_count = 0
        url = 'https://www.104.com.tw/jobs/search/list'
        query = f'ro=0&kwop=7&keyword={keyword}&expansionType=area,spec,com,job,wf,wktm&mode=s&jobsource=2018indexpoc'
        if filter_params:
            query += ''.join([f'&{key}={value}' for key, value in filter_params.items()])

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.92 Safari/537.36',
            'Referer': 'https://www.104.com.tw/jobs/search/',
        }

        sort_dict = {
            '符合度': '1',
            '日期': '2',
            '經歷': '3',
            '學歷': '4',
            '應徵人數': '7',
            '待遇': '13',
        }
        sort_params = f"&order={sort_dict.get(sort_type, '1')}"
        sort_params += '&asc=1' if is_sort_asc else '&asc=0'
        query += sort_params

        page = 1
        while len(jobs) < max_num:
            params = f'{query}&page={page}'
            r = requests.get(url, params=params, headers=headers)
            if r.status_code != requests.codes.ok:
                print('請求失敗', r.status_code)
                data = r.json()
                print(data['status'], data['statusMsg'], data['errorMsg'])
                break

            data = r.json()
            total_count = data['data']['totalCount']
            jobs.extend(data['data']['list'])

            if (page == data['data']['totalPage']) or (data['data']['totalPage'] == 0):
                break
            page += 1
            time.sleep(random.uniform(3, 5))

        return total_count, jobs[:max_num]

    def get_job(self, job_id):
        url = f'https://www.104.com.tw/job/ajax/content/{job_id}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.92 Safari/537.36',
            'Referer': f'https://www.104.com.tw/job/{job_id}'
        }
        r = requests.get(url, headers=headers)
        if r.status_code != requests.codes.ok:
            print('請求失敗', r.status_code)
            return
        return r.json()['data']

    def search_job_transform(self, job_data):
        appear_date = job_data['appearDate']
        apply_num = int(job_data['applyCnt'])
        company_addr = f"{job_data['jobAddrNoDesc']} {job_data['jobAddress']}"
        job_url = f"https:{job_data['link']['job']}"
        job_company_url = f"https:{job_data['link']['cust']}"
        job_analyze_url = f"https:{job_data['link']['applyAnalyze']}"
        job_id = job_url.split('/job/')[-1].split('?')[0]
        salary_high = int(job_data['salaryLow'])
        salary_low = int(job_data['salaryHigh'])

        return {
            'job_id': job_id,
            'type': job_data['jobType'],
            'name': job_data['jobName'],
            'appear_date': appear_date,
            'apply_num': apply_num,
            'apply_text': job_data['applyDesc'],
            'company_name': job_data['custName'],
            'company_addr': company_addr,
            'job_url': job_url,
            'job_analyze_url': job_analyze_url,
            'job_company_url': job_company_url,
            'lon': job_data['lon'],
            'lat': job_data['lat'],
            'education': job_data['optionEdu'],
            'period': job_data['periodDesc'],
            'salary': job_data['salaryDesc'],
            'salary_high': salary_high,
            'salary_low': salary_low,
            'tags': job_data['tags'],
        }

def generate_content_msg(job_name):
    job104_spider = Job104Spider()
    filter_params = {'area': '6001001000,6001005000'}
    if job_name == "":
        job_name = "iOS"

    total_count, jobs = job104_spider.search(job_name, max_num=3, filter_params=filter_params)
    content_msg = f'你現在是一位專業的求職分析師, 使用以下數據來撰寫分析報告:\n{jobs}\n'
    content_msg += f'薪水最高的公司: {{公司名}}{{代號}} {{日期}}{{薪水}}, 薪水最低的公司: {{公司名}}{{代號}} {{日期}}{{薪水}}。\n'
    content_msg += '請給出完整的趨勢分析報告，顯示前1-10筆公司或職缺報告，以下是範例格式:\n'
    content_msg += """1. **CRD 手機遊戲工程師【歡迎應屆畢業生】** 
    - 104職缺代碼: xxxx
    - 公司名稱: 神來也麻將_慧邦科技股份有限公司
    - 工作地點: 台北市中正區
    - 應徵人數描述: {{job['apply_text']}}
    - 薪資待遇: 待遇面議
    - 詳細頁連結:http://xxx.xxx.xx"""
    content_msg += '，必使用台灣繁體中文。非簡中'
    return content_msg

def one04_gpt(job_name):
    content_msg = generate_content_msg(job_name)
    msg = [
        {"role": "system", "content": "你現在是一位專業的職缺分析師, 使用以下數據來撰寫分析報告。(回答時 length < 2000 tokens)"},
        {"role": "user", "content": content_msg}
    ]
    return get_reply(msg)
