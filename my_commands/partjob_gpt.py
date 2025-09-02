import os
import openai
from groq import Groq
import requests
import time
import random
from bs4 import BeautifulSoup
import json
"""
```
{
  "@context": "http://schema.org",
  "@type": "JobPosting",
  "baseSalary": {
    "@type": "MonetaryAmount",
    "currency": "TWD",
    "value": {
      "@type": "QuantitativeValue",
      "value": 195,
      "unitText": "HOUR"
    }
  },
  "description": "✨可週領7000/可週領7000/可週領7000\r\n✨早班195-205/中班A203-213/中班B208-218/晚班219-229/大夜班233-243\r\n✨六、日加班一小時多四十元\r\n✨加好友詢問較快：【@free-jobs】或點擊複製：https://lin.ee/raMJn7J\r\n☎ＬＩＮＥ通話：https://lin.ee/ltcTyd5\r\n①工作地點：楊梅幼獅工業區(近埔心牧場)\r\n②工作內容：撿貨、理貨\r\n③工作制度：8H(日班＆午班A＆午班B＆晚班＆大夜班)\r\n④休假制度：週排休二日\r\n⑤可週領\r\n⑥上班時段+薪資說明：【各班制要視各倉缺額】\r\n✨班制\t\r\n日班-->08:00~17:00；09:00~18:00 //34,320~36,080【時薪195-205】\r\n午班A-->13:30~22:30  //35,728~37,488【時薪203-213】\r\n午班B-->15:00~24:00  //36,608~38,368【時薪208-218】\r\n晚班-->20:00~05:00  //38,544~40,304【時薪219-229】\r\n大夜班-->00:00~08:00  //41,008~42,768【時薪233-243】\r\n✨出勤24天(12H計)-->5萬9~6萬2以上\t\r\n6萬1~6萬5以上\t\r\n6萬2~6萬8以上\r\n6萬9~7萬3以上\t\r\n7萬5多~7萬9以上\r\n▬▬▬▬▬▬▬▬▬▬▬▬▬【應徵方式】▬▬▬▬▬▬▬▬▬▬▬▬▬\r\n1.點這加好友：https://lin.ee/raMJn7J 【可詢問薪資和工作環境】\r\n2.LINE通話：https://lin.ee/ltcTyd5　【和李先生通話】\r\n2.來電：0977-440-310 李先生\r\n3.加 LINE好友【@free-jobs】免費找工作\r\n4.fb搜尋【sam li】每日更新職缺唷!!\r\n5.不喜歡嗎?更多職缺請上：https://lin.ee/raMJn7J或加 LINE好友【@free-jobs】幫您免費找工作或來電☎：https://lin.ee/ltcTyd5　詢問更多相關問題\r\n***所有諮詢、介紹服務皆免收費，讓您安心找工作***\r\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\r\n",
  "hiringOrganization": {
    "@type": "Organization",
    "name": "天泰人力銀行",
    "url": "http://worknowapp.com/companies/6f012d11-8fef-4f3b-9227-4e8b4e062173",
    "description": "天泰人力銀行2011年創立，由各職階人才需求供應出發，到人才評鑑、培育及推薦及勞資顧問服務。始終以專業化、制度化之信念，提供各產業人力資源管理之服務，並建立一套健全的管理制度，致力於整合企業界於人力資源方面需求上，所必要的整體性服務，期以專業人力資源管理的角度，減少企業的人事成本外，更要創造人力資本。\r\n除了精心選擇派遣企業與其職缺之外，天泰人力銀行更努力於保障職工安全，依照契約保障聘僱與任用、要求合法、安全的工作規則與環境。 \r\n長期來深受要派企業與派遣員工的一致肯定與信賴。未來天泰人力銀行將透過專業、細緻與效率的服務，讓各企業感到天泰人力銀行不僅是人力派遣、人才推薦，而是專業的人資管理顧問，更是值得仰賴的策略伙伴，期待有您的加入與天泰人力銀行一起在努力，成就個人，創造公司價值。",
    "sameAs": ""
  },
  "datePosted": "2024-06-20T10:34:07+08:00",
  "validThrough": "2024-06-27T23:59:59+08:00",
  "employmentType": "PART_TIME",
  "industry": "一般",
  "jobLocation": {
    "@type": "Place",
    "address": {
      "@type": "PostalAddress",
      "addressLocality": "桃園",
      "addressRegion": "楊梅區",
      "postalCode": "000",
      "streetAddress": "326桃園市楊梅區梅獅路"
    },
    "geo": {
      "@type": "GeoCoordinates",
      "latitude": "24.919555",
      "longitude": "121.180387"
    }
  },
  "salaryCurrency": "TWD",
  "title": "DC11-楊梅幼獅✨週領7千/網購物流撿貨員",
  "url": "http://worknowapp.com/jobs/a4bb4cb6-ac0b-45c2-82f5-2e403f3291b4",
  "name": "DC11-楊梅幼獅✨週領7千/網購物流撿貨員"
}
```
"""
# 設定 API 金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 建立 GPT 模型
def get_reply(messages):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=messages
        )
        reply = response["choices"][0]["message"]["content"]
    except openai.OpenAIError as openai_err:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=5000,
                temperature=1.2
            )
            reply = response.choices[0].message.content
        except Exception as groq_err:
            time.sleep(15)  # 等待一段時間再重試
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=3000,
                temperature=1.2
            )
            reply = response.choices[0].message.content
        except Groq.GroqError as groq_err:
            reply = f"OpenAI API 發生錯誤: {openai_err.error.message}，GROQ API 發生錯誤: {groq_err.message}"
    return reply

class PartJobSpider:
    def search(self, keyword, max_num=10):
        jobs = []
        total_count = 0

        start_page = 1
        page = start_page
        while len(jobs) < max_num:
            url = f"https://worknowapp.com/regions/%E6%A1%83%E5%9C%92?q={keyword}&page={page}"
            response = requests.get(url)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.content, 'html.parser')
            job_listings = soup.find_all('li', class_='job-item')

            if not job_listings:
                break

            for job in job_listings:
                script = job.find('script', type='application/ld+json')
                if script:
                    job_data = json.loads(script.string)
                    title = job_data.get('title', '')
                    company = job_data.get('hiringOrganization', {}).get('name', '')
                    category = job.find('span', class_='label-part-time-type').text.strip() if job.find('span', class_='label-part-time-type') else ''
                    salary = job_data.get('baseSalary', {}).get('value', {}).get('value', '')
                    unitText = job_data.get('baseSalary', {}).get('value', {}).get('unitText', '')
                    description = job_data.get('description', '')
                    location = job_data.get('jobLocation', {}).get('address', {}).get('addressRegion', '')
                    street_address = job_data.get('jobLocation', {}).get('address', {}).get('streetAddress', '')
                    posted = job.find('time').text.strip() if job.find('time') else ''
                    link = job_data.get('url', '')

                    jobs.append({
                        '職缺標題': title,
                        '公司名稱': company,
                        '分類': category,
                        '薪資': salary,
                        '薪水單位': unitText,
                        '地點': location,
                        '街道地址': street_address,
                        '發布時間': posted,
                        '連結': link,
                        '說明欄': description
                    })

            page += 1
            time.sleep(random.uniform(3, 5))

        total_count = len(jobs)
        return total_count, jobs[:max_num]

def generate_content_msg(keyword):
    partjob_spider = PartJobSpider()

    if keyword == "":
        keyword = "桃園"
        
    find_max_num = 10
    total_count, jobs = partjob_spider.search(keyword, max_num=find_max_num)

    print('搜尋結果職缺總數：', total_count)
    print(f"找到的前的工作 {jobs}。")

    content_msg = f'你現在是一位專業的求職分析師, 使用以下數據來撰寫分析報告:\n'
    content_msg += f'{jobs}\n'
    content_msg += f'薪水最高的公司: {{公司名}}{{代號}} {{日期}}{{薪水}}, 薪水最低的公司: {{公司名}}{{代號}} {{日期}}{{薪水}}。\n'
    content_msg += f'請給出完整的趨勢分析報告，顯示前1-{find_max_num}筆公司或職缺報告\n'
    content_msg += '以下是範例格式:\n'
    content_msg += """1. **打零工名稱** 
    - 公司名稱: 公司名稱 (長/短期兼職)
    - 分類: {emojia:以分類作適合的標示}(分類)
    - 分區: 桃園區 ／ 工作地點: 街道地址
    - 薪資待遇: 薪資待遇 / ({{薪水單位}})
    - 工作時間: 星期(1-5), 時間
    - 領錢周期： 日領
    - 電話:{{電話}} {{名子}}}/ LINE: {{LINEID}}
    - 詳細頁連結
    """
    content_msg += f'至少{find_max_num}筆文長不簡略，必使用台灣繁體中文。非簡中'

    return content_msg

def partjob_gpt(keyword):
    content_msg = generate_content_msg(keyword)
    print(content_msg) 

    msg = [{
        "role": "system",
        "content": f"你現在是一位專業的職缺分析師, 使用以下數據來撰寫分析報告。(回答時 length < 16385 tokens)"
    }, {
        "role": "user",
        "content": content_msg
    }]

    reply_data = get_reply(msg)
    return reply_data

