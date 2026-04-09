import requests, os, time, certifi, urllib3
from bs4 import BeautifulSoup
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 1. 환경 설정 및 세션 초기화
CA_PATH = r'C:\temp\somansa.cer'
API_KEY = '05f7ae68fafc47a0a45ade59ea38edd9'
QUERY, PAGE_SIZE = 'airport', 5
SAVE_DIR = Path(__file__).resolve().parent / "news_results"

def get_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.verify = CA_PATH if os.path.exists(CA_PATH) else certifi.where()
    s.headers.update({'User-Agent': 'Mozilla/5.0...'})
    return s

session = get_session()

def fetch_content(url):
    try:
        res = session.get(url, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            return "\n".join([p.text for p in soup.find_all('p') if len(p.text) > 30])
    except: pass
    return "본문을 불러올 수 없습니다."

# 2. 기사 검색 및 수집
url = f'https://newsapi.org/v2/everything?q={QUERY}&language=en&pageSize={PAGE_SIZE}&sortBy=relevancy&apiKey={API_KEY}'
print(f"'{QUERY}' 검색 중...")

try:
    data = session.get(url).json()
    if data.get('status') != 'ok': raise Exception(data.get('message'))
    
    SAVE_DIR.mkdir(exist_ok=True)
    for i, art in enumerate(data.get('articles', [])):
        print(f"[{i+1}] 수집: {art['title'][:30]}...")
        content = fetch_content(art['url'])
        
        with open(SAVE_DIR / f"article_{i+1}.txt", "w", encoding="utf-8") as f:
            f.write(f"Title: {art['title']}\nSource: {art['source']['name']}\nURL: {art['url']}\n{'-'*50}\n\n{content}")
        time.sleep(1)
    print(f"\n[완료] {len(data.get('articles', []))}개 저장됨: {SAVE_DIR}")

except Exception as e:
    print(f"오류 발생: {e}")