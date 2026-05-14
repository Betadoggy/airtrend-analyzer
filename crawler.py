import requests
import os
import certifi
from pathlib import Path
from newspaper import Article
from datetime import datetime, timedelta  # 날짜 계산을 위해 추가

class NewsCrawler:
    def __init__(self, api_key, save_dir="data"):
        self.api_key = api_key
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 1. 인증서 경로 설정
        self.ca_path = r'C:\temp\somansa.cer'
        # 파일이 존재하면 해당 경로 사용, 없으면 기본 파이썬 인증서 사용
        self.verify = self.ca_path if os.path.exists(self.ca_path) else certifi.where()
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def fetch_clean_content(self, url):
        """본문 추출 시에도 인증서 적용"""
        try:
            # 아래와 같이 직접 처리하는 것입니다.
            article = Article(url, language='en')
            article.download() # 내부적으로 자체 검증을 시도함
            article.parse()
            return article.text
        except Exception:
            return None

    def run(self, query, page_size=10):
        # 2. 날짜 범위 설정 (최근 2년)
        two_years_ago = (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
        current_day = datetime.now().strftime('%Y-%m-%d')

        url = (
            f'https://newsapi.org/v2/everything?q={query}'
            f'&from={two_years_ago}'
            f'&to={current_day}'
            f'&pageSize={page_size}'
            f'&sortBy=relevancy'
            f'&apiKey={self.api_key}'
        )

        print(f">>> 기간 설정: {two_years_ago} ~ {current_day}")
        print(f">>> '{query}' 키워드로 기사 검색 중...")
        
        # 3. requests 호출 시 verify 옵션에 인증서 경로 전달
        res = requests.get(url, verify=self.verify)
        data = res.json()

        if data.get('status') != 'ok':
            print(f"오류 발생: {data.get('message')}")
            return

        articles = data.get('articles', [])
        count = 0

        for i, art in enumerate(articles):
            print(f"[{i+1}/{len(articles)}] 본문 추출 중: {art['title'][:50]}...")
            content = self.fetch_clean_content(art['url'])
            
            if content and len(content) > 100:
                file_path = self.save_dir / f"article_{query.replace(' ', '_')}_{i+1}.txt"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"Title: {art['title']}\nSource: {art['source']['name']}\nURL: {art['url']}\n{'-'*50}\n\n{content}")
                count += 1
            else:
                print(f"   (건너뜀) 본문 추출 실패 또는 내용 부족")

        print(f"\n[수집 완료] 총 {count}개 저장됨.")