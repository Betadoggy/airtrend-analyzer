import requests
import os
import certifi
import time
from pathlib import Path
from newspaper import Article
from datetime import datetime, timedelta

class NewsCrawler:
    def __init__(self, api_key, save_dir="data"):
        self.api_key = api_key
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.ca_path = r'C:\temp\somansa.cer'
        self.verify = self.ca_path if os.path.exists(self.ca_path) else certifi.where()
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def fetch_clean_content(self, url):
        """본문 추출 (성공 시에만 텍스트 반환)"""
        try:
            article = Article(url, language='en')
            article.download()
            article.parse()
            return article.text
        except Exception:
            return None

    def run(self, query, page_size=1000):
        # 1. 날짜 설정 (최근 2년)
        two_years_ago = (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
        
        total_collected = 0
        page = 1
        # NewsAPI는 유료 플랜이라도 통상 1,000~2,000번 이후 페이지는 제한될 수 있으므로 안전장치 설정
        max_pages = 100 

        print(f"\n>>> '{query}' 검색 시작 (목표: {page_size}개)")

        while total_collected < page_size and page <= max_pages:
            url = (
                f'https://newsapi.org/v2/everything?q={query}'
                f'&from={two_years_ago}'
                f'&pageSize=100'  # 한 페이지당 최대 개수
                f'&page={page}'
                f'&sortBy=relevancy'
                f'&apiKey={self.api_key}'
            )

            print(f"  [Request] API Page {page} 호출 중...")
            res = requests.get(url, verify=self.verify)
            data = res.json()

            if data.get('status') != 'ok':
                print(f"  [Error] {data.get('message')}")
                break

            articles = data.get('articles', [])
            if not articles:
                print("  - 더 이상 검색 결과가 없습니다.")
                break

            for art in articles:
                if total_collected >= page_size:
                    break

                # 실제 본문 수집 시도
                content = self.fetch_clean_content(art['url'])

                # 유효한 본문이 있는 경우에만 파일 저장 및 카운트 증가
                if content and len(content.strip()) > 200:
                    total_collected += 1
                    
                    # 파일명 특수문자 제거
                    safe_query = "".join([c for c in query if c.isalnum() or c in (' ', '_')]).replace(' ', '_')
                    file_path = self.save_dir / f"article_{safe_query}_{total_collected}.txt"

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(f"Title: {art['title']}\n")
                        f.write(f"Source: {art['source']['name']}\n")
                        f.write(f"URL: {art['url']}\n")
                        f.write(f"{'-'*50}\n\n")
                        f.write(content)
                    
                    # 수집 진행률 표시
                    if total_collected % 10 == 0:
                        print(f"    > 현재 {total_collected}/{page_size} 완료...")
                else:
                    # 추출 실패 시 카운트를 올리지 않고 다음 기사로 진행
                    continue

            # 한 페이지를 다 검사했는데 아직 모자라면 다음 페이지로
            page += 1
            time.sleep(0.1) # 짧은 지연시간

        print(f">>> [최종 완료] '{query}' 그룹: 총 {total_collected}개 저장됨.")