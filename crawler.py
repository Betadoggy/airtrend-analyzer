import requests
import os
import certifi
import time
import re
import html as _html
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
        # 1) 우선 newspaper로 시도
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text or ''
            if text and len(text.strip()) > 200:
                return text
        except Exception as e:
            print(f"    [Article Error] {e}")

        # 2) 대체: HTML에서 <p> 태그를 모아 텍스트 구성
        try:
            res = requests.get(url, headers=self.headers, verify=self.verify, timeout=20)
            res.raise_for_status()
            html_text = res.text

            # 간단한 p 태그 추출(완벽하진 않음)
            paras = re.findall(r'<p[^>]*>(.*?)</p>', html_text, flags=re.S | re.I)
            cleaned = []
            for p in paras:
                # 태그 제거
                p_text = re.sub(r'<[^>]+>', '', p)
                p_text = _html.unescape(p_text).strip()
                if len(p_text) > 50:
                    cleaned.append(p_text)

            text2 = '\n\n'.join(cleaned)
            if text2 and len(text2.strip()) > 200:
                print(f"    [HTML Fallback] URL={url} extracted {len(text2)} chars")
                return text2
        except Exception as e:
            print(f"    [Fallback Error] {e}")

        return None

    def is_truncated_preview(self, text):
        if not isinstance(text, str):
            return False
        return '[+' in text and 'chars]' in text

    def run(self, query, page_size=1000, language='en'):
        # 날짜 설정 (최근 3년)
        two_years_ago = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')
        
        total_collected = 0
        page = 1
        max_pages = 30 

        print(f"\n>>> '{query}' 검색 시작 (목표: {page_size}개, 언어: {language})")

        while total_collected < page_size and page <= max_pages:
            params = {
                'q': query,
                'from': two_years_ago,
                'pageSize': 100,
                'page': page,
                'sortBy': 'relevancy',
                'language': language,
                'apiKey': self.api_key,
            }

            print(f"  [Request] API Page {page} 호출 중... query={query}")
            try:
                res = requests.get('https://newsapi.org/v2/everything', params=params, verify=self.verify, headers=self.headers, timeout=30)
                res.raise_for_status()
            except requests.RequestException as exc:
                print(f"  [Request Error] {exc}")
                break

            try:
                data = res.json()
            except ValueError:
                print("  [Error] JSON 파싱 실패")
                break

            if data.get('status') != 'ok':
                print(f"  [Error] {data.get('message')} (code={data.get('code')})")
                break

            articles = data.get('articles', [])
            total_results = data.get('totalResults', 'unknown')
            print(f"  [Response] status=ok, totalResults={total_results}, articles={len(articles)}")

            if not articles:
                print("  - 더 이상 검색 결과가 없습니다.")
                break

            for art in articles:
                if total_collected >= page_size:
                    break

                content = self.fetch_clean_content(art['url'])
                if not content or len(content.strip()) <= 200:
                    api_fallback = art.get('content') or art.get('description') or ''
                    if api_fallback and not self.is_truncated_preview(api_fallback):
                        content = api_fallback
                        print(f"    [Fallback] API content 사용: URL={art.get('url')}")
                    elif api_fallback:
                        print(f"    [Truncated fallback] URL={art.get('url')} skipped API preview")

                if content and len(content.strip()) > 200:
                    total_collected += 1
                    safe_query = "".join([c for c in query if c.isalnum() or c in (' ', '_')]).replace(' ', '_')
                    file_path = self.save_dir / f"article_{safe_query}_{total_collected}.txt"

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(f"Title: {art['title']}\n")
                        f.write(f"Source: {art['source']['name']}\n")
                        f.write(f"URL: {art['url']}\n")
                        f.write(f"{'-'*50}\n\n")
                        f.write(content)

                    if total_collected % 10 == 0:
                        print(f"    > 현재 {total_collected}/{page_size} 완료...")
                else:
                    print(f"    [Skip] URL={art.get('url')} title={art.get('title')} (본문 없음 또는 너무 짧음)")
                    continue

            page += 1
            time.sleep(0.1)

        print(f">>> [최종 완료] '{query}' 그룹: 총 {total_collected}개 저장됨.")