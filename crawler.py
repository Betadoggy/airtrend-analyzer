import requests
import os
import certifi
import time
import re
import html as _html
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

class NewsCrawler:
    def __init__(self, api_key, save_dir="data"):
        self.api_key = api_key
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.ca_path = r'C:\temp\somansa.cer'
        self.verify = self.ca_path if os.path.exists(self.ca_path) else certifi.where()
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def contains_korean(self, text, min_chars=30, min_ratio=0.1):
        if not isinstance(text, str):
            return False
        korean_chars = re.findall(r'[\uac00-\ud7a3]', text)
        if len(korean_chars) < min_chars:
            return False
        return len(korean_chars) / max(1, len(text)) >= min_ratio

    def _get_clean_text(self, element):
        for bad in element(['script', 'style', 'noscript', 'header', 'footer', 'nav', 'aside', 'form']):
            bad.decompose()
        paragraphs = []
        for p in element.find_all('p'):
            p_text = p.get_text(separator=' ', strip=True)
            if len(p_text) > 50:
                paragraphs.append(p_text)
        return '\n\n'.join(paragraphs)

    def _extract_body_text(self, soup):
        if soup is None:
            return None

        selectors = [
            'article',
            'main',
            'div[id*="content"]',
            'div[class*="content"]',
            'div[class*="article"]',
            'div[class*="post"]',
            'div[class*="story"]',
            'section'
        ]

        for selector in selectors:
            for candidate in soup.select(selector):
                text = self._get_clean_text(candidate)
                if text and len(text.strip()) > 200 and self.contains_korean(text):
                    return text

        body = soup.body or soup
        text = self._get_clean_text(body)
        return text if text and len(text.strip()) > 200 and self.contains_korean(text) else None

    def is_truncated_preview(self, text):
        if not isinstance(text, str):
            return False
        return '[+' in text and 'chars]' in text

    def fetch_clean_content(self, url):
        """본문 추출 (성공 시에만 텍스트 반환)"""
        try:
            res = requests.get(url, headers=self.headers, verify=self.verify, timeout=20)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'lxml')
            text2 = self._extract_body_text(soup)
            if text2:
                print(f"    [HTML Extracted] URL={url} extracted {len(text2)} chars")
                return text2
        except Exception as e:
            print(f"    [HTML Extract Error] {e}")

        return None


    def run(self, query, page_size=1000):
        # 1. 날짜 설정 (최근 2년)
        two_years_ago = (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
        
        total_collected = 0
        page = 1
        # NewsAPI는 최대 10,000개 결과까지만 검색 가능 (page * pageSize <= 10,000)
        # PAGE_SIZE=100일 때: max_pages=30 → 3,000개까지 가능
        max_pages = 30 

        print(f"\n>>> '{query}' 검색 시작 (목표: {page_size}개)")

        while total_collected < page_size and page <= max_pages:
            params = {
                'q': query,
                'from': two_years_ago,
                'pageSize': 100,
                'page': page,
                'sortBy': 'relevancy',
                'language': 'ko',
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

                if content and len(content.strip()) > 200 and self.contains_korean(content):
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
                    print(f"    [Skip] URL={art.get('url')} title={art.get('title')} (한국어 본문 없음 또는 너무 짧음)")
                    continue

            page += 1
            time.sleep(0.1)

        print(f">>> [최종 완료] '{query}' 그룹: 총 {total_collected}개 저장됨.")
