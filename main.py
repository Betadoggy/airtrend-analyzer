import os, time
from pathlib import Path
from functools import wraps
# (주의) 환경에 맞게 파일명이나 패키지명을 체크하세요. 
# 여기서는 analyzer.py에 TextAnalyzer가 있다고 가정합니다.
from crawler import NewsCrawler
from analyzer import TextAnalyzer

# 인증서 설정
CERT = r'C:\temp\somansa.cer'
if os.path.exists(CERT):
    os.environ.update({'REQUESTS_CA_BUNDLE': CERT, 'SSL_CERT_FILE': CERT})
    # Hugging Face 토큰나이저/모델 다운로드 시 SSL 경고 방지 및 보안 환경 연동
    os.environ["CURL_CA_BUNDLE"] = CERT
else:
    print(f"경고: 인증서 파일이 없습니다. 기본 CA 번들을 사용합니다: {CERT}")

# 설정값
CONFIG = {
    "API_KEY": '0ae709890d054bbba717b80b3a76c039',
    "PAGE_SIZE": 1100,
    "DATA_FOLDER": "news_data",
    "BASE_QUERY": '("Incheon Airport" OR "Incheon International Airport" OR "ICN airport")'
}

def timer(func):
    """함수 실행 시간을 측정하는 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        s = time.time()
        result = func(*args, **kwargs)
        print(f"[{func.__name__} 완료] 소요 시간: {time.time() - s:.2f}초")
        return result
    return wrapper

@timer
def run_crawling(crawler):
    print("\n>>> 인천공항 해외 뉴스 수집 중...")
    crawler.run(query=CONFIG['BASE_QUERY'], page_size=CONFIG['PAGE_SIZE'], language='en')

@timer
def run_analysis(analyzer):
    # 1. 크롤링된 데이터 로드 (도메인 무관 기사는 로드 시 1차 자동 필터링됨)
    df_raw = analyzer.load_corpus()
    if df_raw is None or df_raw.empty: 
        print("[경고] 로드된 유효 데이터가 없습니다.")
        return
    
    # 2. AI 기반 STEEP 문맥 분류 수행 (★ 규칙 기반 대치)
    df_classified = analyzer.classify_steep_with_ai(df_raw)
    
    # 3. 분류된 데이터셋 안에서 TF-IDF 키워드 추출
    tfidf_results = analyzer.build_tfidf_by_category(df_classified)
    
    # 4. 결과 출력 및 저장
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    
    # 기사별로 무슨 카테고리로 분류되었는지 매칭된 마스터 파일 저장
    df_classified.to_csv(out_dir / "steep_classified_news.csv", encoding="utf-8-sig", index=False)
    
    # 각 카테고리방에서 추출된 알짜배기 TF-IDF 키워드들을 각각 csv로 저장
    for cat, result_df in tfidf_results.items():
        safe_filename = cat.replace(" & ", "_").replace(" ", "_")
        result_df.to_csv(out_dir / f"tfidf_{safe_filename}.csv", encoding="utf-8-sig", index=False)
        print(f" 저장 완료: outputs/tfidf_{safe_filename}.csv")

def main():
    total_start = time.time()
    
    # run_crawling(NewsCrawler(api_key=CONFIG['API_KEY'], save_dir=CONFIG['DATA_FOLDER']))
    run_analysis(TextAnalyzer(data_dir=CONFIG['DATA_FOLDER']))
    
    print(f"\n>>> 전체 작업 완료 (총 {time.time() - total_start:.2f}초)")

if __name__ == "__main__":
    main()