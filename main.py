import os, time
from pathlib import Path
from functools import wraps
from crawler import NewsCrawler
from analyzer import TextAnalyzer

# 인증서 설정
CERT = r'C:\temp\somansa.cer'
os.environ.update({'REQUESTS_CA_BUNDLE': CERT, 'SSL_CERT_FILE': CERT})

# 설정값
CONFIG = {
    "API_KEY": '0ae709890d054bbba717b80b3a76c039',
    "PAGE_SIZE": 100,
    "DATA_FOLDER": "news_data",
    "BASE_QUERY": "(airport OR aviation OR vertiport OR airline)",
    "TOPICS": {
        "environment": '("carbon neutral" OR "net zero" OR SAF OR hydrogen)',
        "technology": '(AI OR Robotics OR "Digital twin" OR Drone OR UAM OR AR OR VR)',
        "war": '("Russia-Ukraine war" OR "Iran war")'
    }
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
def run_crawling(crawler, topics):
    for name, query in topics.items():
        print(f"\n>>> [그룹: {name}] 수집 중...")
        crawler.run(query=f"{CONFIG['BASE_QUERY']} AND {query}", page_size=CONFIG['PAGE_SIZE'])

@timer
def run_analysis(analyzer):
    corpus = analyzer.load_corpus()
    if not corpus: return
    
    df = analyzer.build_tfidf(corpus).T
    out_path = Path("outputs/analysis_result.csv")
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, encoding="utf-8-sig")
    print(f"저장 완료: {out_path}")

def main():
    total_start = time.time()
    
    run_crawling(NewsCrawler(api_key=CONFIG['API_KEY'], save_dir=CONFIG['DATA_FOLDER']), CONFIG['TOPICS'])
    run_analysis(TextAnalyzer(data_dir=CONFIG['DATA_FOLDER']))
    
    print(f"\n>>> 전체 작업 완료 (총 {time.time() - total_start:.2f}초)")

if __name__ == "__main__":
    main()