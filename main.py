import os
from pathlib import Path
from crawler import NewsCrawler
from analyzer import TextAnalyzer

# 모든 HTTP 통신 라이브러리가 이 인증서를 참조하도록 강제 설정
CERT_PATH = r'C:\temp\somansa.cer'
os.environ['REQUESTS_CA_BUNDLE'] = CERT_PATH
os.environ['SSL_CERT_FILE'] = CERT_PATH

# 설정 값
API_KEY = '0ae709890d054bbba717b80b3a76c039' # 본인의 API Key
PAGE_SIZE = 5
DATA_FOLDER = "news_data"

# 쿼리 그룹 설정
BASE_QUERY = "(airport OR aviation OR vertiport OR airline)"

TOPIC_GROUPS = {
    "environment": '("carbon neutral" OR "net zero" OR SAF OR hydrogen)',
    "technology": '(AI OR Robotics OR "Digital twin" OR Drone OR UAM OR AR OR VR)',
    "war": '("Russia-Ukraine war" OR "Iran war")'
}

def main():
    # 1. 수집 단계
    crawler = NewsCrawler(api_key=API_KEY, save_dir=DATA_FOLDER)
    
    print(f">>> 그룹별 뉴스 수집 시작 (기본 키워드: {BASE_QUERY})")
    
    # TOPIC_GROUPS를 순회하며 검색 수행
    for topic_name, topic_query in TOPIC_GROUPS.items():
        # 기본 쿼리와 토픽 쿼리를 결합
        combined_query = f"{BASE_QUERY} AND {topic_query}"
        
        print(f"\n[그룹: {topic_name}] 실행 중...")
        print(f"쿼리: {combined_query}")
        
        # 검색 실행 (결과는 DATA_FOLDER에 파일로 저장됨)
        crawler.run(query=combined_query, page_size=PAGE_SIZE)

    print("\n" + "="*30)
    
    # 2. 분석 단계
    analyzer = TextAnalyzer(data_dir=DATA_FOLDER)
    corpus = analyzer.load_corpus()

    if corpus:
        print(">>> TF-IDF 분석 결과 (상위 키워드 가중치):")
        df_result = analyzer.build_tfidf(corpus)
        
        # 전치(T)해서 보기 좋게 출력
        print(df_result.T)
        
        # outputs 폴더에 CSV 저장 로직
        OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
        OUTPUT_DIR.mkdir(exist_ok=True)  # 폴더가 없으면 생성
        
        save_path = OUTPUT_DIR / "analysis_result.csv"
        
        # 행(index)은 단어, 열(column)은 문서 번호인 상태로 저장 (전치 행렬 .T 사용 추천)
        df_result.T.to_csv(save_path, encoding="utf-8-sig")
        
        print("-" * 50)
        print(f"[저장 완료] 분석 결과가 다음 경로에 저장되었습니다: {save_path}")

if __name__ == "__main__":
    main()