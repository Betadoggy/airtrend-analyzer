import os
from pathlib import Path

# 모든 HTTP 통신 라이브러리가 이 인증서를 참조하도록 강제 설정
CERT_PATH = r'C:\temp\somansa.cer'
os.environ['REQUESTS_CA_BUNDLE'] = CERT_PATH
os.environ['SSL_CERT_FILE'] = CERT_PATH

from crawler import NewsCrawler
from analyzer import TextAnalyzer

# 설정 값
API_KEY = '05f7ae68fafc47a0a45ade59ea38edd9' # 본인의 API Key
SEARCH_QUERY = 'airport'
DATA_FOLDER = "news_data"

def main():
    # 1. 수집 단계
    crawler = NewsCrawler(api_key=API_KEY, save_dir=DATA_FOLDER)
    crawler.run(query=SEARCH_QUERY, page_size=5)

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