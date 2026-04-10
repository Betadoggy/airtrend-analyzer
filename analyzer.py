import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

class TextAnalyzer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)

    def load_corpus(self):
        """폴더 내 모든 txt 파일의 본문만 로드"""
        corpus = []
        file_list = list(self.data_dir.glob("*.txt"))
        
        if not file_list:
            print("분석할 데이터가 없습니다. 먼저 수집을 진행하세요.")
            return None

        for file_path in file_list:
            content = file_path.read_text(encoding="utf-8")
            # 구분선(---) 이후의 본문만 추출
            if "--------------------------------------------------" in content:
                body = content.split("-" * 50)[-1].strip()
                if body:
                    corpus.append(body)
        return corpus

    def build_tfidf(self, corpus):
        """TF-IDF 매트릭스 생성"""
        # stop_words='english'로 기본적인 관사, 전치사 제거
        vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
        matrix = vectorizer.fit_transform(corpus)
        
        df = pd.DataFrame(
            matrix.toarray(), 
            columns=vectorizer.get_feature_names_out()
        )
        return df