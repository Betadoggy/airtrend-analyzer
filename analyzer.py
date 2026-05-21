import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords

# NLTK 데이터 다운로드
required_resources = [
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
    ('corpora/stopwords', 'stopwords')
]

for res_path, res_name in required_resources:
    try:
        nltk.data.find(res_path)
    except LookupError:
        print(f"NLTK 리소스 다운로드 중: {res_name}")
        nltk.download(res_name)

# 영어 불용어만 정의
COMBINED_STOPWORDS = set(nltk_stopwords.words('english'))

# 뉴스 및 공항 특화 불용어 강제 추가
extra_stopwords = {
    'according', 'air', 'aircraft', 'airline', 'airlines', 'airplane', 
    'airport', 'airports', 'airspace', 'also', 'aviation', 'could', 
    'first', 'flight', 'flights', 'last', 'like', 'new', 
    'one', 'reported', 'said', 'says', 'since', 'told', 
    'two', 'use', 'would', 'year', 'years'
}
COMBINED_STOPWORDS.update(extra_stopwords)

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
            else:
                # 구분선이 없으면 전체 텍스트에서 메타데이터(Title/Source/URL) 이후를 사용
                lines = content.splitlines()
                body_lines = []
                found_divider = False
                for line in lines:
                    if found_divider:
                        body_lines.append(line)
                    elif line.strip() == "":
                        found_divider = True
                body = "\n".join(body_lines).strip()

            if body:
                corpus.append(body)

        return corpus

    def extract_nouns(self, text):
        """텍스트에서 단어를 토큰화하고 불용어를 제거"""
        try:
            tokens = word_tokenize(text.lower())
            filtered_tokens = [
                token for token in tokens
                if token.isalpha() and token not in COMBINED_STOPWORDS and len(token) > 2
            ]
            return ' '.join(filtered_tokens)
        except Exception as e:
            print(f"토큰 추출 오류: {e}")
            return ""

    def build_tfidf(self, corpus):
        """TF-IDF 매트릭스 생성 (명사 기반)"""
        # 각 문서에서 명사만 추출
        noun_corpus = [self.extract_nouns(doc) for doc in corpus]
        noun_corpus = [doc for doc in noun_corpus if doc.strip()]

        if not noun_corpus:
            print("TF-IDF를 생성할 유효한 텍스트가 없습니다. 문서에 명사 또는 분석 가능한 내용이 포함되어 있는지 확인하세요.")
            return pd.DataFrame()

        # 1. 커스텀 분석기 내부에서 불용어(COMBINED_STOPWORDS)를 직접 필터링하도록 수정
        def noun_analyzer(text):
            tokens = text.split()
            return [t for t in tokens if t and len(t) > 2 and t not in COMBINED_STOPWORDS]
        
        # 2. 무시되던 stop_words 인자를 제거하여 경고 문구 해결
        vectorizer = TfidfVectorizer(
            analyzer=noun_analyzer,
            min_df=1,  # 최소 1개 문서에서 나타나야 함
            max_df=1.0  # (참고) 100% 문서를 뜻하려면 정수 1 대신 실수 1.0을 쓰는 것이 안전합니다.
        )
        try:
            matrix = vectorizer.fit_transform(noun_corpus)
        except ValueError as e:
            if 'empty vocabulary' in str(e):
                print("TF-IDF 생성 중 빈 어휘집 오류가 발생했습니다. 분석 대상 문서에 충분한 명사/토큰이 포함되어 있는지 확인하세요.")
                return pd.DataFrame()
            raise

        # 전체 피처에 대해 모든 문서의 TF-IDF 합을 계산하고 상위 50개 선택
        feature_names = vectorizer.get_feature_names_out()
        if len(feature_names) == 0:
            print("TF-IDF로 추출된 특성이 없습니다. 더 많은 텍스트 또는 다른 전처리 설정을 확인하세요.")
            return pd.DataFrame()
            
        col_sums = np.asarray(matrix.sum(axis=0)).ravel()
        top_k = min(50, len(feature_names))
        top_idx = np.argsort(col_sums)[::-1][:top_k]
        top_features = feature_names[top_idx]

        # [메모리 최적화]: matrix 전체를 덤프하지 않고, 필요한 열(top_idx)만 먼저 슬라이싱한 뒤 배열로 변환합니다.
        dense_top_matrix = matrix[:, top_idx].toarray()

        # 상위 피처들만 컬럼으로 사용하여 데이터프레임 생성
        df = pd.DataFrame(
            dense_top_matrix,
            columns=top_features
        )
        return df