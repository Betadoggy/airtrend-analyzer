import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# 품사 태깅에 필요한 리소스까지 다운로드
nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng', 'stopwords'], quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk import pos_tag  # 품사 태깅을 위해 추가

COMBINED_STOPWORDS = set(nltk_stopwords.words('english')).union({
    'air', 'aircraft', 'airline', 'airlines', 'airplane', 
    'airport', 'airports', 'airspace', 'aviation',
    'flight', 'flights', 'two', 'use', 'year', 'years'
})

class TextAnalyzer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)

    def load_corpus(self):
        corpus = []
        for file_path in self.data_dir.glob("*.txt"):
            content = file_path.read_text(encoding="utf-8")
            if "--------------------------------------------------" in content:
                body = content.split("-" * 50)[-1].strip()
            else:
                parts = content.split('\n\n', 1)
                body = parts[1].strip() if len(parts) > 1 else content.strip()
            if body: corpus.append(body)
        return corpus if corpus else None

    def extract_only_nouns(self, text):
        """텍스트에서 '진짜 명사'만 추출하고 불용어 및 길이 필터링"""
        try:
            tokens = word_tokenize(text.lower())
            # 1. 단어별 품사 태깅 수행 (결과 예시: [('airport', 'NN'), ('fly', 'VB')])
            tagged_tokens = pos_tag(tokens)
            
            # 2. 품사가 NN(일반명사), NNS(복수명사), NNP(고유명사), NNPS(복수고유명사)인 것만 필터링
            nouns = [
                word for word, tag in tagged_tokens
                if tag in ('NN', 'NNS', 'NNP', 'NNPS') 
                and word.isalpha() 
                and word not in COMBINED_STOPWORDS 
                and len(word) > 2
            ]
            return ' '.join(nouns)
        except Exception as e:
            print(f"명사 추출 오류: {e}")
            return ""

    def build_tfidf(self, corpus):
        if not corpus: return pd.DataFrame()

        # 각 문서에서 '진짜 명사'만 먼저 추출하여 새로운 코퍼스 생성
        noun_corpus = [self.extract_only_nouns(doc) for doc in corpus]
        noun_corpus = [doc for doc in noun_corpus if doc.strip()]

        if not noun_corpus:
            print("TF-IDF를 생성할 유효한 명사 텍스트가 없습니다.")
            return pd.DataFrame()

        # 이미 위에서 명사 추출과 불용어 처리를 완벽히 끝냈으므로 기본 공백 분할만 수행
        vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
        
        try:
            matrix = vectorizer.fit_transform(noun_corpus)
        except ValueError:
            return pd.DataFrame()

        feature_names = vectorizer.get_feature_names_out()
        if len(feature_names) == 0: return pd.DataFrame()

        col_sums = np.asarray(matrix.sum(axis=0)).ravel()
        top_k = min(50, len(feature_names))
        top_idx = np.argsort(col_sums)[::-1][:top_k]
        top_features = feature_names[top_idx]

        return pd.DataFrame(matrix[:, top_idx].toarray(), columns=top_features)