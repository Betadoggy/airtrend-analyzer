import re
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# optional morphological analyzer (better noun extraction)
try:
    from kiwipiepy import Kiwi
except Exception:
    Kiwi = None

# 의존명사나 과도하게 일반적인 단어만 제외
KOREAN_STOPWORDS = {
    '것', '수', '등'  # 의존명사들
}

class TextAnalyzer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        if Kiwi:
            try:
                self.kiwi = Kiwi()
            except Exception:
                self.kiwi = None
        else:
            self.kiwi = None

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

    def extract_korean_nouns(self, text):
        """한국어 텍스트에서 명사와 주요 단어를 추출합니다."""
        try:
            # Prefer POS-based noun extraction when kiwipiepy is available
            if self.kiwi:
                raw_nouns = []
                for token in self.kiwi.tokenize(text):
                    word = getattr(token, 'form', None) or getattr(token, 'text', None)
                    pos = getattr(token, 'tag', None) or getattr(token, 'pos', None)
                    if not word or not pos:
                        continue
                    if pos.startswith("NN") and len(word) > 1 and word not in KOREAN_STOPWORDS:
                        raw_nouns.append(word)
                return " ".join(raw_nouns)

            # Fallback: simple regex-based extraction (existing behavior)
            cleaned = re.sub(r"[^\uac00-\ud7a3A-Z\s]", " ", text)
            words = re.findall(r"[\uac00-\ud7a3]{2,}|[A-Z]{2,}", cleaned)
            nouns = [word for word in words if len(word) > 1 and word not in KOREAN_STOPWORDS]
            return " ".join(nouns)
        except Exception as e:
            print(f"한국어 명사 추출 오류: {e}")
            return ""

    def build_tfidf(self, corpus):
        if not corpus:
            return pd.DataFrame()

        noun_corpus = [self.extract_korean_nouns(doc) for doc in corpus]
        noun_corpus = [doc for doc in noun_corpus if doc.strip()]

        if not noun_corpus:
            print("TF-IDF를 생성할 유효한 한국어 명사 텍스트가 없습니다. 전체 한국어 단어 텍스트로 재시도합니다.")
            noun_corpus = []
            for doc in corpus:
                korean_words = re.findall(r"[\uac00-\ud7a3]{2,}", doc)
                if korean_words:
                    noun_corpus.append(" ".join(korean_words))
            if not noun_corpus:
                return pd.DataFrame()

        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        try:
            matrix = vectorizer.fit_transform(noun_corpus)
        except ValueError:
            return pd.DataFrame()

        feature_names = vectorizer.get_feature_names_out()
        if len(feature_names) == 0:
            return pd.DataFrame()

        col_sums = np.asarray(matrix.sum(axis=0)).ravel()
        top_k = min(50, len(feature_names))
        top_idx = np.argsort(col_sums)[::-1][:top_k]
        top_features = feature_names[top_idx]

        return pd.DataFrame(matrix[:, top_idx].toarray(), columns=top_features)
