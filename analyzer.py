import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import pos_tag, word_tokenize
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

    def extract_nouns(self, text):
        """텍스트에서 명사만 추출하고 불용어 제거"""
        try:
            # 토큰화
            tokens = word_tokenize(text.lower())
            
            # POS 태깅
            pos_tagged = pos_tag(tokens)
            
            # 명사만 필터링 (NN=단수명사, NNS=복수명사, NNP=고유명사(단수), NNPS=고유명사(복수))
            nouns = [word for word, pos in pos_tagged if pos in ('NN', 'NNS', 'NNP', 'NNPS')]
            
            # 불용어 및 짧은 단어 제거
            filtered_nouns = [n for n in nouns if n not in COMBINED_STOPWORDS and len(n) > 2]
            
            return ' '.join(filtered_nouns)
        except Exception as e:
            print(f"명사 추출 오류: {e}")
            return ""

    def build_tfidf(self, corpus):
        """TF-IDF 매트릭스 생성 (명사 기반)"""
        # 각 문서에서 명사만 추출
        noun_corpus = [self.extract_nouns(doc) for doc in corpus]
        
        # 불용어를 다시 한 번 필터링하는 customanalyzer 정의
        def noun_analyzer(text):
            tokens = text.split()
            return [t for t in tokens if t and len(t) > 2]
        
        # TF-IDF 벡터화 (명사 기반, 불용어 제거)
        vectorizer = TfidfVectorizer(
            analyzer=noun_analyzer,
            stop_words=list(COMBINED_STOPWORDS),
            max_features=50,
            min_df=1,  # 최소 1개 문서에서 나타나야 함
            max_df=0.95  # 95% 이상의 문서에 나타나는 단어 제외
        )
        matrix = vectorizer.fit_transform(noun_corpus)
        
        df = pd.DataFrame(
            matrix.toarray(), 
            columns=vectorizer.get_feature_names_out()
        )
        return df