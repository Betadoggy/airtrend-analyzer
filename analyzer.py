import os
import re
import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# AI 제로샷 분류를 위한 Hugging Face 파이프라인 로드
try:
    from transformers import pipeline
except ImportError:
    raise ImportError("Hugging Face transformers 라이브러리가 없습니다. 'pip install transformers torch'를 실행하세요.")

# NLTK 자원 다운로드
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class TextAnalyzer:
    def __init__(self, data_dir="news_data"):
        self.data_dir = Path(data_dir)
        self.lemmatizer = WordNetLemmatizer()
        
        # 1. AI 제로샷 분류 모델 초기화 (BART 대형 모델 사용)
        print("\n>>> AI 문맥 분석 모델 로딩 중 (최초 실행 시 다운로드로 인해 시간이 소요될 수 있습니다)...")
        # 소만사 등 사내 망 환경에서 SSL 인증서 경고가 뜨지 않도록 앞서 os.environ 설정을 완료했습니다.
        self.ai_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # 2. AI가 파악할 거시환경(STEEP) 목표 카테고리 정의
        self.steep_labels = ['Social', 'Technological', 'Environmental', 'Political']
        
        # 3. 기본 및 도메인 맞춤형 불용어 생성 (TF-IDF 추출용)
        self.stop_words = set(stopwords.words('english'))
        domain_stopwords = {
            'according', 'air', 'aircraft', 'airline', 'airlines', 'airport', 'airports', 'also', 
            'aviation', 'city', 'could', 'country', 'day', 'first', 'flights', 'flight', 
            'singapore', 'singaporean', 'international', 'korea', 'korean', 'like', 'million', 'new', 
            'news', 'north', 'one', 'passengers', 'passenger', 'reported', 'said', 'second', 
            'since', 'south', 'time', 'travelers', 'traveler', 'travel', 'two', 
            'world', 'years', 'year', 'vna' # 🌟 이전 단계에서 발견된 원형복원 찌꺼기 불용어 사전 추가
        }
        self.stop_words.update(domain_stopwords)

    def clean_text(self, text):
        """하이픈 깨짐 현상을 보완하고 불용어 제거 없이 원형 복원만 수행"""
        if not isinstance(text, str):
            return ""
        
        # 🌟 단어 쪼개짐(ktre 등) 방지: 하이픈과 슬래시를 공백으로 먼저 치환
        text = text.replace('-', ' ').replace('/', ' ')
        
        # 특수문자 및 숫자 제거, 소문자화
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        
        # 단어 토큰화 및 표제어 추출
        words = text.split()
        cleaned_words = [
            self.lemmatizer.lemmatize(w) for w in words 
            if len(w) > 2
        ]
        return " ".join(cleaned_words)

    def load_corpus(self):
        """크롤링 폴더에서 텍스트를 로드하며 가드레일 키워드를 검증"""
        print(f"\n>>> '{self.data_dir}' 폴더에서 데이터 로드 중...")
        if not self.data_dir.exists():
            print(f"[오류] 데이터 폴더가 존재하지 않습니다: {self.data_dir}")
            return None
            
        file_paths = list(self.data_dir.glob("*.txt"))
        if not file_paths:
            print("[오류] 분석할 txt 파일이 없습니다. 크롤링을 먼저 수행하세요.")
            return None

        data = []
        for path in file_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                title = lines[0].replace("Title: ", "").strip() if len(lines) > 0 else ""
                source = lines[1].replace("Source: ", "").strip() if len(lines) > 1 else ""
                url = lines[2].replace("URL: ", "").strip() if len(lines) > 2 else ""
                content = "".join(lines[5:]) if len(lines) > 5 else ""
                
                # [가드레일 필터링] 공항과 관계없는 순수 외부 노이즈 차단
                required_keywords = ['singapore', 'changi', 'sin', 'changi airport', 'sin airport']
                content_lower = content.lower()
                if not any(kw in content_lower for kw in required_keywords):
                    # 소음 기사는 원천적으로 제외 메시지를 띄우고 수집하지 않음
                    print(f" [가드레일 필터링] 공항 도메인과 무관한 기사 제외: {path.name}")
                    continue

                data.append({
                    "title": title,
                    "source": source,
                    "url": url,
                    "content": content,
                    "clean_text": self.clean_text(content)
                })
            except Exception as e:
                print(f" 파일 읽기 실패 ({path.name}): {e}")
                
        print(f" 총 {len(data)}개의 유효 공항 기사 로드 및 전처리 완료.")
        return pd.DataFrame(data)

    def classify_steep_with_ai(self, df):
        """ [핵심 변경] 단어 빈도가 아닌 AI 모델 기반 딥러닝 문맥 추론으로 STEEP 분류 진행"""
        print("\n>>> AI 모델을 활용한 거시환경(STEEP) 문맥 분석 시작...")
        
        def predict_category(row):
            # 분석 속도 최적화 및 핵심 요약 반영을 위해 타이틀과 본문 앞 1,500자를 결합하여 입력값 생성
            input_text = f"Title: {row['title']}\nContent: {row['content'][:1500]}"
            
            try:
                # 제로샷 추론 수행
                ai_result = self.ai_classifier(input_text, self.steep_labels, multi_label=False)
                best_label = ai_result['labels'][0]
                return best_label
            except Exception as e:
                # 예외 발생 시 가장 기본 카테고리인 'Social' 대치
                return 'Social'

        # 행 단위 데이터 추론 적용
        df['STEEP'] = df.apply(predict_category, axis=1)
        
        print("--- AI 분류 결과 분포 ---")
        print(df['STEEP'].value_counts())
        return df

    def build_tfidf_by_category(self, df, top_n=100):
        """카테고리별 핵심 키워드 추출 시점(최종 단계)에만 불용어 제거 반영"""
        print("\n>>> 카테고리별 TF-IDF 핵심 키워드 추출 중 (불용어 반영)...")
        results = {}
        
        categories = df['STEEP'].unique()
        for cat in categories:
            cat_df = df[df['STEEP'] == cat]
            corpus = cat_df['clean_text'].tolist()
            
            if len(corpus) < 3:
                print(f" [건너뛰기] '{cat}' 카테고리는 문서 수가 부족합니다 ({len(corpus)}개).")
                continue
                
            # token_pattern 조정을 통해 3글자 이상 단어만 필터링하여 wa, ha, prod 완벽 차단
            vectorizer = TfidfVectorizer(
                max_features=100, 
                stop_words=list(self.stop_words),
                token_pattern=r"(?u)\b\w{3,}\b"
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            mean_tfidf = tfidf_matrix.mean(axis=0).tolist()[0]
            feature_names = vectorizer.get_feature_names_out()
            
            keywords_df = pd.DataFrame({
                'Keyword': feature_names,
                'TF-IDF_Score': mean_tfidf
            }).sort_values(by='TF-IDF_Score', ascending=False).head(top_n)
            
            results[cat] = keywords_df
            
        return results