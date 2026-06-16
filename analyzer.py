import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

class TextAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
        print(">>> 문맥 분석을 위한 자연어 처리(NLP) 모델을 로드 중입니다... (최초 1회 다운로드)")
        # 문맥 이해 능력이 뛰어난 BART 모델 기반의 제로샷 분류기 로드
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli",
            device=-1 # CPU 사용 (GPU가 있다면 0 설정 가능)
        )
        
        # 우리가 분류할 STEEP 카테고리 정의 (AI에게 입력할 기준)
        self.categories = [
            "Social & Passenger Experience", 
            "Technological Innovation", 
            "Environmental Sustainability", 
            "Political & Government Policy"
        ]

    def load_corpus(self):
        """저장된 뉴스 원본 데이터를 로드합니다."""
        file_path = self.data_dir / "raw_news.csv" 
        if not file_path.exists():
            print("데이터 파일이 존재하지 않습니다.")
            return None
        return pd.read_csv(file_path)

    def classify_steep_with_context(self, df):
        """[핵심] AI가 단어가 아닌 '문맥'을 읽고 기사를 STEEP으로 분류합니다."""
        print("\n>>> AI가 기사별 문맥을 분석하여 STEEP 카테고리로 분류하는 중...")
        assigned_categories = []

        # tqdm을 사용해 진행 상황을 시각적으로 표시합니다.
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # 기사 제목과 본문 앞부분을 합쳐 맥락 텍스트 생성 (너무 길면 자름)
            title = str(row.get('title', ''))
            content = str(row.get('content', ''))
            context_text = f"{title}. {content[:1000]}" 
            
            if not context_text.strip():
                assigned_categories.append("Unclassified")
                continue
            
            try:
                # AI가 텍스트의 맥락을 분석하여 각 카테고리별 확률 계산
                result = self.classifier(context_text, self.categories, multi_label=False)
                # 가장 확률(Score)이 높은 카테고리의 이름을 가져옴
                best_category = result['labels'][0]
                assigned_categories.append(best_category)
            except Exception as e:
                assigned_categories.append("Unclassified")
                
        df['STEEP_Category'] = assigned_categories
        return df

    def build_tfidf_by_category(self, df):
        """AI가 1차 분류한 방 안에서, 핵심 키워드를 추출하기 위해 TF-IDF를 돌립니다."""
        print("\n>>> 카테고리별 독립적 TF-IDF 키워드 추출 시작...")
        
        # 기본 영어 불용어에, 공항 뉴스에서 '맥락상 당연히 나오는 본질적 노이즈'만 최소한으로 제거
        base_stop_words = ['incheon', 'airport', 'international', 'icn', 'korea', 'south', 'seoul', 'said', 'new']
        stop_words = text.ENGLISH_STOP_WORDS.union(base_stop_words)
        
        category_tfidf_results = {}
        
        for cat in self.categories:
            # AI가 해당 카테고리로 묶어준 기사들만 필터링
            sub_df = df[df['STEEP_Category'] == cat]
            print(f" - [{cat}] 카테고리에 분류된 기사 수: {len(sub_df)}개")
            
            if sub_df.empty or len(sub_df) < 2:
                print(f"   ! 기사 수가 너무 적어 {cat}의 TF-IDF 분석을 건너뜁니다.")
                continue
                
            # 기사 제목과 본문을 합쳐 말뭉치(Corpus) 구축
            corpus = (sub_df['title'].fillna('') + " " + sub_df['content'].fillna('')).tolist()
            
            # TF-IDF 모델 설정
            vectorizer = TfidfVectorizer(
                stop_words=list(stop_words),
                max_features=25,       # 각 카테고리별 핵심 단어 25개 추출
                min_df=2,              # 최소 2개 이상의 기사에서 언급된 단어만 (완전 고립된 노이즈 방지)
                ngram_range=(1, 2)     # "carbon neutral", "smart pass" 같은 연어(Phrases)도 함께 추출
            )
            
            tfidf_matrix = vectorizer.fit_transform(corpus)
            importance = tfidf_matrix.mean(axis=0).getA1()
            words = vectorizer.get_feature_names_out()
            
            # 단어와 가중치(점수)를 데이터프레임으로 결합 및 정렬
            cat_result = pd.DataFrame({'Word': words, 'TF-IDF_Score': importance})
            cat_result = cat_result.sort_values(by='TF-IDF_Score', ascending=False).reset_index(drop=True)
            
            category_tfidf_results[cat] = cat_result
            
        return category_tfidf_results