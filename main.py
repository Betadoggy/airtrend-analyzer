from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 1. 영문 분석 데이터 (Corpus)
corpus = [
  'Data science is an interdisciplinary field that uses scientific methods.',
  'Python is a popular programming language for data analysis.',
  'Machine learning is a subset of artificial intelligence and data science.',
  'Data analysis and machine learning are key parts of data science.'
]

# 2. TF-IDF 벡터라이저 설정
# stop_words='english': 영어의 일반적인 불용어(the, a, is 등)를 자동으로 제외합니다.
vectorizer = TfidfVectorizer(stop_words='english')

# 3. TF-IDF 변환
tfidf_matrix = vectorizer.fit_transform(corpus)

# 4. 결과 시각화를 위한 데이터프레임 생성
words = vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=words)

# 결과를 보기 좋게 전치(Transpose)하여 출력 (행: 단어, 열: 문서 번호)
print("### English TF-IDF Analysis Result ###")
print(df_tfidf.T)