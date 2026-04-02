from pathlib import Path
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SRC_DIR = Path(__file__).resolve().parent / "src"


def load_corpus(file_path: str) -> list[str]:
    path = Path(file_path)
    path = path if path.is_absolute() else SRC_DIR / path

    if not path.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    docs = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not docs:
        raise ValueError("파일에 분석할 텍스트가 없습니다. (빈 줄 제외)")
    return docs


def build_tfidf_df(corpus: list[str]) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    return pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF 분석 스크립트")
    parser.add_argument(
        "file_path",
        nargs="?",
        default="crawling_result.txt",
        help="분석할 텍스트 파일 경로 (상대경로는 src 기준, 기본값: crawling_result.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_tfidf = build_tfidf_df(load_corpus(args.file_path))
    print("English TF-IDF Analysis Result")
    print(df_tfidf.T)


if __name__ == "__main__":
    main()