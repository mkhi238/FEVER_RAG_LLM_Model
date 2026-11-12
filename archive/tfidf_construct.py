from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib
import numpy as np

#Index builder; takes processed corpus and makes tfidfvectorizer to computer document term matrix
path = Path(r"C:\Users\mukun\crisis-claim-analysis\data\processed")

def load_corpus():
  corp_parq = pd.read_parquet(path / 'fever_corpus_parquet')
  return corp_parq

def main():
  corpus_parq = load_corpus()
  meta = corpus_parq[["id", "line"]].reset_index(drop=True)
  text = corpus_parq['text'].astype(str)

  tf_vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=3,
    max_df= 0.7,
    strip_accents="unicode",
    lowercase=True,
    dtype = np.float32,
    token_pattern = r"(?u)\b\w\w+\b",
    max_features = 100000
  )

  X = tf_vectorizer.fit_transform(text)
  X = X.tocsr().astype(np.float32, copy=False)

  joblib.dump(tf_vectorizer, r'C:\Users\mukun\crisis-claim-analysis\artifacts\tfidf\tfidf_vectorizer.joblib', compress=0)
  sparse.save_npz(r"C:\Users\mukun\crisis-claim-analysis\artifacts\tfidf\tfidf_matrix.npz", X, compressed=False)
  meta.to_parquet(r"C:\Users\mukun\crisis-claim-analysis\artifacts\tfidf\corpus_meta.parquet", index=False)

  print(f"TF-IDF built: rows={X.shape[0]:,}, vocab={len(tf_vectorizer.vocabulary_):,}")


if __name__ == "__main__":
  main()