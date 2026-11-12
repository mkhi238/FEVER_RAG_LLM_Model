from pathlib import Path
import pandas as pd
import numpy as np
import joblib

#page2idx.joblib (for TF-IDF rerank)

#Format: "Obama" (id) -> np.array([sentence_row_indices])
#Purpose: map BM25 pages → sentence rows for reranking.
#Stores: only sentence indices, no text.

tfidf = Path(r"C:\Users\mukun\crisis-claim-analysis\artifacts\tfidf")

def main():
  meta = pd.read_parquet(tfidf / "corpus_meta.parquet")
  title = meta["id"].astype(str).to_numpy()
  idx = np.arange(len(title), dtype = np.int64)

  order = np.argsort(title, kind = "stable")
  #titles_sorted = ["Cats", "Cats", "Cats", "Obama", "Obama"]
  title_sorted = title[order]
  idx_sorted = idx[order]

  unique, start_idx = np.unique(title_sorted, return_index=True)
  start_idx = np.append(start_idx, len(title_sorted))

  page2idx = {}
  for i in range(len(unique)):
    start, end = int(start_idx[i]), int(start_idx[i+1])
    page2idx[unique[i]] = idx_sorted[start:end]

  joblib.dump(page2idx, tfidf / "page2idx.joblib", compress = 0)
  print("pages mapped:", len(page2idx))

if __name__ == "__main__":
    main()