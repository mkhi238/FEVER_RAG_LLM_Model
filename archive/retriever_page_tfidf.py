from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import joblib
from pyserini.search.lucene import LuceneSearcher
import unicodedata, re



class PageTFIDFRetriever():
  def __init__(self,bm25_k1 = 1.15,bm25_b = 0.8,use_rm3 = False,max_sentences_per_page = None):
    
    self.vectorizer = joblib.load(str(Path(r"C:\Users\mukun\crisis-claim-analysis\artifacts\tfidf\tfidf_vectorizer.joblib")))
    self.X = sparse.load_npz(str(Path(r"C:\Users\mukun\crisis-claim-analysis\artifacts\tfidf\tfidf_matrix.npz"))).tocsr().astype(np.float32, copy = False)

    meta = pd.read_parquet(str(Path(r"C:\Users\mukun\crisis-claim-analysis\artifacts\tfidf\corpus_meta.parquet")))
    self.meta_id = meta["id"].astype(str).to_numpy()
    self.meta_line_val = meta["line"].astype(np.int32).to_numpy()
    corpus = pd.read_parquet(str(Path(r"C:\Users\mukun\crisis-claim-analysis\data\processed\fever_corpus_parquet")))
    self.corpus_text = corpus['text'].astype(str).to_numpy()

    self.page2idx = joblib.load(str(Path(r"C:\Users\mukun\crisis-claim-analysis\artifacts\tfidf\page2idx.joblib")))
    self.searcher = LuceneSearcher(r"C:\Users\mukun\crisis-claim-analysis\artifacts\bm25_pages\index")
    self.searcher.set_bm25(k1=bm25_k1, b=bm25_b)
    if use_rm3:
      self.searcher.set_rm3()

    self.max_sentences_per_page = max_sentences_per_page

    #query is the claim or question text you want to retrieve evidence for.
  def search(self, query, k = 10, pages = 20):
    #BM25 (Lucene) just gave you the top pages page-level results.
    #Each hit has a docid = the page id (e.g., "Obama", "Cats").
    hits = self.searcher.search(query, k=pages)
    if not hits:
      return []
    candidates = []
    for i in hits:
      #Takes the docid of a BM25 hit (say "Obama").
      #Looks up the sentence indices for that page in page2idx.
      #Result = a NumPy array of sentence row numbers.
      idxs = self.page2idx.get(i.docid)
      if idxs is None:
        continue
      if self.max_sentences_per_page and len(idxs) > self.max_sentences_per_page:
        idxs = idxs[:self.max_sentences_per_page]

      candidates.append(idxs)
    if not candidates:
      return []
      
    cand_idx = np.concatenate(candidates)

    qv = self.vectorizer.transform([query]).astype(np.float32, copy = False) # (1, V)
    X_cand = self.X[cand_idx] # (Ncand, V)
    score = (np.dot(qv, X_cand.T)).toarray().ravel() # (Ncand,1)

    k = max(1, min(int(k), score.size))
    part = np.argpartition(score, -k)[-k:]
    order = part[np.argsort(score[part])[::-1]]
    out = []
    for pos, i_sub in enumerate(order):
        i = int(cand_idx[i_sub])
        out.append({
        "id":   self.meta_id[i],
        "line": int(self.meta_line_val[i]),
        "text": self.corpus_text[i],
        "score": float(score[i_sub]),
            })
    pass