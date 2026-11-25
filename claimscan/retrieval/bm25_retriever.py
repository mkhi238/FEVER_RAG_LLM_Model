"""Sentence-level BM25 over the Lucene index built by claimscan.retrieval.bm25_index.

Was archive/bm25_sentence_retriever.py; only the hard-coded C: index path changed.
Returns the same {"id", "line", "score"} dicts as FAISSRetriever, so the two are
interchangeable and can be fused by HybridRetriever. Note the score direction differs:
BM25 scores are higher = better, FAISS (L2) scores are lower = better. Evidence scoring
(claimscan/neurosymbolic_layer/score.py) uses each hit's rank, not its score, so either direction works there.
"""
from __future__ import annotations

from pathlib import Path

from claimscan.config import config


def split_docid(docid: str):
  if "#" in docid:
    page, line = docid.rsplit("#", 1)
    try:
      return page, int(line)
    except ValueError:
      return page, -1
  return docid, -1


class BM25Retriever:
  def __init__(self, index_dir=config.bm25_index_dir, k1=config.bm25_k1, b=config.bm25_b, use_rm3=config.bm25_rm3):
    from pyserini.search.lucene import LuceneSearcher  # starts the JVM; keep it out of module import

    index_dir = Path(index_dir)
    if not index_dir.exists():
      raise FileNotFoundError(f"BM25 index does not exist at {index_dir}; build it with "
                              f"python -m claimscan.retrieval.bm25_index")
    self.s = LuceneSearcher(str(index_dir))
    self.s.set_bm25(k1, b)
    if use_rm3:
      self.s.set_rm3()
    print(f"Loaded BM25 index {index_dir} ({self.s.num_docs:,} sentences, k1={k1}, b={b}, rm3={use_rm3})")

  @staticmethod
  def _to_dicts(hits):
    out = []
    for hit in hits:
      page, line = split_docid(hit.docid)
      out.append({"id": page, "line": line, "score": float(hit.score)})
    return out

  def search(self, query, k=50):
    return self._to_dicts(self.s.search(query, k=k))

  def search_many(self, queries, k=50, threads=2):
    """search() for many queries, run on Lucene's own thread pool. Same hits as calling search() per query."""
    qids = [str(i) for i in range(len(queries))]
    res = self.s.batch_search(list(queries), qids, k=k, threads=threads)
    return [self._to_dicts(res[q]) for q in qids]
