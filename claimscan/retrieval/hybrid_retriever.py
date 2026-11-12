"""Hybrid retrieval: FAISS (dense) + BM25 (sparse) fused with reciprocal rank fusion.

Finished from the unfinished C:\\...\\archive\\hybrid_retriever\\hybrid_retriever.py.
Same RRF as there: each retriever returns its top `candidates` hits, and a sentence scores
sum(1 / (rrf_k + rank)) over the retrievers that found it (rank is 1-based, rrf_k = 60).

What changed while finishing it:
  - It fuses on the (page id, line) pair both retrievers return, instead of row indices
    into a docid list. The old BM25 (rank_bm25 over the first 5M docids) needed that
    mapping; the Lucene BM25 returns page/line directly and covers all 25.2M sentences.
  - The final `np.argsort()` (no argument, so it crashed) is a plain sort by fused score.

Results look like FAISSRetriever's: {"id", "line", "score"}, sorted best first. "score"
follows the FAISS convention, LOWER = better: the stored bge index is L2 (distances), and
evidence/scoring.py turns it into quality with 1 / (score + e). So score = 1 / rrf_score,
which keeps that formula ranking the fused hits in RRF order. The raw RRF value is in
"rrf_score" (higher = better), and faiss_rank / bm25_rank (1-based, None = not retrieved)
show which retriever found what.
"""
from __future__ import annotations


def rrf_fuse(ranked_lists: dict, rrf_k: int = 60, k: int | None = None) -> list[dict]:
  """Fuse {name: [hit, ...]} ranked best-first into one list sorted by RRF score (best first).

  Ties keep first-seen order (the first list's order wins), so results are deterministic.
  """
  fused = {}
  for name, hits in ranked_lists.items():
    for rank, hit in enumerate(hits, start=1):
      key = (hit["id"], hit["line"])
      entry = fused.get(key)
      if entry is None:
        entry = {"id": hit["id"], "line": hit["line"], "rrf_score": 0.0, **{f"{n}_rank": None for n in ranked_lists}}
        fused[key] = entry
      if entry[f"{name}_rank"] is None:   # count a sentence once per retriever
        entry[f"{name}_rank"] = rank
        entry["rrf_score"] += 1.0 / (rrf_k + rank)
  out = sorted(fused.values(), key=lambda e: e["rrf_score"], reverse=True)
  for e in out:
    e["score"] = 1.0 / e["rrf_score"]     # lower = better, like FAISS L2 distances
  return out if k is None else out[:k]


class HybridRetriever:
  def __init__(self, faiss_retriever, bm25_retriever, rrf_k: int = 60, candidates: int = 100):
    self.faiss = faiss_retriever
    self.bm25 = bm25_retriever
    self.rrf_k = rrf_k
    self.candidates = candidates
    print(f"Hybrid retriever ready (RRF k={rrf_k}, {candidates} candidates per retriever)")

  def search(self, query, k=50):
    n = max(k, self.candidates)
    return rrf_fuse(
      {"faiss": self.faiss.search(query, k=n), "bm25": self.bm25.search(query, k=n)},
      rrf_k=self.rrf_k,
      k=k,
    )
