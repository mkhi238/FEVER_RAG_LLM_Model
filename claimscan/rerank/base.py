"""Reranker interface shared by every backend (cohere, openai, cross_encoder).

Contract: ``rerank(query, docs, top_n)`` returns ``[(doc_index, score), ...]`` sorted by
score descending (higher = more relevant), cut to ``top_n`` if given. ``doc_index`` points
into the ``docs`` list that was passed in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Reranker(ABC):
  name = "reranker"

  @abstractmethod
  def rerank(self, query: str, docs: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
    ...


def sort_scores(scores, top_n: int | None = None) -> list[tuple[int, float]]:
  """Turn one score per doc into [(index, score)] sorted descending (ties keep input order)."""
  ranked = sorted(((i, float(s)) for i, s in enumerate(scores)), key=lambda t: (-t[1], t[0]))
  return ranked[:top_n] if top_n else ranked


def rerank_hits(reranker: Reranker, claim: str, hits: list[dict], corpus_lookup, top_n: int | None = None) -> list[dict]:
  """Reorder retriever hits ``{"id", "line", "score"}`` by reranker score.

  Each returned hit is a copy with ``rerank_score`` added; the original ``score`` is left
  as the retriever returned it. Hits whose sentence text is not in
  the corpus cannot be scored: they go after the scored ones with ``rerank_score=None``.
  """
  from claimscan.evidence.select import _lookup_text

  texts, scored_hits, missing = [], [], []
  for hit in hits:
    text = _lookup_text(corpus_lookup, hit["id"], hit["line"])
    if text is None:
      missing.append(dict(hit, rerank_score=None))
    else:
      texts.append(text)
      scored_hits.append(hit)

  ranked = reranker.rerank(claim, texts, top_n) if texts else []
  out = [dict(scored_hits[i], rerank_score=float(s)) for i, s in ranked]
  return out + missing if not top_n else out[:top_n]
