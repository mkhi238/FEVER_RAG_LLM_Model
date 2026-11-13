"""Reranker served by our llama.cpp layer (Qwen3-Reranker-0.6B GGUF on llama-server --reranking).

Delegates to ``claimscan.llm.client.rerank(query, documents, top_n)``. That function is written by
the user and raises NotImplementedError until it is finished; this wrapper only maps its output.
"""
from __future__ import annotations

from claimscan.rerank.base import Reranker


class LlamaCppReranker(Reranker):
  name = "llamacpp"

  def rerank(self, query, docs, top_n=None):
    if not docs:
      return []
    from claimscan.llm import client  # looked up at call time

    ranked = [(int(i), float(s)) for i, s in client.rerank(query, list(docs), top_n)]
    ranked.sort(key=lambda t: (-t[1], t[0]))
    return ranked[:top_n] if top_n else ranked
