"""Cohere Rerank over plain HTTP (httpx). Key comes from env COHERE_API_KEY; it is never logged.

Trial keys are free but limited to ~10 rerank requests/min and 1,000 calls/month, and Cohere's
terms forbid production/commercial use on them (see notes/reranker_research.md).
"""
from __future__ import annotations

import os
import time

from claimscan.rerank.base import Reranker, sort_scores


class CohereReranker(Reranker):
  name = "cohere"

  def __init__(self, model: str = "rerank-v3.5", base_url: str = "https://api.cohere.com",
               api_key: str | None = None, timeout: float = 60.0, max_retries: int = 3,
               transport=None, sleep=time.sleep):
    self.model = model
    self.url = base_url.rstrip("/") + "/v2/rerank"
    self._api_key = api_key
    self.timeout = timeout
    self.max_retries = max_retries
    self._transport = transport  # httpx.MockTransport in tests
    self._sleep = sleep

  def _key(self) -> str:
    key = self._api_key or os.environ.get("COHERE_API_KEY")
    if not key:
      raise RuntimeError("COHERE_API_KEY is not set. Export it or use another rerank.backend.")
    return key

  def rerank(self, query, docs, top_n=None):
    if not docs:
      return []
    if len(docs) > 1000:
      raise ValueError("Cohere recommends at most 1000 documents per request")
    import httpx

    body = {"model": self.model, "query": query, "documents": list(docs)}
    if top_n:
      body["top_n"] = int(top_n)
    headers = {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"}
    with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
      for attempt in range(self.max_retries + 1):
        resp = client.post(self.url, json=body, headers=headers)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
          self._sleep(min(2 ** attempt * 2, 30))
          continue
        break
    if resp.status_code != 200:
      raise RuntimeError(f"Cohere rerank failed: HTTP {resp.status_code}: {resp.text[:200]}")
    results = resp.json()["results"]
    ranked = [(int(r["index"]), float(r["relevance_score"])) for r in results]
    ranked.sort(key=lambda t: (-t[1], t[0]))
    return ranked[:top_n] if top_n else ranked
