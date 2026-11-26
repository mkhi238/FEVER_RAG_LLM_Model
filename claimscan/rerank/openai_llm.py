"""LLM reranker over an OpenAI-compatible /chat/completions endpoint (OpenAI, or a llama.cpp chat server).

Key from env OPENAI_API_KEY (optional when base_url points at a local server). Never logged.

Modes:
  pointwise (default): one call per doc, the model answers a single integer 0-10. Parallel,
    order-independent (no position bias), deterministic-ish at temperature 0. Unparseable
    answers score 0.0 and are counted in ``parse_failures``.
  listwise (RankGPT style): one call, the model returns a ranking like "[3] > [1] > [2]".
    Cheaper, but position-biased; docs the model omits keep their input order at the end.
Reasoning models reject ``temperature`` and small ``max_tokens``; pass temperature=None and a
bigger max_tokens (or max_tokens_field="max_completion_tokens") for those.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor

from claimscan.rerank.base import Reranker, sort_scores

_POINT_SYSTEM = (
  "You judge whether a Wikipedia sentence helps verify a claim (supports or refutes it). "
  "Answer with ONE integer from 0 (irrelevant) to 10 (directly decisive). Output only the integer."
)
_LIST_SYSTEM = (
  "You rank Wikipedia sentences by how useful they are for verifying a claim (supporting or refuting). "
  "Output only the ranking, most useful first, as identifiers joined by ' > ', e.g. [2] > [1] > [3]. "
  "Include every identifier exactly once."
)
_INT_RE = re.compile(r"\d+(?:\.\d+)?")
_ID_RE = re.compile(r"\[(\d+)\]")


def parse_score(text: str | None) -> float | None:
  """First number in the reply, clamped to 0-10; None if there is no number."""
  m = _INT_RE.search(text or "")
  return None if m is None else min(max(float(m.group()), 0.0), 10.0)


def parse_ranking(text: str | None, n: int) -> list[int]:
  """0-based doc indices from '[3] > [1]'; drops out-of-range and repeats."""
  seen, order = set(), []
  for m in _ID_RE.finditer(text or ""):
    i = int(m.group(1)) - 1
    if 0 <= i < n and i not in seen:
      seen.add(i)
      order.append(i)
  return order


class OpenAIReranker(Reranker):
  name = "openai"

  def __init__(self, model: str = "gpt-5-nano", base_url: str = "https://api.openai.com/v1",
               mode: str = "pointwise", api_key: str | None = None, timeout: float = 60.0,
               batch_size: int = 8, temperature: float | None = 0.0, max_tokens: int = 16,
               max_tokens_field: str = "max_tokens", transport=None):
    if mode not in ("pointwise", "listwise"):
      raise ValueError(f"mode must be pointwise or listwise, got {mode!r}")
    self.model, self.mode = model, mode
    self.url = base_url.rstrip("/") + "/chat/completions"
    self._api_key = api_key
    self.timeout, self.batch_size = timeout, max(1, batch_size)
    self.temperature, self.max_tokens, self.max_tokens_field = temperature, max_tokens, max_tokens_field
    self._transport = transport
    self.parse_failures = 0

  def _headers(self) -> dict:
    key = self._api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
      if "api.openai.com" in self.url:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it or use another rerank.backend.")
      return {"Content-Type": "application/json"}  # local OpenAI-compatible server, no key needed
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

  def _chat(self, client, headers, system: str, user: str, max_tokens: int) -> str:
    body = {"model": self.model, self.max_tokens_field: max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if self.temperature is not None:
      body["temperature"] = self.temperature
    resp = client.post(self.url, json=body, headers=headers)
    if resp.status_code != 200:
      raise RuntimeError(f"chat completion failed: HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"] or ""

  def rerank(self, query, docs, top_n=None):
    if not docs:
      return []
    import httpx

    headers = self._headers()
    with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
      if self.mode == "listwise":
        scores = self._listwise(client, headers, query, docs)
      else:
        scores = self._pointwise(client, headers, query, docs)
    return sort_scores(scores, top_n)

  def _pointwise(self, client, headers, query, docs):
    def one(doc):
      reply = self._chat(client, headers, _POINT_SYSTEM, f"Claim: {query}\nSentence: {doc}\nScore:", self.max_tokens)
      return parse_score(reply)

    with ThreadPoolExecutor(max_workers=self.batch_size) as pool:
      raw = list(pool.map(one, docs))
    self.parse_failures += sum(s is None for s in raw)
    return [0.0 if s is None else s for s in raw]

  def _listwise(self, client, headers, query, docs):
    n = len(docs)
    listing = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
    reply = self._chat(client, headers, _LIST_SYSTEM, f"Claim: {query}\n\nSentences:\n{listing}\n\nRanking:",
                       max(self.max_tokens, 6 * n + 16))
    order = parse_ranking(reply, n)
    if len(order) < n:
      self.parse_failures += 1
      placed = set(order)
      order += [i for i in range(n) if i not in placed]
    scores = [0.0] * n
    for rank, i in enumerate(order):
      scores[i] = float(n - rank)
    return scores
