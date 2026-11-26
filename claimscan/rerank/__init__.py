"""Reranker layer: pick a backend with ``rerank.backend`` in configs/default.yaml.

    from claimscan.rerank import load_reranker
    rr = load_reranker()          # None when backend is "none"
"""
from __future__ import annotations

from claimscan.config import config
from claimscan.rerank.base import Reranker, rerank_hits

BACKENDS = ("none", "cross_encoder", "cohere", "openai")


def load_reranker(cfg=config, backend: str | None = None) -> Reranker | None:
  """Build the reranker named by ``backend`` (default ``cfg.rerank_backend``); None for "none"."""
  backend = (backend or cfg.rerank_backend).lower()
  if backend == "none":
    return None
  if backend == "cross_encoder":
    from claimscan.rerank.cross_encoder import CrossEncoderReranker
    return CrossEncoderReranker(cfg.rerank_cross_encoder_model, device=cfg.rerank_device,
                                batch_size=cfg.rerank_batch_size)
  if backend == "cohere":
    from claimscan.rerank.cohere import CohereReranker
    return CohereReranker(cfg.rerank_cohere_model, base_url=cfg.rerank_cohere_base_url)
  if backend == "openai":
    from claimscan.rerank.openai_llm import OpenAIReranker
    model = cfg.rerank_openai_model
    # reasoning models (gpt-5*, o1/o3/o4*) reject temperature and need room for hidden reasoning tokens
    reasoning = model.startswith(("gpt-5", "o1", "o3", "o4"))
    extra = dict(temperature=None, max_tokens=1024, max_tokens_field="max_completion_tokens") if reasoning else {}
    return OpenAIReranker(model, base_url=cfg.rerank_openai_base_url,
                          mode=cfg.rerank_openai_mode, batch_size=cfg.rerank_batch_size, **extra)
  raise ValueError(f"Unknown rerank backend {backend!r}; choose from {BACKENDS}")
