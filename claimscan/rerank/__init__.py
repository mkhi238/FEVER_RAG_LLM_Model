"""Reranker layer: pick a backend with ``rerank.backend`` in configs/default.yaml.

    from claimscan.rerank import load_reranker
    rr = load_reranker()          # None when backend is "none"
"""
from __future__ import annotations

from claimscan.config import config
from claimscan.rerank.base import Reranker, rerank_hits

BACKENDS = ("none", "cross_encoder", "cohere", "openai", "llamacpp")


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
    return OpenAIReranker(cfg.rerank_openai_model, base_url=cfg.rerank_openai_base_url,
                          mode=cfg.rerank_openai_mode, batch_size=cfg.rerank_batch_size)
  if backend == "llamacpp":
    from claimscan.rerank.llamacpp import LlamaCppReranker
    return LlamaCppReranker()
  raise ValueError(f"Unknown rerank backend {backend!r}; choose from {BACKENDS}")
