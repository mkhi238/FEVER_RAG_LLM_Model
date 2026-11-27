"""Local sentence_transformers CrossEncoder reranker. CPU by default so it never competes for the GPU."""
from __future__ import annotations

from claimscan.rerank.base import Reranker, sort_scores


class CrossEncoderReranker(Reranker):
  name = "cross_encoder"

  def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu",
               batch_size: int = 32, max_length: int = 256, model_obj=None):
    self.model_name, self.device, self.batch_size, self.max_length = model, device, batch_size, max_length
    self._model = model_obj  # injectable for tests; loaded on first use otherwise

  def _load(self):
    if self._model is None:
      from sentence_transformers import CrossEncoder
      self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
    return self._model

  def rerank(self, query, docs, top_n=None):
    if not docs:
      return []
    scores = self._load().predict([(query, d) for d in docs], batch_size=self.batch_size,
                                  show_progress_bar=False)
    return sort_scores(scores, top_n)
