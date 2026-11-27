"""Holds the loaded models for the API (was dependancy.py; main.py imported api.dependencies).

Loads the verifier and the retriever once. Sentence text comes from the SQLite sentence store
(claimscan.evidence.store), so the 4 GB corpus is never loaded into RAM.
"""
import logging
import threading

from claimscan.config import config
from claimscan.pipeline import Services, load_retriever, load_verifier

log = logging.getLogger("claimscan.api")


def load_aggregator(cfg=config):
  """Structured LLM for the rear aggregator, or None when disabled (or langchain is missing: fail open)."""
  if not cfg.aggregator_enabled:
    return None
  try:
    from claimscan.aggregation import build_llm
    return build_llm(cfg)
  except Exception as e:
    log.warning("rear aggregator disabled, could not build the LLM client: %r", e)
    return None


def load_services(cfg=config) -> Services:
  from claimscan.evidence.store import SentenceStore
  from claimscan.rerank import load_reranker
  return Services(model=load_verifier(cfg), retriever=load_retriever(cfg), corpus_df=SentenceStore(cfg.sentence_store),
                  reranker=load_reranker(cfg))


class ModelService:
  def __init__(self):
    self.services: Services | None = None
    self.aggregator_llm = None   # structured runnable for the rear aggregator; None = off
    # One claim at a time: inference is CPU-bound (parallel requests would only thrash the cores),
    # and the BM25 searcher (Lucene via pyserini) is not thread-safe.
    self.lock = threading.Lock()

  def load(self):
    self.services = load_services(config)
    self.aggregator_llm = load_aggregator(config)
    print("Load Complete", flush=True)

  def is_loaded(self):
    return self.services is not None


model_service = ModelService()
