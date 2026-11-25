import importlib

import pytest

MODULES = [
  "claimscan.config",
  "claimscan.data.fever_db",
  "claimscan.data.posttrain_store",
  "claimscan.neurosymbolic_layer",
  "claimscan.neurosymbolic_layer.score",
  "claimscan.evidence.select",
  "claimscan.retrieval.hybrid_retriever",
  "claimscan.retrieval.faiss_batch",
  "claimscan.retrieval.bm25_retriever",
  "claimscan.retrieval.bm25_index",
  "claimscan.eval",
  "claimscan.eval.gold",
  "claimscan.eval.fever_eval",
  "claimscan.verifier.deberta_verifier",
  "claimscan.verifier.train",
  "claimscan.pipeline",
  "claimscan.api.schemas",
  "claimscan.api.deps",
  "claimscan.api.main",
  "claimscan.aggregation",
  "claimscan.instruct.schemas",
  "claimscan.instruct.chat_clients",
  "claimscan.instruct.rear_aggregator",
  "claimscan.agents",
  "claimscan.graph",
  "claimscan.rerank",
  "claimscan.rerank.base",
  "claimscan.rerank.cohere",
  "claimscan.rerank.openai_llm",
  "claimscan.rerank.cross_encoder",
  "claimscan.rerank.llamacpp",
  "claimscan.serve",
]
NEEDS_FAISS = [
  "claimscan.retrieval.faiss_retriever",
  "claimscan.retrieval.faiss_indexer",
  "claimscan.retrieval.faiss_embeddings",
  "claimscan.eval.eval_retriever",
  "claimscan.eval.benchmark",
  "claimscan.tracking",
  "claimscan.verifier.prepare",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
  importlib.import_module(name)


@pytest.mark.parametrize("name", NEEDS_FAISS)
def test_import_retrieval(name):
  pytest.importorskip("faiss")
  pytest.importorskip("sentence_transformers")
  importlib.import_module(name)
