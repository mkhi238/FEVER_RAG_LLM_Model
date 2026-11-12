import importlib

import pytest

MODULES = [
  "claimscan.config",
  "claimscan.data.common",
  "claimscan.data.fever_claim",
  "claimscan.data.fever_evidence",
  "claimscan.data.fever_wiki_dump",
  "claimscan.data.liar",
  "claimscan.data.unify",
  "claimscan.evidence.scoring",
  "claimscan.evidence.select",
  "claimscan.verifier.model",
  "claimscan.verifier.train",
  "claimscan.pipeline",
  "claimscan.api.schemas",
  "claimscan.api.deps",
  "claimscan.api.main",
  "claimscan.agents",
  "claimscan.graph",
  "claimscan.rerank",
  "claimscan.serve",
]
NEEDS_FAISS = [
  "claimscan.retrieval.faiss_retriever",
  "claimscan.retrieval.faiss_indexer",
  "claimscan.retrieval.faiss_embeddings",
  "claimscan.retrieval.eval_recall",
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
