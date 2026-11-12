"""Retrieve -> score evidence -> verify, for one claim.

Was rag/Inference/test_rag.py. The model, retriever and corpus used to be module
globals set in ``__main__`` (or patched in by run_experiments.py); they now travel in
a ``Services`` object from ``load_services()``. Decision logic is unchanged.
The demo claim list moved to scripts/rag_demo.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from claimscan.config import ClaimScanConfig, config
from claimscan.evidence.select import get_evidence
from claimscan.verifier.model import FEVERVerifier

LABEL_MAP = {0: "TRUE", 1: "FALSE", 2: "NOT ENOUGH INFO"}


@dataclass
class Services:
  model: FEVERVerifier
  retriever: "FAISSRetriever | BM25Retriever | HybridRetriever"
  corpus_df: pd.DataFrame


def load_verifier(cfg: ClaimScanConfig = config) -> FEVERVerifier:
  model = FEVERVerifier()
  # map_location: checkpoint was saved on GPU; inference here runs on CPU as before
  model.load_state_dict(torch.load(cfg.verifier_checkpoint, map_location="cpu"))
  model.eval()
  return model


def load_retriever(cfg: ClaimScanConfig = config, kind: str | None = None):
  """Build the retriever named by ``kind`` (default ``cfg.retriever_kind``): faiss | bm25 | hybrid."""
  kind = kind or cfg.retriever_kind
  # imported here so the verifier can be used without faiss / sentence-transformers / pyserini installed
  if kind == "faiss":
    from claimscan.retrieval.faiss_retriever import FAISSRetriever
    return FAISSRetriever(
      cfg.retriever_index_dir,
      cfg.retriever_model,
      index_file=cfg.retriever_index_file,
      docids_file=cfg.retriever_docids_file,
      nprobe=cfg.retriever_nprobe,
      mmap=cfg.retriever_mmap,
    )
  if kind == "bm25":
    from claimscan.retrieval.bm25_retriever import BM25Retriever
    return BM25Retriever(cfg.bm25_index_dir, cfg.bm25_k1, cfg.bm25_b, cfg.bm25_rm3)
  if kind == "hybrid":
    from claimscan.retrieval.hybrid_retriever import HybridRetriever
    return HybridRetriever(load_retriever(cfg, "faiss"), load_retriever(cfg, "bm25"),
                           rrf_k=cfg.hybrid_rrf_k, candidates=cfg.hybrid_candidates)
  raise ValueError(f"Unknown retriever kind {kind!r}; use faiss, bm25 or hybrid")


def load_services(cfg: ClaimScanConfig = config) -> Services:
  model = load_verifier(cfg)
  retriever = load_retriever(cfg)
  corpus_df = pd.read_parquet(cfg.corpus)
  return Services(model=model, retriever=retriever, corpus_df=corpus_df)


def verify(model, claim, evidence):
  """Run the verifier on (claim, evidence[:3]). Returns (prediction, probs)."""
  encoding = model.encode_text(claim, evidence[:3])

  with torch.no_grad():
    logits = model(encoding['input_ids'], encoding['attention_mask'])
    #collects the highest liklehood outcome
    prediction_idx = torch.argmax(logits, dim=1).item()
    probs = torch.softmax(logits, dim=1)[0]

  return LABEL_MAP[prediction_idx], probs


def fact_check(claim, services: Services):

  evidence_results = services.retriever.search(claim, k=30)

  print(f"Retrieved {len(evidence_results)} results from FAISS")

  evidence = get_evidence(claim, evidence_results, services.corpus_df)
  if not evidence:
    return "NOT ENOUGH INFO", "No relevant evidence found in the corpus."

  prediction, probs = verify(services.model, claim, evidence)
  print(f"\nModel predictions:")
  print(f"  TRUE: {probs[0]:.4f}")
  print(f"  FALSE: {probs[1]:.4f}")
  print(f"  NOT ENOUGH INFO: {probs[2]:.4f}")

  confidence = torch.max(probs).item()
  if confidence < 0.5:
    original_prediction = prediction
    if prediction != "NOT ENOUGH INFO":
      prediction = "NOT ENOUGH INFO"
      print (f"Low confidence ({confidence:.3f}) of original {original_prediction} prediction - considered now to be NOT ENOUGH INFO")

  if prediction == "TRUE":
    response = f"This statement is TRUE supported by the following evidence: {evidence[0]}"

  elif prediction == "FALSE":
    response = f"This statement is FALSE refuted by the following evidence: {evidence[0]}"

  else:
    response = f"There is insufficient evidence to prove this statements truth"

  return prediction, response



def fact_check_cleaned(claim, services: Services, k=30, conf_threshold=0.5, return_evidence=False):
    """Fact check without debug prints - for MLflow experiments and claimscan.eval.

    With ``return_evidence=True`` returns (prediction, ["Page#line", ...]) for the evidence
    sentences sent to the verifier; the decision logic is the same either way.
    """
    evidence_results = services.retriever.search(claim, k=k)
    picked = get_evidence(claim, evidence_results, services.corpus_df, with_ids=True)
    evidence = [text for text, _ in picked]
    evidence_ids = [docid for _, docid in picked]

    if not evidence:
        return ("NOT ENOUGH INFO", []) if return_evidence else "NOT ENOUGH INFO"

    prediction, probs = verify(services.model, claim, evidence)

    confidence = torch.max(probs).item()
    if confidence < conf_threshold and prediction != "NOT ENOUGH INFO":
        prediction = "NOT ENOUGH INFO"

    return (prediction, evidence_ids) if return_evidence else prediction
