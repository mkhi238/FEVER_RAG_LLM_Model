"""Retrieve -> score evidence -> verify, for one claim.

Was rag/Inference/test_rag.py. The model, retriever and corpus used to be module
globals set in ``__main__`` (or patched in by run_experiments.py); they now travel in
a ``Services`` object from ``load_services()``. Decision logic is unchanged.
The demo claim list moved to scripts/rag_demo.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import torch

from claimscan.config import ClaimScanConfig, config
from claimscan.evidence.select import _lookup_text, get_evidence
from claimscan.verifier.model import FEVERVerifier

LABEL_MAP = {0: "TRUE", 1: "FALSE", 2: "NOT ENOUGH INFO"}


@dataclass
class Services:
  model: FEVERVerifier
  retriever: "FAISSRetriever | BM25Retriever | HybridRetriever"
  corpus_df: "pd.DataFrame | SentenceStore"   # SentenceStore: per-request lookup, used by the API


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



@dataclass
class Verdict:
  label: str
  confidence: float                 # max verifier probability (0.0 when no evidence was found)
  probs: dict | None                # {"TRUE": p, "FALSE": p, "NOT ENOUGH INFO": p}, None without evidence
  evidence: list                    # [(text, "Page#line"), ...] sent to the verifier, best first
  downgraded: bool = False          # TRUE/FALSE turned into NEI by the confidence threshold
  # every retrieved hit, best first: {"rank", "id", "line", "docid", "score", "text"} (text None if not in the corpus).
  # This is what an LLM layer reads to synthesise over the top k instead of the verifier's 3 sentences.
  hits: list = field(default_factory=list)


def fact_check_detailed(claim, services: Services, k=30, conf_threshold=0.5) -> Verdict:
  """The evaluated decision logic, with everything the API needs to show its work.

  ``services.corpus_df`` is a corpus DataFrame (raw or ``index_corpus``-ed) or anything with a
  ``for_hits(hits)`` method returning one, such as ``claimscan.evidence.store.SentenceStore``.
  """
  evidence_results = services.retriever.search(claim, k=k)
  corpus = services.corpus_df
  if hasattr(corpus, "for_hits"):
    corpus = corpus.for_hits(evidence_results)
  picked = get_evidence(claim, evidence_results, corpus, with_ids=True)
  evidence = [text for text, _ in picked]
  hits = [{"rank": r, "id": h["id"], "line": int(h["line"]), "docid": f"{h['id']}#{h['line']}",
           "score": float(h.get("score", 0.0)), "text": _lookup_text(corpus, h["id"], h["line"])}
          for r, h in enumerate(evidence_results)]

  if not evidence:
    return Verdict("NOT ENOUGH INFO", 0.0, None, [], hits=hits)

  prediction, probs = verify(services.model, claim, evidence)

  confidence = torch.max(probs).item()
  downgraded = confidence < conf_threshold and prediction != "NOT ENOUGH INFO"
  if downgraded:
    prediction = "NOT ENOUGH INFO"

  return Verdict(prediction, confidence, {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}, picked, downgraded,
                 hits=hits)


def fact_check_cleaned(claim, services: Services, k=30, conf_threshold=0.5, return_evidence=False):
    """Fact check without debug prints - for MLflow experiments and claimscan.eval.

    With ``return_evidence=True`` returns (prediction, ["Page#line", ...]) for the evidence
    sentences sent to the verifier; the decision logic is the same either way.
    """
    v = fact_check_detailed(claim, services, k=k, conf_threshold=conf_threshold)
    return (v.label, [docid for _, docid in v.evidence]) if return_evidence else v.label
