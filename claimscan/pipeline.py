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
  retriever: "FAISSRetriever"
  corpus_df: pd.DataFrame


def load_verifier(cfg: ClaimScanConfig = config) -> FEVERVerifier:
  model = FEVERVerifier()
  # map_location: checkpoint was saved on GPU; inference here runs on CPU as before
  model.load_state_dict(torch.load(cfg.verifier_checkpoint, map_location="cpu"))
  model.eval()
  return model


def load_retriever(cfg: ClaimScanConfig = config):
  # imported here so the verifier can be used without faiss / sentence-transformers installed
  from claimscan.retrieval.faiss_retriever import FAISSRetriever
  return FAISSRetriever(
    cfg.retriever_index_dir,
    cfg.retriever_model,
    index_file=cfg.retriever_index_file,
    docids_file=cfg.retriever_docids_file,
    nprobe=cfg.retriever_nprobe,
  )


def load_services(cfg: ClaimScanConfig = config) -> Services:
  model = load_verifier(cfg)
  retriever = load_retriever(cfg)
  corpus_df = pd.read_parquet(cfg.corpus)
  return Services(model=model, retriever=retriever, corpus_df=corpus_df)


def verify(model, claim, evidence):
  """Run the verifier on (claim, evidence[:3]). Returns (prediction, probs)."""
  #.item(): Converts a single-value tensor to a Python number, torch.argmax(logits, dim=1) returns tensor [2], .item returns 2
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


#1. Retrieval: Get k=10 candidates from FAISS
#2. Quality scoring: Score all 10, sort by quality
#3. Select top 3: Pick the 3 highest quality pieces
#4. Model inference: Send those 3 to DeBERTa

#If k=10 and the right evidence is ranked #15 in FAISS, you never see it. Your quality scorer can't pick what it never receives.
#If k=50 and the right evidence is at #15, your quality scorer gets it, scores it highly, and includes it in the top 3 sent to the model.


def fact_check_cleaned(claim, services: Services, k=30, conf_threshold=0.5):
    """Fact check without debug prints - for MLflow experiments"""
    evidence_results = services.retriever.search(claim, k=k)
    evidence = get_evidence(claim, evidence_results, services.corpus_df)

    if not evidence:
        return "NOT ENOUGH INFO"

    prediction, probs = verify(services.model, claim, evidence)

    confidence = torch.max(probs).item()
    if confidence < conf_threshold and prediction != "NOT ENOUGH INFO":
        prediction = "NOT ENOUGH INFO"

    return prediction
