"""End-to-end FEVER metric on the fact-check pipeline.

Uses the FEVER metric from huggingface/evaluate PR #710 (vendored, unchanged, in
metrics/fever/), loaded with evaluate.load(<local dir>) so it works offline and is
pinned. This adapter turns our labels / gold / pipeline output into its inputs and
applies two conventions of the official FEVER scorer that the metric leaves to the caller:

  - evidence is compared as "Page#line" ids (not sentence text), and only the first
    MAX_EVIDENCE (5) predicted sentences count;
  - NOT ENOUGH INFO claims get the gold evidence sets [[]], so a correct NEI label
    counts toward fever_score (the metric's loop gives no credit when there are no sets).

Note the metric's evidence precision includes evidence predicted for NEI claims, and its
recall is against the union of all gold sets, so recall is low for claims with several
alternative sets. fever_score and label_accuracy are the headline numbers.

Run: python -m claimscan.eval.fever_eval --max-claims 200 [--retriever hybrid]
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

LABEL_TO_FEVER = {"TRUE": "SUPPORTED", "FALSE": "REFUTED", "NOT ENOUGH INFO": "NOT ENOUGH INFO"}
MAX_EVIDENCE = 5
METRIC_DIR = Path(__file__).resolve().parent / "metrics" / "fever"


def load_fever_metric():
  import evaluate
  return evaluate.load(str(METRIC_DIR))


def to_prediction(label: str, evidence_ids) -> dict:
  from claimscan.eval.gold import norm_title
  ids = []
  for d in list(evidence_ids)[:MAX_EVIDENCE]:
    page, _, line = d.rpartition("#")
    ids.append(f"{norm_title(page)}#{line}")
  return {"label": LABEL_TO_FEVER[label], "evidence": ids}


def to_reference(label: str, evidence_sets) -> dict:
  if label == "NOT ENOUGH INFO" or not evidence_sets:
    evidence_sets = [[]]
  return {"label": LABEL_TO_FEVER[label], "evidence_sets": [list(s) for s in evidence_sets]}


def fever_scores(predictions: list[dict], references: list[dict], metric=None) -> dict:
  """predictions / references already in metric format (see to_prediction / to_reference)."""
  metric = metric or load_fever_metric()
  return metric.compute(predictions=predictions, references=references)


def evaluate_pipeline(max_claims=200, retriever=None, k=30, conf_threshold=0.5, split="validation",
                      seed=0, log_every=25, services=None):
  """Run fact_check_cleaned on a random sample of FEVER claims and score it."""
  from claimscan.config import config
  from claimscan.eval.gold import load_claims, load_gold_sets
  from claimscan.pipeline import Services, fact_check_cleaned, load_retriever, load_verifier

  claims = load_claims(split)
  if max_claims is not None and max_claims < len(claims):
    claims = claims.sample(n=max_claims, random_state=seed).reset_index(drop=True)
  gold_sets = load_gold_sets(claims["id"])
  print(f"[fever] {len(claims):,} {split} claims "
        f"({claims['label_text'].value_counts().to_dict()})")

  if services is None:
    from claimscan.evidence.store import SentenceStore
    print("[fever] loading verifier, retriever, sentence store...")
    corpus = SentenceStore(config.sentence_store)
    from claimscan.rerank import load_reranker
    services = Services(model=load_verifier(config), retriever=load_retriever(config, retriever),
                        corpus_df=corpus, reranker=load_reranker(config))

  metric = load_fever_metric()
  preds, refs = [], []
  t0 = time.perf_counter()
  for i, row in enumerate(claims.itertuples(index=False), start=1):
    label, ev_ids = fact_check_cleaned(row.claim, services, k=k, conf_threshold=conf_threshold,
                                       return_evidence=True)
    preds.append(to_prediction(label, ev_ids))
    refs.append(to_reference(row.label_text, gold_sets.get(int(row.id), [])))
    if i % log_every == 0:
      partial = metric.compute(predictions=preds, references=refs)
      print(f"[fever] {i}/{len(claims)}  fever_score={partial['fever_score']:.3f}  "
            f"label_acc={partial['label_accuracy']:.3f}  {time.perf_counter() - t0:.0f}s")

  results = metric.compute(predictions=preds, references=refs)
  print("[fever] " + "  ".join(f"{k}={v:.3f}" for k, v in results.items()))
  return results


def main(argv=None):
  ap = argparse.ArgumentParser(description="FEVER metric (label accuracy, FEVER score, evidence P/R/F1) on the pipeline")
  ap.add_argument("--max-claims", type=int, default=200)
  ap.add_argument("--retriever", choices=["faiss", "bm25", "hybrid"], default=None,
                  help="default: retriever.kind from config")
  ap.add_argument("--k", type=int, default=30)
  ap.add_argument("--conf-threshold", type=float, default=0.5)
  ap.add_argument("--split", default="validation")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args(argv)
  random.seed(args.seed)
  evaluate_pipeline(args.max_claims, args.retriever, args.k, args.conf_threshold, args.split, args.seed)


if __name__ == "__main__":
  main()
