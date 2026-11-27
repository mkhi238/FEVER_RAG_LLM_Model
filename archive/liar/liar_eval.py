"""Out-of-domain eval of the fact-check pipeline on LIAR (PolitiFact) statements.

LIAR has no evidence ids, so this reports label metrics only (accuracy, macro-F1 over the gold
classes, how often the system abstains with NOT ENOUGH INFO) plus how often the retrieved
Wikipedia evidence looks related to the claim at all. Label mapping: claimscan/data/liar.py.

Two evidence modes:
  retrieved      the normal pipeline: retrieve -> evidence scoring -> verify (fact_check_cleaned,
                 decision logic unchanged). Memory-light: hits are fetched first, the retriever
                 is freed, and only the corpus rows of the hit pages are read from parquet
                 (not the 4 GB corpus).
  justification  LIAR-PLUS: PolitiFact's justification text as "oracle" evidence, top-3
                 sentences by the same quality scorer, then the same verify + threshold.
                 Shows what the verifier does when the evidence is on topic.

Relevance heuristic (retrieved mode): a claim "has related evidence" when some evidence
sentence sent to the verifier (plus its page title) shares >= RELATED_MIN_SHARED content
words with the claim. Lexical, so an upper bound on real relevance; check per-claim output.

Run: python -m claimscan.eval.liar_eval --max-claims 100 [--evidence justification] [--retriever bm25]
Per-claim rows: <artifacts_root>/liar/eval/liar_<mode>_<retriever>_<n>.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time

import pandas as pd

LABELS = ["TRUE", "FALSE", "NOT ENOUGH INFO"]
RELATED_MIN_SHARED = 2
MIN_JUSTIFICATION_OVERLAP = 0.2   # share of the claim's content words found in its justification
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


class PrecomputedRetriever:
  """Serves hits fetched earlier, so the real retriever can be freed before the verifier loads."""

  def __init__(self, hits_by_claim: dict):
    self.hits = hits_by_claim

  def search(self, query, k=50):
    return self.hits[query][:k]


def shared_content_words(claim: str, text: str, page: str | None = None) -> int:
  from claimscan.neurosymbolic_layer import content_words, title_words
  words = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", text)} | title_words(page)
  return len(content_words(claim) & words)


def _content_words(text: str) -> set:
  from claimscan.neurosymbolic_layer import content_words
  return content_words(text)


def split_sentences(text: str) -> list[str]:
  return [s.strip() for s in _SENT.split(text or "") if len(s.split()) >= 3]


def label_metrics(gold: list[str], pred: list[str]) -> dict:
  """accuracy, macro-F1 over the gold classes (NEI predictions count as wrong for TRUE/FALSE
  gold), coverage (share not NEI) and accuracy on the claims it did decide."""
  from sklearn.metrics import f1_score
  classes = [l for l in LABELS if l in set(gold)]
  n = len(gold)
  correct = sum(g == p for g, p in zip(gold, pred))
  decided = [(g, p) for g, p in zip(gold, pred) if p != "NOT ENOUGH INFO"]
  return {
    "n": n,
    "accuracy": correct / n if n else 0.0,
    "macro_f1": float(f1_score(gold, pred, labels=classes, average="macro", zero_division=0)) if n else 0.0,
    "coverage": len(decided) / n if n else 0.0,
    "decided_accuracy": sum(g == p for g, p in decided) / len(decided) if decided else 0.0,
    "pred_counts": {l: pred.count(l) for l in LABELS},
    "gold_counts": {l: gold.count(l) for l in LABELS},
    "majority_baseline": max(gold.count(l) for l in classes) / n if n else 0.0,
  }


def _corpus_for_pages(pages: set[str]):
  """Only the corpus rows of ``pages``, indexed by (id, line). Streams the parquet by row group."""
  import pyarrow as pa
  import pyarrow.compute as pc
  import pyarrow.parquet as pq
  from claimscan.config import config
  from claimscan.evidence.select import index_corpus
  # One batch at a time, single-threaded: pyarrow's dataset scanner reads ahead several row
  # groups in parallel and went past 4.5 GB RSS on this machine.
  wanted = pa.array(sorted(pages))
  parts = []
  for batch in pq.ParquetFile(str(config.corpus)).iter_batches(
      batch_size=500_000, columns=["id", "line", "text"], use_threads=False):
    mask = pc.is_in(batch.column(0), value_set=wanted)
    if pc.any(mask).as_py():
      parts.append(batch.filter(mask))
  return index_corpus(pa.Table.from_batches(parts).to_pandas())


def _retrieve_all(claims: list[str], kind: str | None, k: int) -> dict:
  from claimscan.config import config
  from claimscan.pipeline import load_retriever
  retriever = load_retriever(config, kind)
  t0 = time.perf_counter()
  if hasattr(retriever, "search_many"):
    hits = dict(zip(claims, retriever.search_many(claims, k=k)))
  else:
    hits = {c: retriever.search(c, k=k) for c in claims}
  print(f"[liar] retrieved {len(claims)} claims in {time.perf_counter() - t0:.0f}s")
  del retriever
  gc.collect()
  return hits


def evaluate_liar(max_claims=100, split="test", label_map=None, evidence="retrieved", retriever=None,
                  k=30, conf_threshold=0.5, seed=0, out_dir=None):
  import torch
  from claimscan.config import config
  from claimscan.data.liar import liar_settings, load_liar
  from claimscan.neurosymbolic_layer import score_evidence_quality
  from claimscan.pipeline import Services, fact_check_cleaned, load_verifier, verify

  df = load_liar(split, label_map, with_justification=(evidence == "justification"))
  if evidence == "justification":
    # LIAR-PLUS (master) pairs many statements with another statement's justification: share of
    # rows whose justification has < 20% of the claim's content words: train 87%, val 18%, test 52%.
    df = df[df["justification"].str.len() > 0]
    aligned = df.apply(lambda r: shared_content_words(r.claim, r.justification) / max(
      len(_content_words(r.claim)), 1) >= MIN_JUSTIFICATION_OVERLAP, axis=1)
    print(f"[liar] {int((~aligned).sum())}/{len(df)} justifications look misaligned (dropped)")
    df = df[aligned]
  df = df.drop_duplicates("claim")
  if max_claims is not None and max_claims < len(df):
    df = df.sample(n=max_claims, random_state=seed)
  df = df.reset_index(drop=True)
  claims = df["claim"].tolist()
  print(f"[liar] {len(df)} {split} claims, evidence={evidence}, gold {df['label_text'].value_counts().to_dict()}")

  rows = []
  if evidence == "retrieved":
    # hits cached on disk: retrieval is the slow part (cold BM25+RM3 ~6-11 s/claim)
    cache = liar_settings()["dir"] / "eval" / f"hits_{retriever or config.retriever_kind}_{split}_s{seed}_n{len(df)}_k{k}.json"
    if cache.exists():
      hits = json.loads(cache.read_text(encoding="utf-8"))
    else:
      hits = _retrieve_all(claims, retriever, k)
      cache.parent.mkdir(parents=True, exist_ok=True)
      cache.write_text(json.dumps(hits), encoding="utf-8")
    pages = {h["id"] for hs in hits.values() for h in hs}
    t0 = time.perf_counter()
    corpus = _corpus_for_pages(pages)
    print(f"[liar] corpus rows for {len(pages):,} hit pages: {len(corpus):,} ({time.perf_counter() - t0:.0f}s)")
    services = Services(model=load_verifier(config), retriever=PrecomputedRetriever(hits), corpus_df=corpus)
    for row in df.itertuples(index=False):
      pred, ev_ids = fact_check_cleaned(row.claim, services, k=k, conf_threshold=conf_threshold,
                                        return_evidence=True)
      ev = []
      for d in ev_ids:
        page, _, line = d.rpartition("#")
        text = corpus.loc[(page, int(line))]["text"]
        text = text if isinstance(text, str) else text.iloc[0]
        ev.append({"docid": d, "text": text, "shared": shared_content_words(row.claim, text, page)})
      rows.append({"json_id": row.json_id, "claim": row.claim, "liar_label": row.liar_label,
                   "gold": row.label_text, "pred": pred, "evidence": ev,
                   "related": any(e["shared"] >= RELATED_MIN_SHARED for e in ev)})
  else:
    model = load_verifier(config)
    for row in df.itertuples(index=False):
      sents = split_sentences(row.justification)
      scored = sorted(((score_evidence_quality(row.claim, s, i), s) for i, s in enumerate(sents)), reverse=True)
      ranked = [s for _, s in scored[:3]]
      pred, probs = verify(model, row.claim, ranked)
      if torch.max(probs).item() < conf_threshold and pred != "NOT ENOUGH INFO":
        pred = "NOT ENOUGH INFO"
      rows.append({"json_id": row.json_id, "claim": row.claim, "liar_label": row.liar_label,
                   "gold": row.label_text, "pred": pred, "evidence": [{"text": s} for s in ranked]})

  results = label_metrics([r["gold"] for r in rows], [r["pred"] for r in rows])
  if evidence == "retrieved":
    results["with_any_evidence"] = sum(bool(r["evidence"]) for r in rows) / len(rows)
    results["with_related_evidence"] = sum(r["related"] for r in rows) / len(rows)
  by_label = pd.DataFrame(rows).groupby("liar_label")["pred"].value_counts().unstack(fill_value=0)
  print(by_label.to_string())
  print("[liar] " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in results.items()))

  out_dir = out_dir or liar_settings()["dir"] / "eval"
  out_dir.mkdir(parents=True, exist_ok=True)
  rname = (retriever or config.retriever_kind) if evidence == "retrieved" else "none"
  tag = f"liar_{evidence}_{rname}_{len(rows)}"
  with open(out_dir / f"{tag}.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
      f.write(json.dumps(r, ensure_ascii=False) + "\n")
  with open(out_dir / f"{tag}_summary.json", "w", encoding="utf-8") as f:
    json.dump({"split": split, "label_map": label_map, "k": k, "conf_threshold": conf_threshold,
               "seed": seed, **results}, f, indent=2)
  print(f"[liar] per-claim rows -> {out_dir / tag}.jsonl")
  return results


def main(argv=None):
  ap = argparse.ArgumentParser(description="Out-of-domain label accuracy / macro-F1 on LIAR")
  ap.add_argument("--max-claims", type=int, default=100)
  ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
  ap.add_argument("--label-map", default=None, help="binary | binary_strict | three_way (default: config)")
  ap.add_argument("--evidence", default="retrieved", choices=["retrieved", "justification"])
  ap.add_argument("--retriever", choices=["faiss", "bm25", "hybrid"], default=None,
                  help="default: retriever.kind from config")
  ap.add_argument("--k", type=int, default=30)
  ap.add_argument("--conf-threshold", type=float, default=0.5)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args(argv)
  evaluate_liar(args.max_claims, args.split, args.label_map, args.evidence, args.retriever,
                args.k, args.conf_threshold, args.seed)


if __name__ == "__main__":
  main()
