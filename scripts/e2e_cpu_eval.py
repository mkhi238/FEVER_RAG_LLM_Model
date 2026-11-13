"""CPU-only end-to-end check of the ClaimScan pipeline on a small set of claims.

    python scripts/e2e_cpu_eval.py                       # 6 hand claims + 3 FEVER val claims per label
    python scripts/e2e_cpu_eval.py --per-label 5 --retrievers faiss hybrid --no-rerank

Safe to run while the GPU is busy training: CUDA is hidden, torch uses 4 threads, the
corpus is NOT loaded whole (only the parquet rows of each claim's retrieved pages are read),
and FAISS is memory-mapped. Peak RAM is about 5 GB, so it refuses to start below --min-free-gb.

For every claim and retriever it records: the prediction from the repo's own
pipeline.fact_check_cleaned (decision logic unchanged), the verifier probabilities, the three
evidence sentences, whether the gold sentences were in the top-k hits, and timings. For FEVER
claims it also feeds the GOLD evidence straight into the verifier (verifier upper bound).
Results go to <artifacts_root>/eval/e2e/e2e_<timestamp>.json plus a printed scorecard.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # before torch is imported anywhere

HAND_CLAIMS = [  # (claim, expected label)
  ("Barack Obama was born in Hawaii", "TRUE"),
  ("Tokyo is the capital of Japan", "TRUE"),
  ("The Eiffel Tower is in London", "FALSE"),
  ("George Orwell wrote 1984", "TRUE"),
  ("Michael Jordan played for the Chicago Bulls", "TRUE"),
  ("The weather is nice today", "NOT ENOUGH INFO"),
]


def mem_available_gb() -> float:
  with open("/proc/meminfo") as f:
    for line in f:
      if line.startswith("MemAvailable:"):
        return int(line.split()[1]) / 1024 / 1024
  return float("nan")


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--per-label", type=int, default=3, help="FEVER validation claims per label (default 3)")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--retrievers", nargs="+", default=["faiss", "hybrid"], choices=["faiss", "bm25", "hybrid"])
  p.add_argument("--k", type=int, default=30, help="hits retrieved per claim")
  p.add_argument("--no-hand", action="store_true", help="skip the 6 hand-written claims")
  p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder reranker check")
  p.add_argument("--claim", action="append", default=[], help="extra claim text (repeatable; no gold label)")
  p.add_argument("--threads", type=int, default=4)
  p.add_argument("--min-free-gb", type=float, default=6.0, help="refuse to start with less MemAvailable")
  p.add_argument("--out", default=None, help="results JSON (default <artifacts_root>/eval/e2e/e2e_<time>.json)")
  return p.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  free = mem_available_gb()
  if free < args.min_free_gb:
    sys.exit(f"Only {free:.1f} GB RAM available (< {args.min_free_gb} GB). Not starting; free memory or lower --min-free-gb.")

  import pyarrow.parquet as pq
  import torch
  torch.set_num_threads(args.threads)
  assert not torch.cuda.is_available(), "GPU is visible; refusing to run"

  from claimscan import pipeline
  from claimscan.config import config
  from claimscan.eval.gold import load_claims, load_gold_sets, norm_title
  from claimscan.evidence.select import _lookup_text, get_evidence, index_corpus

  t0 = time.time()

  def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

  def corpus_rows(pages):
    t = pq.read_table(str(config.corpus), columns=["id", "line", "text"], filters=[("id", "in", sorted(set(pages)))])
    return index_corpus(t.to_pandas())

  class Cached:
    """Lets fact_check_cleaned reuse hits already fetched (so retrieval is timed and run once)."""
    def __init__(self, hits):
      self.hits = hits

    def search(self, q, k=30):
      return self.hits[:k]

  # ---------- claims: (fever_id or None, text, expected label or None) ----------
  claims = [] if args.no_hand else [(None, c, lab) for c, lab in HAND_CLAIMS]
  claims += [(None, c, None) for c in args.claim]
  if args.per_label > 0:
    val = load_claims("validation")
    for lab in ["TRUE", "FALSE", "NOT ENOUGH INFO"]:
      sub = val[val["label_text"] == lab].sample(args.per_label, random_state=args.seed)
      claims += [(int(r.id), r.claim, lab) for r in sub.itertuples()]
  gold = load_gold_sets([cid for cid, _, _ in claims if cid is not None])
  log(f"{len(claims)} claims; retrievers {args.retrievers}; RAM available {free:.1f} GB")

  # ---------- load models ----------
  log("loading verifier (CPU)")
  model = pipeline.load_verifier()
  retrievers, load_errors = {}, {}
  faiss_r = None
  if {"faiss", "hybrid"} & set(args.retrievers):
    log("loading FAISS (mmap)")
    faiss_r = pipeline.load_retriever(kind="faiss")
    if "faiss" in args.retrievers:
      retrievers["faiss"] = faiss_r
  if {"bm25", "hybrid"} & set(args.retrievers):
    try:
      import jnius_config
      jnius_config.add_options("-Xmx2g")
      from claimscan.retrieval.bm25_retriever import BM25Retriever
      from claimscan.retrieval.hybrid_retriever import HybridRetriever
      log("loading BM25 (Lucene, -Xmx2g)")
      bm25 = BM25Retriever(config.bm25_index_dir, config.bm25_k1, config.bm25_b, config.bm25_rm3)
      if "bm25" in args.retrievers:
        retrievers["bm25"] = bm25
      if "hybrid" in args.retrievers:
        retrievers["hybrid"] = HybridRetriever(faiss_r, bm25, rrf_k=config.hybrid_rrf_k,
                                               candidates=config.hybrid_candidates)
    except Exception as e:  # report and carry on with what loaded
      load_errors["bm25/hybrid"] = f"{type(e).__name__}: {e}"
      log(f"BM25/HYBRID FAILED TO LOAD: {load_errors['bm25/hybrid']}")
  reranker = None
  if not args.no_rerank:
    try:
      from claimscan.rerank import load_reranker
      from claimscan.rerank.base import rerank_hits
      reranker = load_reranker(backend="cross_encoder")
    except Exception as e:
      load_errors["reranker"] = f"{type(e).__name__}: {e}"
      log(f"RERANKER FAILED: {load_errors['reranker']}")
  log("models loaded")

  # ---------- run ----------
  rows = []
  for cid, claim, lab in claims:
    print("\n" + "=" * 100)
    print(f"CLAIM: {claim}" + (f"   [FEVER {cid}]" if cid else "") + (f"   expected={lab}" if lab else ""))
    gold_ids = {d for s in gold.get(cid, []) for d in s} if cid is not None else set()
    row = dict(claim=claim, fever_id=cid, expected=lab, gold_evidence=sorted(gold_ids), runs={})
    try:
      for name, r in retrievers.items():
        t = time.time(); hits = r.search(claim, k=args.k); t_ret = time.time() - t
        t = time.time(); corp = corpus_rows([h["id"] for h in hits]); t_corp = time.time() - t
        t = time.time()
        svc = pipeline.Services(model=model, retriever=Cached(hits), corpus_df=corp)
        pred, ev_ids = pipeline.fact_check_cleaned(claim, svc, k=args.k, return_evidence=True)
        ev = get_evidence(claim, hits, corp)
        probs = [round(x, 4) for x in pipeline.verify(model, claim, ev)[1].tolist()] if ev else None
        t_ver = time.time() - t
        hit_ids = {f"{norm_title(h['id'])}#{h['line']}" for h in hits}
        run = dict(pred=pred, correct=(pred == lab) if lab else None, probs=probs,
                   evidence=[dict(id=i, text=x[:300]) for i, x in zip(ev_ids, ev)],
                   gold_in_hits=[len(gold_ids & hit_ids), len(gold_ids)] if gold_ids else None,
                   top5=[(h["id"], h["line"], round(h["score"], 4)) for h in hits[:5]],
                   secs=dict(retrieve=round(t_ret, 1), corpus=round(t_corp, 1), verify=round(t_ver, 1)))
        if name == "faiss" and reranker is not None:
          t = time.time()
          rr = rerank_hits(reranker, claim, hits, corp, top_n=3)
          run["rerank_top3"] = [(h["id"], h["line"], round(h["rerank_score"], 2)) for h in rr]
          run["secs"]["rerank"] = round(time.time() - t, 1)
        row["runs"][name] = run
        print(f"  [{name}] {pred}" + (f" ({'OK' if run['correct'] else 'WRONG'})" if lab else "")
              + f"  probs(T/F/NEI)={probs}  gold-in-hits={run['gold_in_hits']}  secs={run['secs']}")
        for e in run["evidence"]:
          print(f"      [{e['id']}] {e['text'][:140]}")
      if gold_ids:  # verifier upper bound on the gold sentences
        pairs = [d.rsplit("#", 1) for d in sorted(gold_ids)]
        gc = corpus_rows([p for p, _ in pairs])
        texts = [x for x in (_lookup_text(gc, p, int(l)) for p, l in pairs) if x][:3]
        if texts:
          gp, gprobs = pipeline.verify(model, claim, texts)
          row["verifier_on_gold"] = dict(pred=gp, correct=gp == lab, probs=[round(x, 4) for x in gprobs.tolist()])
          print(f"  [verifier on GOLD evidence] {gp} ({'OK' if gp == lab else 'WRONG'})  probs={row['verifier_on_gold']['probs']}")
    except Exception as e:
      traceback.print_exc()
      row["error"] = f"{type(e).__name__}: {e}"
    rows.append(row)

  # ---------- scorecard ----------
  def frac(xs):
    xs = [x for x in xs if x is not None]
    return f"{sum(xs)}/{len(xs)}" if xs else "-"

  card = {}
  for name in retrievers:
    runs = [(r, r["runs"].get(name)) for r in rows if name in r["runs"]]
    card[name] = dict(
      hand=frac([run["correct"] for r, run in runs if r["fever_id"] is None]),
      fever=frac([run["correct"] for r, run in runs if r["fever_id"] is not None]),
      **{f"fever_{lab}": frac([run["correct"] for r, run in runs if r["fever_id"] is not None and r["expected"] == lab])
         for lab in ["TRUE", "FALSE", "NOT ENOUGH INFO"]},
      gold_in_hits=frac([run["gold_in_hits"][0] > 0 if run["gold_in_hits"] else None for r, run in runs]),
      mean_secs=round(sum(sum(run["secs"].values()) for _, run in runs) / max(len(runs), 1), 1),
    )
  card["verifier_on_gold"] = frac([r["verifier_on_gold"]["correct"] for r in rows if "verifier_on_gold" in r])
  errors = [r for r in rows if "error" in r]

  print("\n\nSCORECARD")
  for name in retrievers:
    print(f"  {name:7} {card[name]}")
  print(f"  verifier on gold evidence: {card['verifier_on_gold']}")
  if errors or load_errors:
    print(f"  ERRORS: {len(errors)} claims failed; load errors: {load_errors}")

  out = args.out or (config.eval_dir / "e2e" / f"e2e_{datetime.now():%Y%m%d_%H%M%S}.json")
  os.makedirs(os.path.dirname(str(out)), exist_ok=True)
  with open(out, "w") as f:
    json.dump(dict(args=vars(args), scorecard=card, load_errors=load_errors, claims=rows,
                   total_secs=round(time.time() - t0, 1)), f, indent=1, default=str)
  log(f"done; results in {out}")


if __name__ == "__main__":
  main()
