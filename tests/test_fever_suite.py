"""FEVER claim suite: the shipped hybrid retriever (FAISS + BM25 without RM3, RRF; the benchmark's
``hybrid_norm3``) end to end on the 300-claim benchmark sample.

Not part of the unit tests: it needs the artifacts (cached hits, gold, corpus, verifier checkpoint), so it is
skipped by a plain `pytest` (tests/conftest.py) and run by naming the file:

    pytest tests/test_fever_suite.py --fever-claims 10   # quick (/fever, the default)
    pytest tests/test_fever_suite.py                     # 300 claims (/fever 300)

Compute: retrieval is NOT rerun. Both sizes read the FAISS and BM25 (no RM3) hits that the benchmark cached for the
300-claim sample (<eval_dir>/cache/hits), fuse them exactly as the benchmark does, and rerun only the
end-to-end stage (fact_check_cleaned on CPU, 2 threads). The 10-claim run is a label-balanced slice of the
same sample, so it reuses those hits too (a fresh 10-claim sample would need a full FAISS index pass).
The corpus sentences those hits point at are cut from the 4 GB corpus once and cached in <eval_dir>/fever_suite.
If the cached hits are missing, they are rebuilt once with the benchmark's own _retrieve stage (~15 min).

One test, test_suite_matches_baseline, checks against tests/data/fever_suite_hybrid_norm3_baseline.json
(benchmark run 20261124-220309, hybrid_norm3):
  - any N: per-claim predicted labels, at most MAX_FLIPS differ
  - 300 claims: also label accuracy / FEVER score within MAX_FLIPS claims, and retrieval metrics exactly
Results are written to <eval_dir>/fever_suite/latest_n<N>.json (read by the /fever skill).
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pytest

BASELINE_FILE = Path(__file__).parent / "data" / "fever_suite_hybrid_norm3_baseline.json"
RETRIEVER = "hybrid_norm3"   # = load_retriever("hybrid") with bm25.rm3: false (the shipped default)
BASES = ("faiss", "bm25_norm3")
MAX_FLIPS = 1  # CPU verifier is deterministic in practice; allow one borderline claim to flip
LABELS = ("TRUE", "FALSE", "NOT ENOUGH INFO")

pytestmark = pytest.mark.fever_suite


def balanced_slice(claims, n: int):
  """First claims of each label in sample order, a third per label (same split rule as select_claims)."""
  per = {lab: n // 3 + (1 if i < n % 3 else 0) for i, lab in enumerate(LABELS)}
  keep, seen = [], {lab: 0 for lab in LABELS}
  for i, lab in enumerate(claims["label_text"]):
    if seen[lab] < per[lab]:
      keep.append(i)
      seen[lab] += 1
  return claims.iloc[keep].reset_index(drop=True)


def _services_factory(cfg, B, all_fused: dict, base: dict, out_dir: Path):
  """Verifier on 2 CPU threads + the corpus rows of every page in the 300 claims' top-k hybrid hits.

  The subset is cut from the 4 GB corpus once (the slow part, minutes on D:) and cached as a small parquet,
  so later runs of either size only read ~22k sentences. Lookups are by (page, line), so a superset of the
  pages gives the same evidence as the benchmark's per-run subset.
  """
  import pandas as pd
  import torch
  from claimscan.evidence.select import index_corpus
  from claimscan.pipeline import Services, load_verifier

  torch.set_num_threads(2)
  path = out_dir / f"corpus_{base['claims_key']}_{RETRIEVER}_k{base['k']}.parquet"
  if path.exists():
    corpus = index_corpus(pd.read_parquet(path))
  else:
    pages = {h["id"] for hl in all_fused.values() for h in hl[:base["k"]]}
    corpus = B.load_corpus_subset(pages, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus.reset_index().to_parquet(path)
  model = load_verifier(cfg)
  return lambda retr: Services(model=model, retriever=retr, corpus_df=corpus)


@pytest.fixture(scope="module")
def suite(request):
  pd = pytest.importorskip("pandas")
  from claimscan.config import config as cfg
  from claimscan.eval import benchmark as B
  from claimscan.eval.gold import load_gold_sets

  n = request.config.getoption("--fever-claims")
  base = json.loads(BASELINE_FILE.read_text())
  if not 1 <= n <= len(base["predictions"]):
    pytest.fail(f"--fever-claims must be 1..{len(base['predictions'])}, got {n}")
  for p in (cfg.verifier_output_dir, cfg.corpus):
    if not Path(p).exists():
      pytest.skip(f"artifact missing: {p}")

  cache = Path(cfg.eval_dir) / "cache"
  claims_file = cache / base["claims_file"]
  if not claims_file.exists():
    B.run_subprocess(["_select", "--n", "300", "--seed", "0", "--split", "validation", "--out", str(claims_file)])
  sample = pd.read_parquet(claims_file)
  assert B.claims_key(sample["id"]) == base["claims_key"], "claim sample differs from the baseline's"

  hits = {}
  for name in BASES:
    path = B._hits_path(cache, name, base["claims_key"], base["candidates"])
    if not path.exists():
      B.run_subprocess(["_retrieve", "--retriever", name, "--claims-file", str(claims_file), "--out", str(path),
                        "--candidates", str(base["candidates"]), "--threads", "2", "--latency-samples", "0"],
                       env_extra={"_JAVA_OPTIONS": "-Xmx2g", "OMP_NUM_THREADS": "2"})
    hits[name], _ = B.read_hits(path)
  assert cfg.hybrid_rrf_k == base["rrf_k"], "hybrid.rrf_k changed since the baseline"
  assert cfg.bm25_rm3 is False, "bm25.rm3 is on: the shipped hybrid (and this baseline) is BM25 without RM3"
  a, b = B.FUSED[RETRIEVER]
  all_fused = B.fuse_hits(hits[a], hits[b], "faiss", "bm25", base["rrf_k"], base["candidates"])

  claims = sample if n == len(sample) else balanced_slice(sample, n)
  ids = [int(i) for i in claims["id"]]
  fused = {i: all_fused[i] for i in ids}
  gold = load_gold_sets(claims["id"])

  t = time.perf_counter()
  retrieval = B.retrieval_metrics(fused, gold)
  out_dir = Path(cfg.eval_dir) / "fever_suite"
  services_factory = _services_factory(cfg, B, all_fused, base, out_dir)
  e2e = B.e2e_stage(claims, {RETRIEVER: fused}, out_dir / f"e2e_n{n}.json", k=base["k"],
                    conf_threshold=base["conf_threshold"], threads=2, services_factory=services_factory,
                    gold=gold, cfg=cfg)[RETRIEVER]
  preds = pd.read_parquet(out_dir / f"e2e_n{n}.predictions.parquet")
  pred = {int(r.id): r.pred for r in preds.itertuples()}
  flips = [{"id": i, "gold": base["predictions"][str(i)]["gold"], "baseline": base["predictions"][str(i)]["pred"],
            "now": pred[i]} for i in ids if pred[i] != base["predictions"][str(i)]["pred"]]

  result = {"timestamp": datetime.now().isoformat(timespec="seconds"), "n_claims": n, "retriever": RETRIEVER,
            "labels": claims["label_text"].value_counts().to_dict(), "retrieval": retrieval, "e2e": e2e,
            "flips_vs_baseline": flips, "baseline": {"retrieval": base["retrieval"], "e2e": base["e2e"]},
            "wall_clock_s": round(time.perf_counter() - t, 1)}
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / f"latest_n{n}.json").write_text(json.dumps(result, indent=2))
  return result, base


def test_suite_matches_baseline(suite):
  """Per-claim predictions (any N), and at the full 300 claims also the numbers: label accuracy and FEVER score
  (within MAX_FLIPS claims) and retrieval metrics (exactly). The numbers are in latest_n<N>.json either way."""
  result, base = suite
  problems = []
  if len(result["flips_vs_baseline"]) > MAX_FLIPS:
    problems.append(f"{len(result['flips_vs_baseline'])} claims changed prediction: {result['flips_vs_baseline']}")
  if result["n_claims"] == len(base["predictions"]):
    tol = MAX_FLIPS / result["n_claims"] + 1e-9
    for m in ("label_accuracy", "fever_score"):
      if abs(result["e2e"][m] - base["e2e"][m]) > tol:
        problems.append(f"{m} {result['e2e'][m]:.3f} vs baseline {base['e2e'][m]:.3f}")
    for m, v in base["retrieval"].items():
      if abs(result["retrieval"][m] - v) > 1e-9:
        problems.append(f"retrieval {m} {result['retrieval'][m]:.4f} vs baseline {v:.4f}")
  assert not problems, "\n".join(problems)


# --- Verifier with gold evidence --------------------------------------------------------------------------------
# Measures the DeBERTa verifier on its own: the suite's TRUE/FALSE claims (NEI claims have no gold evidence) get the
# gold sentences (first FEVER evidence set, at most 3) instead of retrieved ones. Compared with the suite's own
# accuracy on the same claims, read from the baseline file. A measurement, not an accuracy check. It is NOT part of
# /fever (tests/conftest.py deselects it): run it by naming it with -k or -m gold_evidence:
#
#     pytest tests/test_fever_suite.py -k gold_evidence -s
#
#     pytest tests/test_fever_suite.py -k gold_evidence --verifier MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
#     pytest tests/test_fever_suite.py -k gold_evidence --verifier <dir>/trained_model/model.pt   # another checkpoint
#     add --gold-split train for 200 TRUE/FALSE train claims instead of the suite's (a leakage check)
#
# Writes <eval_dir>/fever_suite/latest_oracle.json (ours) or latest_oracle_<model>.json (a HuggingFace NLI model).

NLI_TO_FEVER = {"entailment": "TRUE", "contradiction": "FALSE", "neutral": "NOT ENOUGH INFO"}


def _hf_nli_verifier(name: str):
  """(claim, evidence) -> label with an off-the-shelf NLI model: premise = evidence, hypothesis = claim."""
  import torch
  from transformers import AutoModelForSequenceClassification, AutoTokenizer
  tok = AutoTokenizer.from_pretrained(name)
  model = AutoModelForSequenceClassification.from_pretrained(name).eval()
  labels = {i: NLI_TO_FEVER[l.lower()] for i, l in model.config.id2label.items()}

  def predict(claim, evidence):
    enc = tok(" ".join(evidence), claim, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
      return labels[int(model(**enc).logits.argmax(-1))]
  return predict


def _load_checkpoint(path: str):
  """Our FEVERVerifier from another checkpoint (e.g. a retrained verifier/deberta_v2/trained_model/model.pt)."""
  import torch
  from claimscan.verifier.deberta_verifier import FEVERVerifier
  model = FEVERVerifier()
  model.load_state_dict(torch.load(path, map_location="cpu"))
  return model.eval()


@pytest.fixture(scope="module")
def gold_evidence_run(request):
  pd = pytest.importorskip("pandas")
  import torch
  from claimscan.config import config as cfg
  from claimscan.eval import benchmark as B
  from claimscan.eval.gold import load_claims, load_gold_sets
  from claimscan.evidence.select import _lookup_text, index_corpus
  from claimscan.pipeline import load_verifier, verify

  for p in (cfg.verifier_checkpoint, cfg.corpus):
    if not Path(p).exists():
      pytest.skip(f"artifact missing: {p}")
  base = json.loads(BASELINE_FILE.read_text())
  claims_file = Path(cfg.eval_dir) / "cache" / base["claims_file"]
  if not claims_file.exists():
    pytest.skip(f"claim sample missing: {claims_file} (run the full suite once)")
  sample = pd.read_parquet(claims_file)
  assert B.claims_key(sample["id"]) == base["claims_key"], "claim sample differs from the baseline's"

  split = request.config.getoption("--gold-split")
  if split == "validation":
    gold = load_gold_sets(sample["id"])
    claims = sample[sample["id"].astype(int).isin(gold.keys())]
  else:  # 200 TRUE/FALSE train claims (seen by FEVER-NLI models): 100 per label, seed 0, with gold evidence
    pool = load_claims("train")
    pool = pool[pool["label_text"] != "NOT ENOUGH INFO"].sample(frac=1, random_state=0)
    gold = load_gold_sets(pool["id"])
    pool = pool[pool["id"].isin(gold.keys())]
    claims = pd.concat([g.head(100) for _, g in pool.groupby("label_text")])
    gold = {int(i): gold[int(i)] for i in claims["id"]}
  # gold sentences cut from the 4 GB corpus once (minutes on D:) and cached, like the suite's corpus subset
  out_dir = Path(cfg.eval_dir) / "fever_suite"
  path = out_dir / (f"corpus_{base['claims_key']}_gold.parquet" if split == "validation" else "corpus_train200_gold.parquet")
  if path.exists():
    corpus = index_corpus(pd.read_parquet(path))
  else:
    corpus = B.load_corpus_subset({d.rsplit("#", 1)[0] for sets in gold.values() for d in sets[0][:3]}, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus.reset_index().to_parquet(path)
  torch.set_num_threads(2)
  name = request.config.getoption("--verifier")
  if name == "ours" or name.endswith(".pt"):
    model = load_verifier(cfg) if name == "ours" else _load_checkpoint(name)
    predict = lambda claim, evidence: verify(model, claim, evidence)[0]
  else:
    predict = _hf_nli_verifier(name)

  t = time.perf_counter()
  rows = []
  for c in claims.itertuples():
    cid = int(c.id)
    docids = gold[cid][0][:3]
    hits = [{"id": d.rsplit("#", 1)[0], "line": int(d.rsplit("#", 1)[1])} for d in docids]
    evidence = [s for s in (_lookup_text(corpus, h["id"], h["line"]) for h in hits) if s is not None]
    pred = predict(c.claim, evidence) if evidence else None
    rows.append({"id": cid, "gold": c.label_text, "pred_gold_evidence": pred,
                 "pred_suite": base["predictions"][str(cid)]["pred"] if split == "validation" else None, "evidence": docids,
                 "sentences_found": len(evidence)})

  scored = [r for r in rows if r["pred_gold_evidence"] is not None]
  acc = lambda key: round(sum(r[key] == r["gold"] for r in scored) / len(scored), 4)
  result = {"timestamp": datetime.now().isoformat(timespec="seconds"), "verifier": name, "split": split,
            "claims": len(rows), "claims_scored": len(scored),
            "accuracy_gold_evidence": acc("pred_gold_evidence"),
            "accuracy_suite_retrieved_evidence": acc("pred_suite") if split == "validation" else None,
            "wall_clock_s": round(time.perf_counter() - t, 1), "per_claim": rows}
  tag = Path(name).parent.parent.name if name.endswith(".pt") else name.replace("/", "__")
  out_name = "latest_oracle.json" if name == "ours" else f"latest_oracle_{tag}.json"
  if split != "validation":
    out_name = out_name.replace(".json", f"_{split}.json")
  (out_dir / out_name).write_text(json.dumps(result, indent=2))
  print(f"\nverifier on {len(scored)} TRUE/FALSE {split} claims: gold evidence {result['accuracy_gold_evidence']:.1%}")
  return result


@pytest.mark.gold_evidence
def test_verifier_with_gold_evidence(gold_evidence_run):
  r = gold_evidence_run
  assert r["claims_scored"] > 0.9 * r["claims"], f"gold sentences missing from the corpus for many claims: {r}"
