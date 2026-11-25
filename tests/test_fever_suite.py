"""FEVER claim suite: the shipped hybrid retriever (FAISS + BM25 without RM3, RRF; the benchmark's
``hybrid_norm3``) end to end on the 300-claim benchmark sample.

Not part of the unit tests: it needs the artifacts (cached hits, gold, corpus, verifier checkpoint), so it is
deselected by default (``-m "not fever_suite"`` in pyproject) and run explicitly:

    python -m pytest -m fever_suite tests/test_fever_suite.py                    # 300 claims (/fever-300)
    python -m pytest -m fever_suite tests/test_fever_suite.py --fever-claims 10  # fast (/fever-10)

Compute: retrieval is NOT rerun. Both sizes read the FAISS and BM25 (no RM3) hits that the benchmark cached for the
300-claim sample (<eval_dir>/cache/hits), fuse them exactly as the benchmark does, and rerun only the
end-to-end stage (fact_check_cleaned on CPU, 2 threads). The 10-claim run is a label-balanced slice of the
same sample, so it reuses those hits too (a fresh 10-claim sample would need a full FAISS index pass).
The corpus sentences those hits point at are cut from the 4 GB corpus once and cached in <eval_dir>/fever_suite.
If the cached hits are missing, they are rebuilt once with the benchmark's own _retrieve stage (~15 min).

Checked against tests/data/fever_suite_hybrid_norm3_baseline.json (benchmark run 20261124-220309, hybrid_norm3):
  - per-claim predicted labels: at most MAX_FLIPS differ
  - 300 claims: retrieval metrics exactly, and the FEVER metric within MAX_FLIPS claims
Results are written to <eval_dir>/fever_suite/latest_n<N>.json (read by the /fever-300 and /fever-10 skills).
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


def test_predictions_match_baseline(suite):
  result, _ = suite
  assert len(result["flips_vs_baseline"]) <= MAX_FLIPS, result["flips_vs_baseline"]


def test_fever_metric_matches_baseline(suite):
  result, base = suite
  if result["n_claims"] != len(base["predictions"]):
    pytest.skip("baseline FEVER metric is for the full 300-claim sample")
  tol = MAX_FLIPS / result["n_claims"] + 1e-9
  for m in ("label_accuracy", "fever_score"):
    assert result["e2e"][m] == pytest.approx(base["e2e"][m], abs=tol), m


def test_retrieval_matches_baseline(suite):
  result, base = suite
  if result["n_claims"] != len(base["predictions"]):
    pytest.skip("baseline retrieval metrics are for the full 300-claim sample")
  for m, v in base["retrieval"].items():
    assert result["retrieval"][m] == pytest.approx(v, abs=1e-9), m
