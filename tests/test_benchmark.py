"""Retriever benchmark: list-major FAISS scan, metrics, RRF from cache, e2e stage, report, MLflow."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from claimscan.eval import benchmark as B


def hits(*docids):
  return [{"id": d.rsplit("#", 1)[0], "line": int(d.rsplit("#", 1)[1]), "score": float(i)} for i, d in enumerate(docids)]


# ---------------------------------------------------------------- FAISS list-major scan

@pytest.fixture(scope="module")
def ivf_file():
  faiss = pytest.importorskip("faiss")
  rng = np.random.default_rng(0)
  d = 24
  xb = rng.standard_normal((6000, d)).astype("float32")
  idx = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 32)
  idx.train(xb)
  idx.add(xb)
  fd, path = tempfile.mkstemp(suffix=".bin")
  os.close(fd)
  faiss.write_index(idx, path)
  yield idx, path, rng.standard_normal((40, d)).astype("float32")
  os.remove(path)


@pytest.mark.parametrize("nprobe,k", [(1, 5), (4, 10), (32, 20)])
def test_list_scan_matches_faiss_search(ivf_file, nprobe, k):
  from claimscan.retrieval.faiss_batch import IVFListScanner
  idx, path, xq = ivf_file
  idx.nprobe = nprobe
  D0, I0 = idx.search(xq, k)
  D1, I1 = IVFListScanner(path, nprobe=nprobe).search(xq, k, log_every=0)
  np.testing.assert_allclose(D1, D0, rtol=1e-4, atol=1e-4)
  assert (I1 == I0).mean() > 0.99            # only exact-distance ties may swap


def test_list_scan_pads_with_minus_one_when_lists_are_short(ivf_file):
  from claimscan.retrieval.faiss_batch import IVFListScanner
  _, path, xq = ivf_file
  D, I = IVFListScanner(path, nprobe=1).search(xq[:3], 2000, log_every=0)
  assert (I == -1).any() and np.isinf(D[I == -1]).all()


def test_docids_take_and_pages(tmp_path):
  from claimscan.retrieval.faiss_batch import DocIds
  p = tmp_path / "docids.csv"
  p.write_text("docid\nA_b#0\nA_b#1\nC#0\nD#x#2\n")
  ids = DocIds(p)
  assert len(ids) == 4 and ids.take([2, 0]) == ["C#0", "A_b#0"]
  assert ids.pages() == {"A_b", "C", "D#x"}


# ---------------------------------------------------------------- metrics

def test_retrieval_metrics_hand_computed():
  gold = {1: [["A#0", "A#1"], ["B#0"]], 2: [["Z#9"]]}
  h = {1: hits("A#0", "X#0", "A#1", "B#0", "Y#0"), 2: hits("Q#0", "R#0")}
  m = B.retrieval_metrics(h, gold, k_list=(1, 3, 5))
  assert m["n"] == 2
  assert m["hit@1"] == 0.5                   # claim 1 yes, claim 2 no
  assert m["recall_set@1"] == 0.0            # no full set in top 1
  assert m["recall_set@3"] == 0.5            # {A#0, A#1} complete in top 3 for claim 1
  assert m["recall@5"] == pytest.approx((3 / 3 + 0) / 2)
  assert m["precision@5"] == pytest.approx((3 / 5 + 0) / 2)
  p, r = 3 / 5, 1.0
  assert m["f1@5"] == pytest.approx((2 * p * r / (p + r)) / 2)
  assert m["mrr"] == pytest.approx(0.5)


def test_retrieval_metrics_dedupes_and_normalizes_titles():
  m = B.retrieval_metrics({1: hits("New York#0", "New_York#0", "B#1")}, {1: [["New_York#0", "B#1"]]}, k_list=(2,))
  assert m["recall_set@2"] == 1.0 and m["precision@2"] == 1.0


def test_faiss_covered_needs_a_whole_set():
  gold = {1: [["A#0", "B#0"]], 2: [["A#0", "Z#0"], ["A#3"]], 3: [["Z#1"]]}
  assert B.faiss_covered(gold, {"A", "B"}) == {1, 2}


# ---------------------------------------------------------------- caching and fusion

def test_hits_roundtrip(tmp_path):
  p = tmp_path / "h" / "faiss_x_c100.jsonl"
  B.write_hits(p, [7, 8], [hits("A#0"), []], {"total_s": 1.5})
  h, t = B.read_hits(p)
  assert h == {7: hits("A#0"), 8: []} and t["total_s"] == 1.5


def test_fuse_from_cache_equals_hybrid_retriever():
  from claimscan.retrieval.hybrid_retriever import HybridRetriever

  class Stub:
    def __init__(self, r):
      self.r = r

    def search(self, q, k=50):
      return self.r[:k]

  f = hits(*[f"F#{i}" for i in range(150)])
  b = hits("F#3", "X#0", "F#120", *[f"Y#{i}" for i in range(150)])
  live = HybridRetriever(Stub(f), Stub(b), rrf_k=60, candidates=100).search("q", k=30)
  cached = B.fuse_hits({1: f}, {1: b}, "faiss", "bm25", rrf_k=60, candidates=100)[1][:30]
  assert live == cached


def test_cached_retriever_truncates_and_handles_unknown():
  r = B.CachedRetriever({"c": hits("A#0", "B#0", "C#0")})
  assert len(r.search("c", k=2)) == 2 and r.search("other") == []


# ---------------------------------------------------------------- end-to-end stage with stubs

class StubModel:
  """Says TRUE with confidence 0.9 when the evidence mentions 'yes', NEI otherwise."""

  def encode_text(self, claim, evidence):
    import torch
    flag = any("yes" in e for e in evidence)
    return {"input_ids": torch.tensor([[int(flag)]]), "attention_mask": torch.ones(1, 1)}

  def __call__(self, ids, mask):
    import torch
    return torch.tensor([[3.0, 0.0, 0.0]]) if ids.item() else torch.tensor([[0.0, 0.0, 3.0]])


def test_e2e_stage_scores_with_fever_metric(tmp_path):
  pytest.importorskip("evaluate")
  from claimscan.evidence.select import index_corpus
  from claimscan.pipeline import Services

  corpus = index_corpus(pd.DataFrame({"id": ["A", "A", "N"], "line": [0, 1, 0],
                                      "text": ["A yes fact", "A more", "noise text"]}))
  claims = pd.DataFrame({"id": [1, 2], "claim": ["claim about A", "unverifiable"],
                         "label_text": ["TRUE", "NOT ENOUGH INFO"]})
  gold = {1: [["A#0"]]}
  hits_by_name = {"good": {1: hits("A#0"), 2: hits("N#0")}, "bad": {1: hits("N#0"), 2: hits("N#0")}}
  out = tmp_path / "e2e.json"
  res = B.e2e_stage(claims, hits_by_name, out, k=30, conf_threshold=0.5, threads=1, gold=gold,
                    services_factory=lambda r: Services(model=StubModel(), retriever=r, corpus_df=corpus))
  assert res["good"]["label_accuracy"] == 1.0 and res["good"]["fever_score"] == 1.0
  assert res["bad"]["label_accuracy"] == 0.5 and res["bad"]["fever_score"] == 0.5
  assert res["good"]["ms_per_claim"] >= 0
  assert json.loads(out.read_text()).keys() == {"good", "bad"}
  preds = pd.read_parquet(out.with_suffix(".predictions.parquet"))
  assert len(preds) == 4 and set(preds["retriever"]) == {"good", "bad"}


# ---------------------------------------------------------------- report and MLflow

def fake_summary():
  ret = {"n": 10, "mrr": 0.4, **{f"{m}@{k}": 0.1 * (i + 1) for i, k in enumerate(B.K_LIST)
                                 for m in ("hit", "recall_set", "recall", "precision", "f1")}}
  e2e = {"label_accuracy": 0.6, "fever_score": 0.4, "evidence_precision": 0.3, "evidence_recall": 0.5,
         "evidence_f1": 0.35, "ms_per_claim": 120.0}
  return {"timestamp": "20261112-000000", "n_claims": 12, "n_retrieval_claims": 10, "split": "validation",
          "seed": 0, "k_list": list(B.K_LIST), "candidates": 100, "nprobe": 64, "rrf_k": 60, "e2e_k": 30,
          "conf_threshold": 0.5, "faiss_coverage_only": True,
          "retrievers": {"faiss": {"retrieval": ret, "e2e": e2e, "time": {"ms_per_claim": 900.0, "total_s": 10.8}},
                         "bm25": {"retrieval": ret, "e2e": None,
                                  "time": {"ms_per_claim": 300.0, "total_s": 3.6, "latency_ms_median": 7000.0}}}}


def test_chart_and_markdown(tmp_path):
  pytest.importorskip("matplotlib")
  s = fake_summary()
  B.plot_summary(s, tmp_path / "b.png")
  assert (tmp_path / "b.png").stat().st_size > 10_000
  md = B.summary_markdown(s)
  assert "| faiss |" in md and "| bm25 |" in md and "R_set@50" in md


def test_tracking_uri_turns_a_path_into_sqlite(tmp_path, monkeypatch):
  from claimscan import tracking
  from claimscan.config import config
  monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
  cfg = config.__class__(**{**config.__dict__, "mlflow_tracking_uri": str(tmp_path / "m" / "mlflow.db")})
  assert tracking.tracking_uri(cfg) == f"sqlite:///{(tmp_path / 'm' / 'mlflow.db').resolve()}"
  monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x:5000")
  assert tracking.tracking_uri(cfg) == "http://x:5000"


def test_log_to_mlflow_creates_parent_and_nested_runs(tmp_path, monkeypatch):
  pytest.importorskip("mlflow")
  monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
  # SQLite schema creation takes ~70 s on drvfs, so use RAM-backed /dev/shm when it exists
  base = Path(tempfile.mkdtemp(dir="/dev/shm" if os.path.isdir("/dev/shm") else None))
  try:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    s = fake_summary()
    (run_dir / "summary.json").write_text(json.dumps(s))
    uri = f"sqlite:///{base / 'mlflow.db'}"
    from claimscan.config import config
    cfg = config.__class__(**{**config.__dict__, "mlflow_artifact_dir": base / "art"})
    run_id = B.log_to_mlflow(s, run_dir, experiment="test_bench", uri=uri, cfg=cfg)
    import mlflow
    runs = mlflow.search_runs(experiment_names=["test_bench"])
    assert len(runs) == 3 and run_id in set(runs["run_id"])
    child = runs[runs["tags.mlflow.runName"] == "faiss"].iloc[0]
    assert child["metrics.retrieval_recall_set_at_5"] == pytest.approx(0.2)
    assert child["metrics.e2e_fever_score"] == pytest.approx(0.4)
    assert child["metrics.time_ms_per_claim"] == pytest.approx(900.0)
    arts = [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)]
    assert "summary.json" in arts
  finally:
    import shutil
    shutil.rmtree(base, ignore_errors=True)
