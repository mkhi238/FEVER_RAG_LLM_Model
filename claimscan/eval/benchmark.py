"""Retriever benchmark: retrieval metrics, end-to-end FEVER metric and wall-clock time, charted and logged to MLflow.

Compares faiss, bm25 (RM3), bm25_norm3, hybrid (faiss+bm25 RRF) and hybrid_norm3 on the same
claim sample from FEVER validation:

  retrieval (SUPPORTS/REFUTES claims, k in 1 5 10 20 50), gold = FEVER evidence sets with the line-0 fix:
    hit@k         some gold sentence in the top k
    recall_set@k  some complete gold evidence set in the top k (what the FEVER score needs)
    recall@k      share of the claim's gold sentences in the top k
    precision@k   share of the top k that are gold sentences
    f1@k          per-claim F1 of the two above
    mrr           1 / rank of the first gold sentence (within the top 50)
  end-to-end (all sampled claims, through fact_check_cleaned unchanged, k=30, conf 0.5):
    label_accuracy, fever_score, evidence_precision / recall / f1  (the FEVER metric, claimscan/eval/metrics/fever)
  wall-clock: per stage total and ms per claim; BM25 single-query latency on a few cold queries.

Memory: each stage runs in its own subprocess and caches its hits under <artifacts>/eval/cache,
so the peak is one retriever (~2.5 GB for BM25's JVM) and reruns skip finished stages.
FAISS uses claimscan.retrieval.faiss_batch (one sequential pass over the 26 GB index, ~5 min on D:).
The hybrids are fused from the cached FAISS and BM25 lists exactly as HybridRetriever.search does.

Run:  python -m claimscan.eval.benchmark --n 300
      python -m claimscan.eval.benchmark --n 60 --retrievers bm25 --no-e2e     (quick)
Outputs: <artifacts>/eval/runs/<timestamp>/{summary.json, summary.md, benchmark.png, predictions.parquet}
and one MLflow run (experiment retriever_eval) with a nested run per retriever.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from claimscan.config import config

BASE_RETRIEVERS = ("faiss", "bm25", "bm25_norm3")
FUSED = {"hybrid": ("faiss", "bm25"), "hybrid_norm3": ("faiss", "bm25_norm3")}
ALL_RETRIEVERS = BASE_RETRIEVERS + tuple(FUSED)
K_LIST = (1, 5, 10, 20, 50)
LABELS = ("TRUE", "FALSE", "NOT ENOUGH INFO")


# ----------------------------------------------------------------------------- claims

def claims_key(ids) -> str:
  return hashlib.sha1(",".join(str(i) for i in sorted(ids)).encode()).hexdigest()[:10]


def faiss_covered(gold_sets: dict, pages: set) -> set:
  """Claim ids with at least one gold evidence set whose pages are all in the FAISS index."""
  return {cid for cid, sets in gold_sets.items()
          if any(all(d.rsplit("#", 1)[0] in pages for d in s) for s in sets)}


def select_claims(n: int, seed: int = 0, split: str = "validation", faiss_coverage_only: bool = True,
                  cfg=config) -> pd.DataFrame:
  """n claims, a third per label. SUPPORTS/REFUTES claims are limited to ones FAISS can answer
  (some gold set fully inside its 17.1M sentences) so retrievers are compared on quality, not coverage."""
  from claimscan.eval.gold import load_claims, load_gold_sets

  claims = load_claims(split)
  gold = load_gold_sets(claims["id"])
  eligible = claims["label_text"].eq("NOT ENOUGH INFO") | claims["id"].isin(gold.keys())
  if faiss_coverage_only:
    from claimscan.retrieval.faiss_batch import DocIds
    pages = DocIds(cfg.retriever_index_dir / cfg.retriever_docids_file).pages()
    covered = faiss_covered(gold, pages)
    del pages
    eligible &= claims["label_text"].eq("NOT ENOUGH INFO") | claims["id"].isin(covered)
  pool = claims[eligible]
  per = {lab: n // 3 + (1 if i < n % 3 else 0) for i, lab in enumerate(LABELS)}
  parts = [pool[pool["label_text"] == lab].sample(n=min(per[lab], int((pool["label_text"] == lab).sum())),
                                                  random_state=seed) for lab in LABELS]
  return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ----------------------------------------------------------------------------- retrieval stages

def _hits_path(cache: Path, name: str, key: str, candidates: int) -> Path:
  return cache / "hits" / f"{name}_{key}_c{candidates}.jsonl"


def write_hits(path: Path, ids, hits, timing: dict):
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(".tmp")
  with open(tmp, "w", encoding="utf-8") as f:
    for cid, h in zip(ids, hits):
      f.write(json.dumps({"id": int(cid), "hits": h}) + "\n")
  path.with_suffix(".timing.json").write_text(json.dumps(timing, indent=2))
  os.replace(tmp, path)


def read_hits(path: Path) -> tuple[dict[int, list], dict]:
  hits = {}
  with open(path, encoding="utf-8") as f:
    for line in f:
      row = json.loads(line)
      hits[row["id"]] = row["hits"]
  timing = json.loads(path.with_suffix(".timing.json").read_text())
  return hits, timing


def retrieve_stage(name: str, claims: pd.DataFrame, out: Path, candidates: int, threads: int,
                   latency_samples: int, cfg=config):
  """Run one base retriever over all claims and cache the hits (called in a subprocess)."""
  queries = claims["claim"].tolist()
  t0 = time.perf_counter()
  timing: dict = {"n": len(queries)}
  if name == "faiss":
    from claimscan.retrieval.faiss_batch import batch_search
    hits, detail = batch_search(queries, cfg.retriever_index_dir, cfg.retriever_model, cfg.retriever_index_file,
                                cfg.retriever_docids_file, nprobe=cfg.retriever_nprobe, k=candidates)
    timing.update(detail)
    timing["mode"] = "batched list-major IVF scan (one pass over the index)"
  elif name in ("bm25", "bm25_norm3"):
    from claimscan.retrieval.bm25_retriever import BM25Retriever
    r = BM25Retriever(cfg.bm25_index_dir, cfg.bm25_k1, cfg.bm25_b, use_rm3=(name == "bm25"))
    lat = []
    for q in queries[:latency_samples]:  # cold single queries, as the online pipeline would see them
      t = time.perf_counter()
      r.search(q, k=candidates)
      lat.append((time.perf_counter() - t) * 1000)
    timing["latency_ms"] = lat
    t1 = time.perf_counter()
    hits = r.search_many(queries, k=candidates, threads=threads)
    timing["batch_s"] = time.perf_counter() - t1
    timing["mode"] = f"Lucene batch_search, {threads} threads"
  else:
    raise ValueError(f"not a base retriever: {name}")
  timing["total_s"] = time.perf_counter() - t0
  timing["ms_per_claim"] = timing["total_s"] * 1000 / max(len(queries), 1)
  write_hits(out, claims["id"].tolist(), hits, timing)


def fuse_hits(a: dict, b: dict, a_name: str, b_name: str, rrf_k: int, candidates: int) -> dict:
  """RRF of two cached hit lists per claim, exactly as HybridRetriever.search does with these candidates."""
  from claimscan.retrieval.hybrid_retriever import rrf_fuse
  return {cid: rrf_fuse({a_name: a[cid][:candidates], b_name: b[cid][:candidates]}, rrf_k=rrf_k)
          for cid in a}


def run_subprocess(args: list[str], env_extra: dict | None = None):
  env = dict(os.environ, **(env_extra or {}))
  cmd = ["nice", "-n", "10", sys.executable, "-m", "claimscan.eval.benchmark", *args]
  print(f"[bench] $ {' '.join(cmd[3:])}", flush=True)
  subprocess.run(cmd, check=True, env=env)


# ----------------------------------------------------------------------------- metrics

def hit_docids(hits) -> list[str]:
  """Ranked, de-duplicated "Page#line" ids."""
  from claimscan.eval.gold import docid
  seen, out = set(), []
  for h in hits:
    d = docid(h["id"], h["line"])
    if d not in seen:
      seen.add(d)
      out.append(d)
  return out


def retrieval_metrics(hits: dict, gold_sets: dict, k_list=K_LIST) -> dict:
  """Mean retrieval metrics over claims that have gold sets (see module docstring)."""
  rows = []
  for cid, sets in gold_sets.items():
    if cid not in hits or not sets:
      continue
    ranked = hit_docids(hits[cid])
    gold = set().union(*map(set, sets))
    row = {}
    first = next((i for i, d in enumerate(ranked[:max(k_list)]) if d in gold), None)
    row["mrr"] = 0.0 if first is None else 1.0 / (first + 1)
    for k in k_list:
      top = set(ranked[:k])
      tp = len(top & gold)
      p, r = tp / k, tp / len(gold)
      row[f"hit@{k}"] = float(tp > 0)
      row[f"recall_set@{k}"] = float(any(set(s) <= top for s in sets))
      row[f"recall@{k}"] = r
      row[f"precision@{k}"] = p
      row[f"f1@{k}"] = 0.0 if tp == 0 else 2 * p * r / (p + r)
    rows.append(row)
  if not rows:
    return {"n": 0}
  out = pd.DataFrame(rows).mean().to_dict()
  out["n"] = len(rows)
  return out


# ----------------------------------------------------------------------------- end-to-end stage

class CachedRetriever:
  """Serves cached hits to fact_check_cleaned, keyed by claim text."""

  def __init__(self, by_claim: dict[str, list]):
    self.by_claim = by_claim

  def search(self, query, k=50):
    return self.by_claim.get(query, [])[:k]


def load_corpus_subset(pages: set, cfg=config) -> pd.DataFrame:
  """Only the corpus rows of the given pages, indexed by (id, line), streamed from the parquet."""
  import pyarrow as pa
  import pyarrow.dataset as ds
  from claimscan.evidence.select import index_corpus

  # stream record batches and keep only matches; to_table() over 25M rows held ~2 GB in the arrow pool
  scanner = ds.dataset(str(cfg.corpus), format="parquet").scanner(
    columns=["id", "line", "text"], filter=ds.field("id").isin(pa.array(sorted(pages), type=pa.string())),
    batch_size=65_536, use_threads=False)
  parts = [b.to_pandas() for b in scanner.to_batches() if b.num_rows]
  del scanner
  pa.default_memory_pool().release_unused()
  df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["id", "line", "text"])
  return index_corpus(df)


def load_decider(spec: str | None):
  """'package.module:function' -> callable(claim, verdict) returning a label, or (label, ["Page#line", ...]).

  The verdict is claimscan.pipeline.Verdict: the verifier's label/probs/evidence plus ``hits`` (all top-k
  sentences with text). Lets an LLM layer be scored on exactly the same claims and metric as the verifier.
  """
  if not spec:
    return None
  import importlib
  mod, _, fn = spec.partition(":")
  if not fn:
    raise ValueError(f"--decider must look like package.module:function, got {spec!r}")
  return getattr(importlib.import_module(mod), fn)


def e2e_stage(claims: pd.DataFrame, hits_by_name: dict[str, dict], out: Path, k: int, conf_threshold: float,
              threads: int, services_factory=None, gold=None, cfg=config, decider=None):
  """fact_check_cleaned (unchanged) on cached hits for each retriever, scored with the FEVER metric.

  With ``decider`` (see load_decider) the final label comes from decider(claim, verdict) instead.
  """
  from claimscan.eval.fever_eval import load_fever_metric, to_prediction, to_reference
  from claimscan.eval.gold import load_gold_sets
  from claimscan.pipeline import Services, fact_check_cleaned, fact_check_detailed, load_verifier

  gold = load_gold_sets(claims["id"]) if gold is None else gold
  if services_factory is None:
    import torch
    torch.set_num_threads(threads)
    pages = {h["id"] for hits in hits_by_name.values() for hl in hits.values() for h in hl[:k]}
    t = time.perf_counter()
    corpus = load_corpus_subset(pages, cfg)
    print(f"[bench] corpus subset: {len(corpus):,} sentences from {len(pages):,} pages "
          f"in {time.perf_counter() - t:.0f}s", flush=True)
    model = load_verifier(cfg)
    services_factory = lambda retr: Services(model=model, retriever=retr, corpus_df=corpus)  # noqa: E731

  metric = load_fever_metric()
  results, preds_rows = {}, []
  for name, hits in hits_by_name.items():
    services = services_factory(CachedRetriever({c: hits.get(int(i), []) for i, c in zip(claims["id"], claims["claim"])}))
    preds, refs, times = [], [], []
    for row in claims.itertuples(index=False):
      t = time.perf_counter()
      if decider is None:
        label, ev = fact_check_cleaned(row.claim, services, k=k, conf_threshold=conf_threshold, return_evidence=True)
      else:
        v = fact_check_detailed(row.claim, services, k=k, conf_threshold=conf_threshold)
        res = decider(row.claim, v)
        label, ev = res if isinstance(res, tuple) else (res, [d for _, d in v.evidence])
      times.append((time.perf_counter() - t) * 1000)
      preds.append(to_prediction(label, ev))
      refs.append(to_reference(row.label_text, gold.get(int(row.id), [])))
      preds_rows.append({"retriever": name, "id": int(row.id), "label": row.label_text, "pred": label,
                         "evidence": ev, "ms": times[-1]})
    m = metric.compute(predictions=preds, references=refs)
    m["ms_per_claim"] = float(np.mean(times))
    results[name] = m
    print(f"[bench] e2e {name:<13} " + "  ".join(f"{a}={b:.3f}" for a, b in m.items()), flush=True)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(results, indent=2))
  pd.DataFrame(preds_rows).to_parquet(out.with_suffix(".predictions.parquet"))
  return results


# ----------------------------------------------------------------------------- report

COLORS = {"faiss": "#2a78d6", "bm25": "#eb6834", "bm25_norm3": "#1baf7a", "hybrid": "#eda100",
          "hybrid_norm3": "#e87ba4"}  # categorical slots 1-5, fixed per retriever
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def plot_summary(summary: dict, path: Path, chart_k: int = 5):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  names = [n for n in summary["retrievers"]]
  R = summary["retrievers"]
  fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=SURFACE)
  for ax in axes.flat:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
      ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
      ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

  # (a) recall_set@k curves, direct-labeled at the right end (labels spread so they don't collide)
  ax = axes[0, 0]
  ks = summary["k_list"]
  ends = []
  for n in names:
    ret = R[n].get("retrieval", {})
    if not ret.get("n"):
      continue
    ys = [ret[f"recall_set@{k}"] for k in ks]
    ax.plot(ks, ys, color=COLORS.get(n, INK), linewidth=2, marker="o", markersize=5, label=n)
    ends.append([ys[-1], ys[-1], n])
  ends.sort()
  for i in range(1, len(ends)):  # min 0.045 between label centres
    ends[i][1] = max(ends[i][1], ends[i - 1][1] + 0.045)
  for y, ly, n in ends:
    ax.annotate(f"{n} {y:.2f}", (ks[-1], y), xytext=(ks[-1] * 1.12, ly), textcoords="data",
                va="center", fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6))
  ax.set_xscale("log")
  ax.set_xlim(ks[0] * 0.8, ks[-1] * 3.2)
  ax.set_xticks(ks, [str(k) for k in ks])
  ax.set_ylim(0, 1)
  ax.set_xlabel("k (retrieved sentences)", color=INK)
  ax.set_title("Full evidence set retrieved (recall_set@k)", loc="left", color=INK, fontsize=11)
  ax.legend(frameon=False, fontsize=8, loc="upper left")

  def grouped(ax, metrics, getter, title, fmt="{:.2f}"):
    x = np.arange(len(metrics))
    shown = [n for n in names if getter(n) is not None]
    w = 0.8 / max(len(shown), 1)
    for i, n in enumerate(shown):
      vals = [getter(n).get(m, np.nan) for m in metrics]
      ax.bar(x + (i - (len(shown) - 1) / 2) * w, vals, w * 0.92, color=COLORS.get(n, INK), label=n)
    for j, m in enumerate(metrics):  # label only the best bar per metric; summary.md has every number
      vals = [getter(n).get(m, np.nan) for n in shown]
      if shown and np.isfinite(vals).any():
        i = int(np.nanargmax(vals))
        ax.annotate(f"best {fmt.format(vals[i])}", (x[j] + (i - (len(shown) - 1) / 2) * w, vals[i]),
                    xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncols=min(len(shown), 3), loc="upper right")

  grouped(axes[0, 1], [f"precision@{chart_k}", f"recall@{chart_k}", f"f1@{chart_k}", f"hit@{chart_k}"],
          lambda n: R[n].get("retrieval") if R[n].get("retrieval", {}).get("n") else None,
          f"Retrieval quality at k={chart_k} (sentence level)")
  grouped(axes[1, 0], ["label_accuracy", "fever_score", "evidence_precision", "evidence_recall", "evidence_f1"],
          lambda n: R[n].get("e2e"), "End-to-end through the verifier (FEVER metric)")
  axes[1, 0].tick_params(axis="x", labelsize=8)

  # (d) wall-clock: retrieval ms per claim, horizontal bars on a log axis anchored at a power of ten
  ax = axes[1, 1]
  vals = [R[n]["time"]["ms_per_claim"] for n in names]
  y = np.arange(len(names))[::-1]
  lo = 10 ** np.floor(np.log10(max(min(vals), 1e-3)))
  ax.barh(y, [v - lo for v in vals], left=lo, color=[COLORS.get(n, INK) for n in names], height=0.6)
  for yi, v, n in zip(y, vals, names):
    lab = f"{v:,.0f} ms"
    if R[n]["time"].get("latency_ms_median") is not None:
      lab += f"  (one cold query: {R[n]['time']['latency_ms_median']:,.0f} ms)"
    ax.annotate(lab, (v, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
  ax.set_xscale("log")
  ax.set_xlim(lo, max(vals) * 30)
  ax.set_yticks(y, names)
  ax.grid(axis="y", visible=False)
  ax.grid(axis="x", color=GRID, linewidth=0.8)
  ax.set_xlabel("retrieval ms per claim, batched (log scale)", color=INK)
  e2e_ms = [R[n]["e2e"]["ms_per_claim"] for n in names if R[n].get("e2e")]
  sub = f"evidence + verifier: ~{np.mean(e2e_ms):,.0f} ms/claim" if e2e_ms else "end-to-end not run"
  ax.set_title(f"Wall-clock time  ·  {sub}", loc="left", color=INK, fontsize=11)

  fig.suptitle(f"Retriever benchmark · {summary['n_claims']} FEVER {summary['split']} claims "
               f"({summary['n_retrieval_claims']} with gold evidence) · {summary['timestamp']}",
               x=0.01, ha="left", color=INK, fontsize=12)
  fig.tight_layout()
  fig.savefig(path, dpi=130, facecolor=SURFACE)
  plt.close(fig)


def summary_markdown(summary: dict, chart_k: int = 5) -> str:
  R = summary["retrievers"]
  ks = summary["k_list"]
  lines = [f"# Retriever benchmark {summary['timestamp']}", "",
           f"{summary['n_claims']} FEVER {summary['split']} claims (seed {summary['seed']}), "
           f"{summary['n_retrieval_claims']} with gold evidence for retrieval metrics.", "",
           "| retriever | " + " | ".join(f"R_set@{k}" for k in ks) + f" | P@{chart_k} | R@{chart_k} | F1@{chart_k} | MRR | "
           "label acc | FEVER | ev F1 | ms/claim |",
           "|---" * (len(ks) + 10) + "|"]
  for n, r in R.items():
    ret, e2e = r.get("retrieval", {}), r.get("e2e") or {}
    f = lambda d, key: f"{d[key]:.3f}" if key in d else "–"  # noqa: E731
    lines.append(f"| {n} | " + " | ".join(f(ret, f"recall_set@{k}") for k in ks)
                 + f" | {f(ret, f'precision@{chart_k}')} | {f(ret, f'recall@{chart_k}')} | {f(ret, f'f1@{chart_k}')}"
                 f" | {f(ret, 'mrr')} | {f(e2e, 'label_accuracy')} | {f(e2e, 'fever_score')} | {f(e2e, 'evidence_f1')}"
                 f" | {r['time']['ms_per_claim']:,.0f} |")
  return "\n".join(lines) + "\n"


def _mlflow_name(k: str) -> str:
  return k.replace("@", "_at_")


def log_to_mlflow(summary: dict, run_dir: Path, experiment: str | None = None, uri: str | None = None,
                  cfg=config) -> str:
  """One parent run with the chart/summary as artifacts and a nested run per retriever. Returns the run id."""
  from claimscan.tracking import setup_mlflow

  mlflow = setup_mlflow(experiment or cfg.mlflow_retriever_experiment, cfg, uri=uri)
  params = {"n_claims": summary["n_claims"], "split": summary["split"], "seed": summary["seed"],
            "candidates": summary["candidates"], "nprobe": summary["nprobe"], "rrf_k": summary["rrf_k"],
            "e2e_k": summary["e2e_k"], "conf_threshold": summary["conf_threshold"],
            "faiss_coverage_only": summary["faiss_coverage_only"], "decider": summary.get("decider", "verifier")}
  with mlflow.start_run(run_name=f"benchmark-{summary['timestamp']}") as parent:
    mlflow.log_params(params)
    for name, r in summary["retrievers"].items():
      with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params({**params, "retriever": name})
        metrics = {}
        for k, v in (r.get("retrieval") or {}).items():
          if k != "n":
            metrics[f"retrieval_{_mlflow_name(k)}"] = v
        for k, v in (r.get("e2e") or {}).items():
          metrics[f"e2e_{k}"] = v
        for k in ("ms_per_claim", "total_s", "latency_ms_median"):
          if r["time"].get(k) is not None:
            metrics[f"time_{k}"] = r["time"][k]
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    for f in ("summary.json", "summary.md", "benchmark.png", "predictions.parquet"):
      if (run_dir / f).exists():
        mlflow.log_artifact(str(run_dir / f))
    return parent.info.run_id


# ----------------------------------------------------------------------------- driver

def run(n=300, retrievers=ALL_RETRIEVERS, seed=0, split="validation", candidates=None, e2e=True, e2e_k=30,
        conf_threshold=0.5, threads=2, latency_samples=5, faiss_coverage_only=True, chart_k=5,
        mlflow_log=True, cfg=config, decider: str | None = None) -> dict:
  candidates = candidates or cfg.hybrid_candidates
  cache = Path(cfg.eval_dir) / "cache"
  stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  run_dir = Path(cfg.eval_dir) / "runs" / stamp
  run_dir.mkdir(parents=True, exist_ok=True)
  t_start = time.perf_counter()

  claims_file = cache / f"claims_{split}_n{n}_seed{seed}_{'cov' if faiss_coverage_only else 'all'}.parquet"
  if not claims_file.exists():
    print(f"[bench] selecting {n} claims...", flush=True)
    claims_file.parent.mkdir(parents=True, exist_ok=True)
    run_subprocess(["_select", "--n", str(n), "--seed", str(seed), "--split", split, "--out", str(claims_file)]
                   + ([] if faiss_coverage_only else ["--all-claims"]))
  claims = pd.read_parquet(claims_file)
  key = claims_key(claims["id"])
  print(f"[bench] {len(claims)} claims {claims['label_text'].value_counts().to_dict()}  key={key}", flush=True)

  needed = set()
  for r in retrievers:
    needed.update(FUSED.get(r, (r,)))
  base_hits, timings = {}, {}
  for name in [b for b in BASE_RETRIEVERS if b in needed]:
    path = _hits_path(cache, name, key, candidates)
    if path.exists():
      print(f"[bench] {name}: cached hits {path.name}", flush=True)
    else:
      run_subprocess(["_retrieve", "--retriever", name, "--claims-file", str(claims_file), "--out", str(path),
                      "--candidates", str(candidates), "--threads", str(threads),
                      "--latency-samples", str(latency_samples)],
                     env_extra={"_JAVA_OPTIONS": "-Xmx2g", "OMP_NUM_THREADS": str(threads)})
    base_hits[name], timings[name] = read_hits(path)

  hits_by_name = {}
  for r in retrievers:
    if r in FUSED:
      a, b = FUSED[r]
      t = time.perf_counter()
      hits_by_name[r] = fuse_hits(base_hits[a], base_hits[b], "faiss", "bm25", cfg.hybrid_rrf_k, candidates)
      fuse_ms = (time.perf_counter() - t) * 1000 / max(len(claims), 1)
      timings[r] = {"ms_per_claim": timings[a]["ms_per_claim"] + timings[b]["ms_per_claim"] + fuse_ms,
                    "total_s": timings[a]["total_s"] + timings[b]["total_s"],
                    "mode": f"{a} + {b} + RRF fuse ({fuse_ms:.2f} ms/claim)"}
    else:
      hits_by_name[r] = base_hits[r]

  from claimscan.eval.gold import load_gold_sets
  gold = load_gold_sets(claims["id"])
  summary = {"timestamp": stamp, "n_claims": len(claims), "split": split, "seed": seed, "claims_key": key,
             "k_list": list(K_LIST), "candidates": candidates, "nprobe": cfg.retriever_nprobe,
             "rrf_k": cfg.hybrid_rrf_k, "e2e_k": e2e_k, "conf_threshold": conf_threshold,
             "faiss_coverage_only": faiss_coverage_only, "decider": decider or "verifier", "labels": claims["label_text"].value_counts().to_dict(),
             "retrievers": {}}
  for r in retrievers:
    t = dict(timings[r])
    lat = t.pop("latency_ms", None)
    if lat:
      t["latency_ms_median"] = float(np.median(lat))
      t["latency_ms"] = lat
    summary["retrievers"][r] = {"retrieval": retrieval_metrics(hits_by_name[r], gold), "time": t}
  summary["n_retrieval_claims"] = max((v["retrieval"].get("n", 0) for v in summary["retrievers"].values()), default=0)

  if e2e:
    dtag = "" if not decider else "_d-" + "".join(ch if ch.isalnum() else "-" for ch in decider)
    e2e_out = cache / "e2e" / f"{key}_c{candidates}_k{e2e_k}_t{conf_threshold}_{'-'.join(retrievers)}{dtag}.json"
    if not e2e_out.exists():
      run_subprocess(["_e2e", "--claims-file", str(claims_file), "--out", str(e2e_out), "--candidates", str(candidates),
                      "--k", str(e2e_k), "--conf-threshold", str(conf_threshold), "--threads", str(threads),
                      "--retrievers", *retrievers] + (["--decider", decider] if decider else []),
                     env_extra={"OMP_NUM_THREADS": str(threads)})
    e2e_res = json.loads(e2e_out.read_text())
    for r in retrievers:
      summary["retrievers"][r]["e2e"] = e2e_res.get(r)
    pred = e2e_out.with_suffix(".predictions.parquet")
    if pred.exists():
      pd.read_parquet(pred).to_parquet(run_dir / "predictions.parquet")

  summary["wall_clock_s"] = time.perf_counter() - t_start
  (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
  (run_dir / "summary.md").write_text(summary_markdown(summary, chart_k))
  plot_summary(summary, run_dir / "benchmark.png", chart_k)
  if mlflow_log:
    t = time.perf_counter()
    summary["mlflow_run_id"] = log_to_mlflow(summary, run_dir, cfg=cfg)
    print(f"[bench] MLflow run {summary['mlflow_run_id']} ({time.perf_counter() - t:.0f}s)", flush=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
  print(summary_markdown(summary, chart_k))
  print(f"[bench] outputs: {run_dir}", flush=True)
  summary["run_dir"] = str(run_dir)
  return summary


def _hits_for_e2e(claims, retrievers, candidates, cfg=config):
  cache = Path(cfg.eval_dir) / "cache"
  key = claims_key(claims["id"])
  base = {}
  for r in retrievers:
    for b in FUSED.get(r, (r,)):
      if b not in base:
        base[b], _ = read_hits(_hits_path(cache, b, key, candidates))
  out = {}
  for r in retrievers:
    if r in FUSED:
      a, b = FUSED[r]
      out[r] = fuse_hits(base[a], base[b], "faiss", "bm25", cfg.hybrid_rrf_k, candidates)
    else:
      out[r] = base[r]
  return out


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  sub = ap.add_subparsers(dest="cmd")
  ap.add_argument("--n", type=int, default=300, help="claims to sample (a third per label)")
  ap.add_argument("--retrievers", nargs="+", default=list(ALL_RETRIEVERS), choices=list(ALL_RETRIEVERS))
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--split", default="validation")
  ap.add_argument("--candidates", type=int, default=None, help="hits per retriever (default hybrid.candidates)")
  ap.add_argument("--no-e2e", action="store_true", help="skip the verifier / FEVER metric stage")
  ap.add_argument("--k", type=int, default=30, help="hits passed to fact_check_cleaned")
  ap.add_argument("--conf-threshold", type=float, default=0.5)
  ap.add_argument("--threads", type=int, default=2)
  ap.add_argument("--latency-samples", type=int, default=5)
  ap.add_argument("--all-claims", action="store_true", help="don't restrict to claims FAISS can answer")
  ap.add_argument("--no-mlflow", action="store_true")
  ap.add_argument("--decider", default=None,
                  help="package.module:function(claim, verdict) -> label; replaces the verifier's final decision "
                       "(e.g. an LLM layer). Default: the verifier, unchanged")

  s = sub.add_parser("_select")
  s.add_argument("--n", type=int, required=True); s.add_argument("--seed", type=int, default=0)
  s.add_argument("--split", default="validation"); s.add_argument("--out", required=True)
  s.add_argument("--all-claims", action="store_true")
  s = sub.add_parser("_retrieve")
  for a in ("--retriever", "--claims-file", "--out"):
    s.add_argument(a, required=True)
  s.add_argument("--candidates", type=int, default=100); s.add_argument("--threads", type=int, default=2)
  s.add_argument("--latency-samples", type=int, default=5)
  s = sub.add_parser("_e2e")
  s.add_argument("--claims-file", required=True); s.add_argument("--out", required=True)
  s.add_argument("--candidates", type=int, default=100); s.add_argument("--k", type=int, default=30)
  s.add_argument("--conf-threshold", type=float, default=0.5); s.add_argument("--threads", type=int, default=2)
  s.add_argument("--retrievers", nargs="+", required=True)
  s.add_argument("--decider", default=None)

  a = ap.parse_args(argv)
  if a.cmd == "_select":
    select_claims(a.n, a.seed, a.split, not a.all_claims).to_parquet(a.out)
  elif a.cmd == "_retrieve":
    retrieve_stage(a.retriever, pd.read_parquet(a.claims_file), Path(a.out), a.candidates, a.threads, a.latency_samples)
  elif a.cmd == "_e2e":
    claims = pd.read_parquet(a.claims_file)
    e2e_stage(claims, _hits_for_e2e(claims, a.retrievers, a.candidates), Path(a.out), a.k, a.conf_threshold, a.threads,
              decider=load_decider(a.decider))
  else:
    run(a.n, tuple(a.retrievers), a.seed, a.split, a.candidates, not a.no_e2e, a.k, a.conf_threshold, a.threads,
        a.latency_samples, not a.all_claims, mlflow_log=not a.no_mlflow, decider=a.decider)


if __name__ == "__main__":
  main()
