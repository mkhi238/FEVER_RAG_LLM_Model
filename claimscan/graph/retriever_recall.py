"""Recall@k chart for the five retrievers, from the benchmark's cached hits (no retrieval is re-run).

    python -m claimscan.graph.retriever_recall                      # full gold set in top k
    python -m claimscan.graph.retriever_recall --metric hit         # any gold sentence in top k
    python -m claimscan.graph.retriever_recall --metric recall      # share of gold sentences in top k

Writes claimscan/graph/retriever_recall_<metric>.png. Scores SUPPORTS / REFUTES claims only
(NEI claims have no gold evidence), with claimscan.eval.benchmark.retrieval_metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from claimscan.config import config
from claimscan.eval import benchmark as B
from claimscan.eval.gold import load_gold_sets

OUT_DIR = Path(__file__).resolve().parent
NAMES = {"faiss": "FAISS (bge-small)", "bm25": "BM25 + RM3", "bm25_norm3": "BM25, no RM3",
         "hybrid": "Hybrid, RM3", "hybrid_norm3": "Hybrid, no RM3"}
TITLES = {"recall_set": "Complete gold evidence set in top k",
          "hit": "Any gold sentence in top k",
          "recall": "Share of gold sentences in top k"}


def compute(n=300, seed=0, split="validation", cfg=config) -> tuple[dict, int]:
  """{retriever: retrieval_metrics(...)} from the cached hits of a finished benchmark run, and the claim count."""
  cache = Path(cfg.eval_dir) / "cache"
  claims_file = cache / f"claims_{split}_n{n}_seed{seed}_cov.parquet"
  if not claims_file.exists():
    raise FileNotFoundError(f"{claims_file} missing: run python -m claimscan.eval.benchmark first")
  claims = pd.read_parquet(claims_file)
  key, c = B.claims_key(claims["id"]), cfg.hybrid_candidates
  hits = {name: B.read_hits(B._hits_path(cache, name, key, c))[0] for name in B.BASE_RETRIEVERS}
  for name, (a, b) in B.FUSED.items():
    hits[name] = B.fuse_hits(hits[a], hits[b], "faiss", "bm25", cfg.hybrid_rrf_k, c)
  gold = load_gold_sets(claims["id"])
  metrics = {name: B.retrieval_metrics(hits[name], gold) for name in B.ALL_RETRIEVERS}
  return metrics, metrics["faiss"]["n"]


def plot(metrics: dict, n_claims: int, metric="recall_set", out: Path | None = None) -> Path:
  out = out or OUT_DIR / f"retriever_recall_{metric}.png"
  ks = list(B.K_LIST)
  fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
  for name in B.ALL_RETRIEVERS:
    vals = [metrics[name][f"{metric}@{k}"] for k in ks]
    ax.plot(ks, vals, color=B.COLORS[name], lw=2, marker="o", ms=6, mec="white", mew=1.5,
            label=f"{NAMES[name]}  (MRR {metrics[name]['mrr']:.2f})")

  ax.set_xscale("log")
  ax.set_xticks(ks, [str(k) for k in ks])
  ax.minorticks_off()
  ax.set_ylim(0, 1)
  ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
  ax.set_xlabel("k (top-k retrieved sentences, log scale)")
  ax.set_title(f"{TITLES[metric]}: {n_claims} FEVER validation claims", loc="left", fontsize=12)
  ax.grid(axis="y", color="#e6e6e6", lw=0.8)
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  ax.spines["left"].set_color("#999999")
  ax.spines["bottom"].set_color("#999999")
  ax.tick_params(colors="#555555")
  ax.legend(frameon=False, loc="lower right", fontsize=9)
  fig.tight_layout()
  fig.savefig(out)
  plt.close(fig)
  return out


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--metric", default="recall_set", choices=list(TITLES))
  ap.add_argument("--n", type=int, default=300)
  ap.add_argument("--seed", type=int, default=0)
  a = ap.parse_args(argv)
  metrics, n_claims = compute(a.n, a.seed)
  print(f"wrote {plot(metrics, n_claims, a.metric)}")


if __name__ == "__main__":
  main()
