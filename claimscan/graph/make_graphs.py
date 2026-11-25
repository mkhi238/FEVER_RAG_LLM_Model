"""Regenerate every graph we have data for into a clean <repo>/graphs/ folder.

    python -m claimscan.graph.make_graphs              # all graphs it can make
    python -m claimscan.graph.make_graphs --out /tmp/g # somewhere else

Nothing is retrained or re-retrieved: every graph is drawn from results already on disk.
The folder is owned by this script: old *.png / README.md in it are removed first, then
each graph whose source exists is written, and graphs/README.md lists what was made,
from which run, and what was skipped (and why).

  benchmark.png                 latest <artifacts>/eval/runs/<ts>/summary.json (claimscan.eval.benchmark)
  retriever_recall_set.png      benchmark cached hits: complete gold evidence set in top k
  retriever_hit.png             benchmark cached hits: any gold sentence in top k
  retriever_recall.png          benchmark cached hits: share of gold sentences in top k
  confusion_matrices.png        latest run's predictions.parquet: predicted vs true label per retriever
  verifier_training_curves.png  DeBERTa training curves (verifier output dir, else archive/)
"""
from __future__ import annotations

import argparse
import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from claimscan.config import config

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "graphs"
LABELS = ("TRUE", "FALSE", "NOT ENOUGH INFO")
SHORT = {"TRUE": "TRUE", "FALSE": "FALSE", "NOT ENOUGH INFO": "NEI"}
NAMES = {"faiss": "FAISS (bge-small)", "bm25": "BM25 + RM3", "bm25_norm3": "BM25, no RM3",
         "hybrid": "Hybrid, RM3", "hybrid_norm3": "Hybrid, no RM3"}
INK, MUTED, SURFACE = "#0b0b0b", "#555555", "#fcfcfb"
# sequential blue ramp (steps 100 -> 700), lightest = near zero
SEQ = LinearSegmentedColormap.from_list(
  "seq_blue", ["#f3f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


# ----------------------------------------------------------------------------- sources

def latest_run(cfg=config) -> Path | None:
  """Newest benchmark run dir that finished (has summary.json)."""
  runs = Path(cfg.eval_dir) / "runs"
  done = sorted(p for p in runs.glob("*/") if (p / "summary.json").exists()) if runs.exists() else []
  return done[-1] if done else None


def verifier_curves_source(cfg=config) -> Path | None:
  for p in (Path(cfg.verifier_output_dir) / "training_curves.png",
            Path(cfg.verifier_output_dir) / "trained_model" / "training_curves.png",
            REPO / "archive" / "verifier_training_curves.png"):
    if p.exists():
      return p
  return None


# ----------------------------------------------------------------------------- graphs

def graph_benchmark(out: Path, run: Path) -> str:
  from claimscan.eval.benchmark import plot_summary
  plot_summary(json.loads((run / "summary.json").read_text()), out / "benchmark.png")
  return f"run `{run.name}`"


def graph_recall(out: Path, run: Path | None) -> str:
  from claimscan.graph import retriever_recall as R
  n, seed = 300, 0
  if run is not None:
    s = json.loads((run / "summary.json").read_text())
    n, seed = s.get("n", s.get("config", {}).get("n", n)), s.get("seed", s.get("config", {}).get("seed", seed))
  metrics, n_claims = R.compute(n=n, seed=seed)
  for metric, fname in (("recall_set", "retriever_recall_set.png"), ("hit", "retriever_hit.png"),
                        ("recall", "retriever_recall.png")):
    R.plot(metrics, n_claims, metric, out / fname)
  return f"benchmark cache, {n} claims (seed {seed}), {n_claims} with gold evidence"


def graph_confusion(out: Path, run: Path) -> tuple[str, list[str]]:
  """Row-normalised confusion matrix per retriever (each row = one true label, sums to 100%)."""
  pred = pd.read_parquet(run / "predictions.parquet")
  names = [r for r in NAMES if r in set(pred["retriever"])]
  fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names) + 0.8, 3.6), dpi=150,
                           facecolor=SURFACE, squeeze=False)
  table = ["| retriever | accuracy | " + " | ".join(f"{SHORT[t]} recall" for t in LABELS) + " |",
           "|---|---|" + "---|" * len(LABELS)]
  im = None
  for ax, name in zip(axes[0], names):
    d = pred[pred["retriever"] == name]
    counts = np.array([[((d["label"] == t) & (d["pred"] == p)).sum() for p in LABELS] for t in LABELS])
    share = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    im = ax.imshow(share, cmap=SEQ, vmin=0, vmax=1)
    for i in range(3):
      for j in range(3):
        ax.text(j, i, f"{share[i, j]:.0%}\n({counts[i, j]})", ha="center", va="center", fontsize=8,
                color="white" if share[i, j] > 0.55 else INK)
    # 2px surface gaps between cells
    ax.set_xticks(np.arange(-.5, 3), minor=True)
    ax.set_yticks(np.arange(-.5, 3), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(3), [SHORT[t] for t in LABELS], fontsize=8, color=MUTED)
    ax.set_yticks(range(3), [SHORT[t] for t in LABELS] if ax is axes[0][0] else [], fontsize=8, color=MUTED)
    ax.tick_params(length=0)
    for s in ax.spines.values():
      s.set_visible(False)
    acc = (d["label"] == d["pred"]).mean()
    ax.set_title(f"{NAMES[name]}\naccuracy {acc:.1%}", fontsize=9, color=INK)
    ax.set_xlabel("predicted", fontsize=8, color=MUTED)
    rec = [share[i, i] for i in range(3)]
    table.append(f"| {NAMES[name]} | {acc:.1%} | " + " | ".join(f"{r:.0%}" for r in rec) + " |")
  axes[0][0].set_ylabel("true label", fontsize=8, color=MUTED)
  cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.02, format=matplotlib.ticker.PercentFormatter(1.0))
  cb.outline.set_visible(False)
  cb.ax.tick_params(labelsize=7, colors=MUTED, length=0)
  fig.suptitle(f"End-to-end predictions vs true label: {len(d)} FEVER validation claims per retriever "
               f"(row = true label, sums to 100%)", x=0.01, ha="left", fontsize=10, color=INK)
  fig.savefig(out / "confusion_matrices.png", facecolor=SURFACE, bbox_inches="tight")
  plt.close(fig)
  return f"run `{run.name}`", table


def graph_verifier(out: Path, src: Path) -> str:
  shutil.copyfile(src, out / "verifier_training_curves.png")
  return f"copied from `{src}` (static; regenerated only when the verifier is retrained)"


# ----------------------------------------------------------------------------- main

DESCRIPTIONS = {
  "benchmark.png": "Retriever benchmark summary: retrieval metrics, end-to-end FEVER metrics and time per claim.",
  "retriever_recall_set.png": "Share of claims with a complete gold evidence set in the top k (what the FEVER score needs).",
  "retriever_hit.png": "Share of claims with at least one gold sentence in the top k.",
  "retriever_recall.png": "Share of each claim's gold sentences found in the top k.",
  "confusion_matrices.png": "End-to-end predicted vs true label per retriever, row-normalised.",
  "verifier_training_curves.png": "DeBERTa verifier training loss and accuracy. Unshuffled split: overstates accuracy.",
}


def make_all(out: Path = DEFAULT_OUT, cfg=config) -> dict:
  out.mkdir(parents=True, exist_ok=True)
  for f in list(out.glob("*.png")) + [out / "README.md"]:
    f.unlink(missing_ok=True)
  others = [f.name for f in out.iterdir()]

  run = latest_run(cfg)
  made, skipped, extra = {}, {}, []

  def attempt(files, fn, *args, missing=None):
    if missing:
      for f in files:
        skipped[f] = missing
      return
    try:
      res = fn(out, *args)
      if isinstance(res, tuple):
        res, table = res
        extra.extend(["", "### Confusion matrix numbers", "", *table])
      for f in files:
        made[f] = res
    except Exception as e:  # one broken source must not stop the rest
      traceback.print_exc()
      for f in files:
        skipped[f] = f"failed: {type(e).__name__}: {e}"

  no_run = None if run else "no finished benchmark run (python -m claimscan.eval.benchmark)"
  attempt(["benchmark.png"], graph_benchmark, run, missing=no_run)
  attempt(["retriever_recall_set.png", "retriever_hit.png", "retriever_recall.png"], graph_recall, run)
  attempt(["confusion_matrices.png"], graph_confusion, run,
          missing=no_run or (None if (run / "predictions.parquet").exists()
                             else f"run {run.name} has no predictions.parquet (benchmark ran with --no-e2e)"))
  src = verifier_curves_source(cfg)
  attempt(["verifier_training_curves.png"], graph_verifier, src, missing=None if src else "no training_curves.png found")

  lines = ["# Graphs", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M} by `python -m claimscan.graph.make_graphs` "
           "(or the `/graphs` skill). Don't edit by hand: the folder is rebuilt on every run.", ""]
  for f, src_note in made.items():
    lines += [f"## {f}", "", f"![{f}]({f})", "", DESCRIPTIONS[f], "", f"Source: {src_note}", ""]
  if skipped:
    lines += ["## Not generated", ""] + [f"- `{f}`: {why}" for f, why in skipped.items()]
  lines += extra
  if others:
    lines += ["", f"Other files in this folder (left alone): {', '.join(sorted(others))}"]
  (out / "README.md").write_text("\n".join(lines) + "\n")
  return {"out": str(out), "made": made, "skipped": skipped, "others": others}


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
  a = ap.parse_args(argv)
  r = make_all(a.out)
  print(f"graphs -> {r['out']}")
  for f, s in r["made"].items():
    print(f"  made     {f:<30} {s}")
  for f, s in r["skipped"].items():
    print(f"  skipped  {f:<30} {s}")


if __name__ == "__main__":
  main()
