"""LIAR (Wang 2017): ~12.8k PolitiFact statements with 6 truth labels and speaker metadata.

LIAR has NO evidence sentences, so it is not training data for the verifier. It is used as
  1. an out-of-domain eval set (claimscan.eval.liar_eval), and
  2. a pool of real-world claims for the research agent later (claims only, no gold evidence).
LIAR-PLUS (Alhindi et al. 2018) adds PolitiFact's human justification text per statement;
liar_eval can feed that to the verifier as "oracle" evidence.

Data (fetched by `python -m claimscan.data.liar --download`, then cleaned):
  <artifacts_root>/liar/raw/{train,valid,test}.tsv           original LIAR (from the author's zip)
  <artifacts_root>/liar/liar_plus/{train2,val2,test2}.tsv    LIAR-PLUS (same rows + justification)
  <artifacts_root>/liar/liar_cleaned.csv                     mapped + deduplicated (see main())
The HF `UKPLab/liar` / `liar` datasets use loading scripts that datasets>=4 refuses, so this
reads the original TSVs instead. Downloaded files are data only; nothing in them is executed.

Label mapping (configs/default.yaml -> liar.label_map; a preset name or an explicit dict):
  binary          true, mostly-true -> TRUE;  false, pants-fire -> FALSE;  half-true, barely-true dropped
  binary_strict   true -> TRUE;  false, pants-fire -> FALSE;  everything else dropped
  three_way       as binary, but half-true, barely-true -> NOT ENOUGH INFO
PolitiFact's middle labels mean "partly accurate / misleading", not "no evidence", so
three_way is a stretch; binary is the default.

Data card (checked 2026-11-12):
  rows        train 10,269 / validation 1,284 / test 1,283 (the released files; the paper says
              10,240 / 1,284 / 1,267). Labels (train): half-true 2,123, false 1,998, mostly-true 1,966,
              true 1,683, barely-true 1,657, pants-fire 842. binary keeps ~63% of rows.
  claims      US political statements 2007-2016 (budgets, jobs, taxes, quotes); many hinge on
              statistics or who-said-what, which the 2017 Wikipedia corpus rarely covers.
  LIAR-PLUS   the justification column is misaligned for many rows (another statement's text):
              justification shares < 20% of the claim's content words for 87% of train, 18% of
              validation, 52% of test. liar_eval drops those rows; don't trust it as evidence blindly.
  agent pool  load_claims_for_agent(): 12,810 unique statements, labels kept for analysis only
              (6-way, not verifiable against the corpus, so not a reward signal).
"""
from __future__ import annotations

import argparse
import csv
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from claimscan.config import DEFAULT_CONFIG_FILE, config
from claimscan.data.common import clean_data

LIAR_LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

COLUMNS = ["json_id", "liar_label", "claim", "subjects", "speaker", "speaker_job", "state",
           "party", "barely_true_count", "false_count", "half_true_count", "mostly_true_count",
           "pants_fire_count", "context"]

LABEL_MAPS = {
  "binary": {"true": "TRUE", "mostly-true": "TRUE", "false": "FALSE", "pants-fire": "FALSE"},
  "binary_strict": {"true": "TRUE", "false": "FALSE", "pants-fire": "FALSE"},
  "three_way": {"true": "TRUE", "mostly-true": "TRUE", "false": "FALSE", "pants-fire": "FALSE",
                "half-true": "NOT ENOUGH INFO", "barely-true": "NOT ENOUGH INFO"},
}

SPLIT_FILES = {"train": "train.tsv", "validation": "valid.tsv", "test": "test.tsv"}
PLUS_FILES = {"train": "train2.tsv", "validation": "val2.tsv", "test": "test2.tsv"}

LIAR_ZIP_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
LIAR_PLUS_URL = "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/tsv/{name}"


def liar_settings(path=None) -> dict:
  """The `liar:` section of the config file (config.py does not parse it), with defaults."""
  path = Path(path or os.environ.get("CLAIMSCAN_CONFIG", DEFAULT_CONFIG_FILE))
  with open(path, "r", encoding="utf-8") as f:
    raw = (yaml.safe_load(f) or {}).get("liar") or {}
  return {"dir": config.artifacts_root / raw.get("dir", "liar"),
          "label_map": raw.get("label_map", "binary")}


def resolve_label_map(label_map=None) -> dict:
  """A preset name from LABEL_MAPS or an explicit {liar_label: TRUE|FALSE|NOT ENOUGH INFO} dict."""
  label_map = label_map if label_map is not None else liar_settings()["label_map"]
  if isinstance(label_map, str):
    if label_map not in LABEL_MAPS:
      raise KeyError(f"Unknown LIAR label map {label_map!r}; choose from {sorted(LABEL_MAPS)} or give a dict")
    return dict(LABEL_MAPS[label_map])
  bad = {k: v for k, v in label_map.items() if k not in LIAR_LABELS or v not in config.label_mapping}
  if bad:
    raise ValueError(f"Bad LIAR label map entries {bad}")
  return dict(label_map)


def read_tsv(path) -> pd.DataFrame:
  """One LIAR TSV (14 columns, no header). QUOTE_NONE: statements contain stray quotes."""
  df = pd.read_csv(path, sep="\t", header=None, names=COLUMNS, quoting=csv.QUOTE_NONE,
                   dtype=str, keep_default_na=False)
  df["liar_label"] = df["liar_label"].str.strip().str.lower()
  df = df[df["liar_label"].isin(LIAR_LABELS)].copy()  # guards against a malformed row
  for c in [c for c in COLUMNS if c.endswith("_count")]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
  return df.reset_index(drop=True)


def read_plus_tsv(path) -> pd.DataFrame:
  """LIAR-PLUS: a row-index column, the 14 LIAR columns, then the justification."""
  df = pd.read_csv(path, sep="\t", header=None, names=["row"] + COLUMNS + ["justification"],
                   quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
  return df[["json_id", "justification"]].drop_duplicates("json_id")


def load_liar(split="test", label_map=None, with_justification=False, data_dir=None) -> pd.DataFrame:
  """LIAR rows for one split (train | validation | test), mapped to the system's labels.

  Columns: json_id, claim (original text), label_text (TRUE / FALSE / NOT ENOUGH INFO),
  liar_label (the 6-way label), the metadata columns, split, and with
  ``with_justification=True`` justification ("" when LIAR-PLUS lacks the row).
  Rows whose LIAR label the map drops are removed.
  """
  data_dir = Path(data_dir or liar_settings()["dir"])
  mapping = resolve_label_map(label_map)
  df = read_tsv(data_dir / "raw" / SPLIT_FILES[split])
  df["label_text"] = df["liar_label"].map(mapping)
  df = df.dropna(subset=["label_text"]).copy()
  df["claim"] = df["claim"].str.strip()
  df = df[df["claim"] != ""]
  df["split"] = split
  if with_justification:
    plus = read_plus_tsv(data_dir / "liar_plus" / PLUS_FILES[split])
    df = df.merge(plus, on="json_id", how="left")
    df["justification"] = df["justification"].fillna("")
  return df.reset_index(drop=True)


def load_claims_for_agent(splits=("train", "validation", "test"), data_dir=None) -> pd.DataFrame:
  """Every LIAR statement (all 6 labels kept) as a claim pool for the research agent.

  No gold evidence exists, so these are for rollouts / self-play style training only; the
  6-way label is kept for analysis, not as a reward. Deduplicated on claim text."""
  identity = {l: "TRUE" for l in LIAR_LABELS}  # keep every row; label_text is meaningless here
  df = pd.concat([load_liar(s, identity, data_dir=data_dir) for s in splits], ignore_index=True)
  df = df.drop(columns=["label_text"]).drop_duplicates("claim").reset_index(drop=True)
  return df[["json_id", "claim", "liar_label", "speaker", "party", "context", "subjects", "split"]]


def _fetch(url, dest: Path):
  dest.parent.mkdir(parents=True, exist_ok=True)
  print(f"[liar] {url} -> {dest}")
  with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
    f.write(r.read())


def download(data_dir=None):
  """Fetch the LIAR zip (extract only its TSVs + README) and the LIAR-PLUS TSVs (~10 MB)."""
  data_dir = Path(data_dir or liar_settings()["dir"])
  raw = data_dir / "raw"
  if not all((raw / f).exists() for f in SPLIT_FILES.values()):
    zpath = raw / "liar_dataset.zip"
    _fetch(LIAR_ZIP_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
      for name in list(SPLIT_FILES.values()) + ["README"]:
        (raw / name).write_bytes(z.read(name))  # fixed names only: no paths from the archive
  for name in PLUS_FILES.values():
    if not (data_dir / "liar_plus" / name).exists():
      _fetch(LIAR_PLUS_URL.format(name=name), data_dir / "liar_plus" / name)


def main(argv=None):
  """Write liar_cleaned.csv (claim, label_text, labels, split) to the LIAR artifacts dir.

  Same output as the old UKPLab/liar version: claims normalised by common.clean_data, mapped,
  deduplicated across splits keeping train > validation > test. Was written to the CWD."""
  ap = argparse.ArgumentParser(description="Download and clean LIAR")
  ap.add_argument("--download", action="store_true", help="fetch the TSVs first")
  ap.add_argument("--label-map", default=None, help=f"preset: {', '.join(LABEL_MAPS)} (default: config)")
  args = ap.parse_args(argv)

  out_dir = liar_settings()["dir"]
  if args.download:
    download(out_dir)
  canon = resolve_label_map(args.label_map)
  frames = []
  for split, fname in SPLIT_FILES.items():
    raw = read_tsv(out_dir / "raw" / fname)
    frames.append(clean_data(raw, split, canon, text_col="claim", label_col="liar_label", id_col="json_id"))
  df = pd.concat(frames, ignore_index=True)
  priority = {"train": 0, "validation": 1, "test": 2}
  df["prio"] = df["split"].map(priority)
  df = df.sort_values(["claim", "prio"]).drop_duplicates(subset=["claim"], keep="first")
  df = df.drop("prio", axis=1).reset_index(drop=True)
  df = df.rename(columns={"label": "label_text"})
  df["labels"] = df["label_text"].map(config.label_mapping)
  df = df[["claim", "label_text", "labels", "split"]]
  out = out_dir / "liar_cleaned.csv"
  df.to_csv(out, index=False)
  print(f"[liar] {len(df):,} claims -> {out}  {df['label_text'].value_counts().to_dict()}")


if __name__ == "__main__":
  main()
