"""FEVER claims and gold evidence for evaluation.

Line-0 fix: the HF ``fever`` v1.0 loader stores sentence 0 as -1 (0 is falsy), so in our
evidence parquet / fever_gold.json every SUPPORTS/REFUTES evidence row that says line -1
really means line 0 (no row has line 0; ~37% have -1). NOT ENOUGH INFO claims have no
evidence (url null, line -1). Every loader here maps -1 -> 0 for evidence that has a page.
"""
from __future__ import annotations

import json
import re
import unicodedata

import pandas as pd

from claimscan.config import config

SPLIT_ALIASES = {"val": "validation", "labelled_dev": "validation", "paper_test": "test"}


def norm_title(s: str) -> str:
  s = unicodedata.normalize("NFKC", str(s)).strip()
  return re.sub(r"\s+", "_", s)


def fix_line(line) -> int:
  line = int(line)
  return 0 if line == -1 else line


def docid(page, line) -> str:
  return f"{norm_title(page)}#{int(line)}"


def load_claims(split: str = "validation", path=None) -> pd.DataFrame:
  """Columns id (int), claim (str), label_text (TRUE / FALSE / NOT ENOUGH INFO)."""
  df = pd.read_parquet(path or config.claims_file, columns=["id", "claim", "split", "label_text"])
  df["split"] = df["split"].replace(SPLIT_ALIASES)
  df = df[df["split"] == split].copy()
  df["id"] = df["id"].astype(int)
  df["claim"] = df["claim"].astype(str)
  return df.drop_duplicates("id").reset_index(drop=True)


def load_gold_pairs(path=None) -> dict[int, list[tuple[str, int]]]:
  """{claim id: [(page, line), ...]} from fever_gold.json (SUPPORTS/REFUTES claims only), line-0 fixed."""
  with open(path or config.gold_file, "r", encoding="utf-8") as f:
    raw = json.load(f)
  return {int(k): [(norm_title(p), fix_line(l)) for p, l in pairs] for k, pairs in raw.items()}


def load_gold_sets(claim_ids=None, path=None) -> dict[int, list[list[str]]]:
  """{claim id: [[docid, ...], ...]}: one inner list per FEVER evidence set.

  A set is all rows sharing evidence_annotation_id, matching how the original FEVER
  jsonl groups evidence. NEI claims (no page) get no entry.
  """
  ev = pd.read_parquet(path or config.evidence)
  ev = ev[ev["evidence_wiki_url"].notna() & (ev["evidence_wiki_url"] != "")]
  if claim_ids is not None:
    ev = ev[ev["id"].isin(set(int(c) for c in claim_ids))]
  out: dict[int, list[list[str]]] = {}
  for (cid, _), g in ev.groupby(["id", "evidence_annotation_id"], sort=False):
    s = sorted({docid(p, fix_line(l)) for p, l in zip(g["evidence_wiki_url"], g["evidence_sentence_id"])})
    sets = out.setdefault(int(cid), [])
    if s not in sets:
      sets.append(s)
  return out
