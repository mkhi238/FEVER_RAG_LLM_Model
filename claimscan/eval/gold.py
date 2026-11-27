"""FEVER claims and gold evidence for evaluation.

Read from the SQLite tables in claimscan/data/fever_db.py. Line 0: the HF ``fever`` loader wrote
sentence 0 as -1; the evidence table stores it as 0 already, and fix_line (-1 -> 0) is kept for any
old -1 values. NOT ENOUGH INFO claims have no evidence rows.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from claimscan.data import fever_db

SPLIT_ALIASES = {"val": "validation", "labelled_dev": "validation", "paper_test": "test"}


def norm_title(s: str) -> str:
  s = unicodedata.normalize("NFKC", str(s)).strip()
  return re.sub(r"\s+", "_", s)


def fix_line(line) -> int:
  line = int(line)
  return 0 if line == -1 else line


def docid(page, line) -> str:
  return f"{norm_title(page)}#{int(line)}"


def load_claims(split: str = "validation", db=None) -> pd.DataFrame:
  """Columns id (int), claim (str), split, label_text (TRUE / FALSE / NOT ENOUGH INFO)."""
  df = fever_db.claims(SPLIT_ALIASES.get(split, split), db=db)
  return df[["id", "claim", "split", "label_text"]].drop_duplicates("id").reset_index(drop=True)


def load_gold_pairs(db=None) -> dict[int, list[tuple[str, int]]]:
  """{claim id: [(page, line), ...]} for SUPPORTS/REFUTES claims, in file order, no repeats."""
  out: dict[int, list[tuple[str, int]]] = {}
  ev = fever_db.evidence(db=db)
  for cid, page, line in zip(ev["claim_id"], ev["page"], ev["line"]):
    pairs = out.setdefault(int(cid), [])
    pair = (norm_title(page), fix_line(line))
    if pair not in pairs:
      pairs.append(pair)
  return out


def load_gold_sets(claim_ids=None, db=None) -> dict[int, list[list[str]]]:
  """{claim id: [[docid, ...], ...]}: one inner list per FEVER evidence set (rows sharing set_id).
  NEI claims (no evidence) get no entry."""
  ev = fever_db.evidence(db=db)
  if claim_ids is not None:
    ev = ev[ev["claim_id"].isin(set(int(c) for c in claim_ids))]
  out: dict[int, list[list[str]]] = {}
  for (cid, _), g in ev.groupby(["claim_id", "set_id"], sort=False):
    s = sorted({docid(p, fix_line(l)) for p, l in zip(g["page"], g["line"])})
    sets = out.setdefault(int(cid), [])
    if s not in sets:
      sets.append(s)
  return out
