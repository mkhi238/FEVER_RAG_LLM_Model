"""All FEVER data in one SQLite file (config.sentence_store = <artifacts>/data/processed/fever_sentences.sqlite).

Tables:
  sentences (id, line, text)              25.2M Wikipedia sentences, primary key (id, line)
  claims    (id, split, claim, label)     155k claims; split train/validation/test; label TRUE/FALSE/NOT ENOUGH INFO.
                                          Row order = the old parquet order (eval samples depend on it).
  evidence  (claim_id, set_id, page, line) gold evidence, one row per sentence; set_id groups one evidence set.
                                          Line 0 is stored as 0 (the HF loader wrote it as -1). NEI claims have none.

Build (once):
    python -m claimscan.data.fever_db claims                  # claims + evidence from HF (fever/fever parquet), ~1 min
    python -m claimscan.data.fever_db sentences wiki-pages.zip  # Wikipedia sentences from the FEVER dump, slow

Browse it: sqlite3 <file>, or any SQLite viewer (e.g. DB Browser for SQLite, the VS Code "SQLite Viewer" extension).
"""
import json
import re
import sqlite3
import sys
import unicodedata
import zipfile

import pandas as pd

from claimscan.config import config

DB = config.sentence_store
LABELS = {"SUPPORTS": "TRUE", "REFUTES": "FALSE", "NOT ENOUGH INFO": "NOT ENOUGH INFO"}
HF_SPLITS = {"train": "train", "validation": "labelled_dev", "test": "paper_test"}


def connect(db=None, write=False):
  if write:
    return sqlite3.connect(db or DB)
  return sqlite3.connect(f"file:{db or DB}?mode=ro", uri=True)


def normalize_text(s):
  """Same cleaning the corpus and claims were built with: NFKC, lowercase, one space between words."""
  return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s)).strip().lower())


# ---------- read ----------

def claims(split=None, db=None):
  """Columns id, split, claim, label_text. All splits, or one."""
  q = "SELECT id, split, claim, label AS label_text FROM claims"
  with connect(db) as con:
    if split:
      return pd.read_sql_query(q + " WHERE split = ? ORDER BY rowid", con, params=[split])
    return pd.read_sql_query(q + " ORDER BY rowid", con)


def evidence(db=None):
  """Columns claim_id, set_id, page, line, in file order."""
  with connect(db) as con:
    return pd.read_sql_query("SELECT claim_id, set_id, page, line FROM evidence ORDER BY rowid", con)


def sentences(pages=None, db=None):
  """Columns id, line, text: every sentence of the given pages (or all 25M sentences if pages is None)."""
  with connect(db) as con:
    if pages is None:
      return pd.read_sql_query("SELECT id, line, text FROM sentences", con)
    pages = sorted(set(pages))
    parts = [pd.read_sql_query(f"SELECT id, line, text FROM sentences WHERE id IN ({','.join('?' * len(chunk))})",
                               con, params=chunk)
             for chunk in (pages[i:i + 900] for i in range(0, len(pages), 900))]
  return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["id", "line", "text"])


def iter_sentences(batch=100_000, db=None):
  """All sentences as DataFrames of `batch` rows, in (id, line) order. Low RAM."""
  with connect(db) as con:
    cur = con.execute("SELECT id, line, text FROM sentences")
    while rows := cur.fetchmany(batch):
      yield pd.DataFrame(rows, columns=["id", "line", "text"])


# ---------- build ----------

def write_sentences(rows, db=None):
  """Add (id, line, text) rows to the sentences table (created if missing). Duplicates are ignored."""
  with connect(db, write=True) as con:
    con.execute("CREATE TABLE IF NOT EXISTS sentences (id TEXT NOT NULL, line INTEGER NOT NULL, text TEXT NOT NULL,"
                " PRIMARY KEY (id, line)) WITHOUT ROWID")
    con.executemany("INSERT OR IGNORE INTO sentences VALUES (?, ?, ?)", rows)


def write_claims_and_evidence(claims_df, evidence_df, db=None):
  """Replace the claims and evidence tables. claims_df: id, split, claim, label. evidence_df: claim_id, set_id, page, line."""
  with connect(db, write=True) as con:
    con.execute("DROP TABLE IF EXISTS claims")
    con.execute("DROP TABLE IF EXISTS evidence")
    con.execute("CREATE TABLE claims (id INTEGER, split TEXT, claim TEXT, label TEXT)")
    con.execute("CREATE TABLE evidence (claim_id INTEGER, set_id INTEGER, page TEXT, line INTEGER)")
    con.executemany("INSERT INTO claims VALUES (?, ?, ?, ?)", claims_df[["id", "split", "claim", "label"]].itertuples(index=False))
    con.executemany("INSERT INTO evidence VALUES (?, ?, ?, ?)",
                    evidence_df[["claim_id", "set_id", "page", "line"]].itertuples(index=False))
    con.execute("CREATE INDEX claims_id ON claims (id)")
    con.execute("CREATE INDEX evidence_claim ON evidence (claim_id)")


def download_claims_and_evidence(db=None):
  """FEVER v1.0 train / labelled_dev / paper_test from HF -> claims + evidence tables.
  HF has one row per evidence sentence, so a claim repeats; each claim is kept once (train wins over dev/test)."""
  from huggingface_hub import hf_hub_download

  rows = []
  for split, hf_split in HF_SPLITS.items():
    path = hf_hub_download("fever/fever", f"v1.0/{hf_split}/0000.parquet", repo_type="dataset",
                           revision="refs/convert/parquet")
    df = pd.read_parquet(path)
    df["split"] = split
    rows.append(df)
  raw = pd.concat(rows, ignore_index=True)

  c = raw.dropna(subset=["claim", "label"]).copy()
  c["claim"] = c["claim"].map(normalize_text)
  c["label"] = c["label"].str.upper().map(LABELS)
  c["order"] = c["split"].map({"train": 0, "validation": 1, "test": 2})
  c = c.sort_values(["claim", "order"]).drop_duplicates("claim")

  e = raw[raw["evidence_wiki_url"].notna() & (raw["evidence_wiki_url"] != "")]
  e = pd.DataFrame({"claim_id": e["id"], "set_id": e["evidence_annotation_id"], "page": e["evidence_wiki_url"],
                    "line": e["evidence_sentence_id"].replace(-1, 0)})   # the HF loader wrote line 0 as -1

  write_claims_and_evidence(c, e, db)
  print(f"claims: {len(c):,}  evidence rows: {len(e):,}")


def build_sentences(wiki_zip, db=None):
  """Sentences from the FEVER wiki-pages.zip (jsonl files of {"id", "lines"}; each line is "num<TAB>text<TAB>links...").
  Same cleaning as the corpus the indexes were built from: the whole rest of the line is kept, then normalize_text."""
  total = 0
  with zipfile.ZipFile(wiki_zip) as z:
    for name in sorted(n for n in z.namelist() if n.endswith(".jsonl")):
      rows = []
      for raw in z.open(name):
        page = json.loads(raw)
        try:
          for entry in (page.get("lines") or "").split("\n"):
            if entry.strip():
              num, text = entry.split("\t", 1)
              text = normalize_text(text)
              if text:
                rows.append((page["id"], int(num), text))
        except ValueError:   # a malformed page is skipped whole, as before
          continue
      write_sentences(rows, db)
      total += len(rows)
      print(f"{name}: {total:,} sentences")


if __name__ == "__main__":
  if sys.argv[1:2] == ["claims"]:
    download_claims_and_evidence()
  elif sys.argv[1:2] == ["sentences"] and len(sys.argv) == 3:
    build_sentences(sys.argv[2])
  else:
    print(__doc__)
