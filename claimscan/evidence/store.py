"""Sentence store for serving: (page, line) -> text from the SQLite file (claimscan/data/fever_db.py builds it).

A primary-key lookup takes milliseconds and almost no RAM, instead of loading the 4 GB corpus,
and the file can be mounted read-only into a container.

    python -m claimscan.evidence.store      # row count + a sample lookup

``SentenceStore.for_hits(hits)`` returns the hits' rows as a DataFrame indexed by (id, line),
the same shape as ``index_corpus``, so ``get_evidence`` is unchanged. ``lookup(page, line)`` returns one text.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd

from claimscan.config import config


class SentenceStore:
  def __init__(self, path: str | Path | None = None):
    self.path = Path(path or config.sentence_store)
    if not self.path.exists():
      raise FileNotFoundError(f"{self.path} missing: build it with python -m claimscan.data.fever_db (see its docstring)")
    # read-only and immutable: no locking, safe to share across the API's worker threads
    self.con = sqlite3.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True, check_same_thread=False)

  def for_hits(self, hits) -> pd.DataFrame:
    """Rows for the (id, line) of each hit, indexed by (id, line). Missing sentences are just absent."""
    keys = list(dict.fromkeys((h["id"], int(h["line"])) for h in hits))
    rows = []
    for page, line in keys:
      r = self.con.execute("SELECT text FROM sentences WHERE id = ? AND line = ?", (page, line)).fetchone()
      if r is not None:
        rows.append((page, line, r[0]))
    df = pd.DataFrame(rows, columns=["id", "line", "text"])
    return df.set_index(["id", "line"]).sort_index()

  def lookup(self, page, line):
    """Text of one sentence, or None."""
    r = self.con.execute("SELECT text FROM sentences WHERE id = ? AND line = ?", (page, int(line))).fetchone()
    return r[0] if r else None

  def count(self) -> int:
    return self.con.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]


def main():
  s = SentenceStore()
  t = time.time()
  df = s.for_hits([{"id": "Tokyo", "line": 0}, {"id": "Barack_Obama", "line": 0}])
  print(f"{s.count():,} sentences; sample lookup {1000 * (time.time() - t):.1f} ms")
  print(df)


if __name__ == "__main__":
  main()
