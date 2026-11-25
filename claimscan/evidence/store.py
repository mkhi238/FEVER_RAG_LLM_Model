"""Sentence store for serving: (page, line) -> text in one SQLite file.

The API needs the text of ~30 retrieved sentences per request. Loading the 25M-row corpus
parquet takes ~4 GB of RAM, and filtering the parquet per request takes ~4 s (it has 25
row groups of ~1M rows). A primary-key lookup in SQLite takes milliseconds and almost no RAM,
and the file can be mounted read-only into a container.

    python -m claimscan.evidence.store build      # once, ~20-40 min, writes config.sentence_store
    python -m claimscan.evidence.store check      # row count + a sample lookup

``SentenceStore.for_hits(hits)`` returns the hits' rows as a DataFrame indexed by (id, line),
the same shape as ``index_corpus``, so ``get_evidence`` is unchanged.
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import pandas as pd

from claimscan.config import config

SCHEMA = "CREATE TABLE sentences (id TEXT NOT NULL, line INTEGER NOT NULL, text TEXT NOT NULL, PRIMARY KEY (id, line)) WITHOUT ROWID"


class SentenceStore:
  def __init__(self, path: str | Path | None = None):
    self.path = Path(path or config.sentence_store)
    if not self.path.exists():
      raise FileNotFoundError(f"{self.path} missing: build it with python -m claimscan.evidence.store build")
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

  def count(self) -> int:
    return self.con.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]


def build(corpus: Path = config.corpus, out: Path = config.sentence_store, batch: int = 200_000):
  """Copy (id, line, text) from the corpus parquet into SQLite, streaming batches (low RAM)."""
  import pyarrow.parquet as pq

  tmp = out.with_suffix(".sqlite.partial")
  tmp.unlink(missing_ok=True)
  con = sqlite3.connect(tmp)
  con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-262144;")
  con.execute(SCHEMA)
  f = pq.ParquetFile(str(corpus))
  t0, n = time.time(), 0
  for b in f.iter_batches(batch_size=batch, columns=["id", "line", "text"]):
    d = b.to_pydict()
    con.executemany("INSERT OR IGNORE INTO sentences VALUES (?, ?, ?)",
                    ((i, int(l), t) for i, l, t in zip(d["id"], d["line"], d["text"]) if t is not None))
    con.commit()
    n += b.num_rows
    print(f"[store] {n:,}/{f.metadata.num_rows:,} rows  {time.time() - t0:.0f}s", flush=True)
  con.close()
  tmp.rename(out)
  print(f"[store] wrote {out} ({out.stat().st_size / 1e9:.1f} GB) in {time.time() - t0:.0f}s")


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("cmd", choices=["build", "check"])
  a = ap.parse_args(argv)
  if a.cmd == "build":
    build()
  else:
    s = SentenceStore()
    t = time.time()
    df = s.for_hits([{"id": "Tokyo", "line": 0}, {"id": "Barack_Obama", "line": 0}])
    print(f"{s.count():,} sentences; sample lookup {1000 * (time.time() - t):.1f} ms")
    print(df)


if __name__ == "__main__":
  main()
