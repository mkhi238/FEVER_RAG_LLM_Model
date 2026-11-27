"""Build the sentence-level BM25 (Lucene) index that BM25Retriever searches.

Was archive/bm25_page_constructor.py (JSONL step) plus the pyserini indexing command.
Two steps, both under config.bm25_dir (default <artifacts_root>/sparse/bm25_sentences):

  1. jsonl:  one Lucene document per corpus sentence, written in 1M-sentence parts
             {"id": "Barack_Obama#3", "contents": "Barack Obama [SEP] Barack_Obama [SEP] <sentence>"}
             The page title is repeated in contents as a title boost (as in the archive script).
  2. index:  pyserini.index.lucene over jsonl/ -> index/  (+ docvectors, which RM3 needs)

Run:  python -m claimscan.retrieval.bm25_index            # both steps, full 25.2M-sentence corpus
      python -m claimscan.retrieval.bm25_index --step jsonl
Expect ~6 GB of JSONL, ~5 GB of index and 1-3 h on this machine. The JVM heap can be capped
(JAVA_TOOL_OPTIONS=-Xmx2g); keep --memory-buffer well below it. Finished JSONL parts are
skipped on a rerun, so an interrupted jsonl step resumes.

Changes from the archive script: paths come from config; the corpus is streamed in batches
instead of loaded whole (15 GB RAM); docids use the corpus id unchanged, so they match the
FAISS docids and the corpus join in evidence/select.py. The page_offsets.json line shift
was dropped: that file no longer exists and FAISS uses the unshifted lines.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from claimscan.config import config
from claimscan.data import fever_db

PART_ROWS = 1_000_000


def doc_for(page: str, line: int, text: str) -> dict:
  return {"id": f"{page}#{line}", "contents": f"{page.replace('_', ' ')} [SEP] {page} [SEP] {text}"}


def write_jsonl(db=None, out_dir=config.bm25_jsonl_dir, part_rows=PART_ROWS, limit=None):
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  part, rows_in_part, total, f = 0, 0, 0, None
  skipping = False

  def open_part(n):
    path = out_dir / f"sentences_{n:03d}.jsonl"
    if path.exists():                     # finished on an earlier run
      return None, True
    return open(out_dir / f"sentences_{n:03d}.jsonl.tmp", "w", encoding="utf-8"), False

  def close_part(n, fh):
    if fh is not None:
      fh.close()
      (out_dir / f"sentences_{n:03d}.jsonl.tmp").rename(out_dir / f"sentences_{n:03d}.jsonl")

  f, skipping = open_part(part)
  for batch in fever_db.iter_sentences(100_000, db=db):
    cols = batch.to_dict("list")
    for page, line, text in zip(cols["id"], cols["line"], cols["text"]):
      if page is None or line is None or text is None:
        continue
      if rows_in_part == part_rows:
        close_part(part, f)
        part += 1
        rows_in_part = 0
        f, skipping = open_part(part)
      if not skipping:
        f.write(json.dumps(doc_for(str(page), int(line), str(text)), ensure_ascii=False) + "\n")
      rows_in_part += 1
      total += 1
      if limit is not None and total >= limit:
        break
    if limit is not None and total >= limit:
      break
    if total % 5_000_000 < len(batch):
      print(f"[bm25] {total:,} sentences")
  close_part(part, f)
  print(f"[bm25] jsonl done: {total:,} sentences in {part + 1} parts -> {out_dir}")
  return total


def build_index(jsonl_dir=config.bm25_jsonl_dir, index_dir=config.bm25_index_dir, threads=6, docvectors=True,
                memory_buffer_mb=512):
  cmd = [
    sys.executable, "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", str(jsonl_dir),
    "--index", str(index_dir),
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", str(threads),
    # Lucene's in-RAM write buffer; pyserini's default (4096 MB) overflows a capped JVM heap
    "-memoryBuffer", str(memory_buffer_mb),
  ]
  if docvectors:
    cmd.append("--storeDocvectors")
  print("[bm25] " + " ".join(cmd))
  subprocess.run(cmd, check=True)
  print(f"[bm25] index done -> {index_dir}")


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--step", choices=["jsonl", "index", "all"], default="all")
  ap.add_argument("--threads", type=int, default=6)
  ap.add_argument("--out", type=Path, default=config.bm25_dir, help="writes <out>/jsonl and <out>/index")
  ap.add_argument("--memory-buffer", type=int, default=512, help="Lucene write buffer in MB (keep well under the JVM heap)")
  ap.add_argument("--limit", type=int, default=None, help="only the first N sentences (for a test build)")
  args = ap.parse_args(argv)

  if args.step in ("jsonl", "all"):
    write_jsonl(out_dir=args.out / "jsonl", limit=args.limit)
  if args.step in ("index", "all"):
    build_index(args.out / "jsonl", args.out / "index", threads=args.threads, docvectors=config.bm25_rm3,
                memory_buffer_mb=args.memory_buffer)


if __name__ == "__main__":
  main()
