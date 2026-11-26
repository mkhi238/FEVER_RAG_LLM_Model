"""Add the embedding shards missing from the FAISS index, without rebuilding it or loading it into RAM.

    python -m claimscan.retrieval.faiss_extend            # bge_small: faiss_index_175.bin -> faiss_index_full.bin

faiss_indexer.py built faiss_index_175.bin from shards 0-174 only, and its first shards (0-3) went to
k-means training and were never added. This finds which shards the index holds (by docid), adds every
other shard with the SAME trained centroids (so scores stay comparable), and merges everything into an
on-disk index (faiss.contrib.ondisk): <out>.bin is small, the vectors live in <out>.ivfdata, memory-mapped.
Peak RAM is about one block of shards (~1.5 GB). The old index and docids are left untouched; switch
retriever.index_file / docids_file in configs/default.yaml to use the new ones.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from faiss.contrib.ondisk import merge_ondisk

from claimscan.config import config
from claimscan.retrieval.faiss_indexer import get_files


def _first_docid(csv_path) -> str:
  return str(pd.read_csv(csv_path, nrows=1).astype(str)["docid"][0])


def _missing_shards(emb_files, docid_files, old_docids_csv, ntotal):
  """Shards not in the old index: it holds a contiguous run starting at the shard whose first docid
  equals the old docid list's first entry, totalling ``ntotal`` vectors."""
  first = _first_docid(old_docids_csv)
  start = next(i for i, f in enumerate(docid_files) if _first_docid(f) == first)
  count, end = 0, start
  while count < ntotal:
    count += np.load(emb_files[end], mmap_mode="r").shape[0]
    end += 1
  if count != ntotal:
    raise ValueError(f"shards {start}..{end - 1} hold {count:,} vectors, the index {ntotal:,}")
  print(f"old index = shards {start}..{end - 1} ({ntotal:,} vectors)")
  return [i for i in range(len(emb_files)) if not start <= i < end]


def _empty_like(index):
  """An empty IVFFlat with the same trained centroids and metric as ``index``."""
  ivf = faiss.extract_index_ivf(index)
  quantizer = faiss.clone_index(ivf.quantizer)
  out = faiss.IndexIVFFlat(quantizer, ivf.d, ivf.nlist, ivf.metric_type)
  assert out.is_trained
  return out


def extend(index_dir, old_index="faiss_index_175.bin", old_docids="all_docids_175.csv",
           new_name="faiss_index_full", new_docids="all_docids_full.csv", block_shards=10):
  index_dir = Path(index_dir)
  t0 = time.perf_counter()
  old_path = index_dir / old_index
  old = faiss.read_index(str(old_path), faiss.IO_FLAG_MMAP)
  print(f"old index: {old.ntotal:,} vectors, nlist {faiss.extract_index_ivf(old).nlist}, metric {old.metric_type}")

  emb_files, docid_files = get_files(index_dir, max_shards=10**6)
  missing = _missing_shards(emb_files, docid_files, index_dir / old_docids, old.ntotal)
  print(f"adding {len(missing)} shards: {missing[:6]}{' ...' if len(missing) > 6 else ''}")

  work = index_dir / f"{new_name}_blocks"
  work.mkdir(exist_ok=True)
  out_docids = index_dir / new_docids
  shutil.copyfile(index_dir / old_docids, out_docids)   # old ids 0..ntotal-1 keep their order

  next_id, block_files = old.ntotal, []
  for b in range(0, len(missing), block_shards):
    block = _empty_like(old)
    for s in missing[b:b + block_shards]:
      X = np.load(emb_files[s]).astype(np.float32)
      ids = pd.read_csv(docid_files[s]).astype(str)["docid"]
      if len(X) != len(ids):
        raise ValueError(f"shard {s}: {len(X)} embeddings but {len(ids)} docids")
      block.add_with_ids(X, np.arange(next_id, next_id + len(X), dtype=np.int64))
      ids.to_frame().to_csv(out_docids, mode="a", header=False, index=False)
      next_id += len(X)
      del X
    path = work / f"block_{b // block_shards:03d}.bin"
    faiss.write_index(block, str(path))
    block_files.append(str(path))
    print(f"  block {len(block_files)}: {block.ntotal:,} vectors (total {next_id:,})  {time.perf_counter() - t0:.0f}s")
    del block

  print("merging into the on-disk index...")
  merged = _empty_like(old)
  del old
  merge_ondisk(merged, [str(old_path)] + block_files, str((index_dir / f"{new_name}.ivfdata").resolve()))
  if merged.ntotal != next_id:
    raise ValueError(f"merged {merged.ntotal:,} vectors, expected {next_id:,}")
  faiss.write_index(merged, str(index_dir / f"{new_name}.bin"))
  shutil.rmtree(work)
  print(f"done: {new_name}.bin + .ivfdata + {new_docids}, {merged.ntotal:,} vectors, {time.perf_counter() - t0:.0f}s")
  return merged.ntotal


if __name__ == "__main__":
  extend(config.retriever_index_dir)
