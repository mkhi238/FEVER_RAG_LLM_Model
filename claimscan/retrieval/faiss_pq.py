"""Build a compressed in-RAM FAISS index (IVF-PQ) of the full corpus, for fast CPU search.

    python -m claimscan.retrieval.faiss_pq            # bge_small: -> faiss_index_pq96.bin (uses all_docids_full.csv)

Why: the full on-disk IVF-Flat index (faiss_index_full.bin + 39 GB .ivfdata) takes 30-110 s per cold query from D:.
The RL librarian calls retrieval thousands of times, so it needs < 1 s. IVF-PQ stores each 384-dim vector as
``m`` bytes (96 -> 2.63 GB for 25.2M vectors, 43 ms/query; 48 -> 1.42 GB, 22 ms), small enough to hold in RAM.

How:
- reuse the full index's trained 4096 centroids (same clusters, same L2 metric), so only the PQ codebook is trained;
- train the PQ on a sample spread over the shards;
- add the embedding shards (the .npy files) in the SAME order as all_docids_full.csv (faiss_extend's order:
  the old index's shards first, then the missing ones), so the existing docid list is reused unchanged.
Peak RAM: the codes (~2.6 GB with ids at m=96) plus one shard. Build: 30-70 min from D:. The full index and its files are left untouched.
Used by default: retriever.index_file: faiss_index_pq96.bin, retriever.mmap: false in configs/default.yaml.
"""
from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np

from claimscan.config import config
from claimscan.retrieval.faiss_extend import _missing_shards
from claimscan.retrieval.faiss_indexer import get_files

M = 96            # bytes per vector (384 dims / 96 = 4 dims per sub-quantizer); 48 = half the RAM, ~5 pts less recall
NBITS = 8         # 256 centroids per sub-quantizer
TRAIN_SIZE = 400_000


def shard_order(emb_files, docid_files, old_docids_csv, ntotal_old):
  """Shard indices in all_docids_full.csv order: the old index's contiguous run, then the rest."""
  missing = _missing_shards(emb_files, docid_files, old_docids_csv, ntotal_old)
  old = [i for i in range(len(emb_files)) if i not in set(missing)]
  return old + missing


def empty_pq(full_index, m=M, nbits=NBITS):
  """An empty IVF-PQ with the full index's trained centroids and metric (PQ still untrained)."""
  ivf = faiss.extract_index_ivf(full_index)
  quantizer = faiss.clone_index(ivf.quantizer)
  return faiss.IndexIVFPQ(quantizer, ivf.d, ivf.nlist, m, nbits, ivf.metric_type)


def train_sample(emb_files, size=TRAIN_SIZE, n_shards=20, seed=0):
  """``size`` vectors taken evenly from ``n_shards`` shards spread over the corpus."""
  rng = np.random.default_rng(seed)
  picks = np.linspace(0, len(emb_files) - 1, min(n_shards, len(emb_files))).astype(int)
  per = size // len(picks)
  parts = []
  for s in picks:
    X = np.load(emb_files[s], mmap_mode="r")
    rows = np.sort(rng.choice(len(X), min(per, len(X)), replace=False))
    parts.append(np.asarray(X[rows], dtype=np.float32))
  return np.vstack(parts)


def build(index_dir, full_index="faiss_index_full.bin", old_docids="all_docids_175.csv",
          old_ntotal=None, out_name="faiss_index_pq96.bin", m=M, nbits=NBITS, train_size=TRAIN_SIZE):
  index_dir = Path(index_dir)
  t0 = time.perf_counter()
  full_path = index_dir / full_index
  flags = faiss.IO_FLAG_ONDISK_SAME_DIR | faiss.IO_FLAG_READ_ONLY if full_path.with_suffix(".ivfdata").exists() else 0
  full = faiss.read_index(str(full_path), flags)
  print(f"full index: {full.ntotal:,} vectors, nlist {faiss.extract_index_ivf(full).nlist}, metric {full.metric_type}")

  emb_files, docid_files = get_files(index_dir, max_shards=10**6)
  if old_ntotal is None:   # how many vectors the old 175-shard index held (= rows in its docid list)
    with open(index_dir / old_docids) as f:
      old_ntotal = sum(1 for _ in f) - 1
  order = shard_order(emb_files, docid_files, index_dir / old_docids, old_ntotal)

  index = empty_pq(full, m, nbits)
  del full
  X = train_sample(emb_files, train_size)
  print(f"training PQ (m={m}, nbits={nbits}) on {len(X):,} vectors...")
  index.train(X)
  del X
  print(f"  trained  {time.perf_counter() - t0:.0f}s")

  next_id = 0
  for n, s in enumerate(order, 1):
    X = np.load(emb_files[s]).astype(np.float32)
    index.add_with_ids(X, np.arange(next_id, next_id + len(X), dtype=np.int64))
    next_id += len(X)
    del X
    if n % 10 == 0 or n == len(order):
      print(f"  {n}/{len(order)} shards, {next_id:,} vectors  {time.perf_counter() - t0:.0f}s", flush=True)

  out = index_dir / out_name
  faiss.write_index(index, str(out))
  print(f"done: {out} ({out.stat().st_size / 1e9:.2f} GB), {index.ntotal:,} vectors, {time.perf_counter() - t0:.0f}s")
  return index.ntotal


if __name__ == "__main__":
  build(config.retriever_index_dir)
