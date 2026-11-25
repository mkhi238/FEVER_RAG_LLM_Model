"""Batched FAISS IVFFlat search that reads the index once, in file order.

For evals over hundreds of claims. With the 26 GB bge index memory-mapped from D:,
``index.search`` takes 30-120 s per cold query: each query touches nprobe=64 lists
(~6 MB each) through 4 KB page faults over WSL drvfs. A few hundred queries touch
nearly all 4096 lists anyway, so this module turns the search inside out: assign
every query to its nprobe lists with the (small, in-RAM) coarse quantizer, then read
each needed inverted list once with large sequential reads and score all of its
queries. Results are the same as ``IndexIVFFlat.search`` with the same nprobe (exact
squared-L2 per vector; only the order of exact ties can differ).

Memory: the quantizer, one list at a time and the docids as a pyarrow array (~0.6 GB
instead of ~5 GB as a Python list).
"""
from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path

import numpy as np


def _mapping_start(path: Path) -> int:
  """Start address of the mmap of ``path`` in this process (faiss maps the whole file once)."""
  real = os.path.realpath(path)
  starts = []
  with open("/proc/self/maps") as f:
    for line in f:
      parts = line.split()
      if len(parts) >= 6 and parts[5] == real and int(parts[2], 16) == 0:
        starts.append(int(parts[0].split("-")[0], 16))
  if not starts:
    raise RuntimeError(f"{real} is not memory-mapped in this process")
  return min(starts)


class IVFListScanner:
  """List-major search over an IndexIVFFlat file (L2 metric)."""

  def __init__(self, index_path, nprobe: int = 64):
    import faiss

    self.path = Path(index_path)
    self.index = faiss.read_index(str(self.path), faiss.IO_FLAG_MMAP)
    ivf = faiss.downcast_index(faiss.extract_index_ivf(self.index))
    if not isinstance(ivf, faiss.IndexIVFFlat):
      raise TypeError(f"expected IndexIVFFlat, got {type(ivf).__name__}")
    if ivf.metric_type != faiss.METRIC_L2:
      raise ValueError("IVFListScanner only implements the L2 metric")
    self.ivf, self.d, self.nlist, self.nprobe = ivf, ivf.d, ivf.nlist, nprobe
    self.ntotal = ivf.ntotal

    # file offset of each list's codes and ids, from their addresses inside faiss's mmap
    base = _mapping_start(self.path)
    inv = ivf.invlists
    code_size = inv.code_size
    self.sizes = np.zeros(self.nlist, dtype=np.int64)
    self.code_off = np.zeros(self.nlist, dtype=np.int64)
    self.ids_off = np.zeros(self.nlist, dtype=np.int64)
    for l in range(self.nlist):
      n = inv.list_size(l)
      self.sizes[l] = n
      if n:
        codes = faiss.rev_swig_ptr(inv.get_codes(l), n * code_size)  # a view: no page is touched
        ids = faiss.rev_swig_ptr(inv.get_ids(l), n)
        self.code_off[l] = codes.ctypes.data - base
        self.ids_off[l] = ids.ctypes.data - base
    self._check_offsets(inv, code_size)

  def _check_offsets(self, inv, code_size):
    import faiss

    nonempty = np.flatnonzero(self.sizes)
    with open(self.path, "rb") as f:
      for l in (nonempty[0], nonempty[-1]):
        n = int(min(self.sizes[l], 4))
        want = faiss.rev_swig_ptr(inv.get_ids(int(l)), n).tobytes()
        got = os.pread(f.fileno(), n * 8, int(self.ids_off[l]))
        if want != got:
          raise RuntimeError("could not locate inverted lists in the index file")

  def assign(self, xq: np.ndarray) -> np.ndarray:
    _, lists = self.ivf.quantizer.search(xq, self.nprobe)
    return lists

  def _read_lists(self, lists, out_q):
    fd = os.open(self.path, os.O_RDONLY)
    try:
      for l in lists:
        n = int(self.sizes[l])
        codes = np.frombuffer(os.pread(fd, n * self.d * 4, int(self.code_off[l])), dtype=np.float32)
        ids = np.frombuffer(os.pread(fd, n * 8, int(self.ids_off[l])), dtype=np.int64)
        out_q.put((l, codes.reshape(n, self.d), ids))
    except BaseException as e:  # surface reader errors in the main thread
      out_q.put(e)
    finally:
      os.close(fd)
      out_q.put(None)

  def search(self, xq: np.ndarray, k: int, log_every: int = 256):
    """Same contract as index.search: (distances (nq,k) float32 ascending, labels (nq,k) int64, -1 padded)."""
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    nq = xq.shape[0]
    assign = self.assign(xq)
    by_list: dict[int, list[int]] = {}
    for qi in range(nq):
      for l in assign[qi]:
        if l >= 0 and self.sizes[l]:
          by_list.setdefault(int(l), []).append(qi)
    order = sorted(by_list, key=lambda l: self.code_off[l])  # sequential file order

    D = np.full((nq, k), np.inf, dtype=np.float32)
    I = np.full((nq, k), -1, dtype=np.int64)
    q: queue.Queue = queue.Queue(maxsize=8)  # read ahead while scoring
    threading.Thread(target=self._read_lists, args=(order, q), daemon=True).start()

    t0, done, nbytes = time.perf_counter(), 0, 0
    while True:
      item = q.get()
      if item is None:
        break
      if isinstance(item, BaseException):
        raise item
      l, codes, ids = item
      nbytes += codes.nbytes
      for qi in by_list[l]:
        diff = codes - xq[qi]
        dist = np.einsum("ij,ij->i", diff, diff)
        cand_d = np.concatenate([D[qi], dist])
        cand_i = np.concatenate([I[qi], ids])
        top = np.argpartition(cand_d, k - 1)[:k] if len(cand_d) > k else np.arange(len(cand_d))
        top = top[np.argsort(cand_d[top], kind="stable")]
        D[qi], I[qi] = cand_d[top], cand_i[top]
      done += 1
      if log_every and done % log_every == 0:
        dt = time.perf_counter() - t0
        print(f"[faiss-scan] {done}/{len(order)} lists  {nbytes / 2**30:.1f} GB  "
              f"{nbytes / 2**20 / max(dt, 1e-9):.0f} MB/s  {dt:.0f}s", flush=True)
    I[~np.isfinite(D)] = -1
    return D, I


class DocIds:
  """The FAISS docids CSV as a pyarrow string array (~0.6 GB for 17.1M ids)."""

  def __init__(self, path):
    import pyarrow.csv as pacsv

    t = pacsv.read_csv(str(path), convert_options=pacsv.ConvertOptions(column_types={"docid": "string"}))
    self.arr = t.column("docid").combine_chunks()

  def __len__(self):
    return len(self.arr)

  def take(self, idx):
    import pyarrow as pa
    return self.arr.take(pa.array(np.asarray(idx, dtype=np.int64))).to_pylist()

  def pages(self) -> set[str]:
    """Unique page titles (the part before the last '#'), title-normalized."""
    import pyarrow.compute as pc
    from claimscan.eval.gold import norm_title

    uniq = pc.unique(pc.replace_substring_regex(self.arr, pattern="#[^#]*$", replacement=""))
    return {norm_title(p) for p in uniq.to_pylist()}


def batch_search(queries, index_dir, model_name, index_file, docids_file, nprobe=64, k=100,
                 encode_batch_size=64):
  """Encode ``queries`` like FAISSRetriever.encode_query and search them with IVFListScanner.

  Returns (hits, timings): hits[i] is a list of {"id", "line", "score"} like FAISSRetriever.search.
  """
  from sentence_transformers import SentenceTransformer
  from claimscan.retrieval.bm25_retriever import split_docid

  index_dir = Path(index_dir)
  t0 = time.perf_counter()
  model = SentenceTransformer(model_name, device="cpu")
  # same text and normalization as FAISSRetriever.encode_query
  xq = model.encode([f"query: {q}" for q in queries], batch_size=encode_batch_size,
                    convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
  t_enc = time.perf_counter() - t0
  del model

  t1 = time.perf_counter()
  scanner = IVFListScanner(index_dir / index_file, nprobe=nprobe)
  D, I = scanner.search(xq, k)
  t_scan = time.perf_counter() - t1

  t2 = time.perf_counter()
  docids = DocIds(index_dir / docids_file)
  if len(docids) != scanner.ntotal:
    raise ValueError(f"Mismatch: {len(docids)} docids vs {scanner.ntotal} vectors in index")
  hits = []
  for drow, irow in zip(D, I):
    keep = irow >= 0
    names = docids.take(irow[keep])
    out = []
    for name, score in zip(names, drow[keep]):
      page, line = split_docid(name)
      out.append({"id": str(page), "line": line, "score": round(float(score), 6)})
    hits.append(out)
  t_ids = time.perf_counter() - t2
  return hits, {"encode_s": t_enc, "scan_s": t_scan, "docids_s": t_ids}
