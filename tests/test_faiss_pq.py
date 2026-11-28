import numpy as np
import pandas as pd
import pytest

faiss = pytest.importorskip("faiss")
from claimscan.retrieval import faiss_extend, faiss_indexer, faiss_pq  # noqa: E402


def test_pq_index_matches_full_index_order(tmp_path):
  """Build a tiny full index the real way (indexer -> extend), then the PQ index from the same shards:
  same centroids and metric, same docid order, and vectors still find themselves."""
  rng = np.random.default_rng(0)
  vecs, n, d = {}, 200, 16
  for s in range(5):
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    np.save(tmp_path / f"embeddings_shard_{s:05d}.npy", X)
    pd.DataFrame({"docid": [f"P{s}#{i}" for i in range(n)]}).to_csv(tmp_path / f"docids_shard_{s:05d}.csv", index=False)
    vecs.update({f"P{s}#{i}": X[i] for i in range(n)})

  emb, ids = faiss_indexer.get_files(tmp_path, max_shards=3)
  index = faiss_indexer.create_index(num_clusters=8, dim=d)
  docids = faiss_indexer.process_shards(index, emb, ids, training_sample_size=400)
  faiss_indexer.save_index_and_docids(index, docids, tmp_path)
  faiss_extend.extend(tmp_path, block_shards=2)

  assert faiss_pq.build(tmp_path, m=4, nbits=4, train_size=1000, out_name="faiss_index_pq.bin") == 5 * n
  full = faiss.read_index(str(tmp_path / "faiss_index_full.bin"))
  pq = faiss.read_index(str(tmp_path / "faiss_index_pq.bin"))
  assert isinstance(pq, faiss.IndexIVFPQ)
  assert pq.nlist == full.nlist and pq.metric_type == full.metric_type
  assert np.array_equal(pq.quantizer.reconstruct_n(0, pq.nlist), faiss.extract_index_ivf(full).quantizer.reconstruct_n(0, full.nlist))

  all_ids = pd.read_csv(tmp_path / "all_docids_full.csv").astype(str)["docid"].tolist()
  pq.nprobe = 8
  q = np.stack([vecs[x] for x in all_ids])
  _, hit = pq.search(q, 5)
  found = np.mean([i in row for i, row in enumerate(hit)])
  assert found > 0.9   # compressed, so not exact, but a vector's own docid is almost always in its top 5

  # faiss_batch.batch_search scans IVF-Flat files list by list, and searches an IVF-PQ index directly
  from claimscan.retrieval.faiss_batch import _is_ivf_flat
  assert _is_ivf_flat(tmp_path / "faiss_index_full.bin") and _is_ivf_flat(tmp_path / "faiss_index_175.bin")
  assert not _is_ivf_flat(tmp_path / "faiss_index_pq.bin")


def test_train_sample_size(tmp_path):
  for s in range(3):
    np.save(tmp_path / f"e{s}.npy", np.zeros((100, 8), dtype=np.float32))
  X = faiss_pq.train_sample([tmp_path / f"e{s}.npy" for s in range(3)], size=90)
  assert X.shape == (90, 8) and X.dtype == np.float32
