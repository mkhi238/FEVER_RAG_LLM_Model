import numpy as np
import pandas as pd
import pytest

faiss = pytest.importorskip("faiss")
from claimscan.retrieval import faiss_extend, faiss_indexer  # noqa: E402


def test_extend_adds_missing_shards(tmp_path):
  rng = np.random.default_rng(0)
  vecs, n = {}, 200
  for s in range(5):
    X = rng.standard_normal((n, 16)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    np.save(tmp_path / f"embeddings_shard_{s:05d}.npy", X)
    pd.DataFrame({"docid": [f"P{s}#{i}" for i in range(n)]}).to_csv(tmp_path / f"docids_shard_{s:05d}.csv", index=False)
    vecs.update({f"P{s}#{i}": X[i] for i in range(n)})

  # like the real index: shard 0 only trains it (sample 400 > 200), shards 1-2 added, shards 3-4 never seen
  emb, ids = faiss_indexer.get_files(tmp_path, max_shards=3)
  index = faiss_indexer.create_index(num_clusters=8, dim=16)
  docids = faiss_indexer.process_shards(index, emb, ids, training_sample_size=400)
  faiss_indexer.save_index_and_docids(index, docids, tmp_path)
  assert index.ntotal == 2 * n

  assert faiss_extend.extend(tmp_path, block_shards=2) == 5 * n
  new = faiss.read_index(str(tmp_path / "faiss_index_full.bin"))   # on-disk lists mmap the .ivfdata themselves
  new.nprobe = 8
  all_ids = pd.read_csv(tmp_path / "all_docids_full.csv").astype(str)["docid"].tolist()
  assert len(all_ids) == new.ntotal == 5 * n and len(set(all_ids)) == 5 * n
  assert all_ids[:2 * n] == docids   # old ids keep their positions

  q = np.stack([vecs[d] for d in all_ids])
  _, hit = new.search(q, 1)
  assert [all_ids[i] for i in hit[:, 0]] == all_ids   # every vector finds itself, with its own docid
  assert not (tmp_path / "faiss_index_full_blocks").exists()
