import numpy as np
import pandas as pd
import pytest

from claimscan.config import config

faiss = pytest.importorskip("faiss")
from claimscan.retrieval import faiss_indexer  # noqa: E402


def test_all_dense_indexes_configured():
  assert set(config.dense_indexes) == {"bge_small", "minilm", "e5_base"}
  assert config.dense_index()["dir"] == config.retriever_index_dir
  assert config.dense_index("e5_base")["dim"] == 768
  with pytest.raises(KeyError):
    config.dense_index("nope")


@pytest.mark.skipif(not config.artifacts_root.exists(), reason="artifacts drive not mounted")
@pytest.mark.parametrize("name", ["bge_small", "minilm", "e5_base"])
def test_dense_index_dirs_exist_and_dims_match(name):
  entry = config.dense_index(name)
  shard = sorted(entry["dir"].glob("embeddings_shard_*.npy"))[0]
  assert np.load(shard, mmap_mode="r").shape[1] == entry["dim"]


@pytest.mark.parametrize("dim", [384, 768])
def test_builds_and_searches_ivf_index(tmp_path, dim):
  rng = np.random.default_rng(0)
  for s in range(2):
    X = rng.standard_normal((300, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    np.save(tmp_path / f"embeddings_shard_{s:05d}.npy", X)
    pd.DataFrame({"docid": [f"Page{s}#{i}" for i in range(300)]}).to_csv(tmp_path / f"docids_shard_{s:05d}.csv", index=False)

  emb, ids = faiss_indexer.get_files(tmp_path)
  index = faiss_indexer.create_index(num_clusters=8, dim=dim)
  docids = faiss_indexer.process_shards(index, emb, ids, training_sample_size=300)
  faiss_indexer.save_index_and_docids(index, docids, tmp_path)

  loaded = faiss.read_index(str(tmp_path / "faiss_index_175.bin"))
  assert loaded.ntotal == 600 and loaded.d == dim
  loaded.nprobe = 8
  q = np.load(tmp_path / "embeddings_shard_00001.npy")[:1]
  _, idx = loaded.search(q, 1)
  assert docids[idx[0][0]] == "Page1#0"
