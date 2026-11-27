from pathlib import Path
import numpy as np

INDIR = Path(r"D:\crisis-claim-analysis\artifacts\dense\bge_small_en_v15")
OUT = INDIR / "shard_centroids.npy"

def main():
  emb_path = sorted(INDIR.glob("embeddings_shard_*.npy"))
  if not emb_path:
    raise RuntimeError("No embedding shard found")
  
  centroids = []
  for p in emb_path:
    X = np.load(p).astype(np.float32, copy = False) # shape (N, 384), already L2-normalized (normalize_embeddings=True)
    m = X.mean(axis = 0)
    n = np.linalg.norm(m)
    if n > 0:
      m = m/n # normalize the mean vector to unit length (m = m / ‖m‖₂  (if norm > 0).) to renormalize
    centroids.append(m.astype(np.float32))
  C = np.vstack(centroids)
  np.save(OUT, C)
  print(f"saved centroids: {OUT} with shape {C.shape}")

if __name__ == "__main__":
    main()
    