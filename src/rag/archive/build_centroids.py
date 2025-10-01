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
    m = X.mean(axis = 0) #X has shape (N, 384) (N sentences in the shard, each 384-dim embedding).  X.mean(axis=0) computes the mean along the first axis, so you get back a vector of shape (384,), not a single scalar.
    n = np.linalg.norm(m)
    if n > 0:
      #Your BGE embeddings were saved as normalize_embeddings=True. That means each individual vector has unit length (‖v‖₂ = 1).
      #To use inner product (space="ip") in hnswlib as a proxy for cosine similarity, you want all vectors you compare to also be unit normalized.
      #If you just take the raw mean (m = X.mean(axis=0)), the result will not be unit length. Its magnitude could bias similarity scores.
      #just because the unit vectors are normalized, doesnt mean that the mean of the vectors are normalized (arithmetic mean may not be normalized)
      m = m/n # normalize the mean vector to unit length (m = m / ‖m‖₂  (if norm > 0).) to renormalize
    centroids.append(m.astype(np.float32))
  C = np.vstack(centroids)
  np.save(OUT, C)
  print(f"saved centroids: {OUT} with shape {C.shape}")

if __name__ == "__main__":
    main()
    