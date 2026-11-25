import math
from pathlib import Path 
import numpy as np 
import pandas as pd 
import torch
from sentence_transformers import SentenceTransformer

#Purpose: Takes your 17M Wikipedia sentences and converts each one into a 384-dimensional vector using BGE model. Saves as .npy files (vectors) and .csv files (document IDs).


from claimscan.config import config

PARQUET = config.corpus
OUTDIR = config.retriever_index_dir
MODEL = config.retriever_model
BATCH_SIZE = 2048
SHARD_SIZE = 100000


def main(index_name=None):
  # index_name picks an entry from dense_indexes in configs/default.yaml (default: bge_small,
  # i.e. the original MODEL / OUTDIR above). Run: python -m claimscan.retrieval.faiss_embeddings --index e5_base
  entry = config.dense_index(index_name)
  MODEL, OUTDIR = entry["model"], entry["dir"]

  # (moved from module level so importing this file doesn't crash without a GPU)
  print(torch.cuda.is_available())
  print(torch.cuda.device_count())
  if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

  outdir = Path(OUTDIR)
  outdir.mkdir(parents=True, exist_ok=True)

  df = pd.read_parquet(PARQUET, columns=["id", "line", "text"])
  df = df.sort_values(["id", "line"], kind="mergesort").reset_index(drop=True)
  if not {"id", "line", "text"}.issubset(df.columns):
    raise ValueError("Missing Columns")
  
  text = ("passage: " + df["text"].astype(str)).tolist()
  docids = (df["id"].astype(str) + "#" + df["line"].astype("int64").astype(str)).tolist()

  N_sentences = len(docids)
  print(f"Total sentences: {N_sentences}")

  device = "cuda" if torch.cuda.is_available() else "cpu"

  print(f"Loading model '{MODEL}' on {device}")

  model = SentenceTransformer(MODEL, device = device)
  if device == "cuda":
    model = model.half()
  model.max_seq_length = 128
  dim = model.get_sentence_embedding_dimension()
  print("Device:", next(model.parameters()).device, "Dimensions:", dim)

  num_shards = math.ceil(N_sentences / SHARD_SIZE)
  
  
  for shard in range(num_shards):
    s0 = shard * SHARD_SIZE
    s1 = min((shard + 1) * SHARD_SIZE, N_sentences)
    shard_texts = text[s0:s1]
    shard_docids = docids[s0: s1]

    #shard_file and ids_file = the expected filenames for the current shard
    shard_file = outdir / f"embeddings_shard_{shard:05d}.npy" #checks for npy file of shard
    ids_file   = outdir / f"docids_shard_{shard:05d}.csv" #checks for csv of shard
    if shard_file.exists() and ids_file.exists():
      print(f"[skip] Shard {shard+1}/{num_shards} already present")
      continue

    embs_part = []
    for i in range(0, len(shard_texts), BATCH_SIZE):
      j = min(i + BATCH_SIZE, len(shard_texts)) 
      batch = shard_texts[i:j]
      with torch.inference_mode():
        vectors = model.encode(
          batch,
          convert_to_numpy=True,
          normalize_embeddings=True,
          batch_size = len(batch),
          show_progress_bar = False,
        )
      embs_part.append(vectors)

    X = np.vstack(embs_part)
    X = X.astype(np.float32, copy = False)

    np.save(outdir / f"embeddings_shard_{shard:05d}.npy", X)
    pd.DataFrame({"docid": shard_docids}).to_csv(outdir / f"docids_shard_{shard:05d}.csv", index=False)
    print(f"Shard {shard+1}/{num_shards}: saved {len(shard_texts):,} rows")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Embed the FEVER corpus into .npy shards for one dense index")
    parser.add_argument("--index", default=None, choices=sorted(config.dense_indexes), help="default: bge_small")
    main(parser.parse_args().index)