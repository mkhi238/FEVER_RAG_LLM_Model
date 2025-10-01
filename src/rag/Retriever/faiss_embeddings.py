import math
from pathlib import Path 
import numpy as np 
import pandas as pd 
import torch
from sentence_transformers import SentenceTransformer

#Purpose: Takes your 17M Wikipedia sentences and converts each one into a 384-dimensional vector using BGE model. Saves as .npy files (vectors) and .csv files (document IDs).


PARQUET = r"D:\crisis-claim-analysis\data\processed\fever_corpus_parquet"
OUTDIR = r"D:\crisis-claim-analysis\artifacts\dense\faiss_bge_small_en_v1_5"
MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 2048
SHARD_SIZE = 100000


print(torch.cuda.is_available())   
print(torch.cuda.device_count())  
print(torch.cuda.get_device_name(0))  


def main():
  outdir = Path(OUTDIR)
  outdir.mkdir(parents=True, exist_ok=True)

  df = pd.read_parquet(PARQUET, columns=["id", "line", "text"])
  df = df.sort_values(["id", "line"], kind="mergesort").reset_index(drop=True)
  if not {"id", "line", "text"}.issubset(df.columns):
    raise ValueError("Missing Columns")
  
  #SentenceTransformer.encode(...) is designed to take a list of Python strings, text must be str
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
  #dimensions chosen automatically by model
  dim = model.get_sentence_embedding_dimension()
  print("Device:", next(model.parameters()).device, "Dimensions:", dim)

  #Shards = chunks of your corpus that you save as separate files so you don’t hold everything in RAM at once.
  #Shards are for Lower memory use (encoder saves 100k sentences at a time)
  #Faster + safer I/O: Writing several ~70–80 MB files is smoother than one multi-GB blob.
  #Resumable: If something crashes at shard 3/20, you only redo that shard.
  num_shards = math.ceil(N_sentences / SHARD_SIZE)
  
  
  for shard in range(num_shards):
    #s ranges from 0 to num_shards - 1, not to 100,000. SHARD_SIZE = 100000 is the chunk size, not the loop bound.
    s0 = shard * SHARD_SIZE  #s = 0, 1, 2, ... num_shards-1 (so like 0*100,000, 1*100,000,....)
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
      #torch.inference_mode() - A PyTorch context that disables autograd and some bookkeeping → lower memory + a bit faster. 
      #It doesn’t return anything; it just changes how work inside the block runs.
      #model.encode(...) Runs the sentence embedding model on this mini-batch and returns a NumPy array of shape (batch_size, dim) (for BGE-small, dim=384
      #embs_part.append(vectors) Collect each mini-batch’s matrix in a Python list.
      with torch.inference_mode():
        vectors = model.encode(
          batch,
          convert_to_numpy=True,
          normalize_embeddings=True,
          batch_size = len(batch),
          show_progress_bar = False,
        )
      #embs_part.append(vectors) Collect each mini-batch’s matrix in a Python list.
      embs_part.append(vectors) #we appebd the 2048 to the 100 000

    #X = np.vstack(embs_part) Concatenate all those (batch_size, 384) blocks into one big matrix for the whole shard: shape (num_rows_in_shard, 384)
    #X = X.astype(np.float16, copy=False) Downcast to float16 to cut disk usage by ~2× (384 dims × 2 bytes). You’ll cast back to float32 when building HNSW (hnswlib expects float32).
    X = np.vstack(embs_part)
    X = X.astype(np.float32, copy = False)

    #np.save(...) Takes your big NumPy matrix X (shape ≈ [rows_in_shard, 384]) and writes it to a .npy file.
    #So shard 0 saves as embeddings_shard_00000.npy, shard 1 as embeddings_shard_00001.npy, and so on.
    np.save(outdir / f"embeddings_shard_{shard:05d}.npy", X) #save the 100 000 batch
    #pd.DataFrame(...).to_csv(...)
    #Wraps the shard’s docids in a 1-column DataFrame and saves them as a .csv.
    #The order is identical to the rows in X, so row i in the .npy corresponds to line i in the .csv.
    pd.DataFrame({"docid": shard_docids}).to_csv(outdir / f"docids_shard_{shard:05d}.csv", index=False)
    print(f"Shard {shard+1}/{num_shards}: saved {len(shard_texts):,} rows")


if __name__ == "__main__":
    main()