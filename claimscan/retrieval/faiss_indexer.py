from pathlib import Path
import numpy as np
import pandas as pd
import faiss
import time

# Configuration
EMBEDDING_DIR = r"D:\crisis-claim-analysis\artifacts\dense\faiss_bge_small_en_v1_5"
DIM = 384  # BGE-small-en-v1.5 embedding dimension
NUM_CLUSTERS = 4096
TRAINING_SAMPLE_SIZE = 500000
MAX_SHARDS = 175

#Purpose: Takes all those vector files and builds a searchable FAISS index. Creates 4096 clusters to make search faster (instead of checking all 17M vectors, it only checks relevant clusters).

def get_files(embedding_dir,  max_shards = MAX_SHARDS):
  embedding_dir = Path(embedding_dir)

  embedding_files = sorted(embedding_dir.glob("embeddings_shard_*.npy"))[:max_shards]
  docid_files = sorted(embedding_dir.glob("docids_shard_*.csv"))[:max_shards]

  if not embedding_files:
    raise FileNotFoundError("No embedding files found")
  
  if len(embedding_files) != len(docid_files):
    raise ValueError(f'{len(embedding_files)} embedding files does not equa {len(docid_files)} files')
  
  print(f'{len(embedding_files)} found (limited to {max_shards})')

  return embedding_files, docid_files

def create_index(num_clusters = NUM_CLUSTERS):
  index = faiss.IndexIVFFlat(faiss.IndexFlatIP(DIM), DIM, num_clusters)
  print(f"Created IndexIVFFlat with {num_clusters} clusters")
  return index

#we are using IndexIVFFlat now, so we need to train on samples and get clusters
def process_shards(index, embedding_files, docid_files, training_sample_size = TRAINING_SAMPLE_SIZE):
  print("Collecting training data...")
  training_vectors = []
  training_count = 0

  all_docids = []
  total_added = 0

  for idx, (emb_f, docid_f) in enumerate(zip(embedding_files, docid_files)):
    print(f"Processing shard {idx+1}/{len(embedding_files)}: {emb_f.name}")

    X = np.load(emb_f).astype(np.float32)
    docid_df = pd.read_csv(docid_f).astype(str)
    shard_docids = docid_df['docid'].tolist()
    

    if len(X) != len(shard_docids):
      raise ValueError(f"Shard {idx}: {len(X)} embeddings but {len(shard_docids)} docids")
    
    if training_count < training_sample_size:
      training_vectors.append(X.copy())
      training_count += len(X)
      print(f"    Added to training set: {len(X):,} vectors (training total: {training_count:,})")
      if training_count >= training_sample_size and not index.is_trained:
        print(f"Training index on {training_count:,} vectors...")
        all_training_data = np.vstack(training_vectors)

        train_start = time.perf_counter()
        index.train(all_training_data)
        train_time = time.perf_counter() - train_start
        print(f"Training completed in {train_time:.2f} seconds")

        del training_vectors
        del all_training_data

    
    if index.is_trained:
      all_docids.extend(shard_docids)
      index.add(X)
      print(f"    Added to index: {len(X):,} vectors")
    else:
      print(f"    Skipping add (index not trained yet)")

    total_added += len(X)
    print(f"    Total processed: {total_added:,}")
    del X
  return all_docids


def save_index_and_docids(index, docids, output_dir):
  output_dir = Path(output_dir)

  index_path = output_dir / "faiss_index_175.bin"
  print(f"Saving FAISS index to {index_path}")
  faiss.write_index(index, str(index_path))

  docids_path = output_dir / "all_docids_175.csv"
  print(f"Saving document IDs to {docids_path}")
  pd.DataFrame({'docid': docids}).to_csv(docids_path, index = False)

  print(f"Saved index with {index.ntotal:,} vectors")
  print(f"Files created:")
  print(f"  Index: {index_path}")
  print(f"  DocIDs: {docids_path}")

def main():
  start_time = time.perf_counter()
  
  print("Finding shard files...")
  embedding_files, docid_files = get_files(embedding_dir=EMBEDDING_DIR)

  print("\nCreating FAISS index...")
  index = create_index()

  print("\nProcessing all shards (training + adding in single pass)...")
  all_docids = process_shards(index, embedding_files, docid_files, training_sample_size = TRAINING_SAMPLE_SIZE)
  
  print("\nSaving index and document IDs...")
  save_index_and_docids(index, all_docids, output_dir=EMBEDDING_DIR)
  
  total_time = time.perf_counter() - start_time
  print(f"\n=== COMPLETED ===")
  print(f"Total time: {total_time:.2f} seconds")
  print(f"Built index for {len(all_docids):,} documents")

if __name__ == "__main__":
    main()
