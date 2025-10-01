from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

#Purpose: The actual search interface. Takes a text query, converts it to a vector, finds the most similar document vectors in the index, and returns the matching Wikipedia sentences.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

class FAISSRetriever:
  def __init__(self, index_dir, model_name = MODEL_NAME):
    self.index_dir = Path(index_dir)
    self.model_name = model_name

    self.model = SentenceTransformer(model_name, device = 'cpu')

    index_path = self.index_dir / "faiss_index_175.bin"

    if not index_path.exists():
      raise FileNotFoundError(f"Faiss Index Does Not Exist in {index_path}")
    
    print(f"Loading FAISS index from {index_path}")
    self.index = faiss.read_index(str(index_path))

    # Set search parameters for IVF index
    if hasattr(self.index, 'nprobe'):
      self.index.nprobe = 64  # Search 64 clusters (speed vs accuracy tradeoff)

    docids_path = self.index_dir / "all_docids_175.csv"

    if not docids_path.exists():
      raise FileNotFoundError(f"Docids Do Not Exist in {docids_path}")
    
    print(f"Loading document IDs from {docids_path}")
    docids_df = pd.read_csv(docids_path).astype(str)
    self.docids = docids_df['docid'].tolist()

    if len(self.docids) != self.index.ntotal:
      raise ValueError(f"Mismatch: {len(self.docids)} docids vs {self.index.ntotal} vectors in index")
    
    print(f"Loaded retriever with {self.index.ntotal:,} documents")
    print(f"Model: {model_name}")
    print(f"Index type: {type(self.index).__name__}")
  
  def encode_query(self, query_text):
    vector = self.model.encode([f"query: {query_text}"], convert_to_numpy= True, normalize_embeddings= True)
    return vector.astype(np.float32)
  
  def search(self, query, k = 50):
    query_vector = self.encode_query(query)

    scores, idx = self.index.search(query_vector, k = k)

    results = []
    for score, i in zip(scores[0], idx[0]):
      if i == -1:
        continue

      docid = self.docids[i]

      if "#" in docid:
        page, line = docid.rsplit("#", 1)
        try:
          line = int(line)

        except ValueError:
          line = -1
      else:
        page, line = docid, -1

      results.append({"id": str(page), "line": line, "score": round(float(score),6)})
    
    return results
  
  def batch_multiple_queries(self, queries, k = 50):
    query_vectors = []
    for query in queries:
      vector = self.encode_query(query)
      query_vectors.append(vector[0])
    query_matrix = np.vstack(query_vectors)

    scores, idx = self.index.search(query_matrix, k)
    all_results = []
    for i, (query_scores, query_index) in enumerate(zip(scores, idx)):
      query_results = []
      for score, idx in zip(query_scores, query_index):
        if idx == -1:
          continue
                    
        docid = self.docids[idx]
                
        if "#" in docid:
          page, line = docid.rsplit("#", 1)
          try:
            line = int(line)
          except ValueError:
            line = -1
        else:
          page, line = docid, -1
                
        query_results.append({
                    "id": str(page),
                    "line": line,
                    "score": float(score)
                })
            
      all_results.append(query_results)
    return all_results