from pathlib import Path
import numpy as np, pandas as pd, hnswlib
import os
NUM_THREADS = max(1, (os.cpu_count() or 4) - 1)
INDIRECT = r"D:\crisis-claim-analysis\artifacts\dense\bge_small_en_v15"
DIM = 384 
SPACE = "ip"
M = 24
EF_CONSTRUCTION = 200
EF_SEARCH = 512


#emb shard is [#sentences in shard, 384]
#embeddings_shard_00000.npy → shape (100,000, 384) embeddings_shard_00001.npy → shape (100,000, 384)
# 2048 rust referred to the GPU mini batch per time, but this amount got appended to the 100 000 total 

def main():
    indir = Path(INDIRECT)
    embdeddings = sorted(indir.glob("embeddings_shard_*.npy"))
    id_files  = sorted(indir.glob("docids_shard_*.csv"))
    assert embdeddings and len(embdeddings) == len(id_files)

    
    for shard_id, (embedded_f, id_f) in enumerate(zip(embdeddings, id_files)):
        out_idx = indir / f"hnsw_ip_{shard_id:05d}.index"
        out_labels = indir / f"labels_{shard_id:05d}.csv"

        if out_idx.exists() and out_labels.exists():
            continue
        
        #loading individual 100000 by 384 matrix
        #this 100000 x 384 matrix is the embeddings of all 100000 sentences in a batch
        X = np.load(embedded_f).astype(np.float32, copy = False)
        ids = pd.read_csv(id_f, usecols=["docid"])["docid"].tolist()
        # 100000 == len(ids) == 100000
        assert X.shape[0] == len(ids)
        print(f"Building HNSW for shard {shard_id}: {X.shape[0]:,} vectors")

        #it stores the top 50 of the 512 prio queue, and each vector inserted has 32 connections max of 500 potnetial? and query traverses a path (from prio queue) and find best vectors
        idx = hnswlib.Index(space=SPACE, dim=DIM)
        idx.init_index(max_elements=len(ids), ef_construction=EF_CONSTRUCTION, M=M)

        idx.add_items(X, np.arange(len(ids)), num_threads=NUM_THREADS)
        idx.set_ef(EF_SEARCH)
        idx.save_index(str(out_idx))
        pd.Series(ids, name="docid").to_csv(out_labels, index=False)
        print(f"Saved {out_idx.name} and {out_labels.name}")

#it will return something like [0, 17, 532] (internal row indices). You use those numbers to index into ids to recover the actual document ID
if __name__ == "__main__":
    main()