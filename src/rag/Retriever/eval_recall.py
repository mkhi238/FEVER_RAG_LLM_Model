from pathlib import Path
import json, time, unicodedata, re
import numpy as np
import pandas as pd
from faiss_sentence_retriever import FAISSRetriever  # Updated import
#Emb: Script 1: Wikipedia text → 384-dim vectors (offline, one-time)
#indexer: Script 2: Vectors → FAISS search index (offline, one-time) -simplifies the search a lot since we no longer need to go into all cluses
#find cluster centroids, and will then just map to closest centroids
#Querier: Script 3: Query text → search results (real-time, during training/inference)
#ok so 1 creates the embs, 2 takes embs and makes centroids, 3 actually converts a sentence into a  vector and maps to ac entroid
# --------- PATHS ----------
CLAIMS_DIR = Path(r"D:\crisis-claim-analysis\data\processed\fever_claims_parquet")
GOLD_PATH  = Path(r"D:\crisis-claim-analysis\data\processed\fever_gold.json")
INDEX_DIR = Path(r"D:\crisis-claim-analysis\artifacts\dense\faiss_bge_small_en_v1_5")  # BGE path
MODEL_NAME = "BAAI/bge-small-en-v1.5"  # BGE model
# --------------------------

# --------- KNOBS ----------
K_SENT               = 50      # global top-k to keep
LOG_EVERY            = 50
# --------------------------

def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).strip()
    return re.sub(r"\s+", "_", s)

def load_claims_validation():
    print("[eval] loading claims…")
    path = CLAIMS_DIR
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files in {path}")
        path = files[0]
        print(f"[eval] using claims file: {path.name}")
    
    df = pd.read_parquet(path, columns=["id", "claim", "split"])
    df["split"] = df["split"].replace({"val": "validation", "labelled_dev": "validation", "paper_test": "test"})
    df = df[df["split"] == "validation"].copy()
    df["id"] = df["id"].astype(int)
    df["claim"] = df["claim"].astype(str)
    print(f"[eval] validation claims loaded: {len(df):,}")
    return df.reset_index(drop=True)

def load_gold():
    print("[eval] loading gold evidence…")
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    gold_norm = {}
    for k, pairs in raw.items():
        k_int = int(k)
        gold_norm[k_int] = [(norm_title(pid), int(line)) for pid, line in pairs]
    print(f"[eval] gold entries: {len(gold_norm):,}")
    return gold_norm

def eval_recall_faiss(
    k_list=(1,5,10,20),
    k_sent=K_SENT,
    log_every=LOG_EVERY,
    max_claims=None
):
    t0 = time.perf_counter()
    claims = load_claims_validation()
    gold = load_gold()

    print(f"[eval] initializing FAISS retriever...")
    retr = FAISSRetriever(index_dir=INDEX_DIR, model_name=MODEL_NAME)
    print("[eval] retriever ready.")
    
    # Load available documents from the index
    print("[eval] loading available documents from index...")
    available_docs = set(retr.docids)
    available_pages = {doc.split("#")[0] for doc in available_docs}
    print(f"[eval] index contains {len(available_pages):,} unique pages")
    
    # Filter gold to only include claims where evidence exists in our index
    print("[eval] filtering gold evidence to match available documents...")
    original_gold_count = len(gold)
    filtered_gold = {}
    coverage_stats = {"total_claims": 0, "claims_with_coverage": 0, "evidence_found": 0, "evidence_total": 0}
    
    for cid, pairs in gold.items():
        coverage_stats["total_claims"] += 1
        coverage_stats["evidence_total"] += len(pairs)
        
        # Check which evidence pairs exist in our index
        available_pairs = [(page, line) for page, line in pairs if page in available_pages]
        coverage_stats["evidence_found"] += len(available_pairs)
        
        # Keep claim if at least one evidence document is available
        if available_pairs:
            filtered_gold[cid] = available_pairs
            coverage_stats["claims_with_coverage"] += 1
    
    print(f"[eval] coverage analysis:")
    print(f"  Original claims with gold: {original_gold_count:,}")
    print(f"  Claims with evidence in index: {len(filtered_gold):,}")
    print(f"  Evidence coverage: {coverage_stats['evidence_found']}/{coverage_stats['evidence_total']} ({coverage_stats['evidence_found']/coverage_stats['evidence_total']:.1%})")
    print(f"  Claim coverage: {coverage_stats['claims_with_coverage']}/{coverage_stats['total_claims']} ({coverage_stats['claims_with_coverage']/coverage_stats['total_claims']:.1%})")
    
    # Filter claims to only those with available evidence
    claims = claims[claims["id"].isin(filtered_gold.keys())].reset_index(drop=True)
    if max_claims is not None:
        claims = claims.head(max_claims).copy()
    print(f"[eval] filtered claims for evaluation: {len(claims):,}")
    
    # Use filtered gold for evaluation
    gold = filtered_gold

    recalls_any = {k: 0 for k in k_list}
    total = 0
    t_loop = time.perf_counter()

    for i, row in enumerate(claims.itertuples(index=False), start=1):
        cid = int(row.id)
        q = row.claim
        gold_pairs = set(gold.get(cid, []))
        if not gold_pairs:
            continue

        # Search with FAISS
        t_q = time.perf_counter()
        results = retr.search(q, k=k_sent)
        q_ms = (time.perf_counter() - t_q) * 1000.0

        retrieved_pairs = [(norm_title(r["id"]), int(r["line"])) for r in results]
        retrieved_pages = [p for (p, _) in retrieved_pairs[:20]]
        gold_pages = {p for (p, _) in gold_pairs}
        page_hit = any(p in gold_pages for p in retrieved_pages)

        # First-claim diagnostics
        if i == 1:
            print("[debug] retrieved pages:", retrieved_pages[:10])
            print("[debug] gold pages:", list(gold_pages)[:10])
            
            # off-by-one check (line neighbor)
            off_by_one = False
            if page_hit:
                gold_map = {}
                for p, l in gold_pairs:
                    gold_map.setdefault(p, set()).add(l)
                for p, l in retrieved_pairs[:50]:
                    if p in gold_map and ((l-1) in gold_map[p] or (l+1) in gold_map[p]):
                        off_by_one = True
                        break
            print(f"[debug] page_hit={page_hit}, off_by_one={off_by_one}")
            print("[debug] example retrieved:", retrieved_pairs[:5])
            print("[debug] example gold:", list(gold_pairs)[:5])

        total += 1
        for k in k_list:
            if gold_pairs.intersection(retrieved_pairs[:k]):
                recalls_any[k] += 1

        if i % log_every == 0:
            elapsed = time.perf_counter() - t_loop
            partials = " | ".join([f"R@{k}={recalls_any[k]/max(total,1):.3f}" for k in k_list])
            print(f"[eval] processed {i}/{len(claims)}; evaluated={total}; {partials}; "
                  f"{elapsed:.1f}s since last log; last_q={q_ms:.1f} ms")
            t_loop = time.perf_counter()

    if total == 0:
        print("[eval] No validation claims with gold evidence found.")
        return

    print(f"[eval] Claims evaluated: {total}")
    for k in k_list:
        print(f"[eval] Recall@{k}: {recalls_any[k] / total:.3f}")
    print(f"[eval] done. total time: {time.perf_counter() - t0:.2f}s")

if __name__ == "__main__":
    eval_recall_faiss(
        max_claims=200,
        k_sent=K_SENT,
        k_list=(1,5,10,20),
        log_every=LOG_EVERY
    )