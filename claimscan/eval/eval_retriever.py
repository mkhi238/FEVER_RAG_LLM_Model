"""Recall@k of the retrievers (faiss | bm25 | hybrid) on FEVER validation.

Was claimscan/retrieval/eval_recall.py. Gold now comes from claimscan.eval.gold, which
fixes FEVER's line-0-stored-as--1 quirk; recall numbers from before that fix are too low.

Run: python -m claimscan.eval.eval_retriever --retriever faiss bm25 hybrid --max-claims 200
"""
import time
import pandas as pd
from claimscan.config import config
from claimscan.eval.gold import load_claims, load_gold_pairs, norm_title
from claimscan.pipeline import load_retriever
# --------- PATHS ----------
DOCIDS_PATH = config.retriever_index_dir / config.retriever_docids_file  # FAISS coverage
# --------------------------

# --------- KNOBS ----------
K_SENT               = 50      # global top-k to keep
LOG_EVERY            = 50
# --------------------------

def load_claims_validation():
    print("[eval] loading claims…")
    df = load_claims("validation")
    print(f"[eval] validation claims loaded: {len(df):,}")
    return df

def load_gold():
    print("[eval] loading gold evidence…")
    gold = load_gold_pairs()
    print(f"[eval] gold entries: {len(gold):,}")
    return gold

def faiss_pages(retr=None):
    """Pages the FAISS index covers (shards 0-174 only). Every retriever is scored on the
    claims whose gold pages are in here, so FAISS / BM25 / hybrid numbers are comparable."""
    docids = getattr(retr, "docids", None) or getattr(getattr(retr, "faiss", None), "docids", None)
    if docids is None:
        docids = pd.read_csv(DOCIDS_PATH, usecols=["docid"], dtype=str)["docid"]
    return {norm_title(doc.rsplit("#", 1)[0]) for doc in docids}

def eval_recall(
    retriever="faiss",
    k_list=(1,5,10,20),
    k_sent=K_SENT,
    log_every=LOG_EVERY,
    max_claims=None,
    faiss_coverage_only=True,
):
    """Recall@k on FEVER validation for retriever = faiss | bm25 | hybrid."""
    t0 = time.perf_counter()
    claims = load_claims_validation()
    gold = load_gold()

    print(f"[eval] initializing {retriever} retriever...")
    retr = load_retriever(config, retriever)
    print("[eval] retriever ready.")

    if not faiss_coverage_only:
        available_pages = None
    else:
        print("[eval] loading pages covered by the FAISS index...")
        available_pages = faiss_pages(retr)
    if available_pages is not None:
        print(f"[eval] FAISS index covers {len(available_pages):,} unique pages")
    
    # Filter gold to only include claims where evidence exists in our index
    print("[eval] filtering gold evidence to match available documents...")
    original_gold_count = len(gold)
    filtered_gold = {}
    coverage_stats = {"total_claims": 0, "claims_with_coverage": 0, "evidence_found": 0, "evidence_total": 0}
    
    for cid, pairs in gold.items():
        coverage_stats["total_claims"] += 1
        coverage_stats["evidence_total"] += len(pairs)
        
        # Check which evidence pairs exist in our index
        available_pairs = [(page, line) for page, line in pairs if available_pages is None or page in available_pages]
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

        # Search
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

    print(f"[eval] retriever={retriever}  claims evaluated: {total}")
    for k in k_list:
        print(f"[eval] Recall@{k}: {recalls_any[k] / total:.3f}")
    print(f"[eval] done. total time: {time.perf_counter() - t0:.2f}s")
    return {k: recalls_any[k] / total for k in k_list}

def eval_recall_faiss(**kwargs):
    return eval_recall("faiss", **kwargs)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Recall@k on FEVER validation")
    ap.add_argument("--retriever", nargs="+", choices=["faiss", "bm25", "hybrid"], default=["faiss"],
                    help="one or more; e.g. --retriever faiss bm25 hybrid")
    ap.add_argument("--max-claims", type=int, default=200)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    ap.add_argument("--all-claims", action="store_true",
                    help="don't restrict to claims whose gold pages are in the FAISS index")
    args = ap.parse_args()
    summary = {}
    for name in args.retriever:
        summary[name] = eval_recall(name, k_list=tuple(args.k), k_sent=max(K_SENT, *args.k),
                                    max_claims=args.max_claims, faiss_coverage_only=not args.all_claims)
    if len(summary) > 1:
        print("\n[eval] summary  " + "  ".join(f"R@{k:<4}" for k in args.k))
        for name, r in summary.items():
            if r:
                print(f"[eval] {name:<8} " + "  ".join(f"{r[k]:.3f}" for k in args.k))