import json

import pandas as pd
import pytest

from claimscan.retrieval.hybrid_retriever import HybridRetriever, rrf_fuse


def hits(*docids):
  return [{"id": d.split("#")[0], "line": int(d.split("#")[1]), "score": 1.0} for d in docids]


def test_rrf_scores_and_order():
  fused = rrf_fuse({"faiss": hits("A#0", "B#1"), "bm25": hits("B#1", "C#2")}, rrf_k=60)
  assert [(h["id"], h["line"]) for h in fused] == [("B", 1), ("A", 0), ("C", 2)]
  assert fused[0]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
  assert fused[1]["rrf_score"] == pytest.approx(1 / 61)
  assert fused[0]["score"] == pytest.approx(1 / fused[0]["rrf_score"])   # FAISS L2 direction
  assert [h["score"] for h in fused] == sorted(h["score"] for h in fused)
  assert fused[0]["faiss_rank"] == 2 and fused[0]["bm25_rank"] == 1
  assert fused[2]["faiss_rank"] is None and fused[2]["bm25_rank"] == 2


def test_rrf_tie_keeps_first_list_order_and_counts_duplicates_once():
  fused = rrf_fuse({"faiss": hits("A#0", "A#0"), "bm25": hits("B#0")}, rrf_k=60, k=5)
  assert [h["id"] for h in fused] == ["A", "B"]          # equal 1/61 each: faiss first
  assert fused[0]["rrf_score"] == pytest.approx(1 / 61)


class Stub:
  def __init__(self, results):
    self.results, self.asked = results, None

  def search(self, query, k=50):
    self.asked = k
    return self.results[:k]


def test_hybrid_asks_each_retriever_for_candidates_and_truncates():
  f, b = Stub(hits(*[f"F#{i}" for i in range(200)])), Stub(hits("F#3", "X#0"))
  out = HybridRetriever(f, b, rrf_k=60, candidates=100).search("q", k=10)
  assert f.asked == 100 and b.asked == 100 and len(out) == 10
  assert (out[0]["id"], out[0]["line"]) == ("F", 3)       # found by both
  assert set(out[0]) >= {"id", "line", "score"}           # same shape get_evidence expects


def test_hybrid_hits_rank_in_rrf_order_through_evidence_scoring():
  from claimscan.evidence.select import get_evidence
  corpus = pd.DataFrame({"id": ["A", "B", "C"], "line": [0, 0, 0],
                         "text": ["alpha fact one here", "alpha fact two here", "alpha fact three here"]})
  fused = rrf_fuse({"faiss": hits("C#0", "B#0", "A#0"), "bm25": hits("C#0", "B#0")}, rrf_k=60)
  assert get_evidence("alpha fact", fused, corpus) == ["alpha fact three here", "alpha fact two here", "alpha fact one here"]


def test_bm25_jsonl_and_lucene_search_end_to_end(tmp_path):
  pytest.importorskip("pyserini")
  from claimscan.retrieval import bm25_index
  from claimscan.retrieval.bm25_retriever import BM25Retriever

  corpus = tmp_path / "corpus.parquet"
  pd.DataFrame({
    "id": ["Barack_Obama", "Barack_Obama", "Tokyo", "Cat"],
    "line": [0, 1, 0, 0],
    "text": ["barack obama was born in honolulu hawaii", "he was the 44th president",
             "tokyo is the capital of japan", "cats are small mammals"],
  }).to_parquet(corpus)

  n = bm25_index.write_jsonl(corpus, tmp_path / "jsonl", part_rows=3)
  assert n == 4 and sorted(p.name for p in (tmp_path / "jsonl").iterdir()) == ["sentences_000.jsonl", "sentences_001.jsonl"]
  first = json.loads((tmp_path / "jsonl" / "sentences_000.jsonl").read_text().splitlines()[0])
  assert first == {"id": "Barack_Obama#0", "contents": "Barack Obama [SEP] Barack_Obama [SEP] barack obama was born in honolulu hawaii"}
  assert bm25_index.write_jsonl(corpus, tmp_path / "jsonl", part_rows=3) == 4   # rerun skips finished parts

  bm25_index.build_index(tmp_path / "jsonl", tmp_path / "index", threads=1)
  r = BM25Retriever(tmp_path / "index", use_rm3=False)
  top = r.search("capital of japan", k=2)
  assert (top[0]["id"], top[0]["line"]) == ("Tokyo", 0)
  assert BM25Retriever(tmp_path / "index", use_rm3=True).search("obama honolulu", k=1)[0]["id"] == "Barack_Obama"
