"""SentenceStore: build from a tiny corpus parquet, look sentences up, and drive the pipeline with it."""
import pandas as pd
import torch

from claimscan.evidence.store import SentenceStore, build
from claimscan.pipeline import Services, fact_check_cleaned, fact_check_detailed

CORPUS = pd.DataFrame({"id": ["Tokyo", "Tokyo", "Japan"], "line": [0, 1, 0],
                       "text": ["tokyo is the capital of japan", "it is large", "japan is an island country"]})
HITS = [{"id": "Tokyo", "line": 0, "score": 1.0}, {"id": "Japan", "line": 0, "score": 0.9},
        {"id": "Nowhere", "line": 3, "score": 0.1}]


class Retr:
  def search(self, q, k=30):
    return HITS[:k]


class Verifier:
  def encode_text(self, claim, evidence):
    return {"input_ids": torch.zeros(1, 4, dtype=torch.long), "attention_mask": torch.ones(1, 4, dtype=torch.long)}

  def __call__(self, input_ids, attention_mask):
    return torch.tensor([[5.0, 0.0, 0.0]])


def _store(tmp_path):
  src, out = tmp_path / "corpus.parquet", tmp_path / "s.sqlite"
  CORPUS.to_parquet(src)
  build(src, out, batch=2)
  return SentenceStore(out)


def test_build_and_lookup(tmp_path):
  s = _store(tmp_path)
  assert s.count() == 3
  df = s.for_hits(HITS)
  assert list(df.index) == [("Japan", 0), ("Tokyo", 0)]   # missing sentence is just absent
  assert df.loc[("Tokyo", 0), "text"] == "tokyo is the capital of japan"


def test_store_gives_same_answer_as_dataframe(tmp_path):
  s = _store(tmp_path)
  with_store = fact_check_detailed("Tokyo is the capital of Japan", Services(Verifier(), Retr(), s))
  with_df = fact_check_detailed("Tokyo is the capital of Japan", Services(Verifier(), Retr(), CORPUS))
  assert with_store == with_df
  assert with_store.label == "TRUE" and len(with_store.evidence) == 2
  assert fact_check_cleaned("Tokyo is the capital of Japan", Services(Verifier(), Retr(), s), return_evidence=True) \
    == ("TRUE", [d for _, d in with_store.evidence])


def test_verdict_carries_all_hits_with_text(tmp_path):
  v = fact_check_detailed("Tokyo is the capital of Japan", Services(Verifier(), Retr(), _store(tmp_path)))
  assert [h["docid"] for h in v.hits] == ["Tokyo#0", "Japan#0", "Nowhere#3"]
  assert [h["rank"] for h in v.hits] == [0, 1, 2]
  assert v.hits[0]["text"] == "tokyo is the capital of japan" and v.hits[2]["text"] is None
