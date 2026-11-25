"""Verifier data prep (line-0 fix, lookups) and the train/val split. Toy data only: no artifacts, no GPU."""
import pandas as pd

from claimscan.evidence.select import index_corpus
from claimscan.verifier.prepare import create_training_data, get_negative_text, group_gold_pairs
from claimscan.verifier.train import split_examples

CORPUS = index_corpus(pd.DataFrame({
  "id": ["Barack_Obama", "Barack_Obama", "Paris", "Paris"],
  "line": [0, 1, 0, 1],
  "text": ["Obama was born in Hawaii.", "Obama went to Harvard.", "Paris is in France.", "Paris has museums."],
}))

EVIDENCE = pd.DataFrame({
  "id": [1, 1, 2, 3],
  "evidence_annotation_id": [10, 11, 20, 30],
  "evidence_wiki_url": ["Barack Obama", "Barack_Obama", "Paris", None],
  "evidence_sentence_id": [-1, -1, 1, -1],  # -1 means line 0 (HF loader quirk)
})


class FakeRetriever:
  def __init__(self, hits):
    self.hits = hits

  def search(self, query, k=20):
    return self.hits[:k]


def test_gold_pairs_fix_line0_and_dedupe():
  gold = group_gold_pairs(EVIDENCE)
  assert gold == {1: [("Barack_Obama", 0)], 2: [("Paris", 1)]}  # NEI claim 3 skipped, duplicate set merged


def test_gold_line0_not_used_as_negative():
  retriever = FakeRetriever([{"id": "Barack_Obama", "line": 0}, {"id": "Barack_Obama", "line": 1}])
  negs = get_negative_text("Obama was born in Hawaii", [("Barack_Obama", 0)], CORPUS, retriever)
  assert negs == ["Obama went to Harvard."]


def test_create_training_data_finds_line0_evidence():
  claims = pd.DataFrame({"id": [1, 2], "claim": ["Obama was born in Hawaii", "Paris has no museums"],
                         "label_text": ["TRUE", "FALSE"]})
  retriever = FakeRetriever([{"id": "Paris", "line": 0}])
  ex = create_training_data(claims, group_gold_pairs(EVIDENCE), CORPUS, retriever, per_class=5)
  by_label = {}
  for e in ex:
    by_label.setdefault(e["label"], []).append(e)
  assert [e["evidence"] for e in by_label["TRUE"]] == [["Obama was born in Hawaii."]]
  assert [e["evidence"] for e in by_label["FALSE"]] == [["Paris has museums."]]
  assert len(by_label["NOT ENOUGH INFO"]) == 2


def _examples():
  # Same layout prepare.py writes: all TRUE, then FALSE, then NEI (NEI claims overlap TRUE/FALSE claims).
  ex = [{"claim_id": i, "claim": f"c{i}", "evidence": ["e"], "label": "TRUE"} for i in range(50)]
  ex += [{"claim_id": i, "claim": f"c{i}", "evidence": ["e"], "label": "FALSE"} for i in range(50, 100)]
  ex += [{"claim_id": i, "claim": f"c{i}", "evidence": ["n"], "label": "NOT ENOUGH INFO"} for i in range(0, 150, 2)]
  return ex


def test_split_has_every_class_and_no_claim_leak():
  examples = _examples()
  train, val = split_examples(examples)
  assert len(train) + len(val) == len(examples)
  labels = {"TRUE", "FALSE", "NOT ENOUGH INFO"}
  assert {e["label"] for e in val} == labels and {e["label"] for e in train} == labels
  assert not {e["claim_id"] for e in train} & {e["claim_id"] for e in val}
  assert 0.15 < len(val) / len(examples) < 0.25


def test_split_is_deterministic():
  assert split_examples(_examples()) == split_examples(_examples())
