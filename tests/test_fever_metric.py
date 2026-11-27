"""FEVER metric (vendored from huggingface/evaluate PR #710), its adapter, and the gold loaders."""
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("evaluate")

from claimscan.eval import fever_eval  # noqa: E402
from claimscan.eval.fever_eval import load_fever_metric, to_prediction, to_reference  # noqa: E402
from claimscan.data import fever_db  # noqa: E402
from claimscan.eval.gold import fix_line, load_claims, load_gold_pairs, load_gold_sets  # noqa: E402

# The upstream unittest suite, run unchanged against the vendored copy.
sys.path.insert(0, str(fever_eval.METRIC_DIR))
from test_fever import TestFEVER  # noqa: E402,F401  (pytest collects the unittest class)
sys.path.pop(0)


@pytest.fixture(scope="module")
def metric():
  return load_fever_metric()


def test_vendored_metric_loads_through_evaluate(metric):
  r = metric.compute(predictions=[{"label": "SUPPORTED", "evidence": ["E1", "E2"]}],
                     references=[{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"], ["E3", "E4"]]}])
  assert r["fever_score"] == 1.0 and r["evidence_recall"] == 0.5


def test_labels_map_to_fever_names():
  assert to_prediction("TRUE", [])["label"] == "SUPPORTED"
  assert to_reference("FALSE", [["A#0"]])["label"] == "REFUTED"
  assert to_reference("NOT ENOUGH INFO", [])["label"] == "NOT ENOUGH INFO"


def test_correct_nei_counts_toward_fever_score(metric):
  ref = to_reference("NOT ENOUGH INFO", [])
  assert ref["evidence_sets"] == [[]]
  r = metric.compute(predictions=[to_prediction("NOT ENOUGH INFO", ["X#1"])], references=[ref])
  assert r["fever_score"] == 1.0


def test_only_first_five_predicted_sentences_count(metric):
  ids = [f"P#{i}" for i in range(10)]
  assert len(to_prediction("TRUE", ids)["evidence"]) == 5
  r = metric.compute(predictions=[to_prediction("TRUE", ids)], references=[to_reference("TRUE", [["P#7"]])])
  assert r["label_accuracy"] == 1.0 and r["fever_score"] == 0.0


def test_prediction_ids_are_title_normalized():
  assert to_prediction("TRUE", ["Barack Obama#0"])["evidence"] == ["Barack_Obama#0"]


def test_fix_line_maps_minus_one_to_zero():
  assert fix_line(-1) == 0 and fix_line(7) == 7


def test_gold_sets_group_by_annotation_and_fix_line0(tmp_path):
  claims = pd.DataFrame({"id": [1, 2], "split": "validation", "claim": ["a", "b"], "label": ["TRUE", "NOT ENOUGH INFO"]})
  ev = pd.DataFrame({"claim_id": [1, 1, 1], "set_id": [10, 10, 11],
                     "page": ["Nikolaj_Coster-Waldau", "Fox_Broadcasting_Company", "Nikolaj_Coster-Waldau"],
                     "line": [7, -1, 7]})
  fever_db.write_claims_and_evidence(claims, ev, tmp_path / "f.sqlite")
  sets = load_gold_sets(db=tmp_path / "f.sqlite")
  assert sets == {1: [["Fox_Broadcasting_Company#0", "Nikolaj_Coster-Waldau#7"], ["Nikolaj_Coster-Waldau#7"]]}
  assert list(load_claims("val", db=tmp_path / "f.sqlite")["id"]) == [1, 2]


def test_gold_pairs_fix_line0(tmp_path):
  claims = pd.DataFrame({"id": [5], "split": "train", "claim": ["t"], "label": ["TRUE"]})
  ev = pd.DataFrame({"claim_id": [5, 5, 5], "set_id": [1, 1, 2], "page": ["Tokyo"] * 3, "line": [-1, 3, 0]})
  fever_db.write_claims_and_evidence(claims, ev, tmp_path / "f.sqlite")
  assert load_gold_pairs(db=tmp_path / "f.sqlite") == {5: [("Tokyo", 0), ("Tokyo", 3)]}


def test_evaluate_pipeline_with_stub_services(monkeypatch):
  from claimscan.eval import gold
  claims = pd.DataFrame({"id": [1, 2], "claim": ["a", "b"], "label_text": ["TRUE", "NOT ENOUGH INFO"]})
  monkeypatch.setattr(gold, "load_claims", lambda split: claims)
  monkeypatch.setattr(gold, "load_gold_sets", lambda ids: {1: [["A#0"]]})
  answers = {"a": ("TRUE", ["A#0", "B#3"]), "b": ("NOT ENOUGH INFO", [])}
  import claimscan.pipeline as pipeline
  monkeypatch.setattr(pipeline, "fact_check_cleaned", lambda c, s, **kw: answers[c])
  r = fever_eval.evaluate_pipeline(max_claims=None, services=object())
  assert r["label_accuracy"] == 1.0 and r["fever_score"] == 1.0
  assert r["evidence_precision"] == pytest.approx(0.5) and r["evidence_recall"] == 1.0
