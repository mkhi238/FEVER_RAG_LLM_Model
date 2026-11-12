"""Pins the current evidence-scoring heuristics so the refactor (and later changes) can't drift silently."""
import pandas as pd
import pytest

from claimscan.evidence.scoring import filter_evidence_by_claim_type, score_evidence_quality
from claimscan.evidence.select import get_evidence


def test_claim_type_multipliers():
  assert filter_evidence_by_claim_type("dickens wrote oliver twist", "oliver twist was written by charles dickens") == 1.5
  assert filter_evidence_by_claim_type("obama was born in hawaii", "he was born in honolulu") == 1.3
  assert filter_evidence_by_claim_type("tokyo is the capital of japan", "tokyo is the capital of japan") == 1.3
  assert filter_evidence_by_claim_type("biden is the current president", "biden served as vice president") == 0.6
  assert filter_evidence_by_claim_type("cats can fly", "cats are mammals") == 1.0


def test_quality_score_formula():
  # 1/(0.5+0.01) * (1 + overlap) * type multiplier; overlap = {"tokyo","capital","japan"} all present -> 2.0
  s = score_evidence_quality("Tokyo is the capital of Japan", "Tokyo is the capital of Japan and its largest city", 0.5)
  assert s == pytest.approx(1 / 0.51 * 2.0 * 1.3)


def test_short_and_disambiguation_penalties():
  base = score_evidence_quality("cats can fly", "cats are small furry domestic animals", 0.5)
  assert score_evidence_quality("cats can fly", "cats", 0.5) < base
  assert score_evidence_quality("cats can fly", "cats may refer to disambiguation page entries here", 0.5) < base


def test_get_evidence_joins_corpus_and_keeps_top3():
  corpus = pd.DataFrame({
    "id": ["Tokyo", "Tokyo", "Tokyo", "Tokyo", "Paris"],
    "line": [0, 1, 2, 3, 0],
    "text": [
      "tokyo is the capital of japan",
      "tokyo has many trains running through every district",
      "it rains",
      "tokyo hosted the olympics in the summer of 1964",
      "paris is the capital of france",
    ],
  })
  hits = [{"id": "Tokyo", "line": l, "score": 0.5} for l in range(4)] + [{"id": "Missing", "line": 0, "score": 0.9}]
  out = get_evidence("Tokyo is the capital of Japan", hits, corpus)
  assert len(out) == 3
  assert out[0] == "tokyo is the capital of japan"
  assert "it rains" not in out
