"""Claim span -> best evidence span (claimscan/neurosymbolic_layer/natlog/align.py), with a bag-of-words encoder.
split_spans is stubbed ("|" separates spans), so no spaCy is needed."""
import pytest

from claimscan.neurosymbolic_layer.natlog import align as A
from stub_natlog import bow_encode


@pytest.fixture(autouse=True)
def pipe_spans(monkeypatch):
  monkeypatch.setattr(A, "split_spans", lambda s: [x.strip() for x in s.split("|")])


def test_best_span_over_all_sentences():
  ev = ["Michelle Obama | was born | in Chicago", "Barack Obama | was born | in Hawaii"]
  out = A.align(["Barack Obama", "in Hawaii"], ev, bow_encode)
  assert [(a.claim_span, a.evidence_idx, a.evidence_span) for a in out] == [
    ("Barack Obama", 1, "Barack Obama"), ("in Hawaii", 1, "in Hawaii")]
  assert out[0].sim == pytest.approx(1.0)


def test_multi_granular_candidates_match_longer_spans():
  out = A.align(["was born in Hawaii"], ["Barack Obama | was born | in Hawaii"], bow_encode)
  assert out[0].evidence_span == "was born in Hawaii" and out[0].sim == pytest.approx(1.0)


def test_below_min_sim_is_unaligned():
  out = A.align(["Barack Obama", "in Kenya"], ["Barack Obama | was born | in Hawaii"], bow_encode, min_sim=0.6)
  assert (out[0].evidence_idx, out[0].evidence_span) == (0, "Barack Obama")
  # "in Kenya" / "in Hawaii" share only "in": cosine 0.5 < 0.6
  assert (out[1].evidence_idx, out[1].evidence_span) == (None, None) and out[1].sim == pytest.approx(0.5)


def test_no_evidence():
  assert [(a.evidence_idx, a.evidence_span) for a in A.align(["Barack Obama"], [], bow_encode)] == [(None, None)]
  assert A.align([], ["Barack Obama"], bow_encode) == []
