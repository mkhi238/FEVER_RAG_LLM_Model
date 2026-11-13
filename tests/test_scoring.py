"""Pins the evidence-scoring heuristics (neurosymbolic_layer/, evidence/select.py) so changes can't drift silently."""
import pandas as pd
import pytest

from claimscan.neurosymbolic_layer import (MAX_MULT, MIN_MULT, RANK_C, content_words, filter_evidence_by_claim_type,
                                        score_evidence_quality, title_words)
from claimscan.evidence.select import get_evidence


def test_claim_type_multipliers():
  assert filter_evidence_by_claim_type("dickens wrote oliver twist", "oliver twist was written by charles dickens") == 1.5
  assert filter_evidence_by_claim_type("obama was born in hawaii", "he was born in honolulu") == 1.3
  assert filter_evidence_by_claim_type("tokyo is the capital of japan", "tokyo is the capital of japan") == 1.3
  assert filter_evidence_by_claim_type("biden is the current president", "biden served as vice president") == 0.6
  assert filter_evidence_by_claim_type("cats can fly", "cats are mammals") == 1.0


def test_claim_type_rules_are_whole_word_and_combine():
  # 'was' alone is not past-role evidence; 'known' does not contain the word 'now'
  assert filter_evidence_by_claim_type("biden is now president", "biden was elected in 2020") == 1.0
  assert filter_evidence_by_claim_type("he is known for jazz", "he served in the army") == 1.0
  assert filter_evidence_by_claim_type("the authority approved it", "it was written by staff") == 1.0
  # every matching rule applies (the superlative rule used to return early)
  assert filter_evidence_by_claim_type("the largest city located in texas",
                                       "houston is the largest city located in texas") == pytest.approx(1.2 * 1.3)


def test_hedge_words_are_whole_word():
  plain = score_evidence_quality("cats can fly", "cats live a long and different life", 0)
  hedged = score_evidence_quality("cats can fly", "cats live a long life if fed well", 0)
  assert plain > hedged                      # 'if' still penalised ...
  assert plain == pytest.approx(score_evidence_quality("cats can fly", "cats live a long and happy time", 0))  # ... 'life' is not


def test_quality_score_formula():
  # (1 + overlap) * capital rule / (rank + RANK_C); overlap = {"tokyo","capital","japan"} all present -> 2.0
  s = score_evidence_quality("Tokyo is the capital of Japan", "Tokyo is the capital of Japan and its largest city", 0)
  assert s == pytest.approx(2.0 * 1.3 / RANK_C)


def test_rank_prior_and_clamp():
  ev = "cats are small furry domestic animals"
  assert score_evidence_quality("cats can fly", ev, 0) > score_evidence_quality("cats can fly", ev, 5)
  best = score_evidence_quality("Tokyo is the capital of Japan", "Tokyo is the capital of Japan", 0, page="Tokyo")
  assert best == pytest.approx(MAX_MULT / RANK_C)          # 2 * 1.5 * 1.3 = 3.9, clamped
  worst = score_evidence_quality("cats can fly", "if dogs would be perhaps quiet animals", 0)
  assert worst == pytest.approx(MIN_MULT / RANK_C)         # 0.5 * 0.75 = 0.375, clamped


def test_words_and_titles():
  assert content_words("The UK left the EU in 2020, said Fox.") == {"uk", "left", "eu", "2020", "said", "fox"}
  assert "obama" in content_words("Obama, the president")   # punctuation stripped
  assert title_words("Savages_-LRB-2012_film-RRB-") == {"savages"}
  assert title_words("Barack_Obama") == {"barack", "obama"}


def test_page_title_helps_pronoun_sentences():
  ev = "He was born in Honolulu and raised there by family"
  with_title = score_evidence_quality("Barack Obama was born in Honolulu", ev, 0, page="Barack_Obama")
  other_page = score_evidence_quality("Barack Obama was born in Honolulu", ev, 0, page="Some_Singer")
  assert with_title > other_page


def test_short_and_disambiguation_penalties():
  base = score_evidence_quality("cats can fly", "cats are small furry domestic animals", 0)
  assert score_evidence_quality("cats can fly", "cats", 0) < base
  assert score_evidence_quality("cats can fly", "cats may refer to disambiguation page entries here", 0) < base


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
  assert len(out) == 3                       # only one page: the per-page cap backfills
  assert out[0] == "tokyo is the capital of japan"
  assert "it rains" not in out


def test_get_evidence_at_most_two_per_page():
  corpus = pd.DataFrame({
    "id": ["Tokyo"] * 3 + ["Japan"],
    "line": [0, 1, 2, 0],
    "text": ["tokyo is the capital of japan", "tokyo is a large city in japan",
             "tokyo is the seat of the government of japan", "japan has its capital in the city of tokyo"],
  })
  hits = [{"id": "Tokyo", "line": l, "score": 0.0} for l in range(3)] + [{"id": "Japan", "line": 0, "score": 0.0}]
  ids = [d for _, d in get_evidence("Tokyo is the capital of Japan", hits, corpus, with_ids=True)]
  assert len(ids) == 3 and "Japan#0" in ids and sum(d.startswith("Tokyo#") for d in ids) == 2


def test_get_evidence_ignores_score_direction():
  # same order, BM25-style (higher = better) vs L2-style (lower = better) scores -> same picks
  corpus = pd.DataFrame({"id": ["A", "B", "C", "D"], "line": [0] * 4,
                         "text": ["cats are small furry domestic animals"] * 4})
  order = ["A", "B", "C", "D"]
  bm25 = [{"id": p, "line": 0, "score": 40.0 - i} for i, p in enumerate(order)]
  l2 = [{"id": p, "line": 0, "score": 0.1 * (i + 1)} for i, p in enumerate(order)]
  assert get_evidence("cats can fly", bm25, corpus, with_ids=True) == get_evidence("cats can fly", l2, corpus, with_ids=True)


def test_get_evidence_ids_and_indexed_corpus_match_scan():
  from claimscan.evidence.select import index_corpus
  corpus = pd.DataFrame({"id": ["Tokyo", "Tokyo", "Paris"], "line": [0, 1, 0],
                         "text": ["tokyo is the capital of japan", "tokyo has many trains running", "paris is the capital of france"]})
  hits = [{"id": "Tokyo", "line": 1, "score": 0.5}, {"id": "Tokyo", "line": 0, "score": 0.5},
          {"id": "Nope", "line": 0, "score": 0.1}]
  scanned = get_evidence("tokyo capital japan", hits, corpus, with_ids=True)
  assert scanned[0] == ("tokyo is the capital of japan", "Tokyo#0") and len(scanned) == 2
  assert get_evidence("tokyo capital japan", hits, index_corpus(corpus), with_ids=True) == scanned
  assert get_evidence("tokyo capital japan", hits, corpus) == [t for t, _ in scanned]
