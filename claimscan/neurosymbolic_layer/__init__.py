"""Neurosymbolic constraint layer: keyword/structure rules that re-weight the neural retriever's hits.

    from claimscan.neurosymbolic_layer import score_evidence_quality

  text.py   word helpers (whole-word phrase tests, content words, page-title words)
  rules.py  the rules: hedges, overlap, page-title bonus, claim-type rules, structural penalties
  score.py  score_evidence_quality = retrieval rank prior x rules (used by evidence/select.py)
"""
from claimscan.neurosymbolic_layer.rules import (MAX_MULT, MIN_MULT, PAST_ROLE, SUPERLATIVES,
                                                 filter_evidence_by_claim_type, rule_multiplier,
                                                 structural_penalty)
from claimscan.neurosymbolic_layer.score import RANK_C, score_evidence_quality
from claimscan.neurosymbolic_layer.text import STOPWORDS, content_words, has_any, title_words

__all__ = ["MAX_MULT", "MIN_MULT", "PAST_ROLE", "RANK_C", "SUPERLATIVES", "STOPWORDS", "content_words",
           "filter_evidence_by_claim_type", "has_any", "rule_multiplier", "score_evidence_quality",
           "structural_penalty", "title_words"]
