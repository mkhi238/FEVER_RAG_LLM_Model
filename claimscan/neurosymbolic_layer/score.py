"""Evidence quality score: retrieval rank prior x the symbolic rules (was claimscan/evidence/scoring.py).

score = structural_penalty(rule_multiplier (clamped) / (rank + RANK_C))

  * rank prior: the hit's 0-based position in the retriever's best-first list. Works the same
    for faiss (L2, lower = better), bm25 (higher = better) and hybrid, so no score-direction bugs.
  * rule_multiplier / structural_penalty: see rules.py.
"""
from claimscan.neurosymbolic_layer.rules import rule_multiplier, structural_penalty

RANK_C = 10                 # 1/(rank+10): rank 0 vs rank 10 is a 2x difference


def score_evidence_quality(claim, evidence_text, rank, page=None):
  """Quality of one retrieved sentence for ``claim``; higher = better.

  ``rank``: 0-based position in the retriever's best-first hit list. ``page``: FEVER page id.
  """
  return structural_penalty(rule_multiplier(claim, evidence_text, page) / (rank + RANK_C), evidence_text)
