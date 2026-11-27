"""Align each claim span to its best-matching evidence span (step 2 of the natural-logic proof).

Candidates: every evidence sentence is split with split_spans, plus QA-NatVer's multi-granular chunks (runs of up to
4 spans), so "was born in Hawaii" can match one claim span or a longer stretch of evidence. Claim spans and candidates
are embedded with `encode` (the retriever's bge-small; any list[str] -> 2D array callable) and compared by cosine.
Each claim span takes the best candidate over all evidence sentences; below min_sim it stays unaligned (-> # later).
"""
from dataclasses import dataclass

import numpy as np

from claimscan.neurosymbolic_layer.natlog.spans import multi_granular, split_spans

# Placeholder default, to be tuned with the user (not tuned on data). bge-small squeezes cosines into a high range:
# on a few hand pairs, unrelated short phrases score ~0.57 ("in Hawaii" / "a television network"), alternatives ~0.7
# ("in Hawaii" / "in Rome") and paraphrases 0.88+ ("an Oscar" / "the Academy Award"). 0.6 drops the unrelated ones
# but keeps alternatives aligned, so the operator step can still tag them ¬ instead of losing them to #.
MIN_SIM = 0.6


@dataclass
class Alignment:
  claim_span: str
  evidence_idx: int | None      # which evidence sentence; None = nothing similar enough
  evidence_span: str | None
  sim: float


def _unit(x) -> np.ndarray:
  x = np.asarray(x, dtype=np.float32)
  return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def align(claim_spans: list[str], evidence: list[str], encode, min_sim: float = MIN_SIM) -> list[Alignment]:
  """One Alignment per claim span, in order."""
  cands = [(i, c) for i, sent in enumerate(evidence) for c in multi_granular(split_spans(sent))]
  if not claim_spans or not cands:
    return [Alignment(s, None, None, 0.0) for s in claim_spans]
  emb = _unit(encode(list(claim_spans) + [c for _, c in cands]))
  sims = emb[:len(claim_spans)] @ emb[len(claim_spans):].T
  out = []
  for span, row in zip(claim_spans, sims):
    j = int(row.argmax())
    sim = float(row[j])
    idx, ev = cands[j] if sim >= min_sim else (None, None)
    out.append(Alignment(span, idx, ev, sim))
  return out
