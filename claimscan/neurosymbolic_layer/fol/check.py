"""Check one hypothesis (an FOL unit's text) against the evidence: -> TRUE / FALSE / NOT ENOUGH INFO + operator + sentence.

A checker is any callable check(hypothesis, evidence) -> UnitCheck. Two are built in (natlog.checker in default.yaml):

  VerifierChecker  (default) the pipeline's Verify decision, unchanged: pipeline.verify(model, hypothesis, evidence)
                   (premise = the first 3 evidence sentences joined), argmax, then NEI if max prob < conf_threshold or
                   P(NEI) >= nei_prob_threshold, as in pipeline.fact_check_detailed. It then names the sentence: the one
                   of those 3 with the highest probability for the chosen label (one NLI pass per sentence; attribution
                   only, it never changes the label). Operator: TRUE -> ⊑ (evidence entails the unit), FALSE -> ¬, NEI -> #.
  NatlogChecker    the natural-logic verifier (natlog/verify.py) on the hypothesis: spans -> align -> operators ->
                   automaton; the proof is kept. min_sim and forward_supports are the open questions in docs/NATLOG_FOL.md.

With the fallback decomposition (the whole claim as one atom), VerifierChecker gives exactly the pipeline's label.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LABELS = ("TRUE", "FALSE", "NOT ENOUGH INFO")
OP_FOR_LABEL = {"TRUE": "⊑", "FALSE": "¬", "NOT ENOUGH INFO": "#"}


@dataclass
class UnitCheck:
  label: str                               # TRUE | FALSE | NOT ENOUGH INFO
  op: str                                  # natural-logic operator symbol for the unit vs its evidence
  evidence_idx: int | None = None          # the sentence that decided it (None: no evidence / nothing aligned)
  probs: dict[str, float] | None = None    # VerifierChecker: the verifier's softmax
  proof: list = field(default_factory=list)  # NatlogChecker: (claim span, evidence span, op) steps


def decide(probs, conf_threshold: float = 0.5, nei_prob_threshold: float | None = None) -> str:
  """pipeline.fact_check_detailed's rule on a softmax in LABEL_MAP order (TRUE, FALSE, NEI)."""
  p = [float(x) for x in probs]
  label = LABELS[max(range(3), key=p.__getitem__)]
  if label != LABELS[2] and (max(p) < conf_threshold or (nei_prob_threshold is not None and p[2] >= nei_prob_threshold)):
    label = LABELS[2]
  return label


class VerifierChecker:
  def __init__(self, model, conf_threshold: float = 0.5, nei_prob_threshold: float | None = None,
               attribute: bool = True):
    self.model, self.conf, self.nei_cut, self.attribute = model, conf_threshold, nei_prob_threshold, attribute

  def _probs(self, hypothesis: str, evidence: list[str]):
    from claimscan.pipeline import verify            # lazy: keeps this module light for the tests
    _, probs = verify(self.model, hypothesis, evidence)
    return [float(x) for x in probs]

  def __call__(self, hypothesis: str, evidence: list[str]) -> UnitCheck:
    if not evidence:
      return UnitCheck(LABELS[2], "#")
    probs = self._probs(hypothesis, evidence)
    label = decide(probs, self.conf, self.nei_cut)
    idx = None if label == LABELS[2] else 0     # NEI: no sentence decided it
    seen = evidence[:3]
    if idx is not None and self.attribute and len(seen) > 1:
      col = LABELS.index(label)
      idx = max(range(len(seen)), key=lambda i: self._probs(hypothesis, [seen[i]])[col])
    return UnitCheck(label, OP_FOR_LABEL[label], idx, dict(zip(LABELS, probs)))


class NatlogChecker:
  def __init__(self, nli, encode, min_sim: float | None = None, forward_supports: bool = True):
    from claimscan.neurosymbolic_layer.natlog.align import MIN_SIM
    self.nli, self.encode = nli, encode
    self.min_sim = MIN_SIM if min_sim is None else min_sim
    self.forward_supports = forward_supports

  def __call__(self, hypothesis: str, evidence: list[str]) -> UnitCheck:
    from claimscan.neurosymbolic_layer.natlog.verify import verify_claim
    r = verify_claim(hypothesis, evidence, self.nli, self.encode, self.min_sim, self.forward_supports)
    ops = [op for _, _, op in r.proof]
    # the operator that decided the verdict: the first ¬ for FALSE, the first # for NEI, ≡ if all ≡ else ⊑ for TRUE
    if r.label == "FALSE":
      op = "¬"
    elif r.label == "TRUE":
      op = "≡" if ops and all(o == "≡" for o in ops) else "⊑"
    else:
      op = next((o for o in ops if o in ("#", "⊒", "⊑")), "#")
    return UnitCheck(r.label, op, r.used[0] if r.used else None, None, r.proof)
