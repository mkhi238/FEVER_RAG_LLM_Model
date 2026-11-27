"""Tag an aligned (claim span, evidence span) pair with a natural-logic operator (step 3), using an NLI model.

NLI is run both ways, as in QA-NatVer's span-level tagging:
  fwd  = nli(premise=evidence_span, hypothesis=claim_span)
  back = nli(premise=claim_span, hypothesis=evidence_span)
  both entail          -> ≡ EQUIVALENCE      "Barack Obama" / "Obama"
  fwd entails only     -> ⊑ FORWARD          claim "a dog", evidence "a poodle"
  back entails only    -> ⊒ REVERSE          claim "a poodle", evidence "a dog"
  a contradiction      -> ¬ NEGATION         "was born" / "was not born"
  otherwise / no span  -> # INDEPENDENCE
ALTERNATION (⇃) is not produced yet: a 3-way NLI model can't tell it from ¬.
`nli(premise, hypothesis)` returns "entailment" | "contradiction" | "neutral"; nli_from_verifier() makes one from the
pipeline's HFNLIVerifier (claimscan/verifier/hf_nli.py).
"""
from functools import lru_cache

import torch

from claimscan.neurosymbolic_layer.natlog.automaton import Op

NLI_LABELS = ("entailment", "contradiction", "neutral")   # HFNLIVerifier's column order (LABEL_MAP: TRUE, FALSE, NEI)


def nli_from_verifier(model):
  """nli(premise, hypothesis) -> label, by argmax over the verifier's logits (no NEI threshold), cached per pair."""
  @lru_cache(maxsize=4096)
  def nli(premise: str, hypothesis: str) -> str:
    enc = model.encode_text(hypothesis, [premise])
    with torch.no_grad():
      logits = model(enc["input_ids"], enc["attention_mask"])
    return NLI_LABELS[int(logits.argmax(dim=1).item())]
  return nli


def tag(claim_span: str, evidence_span: str | None, nli) -> Op:
  if evidence_span is None:
    return Op.INDEPENDENCE
  fwd = nli(evidence_span, claim_span)
  back = nli(claim_span, evidence_span)
  if fwd == "entailment" and back == "entailment":
    return Op.EQUIVALENCE
  if "contradiction" in (fwd, back):
    return Op.NEGATION
  if fwd == "entailment":
    return Op.FORWARD
  if back == "entailment":
    return Op.REVERSE
  return Op.INDEPENDENCE


def project(op: Op, claim_span: str) -> Op:
  """The quantifier / monotonicity hook: where projection through the context goes (e.g. "all", "no", "not" flip ⊑
  and ⊒), to be designed by the user. For now it returns `op` unchanged."""
  return op
