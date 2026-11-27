"""A GRPO reward for the decomposer (proposal; nothing trains with it yet, docs/NATLOG_FOL.md).

  fol_reward(reply, evidence, gold_label, checker) -> Reward
    0                               the reply does not parse / validate (decomposition.parse_output)
    format_bonus                    valid, but the composed verdict differs from the gold FEVER label
    format_bonus + 1                valid and the verdict == gold
Evidence = the claim's gold evidence sentences (or the retrieved ones); the checker is the frozen verifier
(check.VerifierChecker), so the decomposer is rewarded for parses whose logic reproduces the gold label.
Several claims in one reply are joined with ∧ (a FEVER claim is one claim). The weights are placeholders for the user.
Known hole: a one-atom "decomposition" (the whole claim) scores like the plain verifier; a bonus for splitting, or a
per-atom evidence-recall term (the vendored FEVER metric), are options for the user to decide.
"""
from __future__ import annotations

from dataclasses import dataclass

from claimscan.neurosymbolic_layer.fol.compose import FOLVerdict, compose
from claimscan.neurosymbolic_layer.fol.decomposition import parse_output
from claimscan.neurosymbolic_layer.fol.formula import And, FOLError


@dataclass
class Reward:
  reward: float
  valid: bool
  label: str | None            # composed verdict (None if invalid)
  error: str = ""
  verdict: FOLVerdict | None = None


def fol_reward(reply: str, evidence: list[str], gold_label: str, checker, quantifier_mode: str = "unit",
               format_bonus: float = 0.2) -> Reward:
  try:
    claims = parse_output(reply)
  except FOLError as e:
    return Reward(0.0, False, None, str(e))
  ents = {k: v for d in claims for k, v in d.entities.items()}
  formula = claims[0].formula if len(claims) == 1 else And(tuple(d.formula for d in claims))
  v = compose(formula, evidence, checker, ents, quantifier_mode)
  return Reward(format_bonus + (1.0 if v.label == gold_label else 0.0), True, v.label, "", v)
