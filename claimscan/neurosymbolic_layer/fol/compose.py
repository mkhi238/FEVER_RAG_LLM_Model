"""Compose unit checks into a verdict with three-valued (Kleene) logic, and say why.

Values: TRUE, FALSE, NEI (NOT ENOUGH INFO = unknown).
  ∧  FALSE if any part is FALSE, TRUE if all are TRUE, else NEI
  ∨  TRUE if any part is TRUE, FALSE if all are FALSE, else NEI
  ¬  swaps TRUE / FALSE, NEI stays NEI
  →  ¬A ∨ B
Units (atoms, quantified subformulas: hypothesis.is_unit) are checked by DeBERTa through a checker (check.py); the
logic above them is this file's, never the model's. The reason names the unit(s) that decided the verdict and the
evidence sentence each was decided on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from claimscan.neurosymbolic_layer.fol.check import UnitCheck
from claimscan.neurosymbolic_layer.fol.formula import And, Formula, Implies, Not, Or, to_string
from claimscan.neurosymbolic_layer.fol.hypothesis import QUANTIFIER_MODES, distribute, is_unit, render

T, F, N = "TRUE", "FALSE", "NOT ENOUGH INFO"
_NEG = {T: F, F: T, N: N}


@dataclass
class UnitResult:
  formula: str                 # the unit in FOL, e.g. born(hinton, 1950)
  hypothesis: str              # what DeBERTa checked
  label: str
  op: str
  evidence_idx: int | None
  evidence: str                # that sentence ("" if none)
  probs: dict | None = None
  proof: list = field(default_factory=list)


@dataclass
class FOLVerdict:
  label: str                   # TRUE | FALSE | NOT ENOUGH INFO
  reason: str                  # one line: which unit decided it, on which sentence
  units: list[UnitResult]      # every unit checked, left to right
  deciding: list[int]          # indices into units of the ones that decided the label
  formula: str                 # the whole formula in FOL

  @property
  def used(self) -> list[int]:
    """Sorted evidence indices the deciding units were decided on."""
    return sorted({self.units[i].evidence_idx for i in self.deciding if self.units[i].evidence_idx is not None})


def _kleene_and(vals):
  return F if F in vals else (T if all(v == T for v in vals) else N)


def _kleene_or(vals):
  return T if T in vals else (F if all(v == F for v in vals) else N)


def compose(formula: Formula, evidence: list[str], checker, entities: dict[str, str] | None = None,
            quantifier_mode: str = "unit") -> FOLVerdict:
  if quantifier_mode not in QUANTIFIER_MODES:
    raise ValueError(f"quantifier_mode must be one of {QUANTIFIER_MODES}, got {quantifier_mode!r}")
  f = distribute(formula) if quantifier_mode == "distribute" else formula
  units: list[UnitResult] = []
  cache: dict[str, UnitCheck] = {}

  def check_unit(node) -> tuple[str, list[int]]:
    hyp = render(node, entities)
    if quantifier_mode == "nei" and not hasattr(node, "pred"):
      c = UnitCheck(N, "#")
    else:
      c = cache.get(hyp) or checker(hyp, evidence)
      cache[hyp] = c
    ev = evidence[c.evidence_idx] if c.evidence_idx is not None and c.evidence_idx < len(evidence) else ""
    units.append(UnitResult(to_string(node), hyp, c.label, c.op, c.evidence_idx, ev, c.probs, c.proof))
    return c.label, [len(units) - 1]

  def ev(node) -> tuple[str, list[int]]:
    """(value, indices of the units that explain the value)."""
    if is_unit(node):
      return check_unit(node)
    if isinstance(node, Not):
      v, why = ev(node.arg)
      return _NEG[v], why
    if isinstance(node, Implies):
      return ev(Or((Not(node.left), node.right)))
    parts = [ev(a) for a in node.args]
    vals = [v for v, _ in parts]
    v = (_kleene_and if isinstance(node, And) else _kleene_or)(vals)
    decisive = F if isinstance(node, And) else T     # one such part settles it
    if v == decisive:
      why = next(w for pv, w in parts if pv == v)
    else:                                             # all parts matter, or the unknown ones do
      why = [i for pv, w in parts if v != N or pv == N for i in w]
    return v, why

  label, deciding = ev(f)
  return FOLVerdict(label, _reason(label, [units[i] for i in deciding], len(units)), units, deciding,
                    to_string(formula))


def _cite(u: UnitResult) -> str:
  return f"[{u.evidence_idx}] \"{u.evidence}\"" if u.evidence_idx is not None else "no sentence"


def _reason(label: str, deciding: list[UnitResult], n_units: int) -> str:
  if not deciding:
    return "nothing to check"
  if label == N:
    return "not found in the evidence: " + "; ".join(f"\"{u.hypothesis}\"" for u in deciding)
  word = {T: "supported", F: "contradicted"}
  parts = [f"\"{u.hypothesis}\" is {word.get(u.label, 'unknown')} by {_cite(u)}"
           + (" (and the claim negates it)" if u.label != label else "") for u in deciding]
  head = f"all {n_units} parts hold: " if label == T and n_units > 1 else ""
  return head + "; ".join(parts)
