"""FOL composition and checkers (claimscan/neurosymbolic_layer/fol/compose.py, check.py, verify.py). No models."""
import importlib.util

import pytest
import torch

from claimscan.neurosymbolic_layer.fol.check import NatlogChecker, UnitCheck, VerifierChecker, decide
from claimscan.neurosymbolic_layer.fol.compose import compose
from claimscan.neurosymbolic_layer.fol.decomposition import parse_output
from claimscan.neurosymbolic_layer.fol.formula import parse
from claimscan.neurosymbolic_layer.fol.verify import build_checker, settings, verify_fol

EVIDENCE = ["Hinton is an American writer of fantasy novels.", "Hinton was born in 1948."]


def table_checker(table):
  """check(hypothesis, evidence) from {hypothesis: (label, evidence_idx)}; anything else is NEI."""
  calls = []
  def check(hyp, evidence):
    calls.append(hyp)
    label, idx = table.get(hyp, ("NOT ENOUGH INFO", None))
    return UnitCheck(label, {"TRUE": "⊑", "FALSE": "¬"}.get(label, "#"), idx)
  check.calls = calls
  return check


HINTON = parse_output({"claims": [{"claim": "Hinton is an American fantasy writer born in 1950.",
  "entities": {"hinton": "Hinton"}, "queries": [], "fol": {"op": "and", "args": [
    {"op": "atom", "pred": "writer", "args": ["hinton", "fantasy"], "text": "Hinton is a fantasy writer"},
    {"op": "atom", "pred": "American", "args": ["hinton"]},
    {"op": "atom", "pred": "born", "args": ["hinton", "1950"], "text": "Hinton was born in 1950"}]}}]})[0]


def test_and_one_refuted_is_false_with_reason():
  check = table_checker({"Hinton is a fantasy writer": ("TRUE", 0), "Hinton is American": ("TRUE", 0),
                         "Hinton was born in 1950": ("FALSE", 1)})
  v = verify_fol(HINTON.claim, EVIDENCE, check, HINTON)
  assert v.label == "FALSE" and v.used == [1]
  assert [u.hypothesis for u in v.units] == ["Hinton is a fantasy writer", "Hinton is American",
                                             "Hinton was born in 1950"]
  assert "\"Hinton was born in 1950\" is contradicted by [1] \"Hinton was born in 1948.\"" == v.reason


def test_and_all_true_and_missing_is_nei():
  table = {"Hinton is a fantasy writer": ("TRUE", 0), "Hinton is American": ("TRUE", 0),
           "Hinton was born in 1950": ("TRUE", 1)}
  v = verify_fol(HINTON.claim, EVIDENCE, table_checker(table), HINTON)
  assert v.label == "TRUE" and v.reason.startswith("all 3 parts hold") and v.used == [0, 1]
  del table["Hinton is American"]
  v = verify_fol(HINTON.claim, EVIDENCE, table_checker(table), HINTON)
  assert v.label == "NOT ENOUGH INFO" and v.reason == "not found in the evidence: \"Hinton is American\""


@pytest.mark.parametrize("formula, a, b, want", [
  ("¬A(rex)", "TRUE", None, "FALSE"),
  ("¬A(rex)", "NOT ENOUGH INFO", None, "NOT ENOUGH INFO"),
  ("A(rex) ∨ B(rex)", "FALSE", "TRUE", "TRUE"),
  ("A(rex) ∨ B(rex)", "FALSE", "NOT ENOUGH INFO", "NOT ENOUGH INFO"),
  ("A(rex) ∨ B(rex)", "FALSE", "FALSE", "FALSE"),
  ("A(rex) → B(rex)", "FALSE", "NOT ENOUGH INFO", "TRUE"),     # false antecedent
  ("A(rex) → B(rex)", "TRUE", "FALSE", "FALSE"),
  ("A(rex) → B(rex)", "TRUE", "NOT ENOUGH INFO", "NOT ENOUGH INFO"),
])
def test_kleene(formula, a, b, want):
  table = {"rex is A": (a, 0)}
  if b:
    table["rex is B"] = (b, 0)
  assert compose(parse(formula), ["e"], table_checker(table)).label == want


def test_every_refuted_part_is_named():
  v = compose(parse("A(rex) ∧ B(rex) ∧ C(rex)"), ["e0", "e1"],
              table_checker({"rex is A": ("FALSE", 0), "rex is B": ("TRUE", 0), "rex is C": ("FALSE", 1)}))
  assert v.label == "FALSE" and v.deciding == [0, 2] and v.used == [0, 1]


def test_negated_unit_reason():
  v = compose(parse("¬A(rex)"), ["rex is A."], table_checker({"rex is A": ("TRUE", 0)}))
  assert v.label == "FALSE" and "negates" in v.reason


def test_quantifier_modes():
  f = parse("∀x (Poodle(x) → Dog(x) ∧ Curly(x))")
  check = table_checker({"Every poodle is dog": ("TRUE", 0), "Every poodle is curly": ("NOT ENOUGH INFO", None)})
  unit = compose(f, ["e"], check, quantifier_mode="unit")
  assert len(unit.units) == 1 and unit.label == "NOT ENOUGH INFO"          # one hypothesis, not in the table
  dist = compose(f, ["e"], check, quantifier_mode="distribute")
  assert [u.hypothesis for u in dist.units] == ["Every poodle is dog", "Every poodle is curly"]
  assert dist.label == "NOT ENOUGH INFO" and dist.deciding == [1]
  nei = compose(parse("A(rex) ∧ ∃x (B(x))"), ["e"], table_checker({"rex is A": ("TRUE", 0)}), quantifier_mode="nei")
  assert nei.label == "NOT ENOUGH INFO"
  with pytest.raises(ValueError):
    compose(f, ["e"], check, quantifier_mode="sometimes")


def test_repeated_units_checked_once():
  check = table_checker({"rex is A": ("TRUE", 0)})
  compose(parse("A(rex) ∧ (A(rex) ∨ B(rex))"), ["e"], check)
  assert check.calls == ["rex is A", "rex is B"]


# ---- the Verify decision, unchanged ----
@pytest.mark.parametrize("probs, nei, want", [
  ([0.9, 0.05, 0.05], None, "TRUE"),
  ([0.45, 0.30, 0.25], None, "NOT ENOUGH INFO"),     # max prob < 0.5
  ([0.70, 0.05, 0.25], 0.2, "NOT ENOUGH INFO"),      # P(NEI) >= nei_prob_threshold
  ([0.05, 0.80, 0.15], 0.2, "FALSE"),
])
def test_decide_matches_pipeline_rule(probs, nei, want):
  assert decide(probs, 0.5, nei) == want


class FakeVerifier:
  """HFNLIVerifier's interface; logits by premise text: entail if the hypothesis' last word is in the premise."""
  def encode_text(self, claim, evidence):
    return {"input_ids": (" ".join(evidence), claim), "attention_mask": None}

  def __call__(self, input_ids, attention_mask):
    premise, hyp = input_ids
    last = hyp.rstrip(".").split()[-1]
    return torch.tensor([[4.0, 0.0, 0.0]] if last in premise else [[0.0, 0.0, 4.0]])


def test_verifier_checker_fallback_equals_pipeline_verify():
  from claimscan.pipeline import LABEL_MAP, verify
  model = FakeVerifier()
  ev = ["Rex lives in Paris.", "Rex is a poodle."]
  for claim in ["Rex is a poodle.", "Rex is a cat."]:
    pipeline_label, _ = verify(model, claim, ev)
    v = verify_fol(claim, ev, VerifierChecker(model, 0.5, 0.2))
    assert v.label == pipeline_label and len(v.units) == 1 and v.units[0].hypothesis == claim
  assert LABEL_MAP[0] == "TRUE"
  v = verify_fol("Rex is a poodle.", ev, VerifierChecker(model, 0.5, 0.2))
  assert v.units[0].evidence_idx == 1 and v.units[0].op == "⊑"       # attribution picks the poodle sentence


def test_verifier_checker_no_evidence():
  assert VerifierChecker(FakeVerifier())("Rex is a dog.", []).label == "NOT ENOUGH INFO"


def test_settings_and_build_checker():
  class Cfg:
    natlog = {"checker": "natlog", "min_sim": 0.5, "tasks": {"define": {"min_sim": 0.7}}}
    verifier_nei_prob_threshold = 0.2
  s = settings(Cfg)
  assert s["quantifier_mode"] == "unit" and s["forward_supports"] is True
  with pytest.raises(ValueError, match="encode"):
    build_checker(None, Cfg)
  assert build_checker(None, Cfg, encode=lambda t: t, task="define").min_sim == 0.7
  assert build_checker(None, Cfg, encode=lambda t: t).min_sim == 0.5
  Cfg.natlog = {}
  c = build_checker(FakeVerifier(), Cfg)
  assert isinstance(c, VerifierChecker) and c.nei_cut == 0.2


needs_spacy = pytest.mark.skipif(importlib.util.find_spec("spacy") is None
                                 or importlib.util.find_spec("en_core_web_sm") is None,
                                 reason="spaCy / en_core_web_sm not installed")


@needs_spacy
def test_natlog_checker_and_forward_supports():
  from stub_natlog import bow_encode, stub_nli
  nli = stub_nli(entails=[("an American fantasy writer", "a fantasy writer"),
                          ("is an American fantasy writer", "is a fantasy writer")])
  hyp, ev = "Hinton is a fantasy writer", ["Hinton is an American fantasy writer."]
  yes = NatlogChecker(nli, bow_encode, 0.3, forward_supports=True)(hyp, ev)
  no = NatlogChecker(nli, bow_encode, 0.3, forward_supports=False)(hyp, ev)
  assert "⊑" in [op for _, _, op in yes.proof]
  assert yes.label == "TRUE" and yes.op == "⊑" and yes.evidence_idx == 0
  assert no.label == "NOT ENOUGH INFO"
