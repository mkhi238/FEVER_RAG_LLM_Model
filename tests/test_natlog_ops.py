"""Aligned pair -> operator (claimscan/neurosymbolic_layer/natlog/ops.py), with a stub NLI. No models."""
import pytest
import torch

from claimscan.neurosymbolic_layer.natlog.automaton import Op
from claimscan.neurosymbolic_layer.natlog.ops import nli_from_verifier, project, tag
from stub_natlog import stub_nli

NLI = stub_nli(entails=[("a poodle", "a dog"), ("Obama", "Barack Obama"), ("Barack Obama", "Obama")],
               contradicts=[("was not born", "was born")])


@pytest.mark.parametrize("claim_span, evidence_span, op", [
  ("Barack Obama", "Obama", Op.EQUIVALENCE),       # entails both ways
  ("in Hawaii", "in Hawaii", Op.EQUIVALENCE),
  ("a dog", "a poodle", Op.FORWARD),               # evidence -> claim only
  ("a poodle", "a dog", Op.REVERSE),               # claim -> evidence only
  ("was born", "was not born", Op.NEGATION),
  ("in Hawaii", "in Kenya", Op.INDEPENDENCE),      # neutral both ways
  ("in Hawaii", None, Op.INDEPENDENCE),            # unaligned
])
def test_tag(claim_span, evidence_span, op):
  assert tag(claim_span, evidence_span, NLI) == op


def test_contradiction_one_way_is_negation():
  nli = lambda p, h: "contradiction" if p == "alive" else "neutral"
  assert tag("dead", "alive", nli) == Op.NEGATION       # fwd only
  assert tag("alive", "dead", nli) == Op.NEGATION       # back only


def test_project_is_a_no_op_for_now():
  assert all(project(op, "all dogs") is op for op in Op)


class FakeVerifier:
  """HFNLIVerifier's interface: encode_text(claim, evidence) and logits in (entailment, contradiction, neutral) order."""
  def __init__(self):
    self.calls = []

  def encode_text(self, claim, evidence):
    self.calls.append((evidence[0], claim))
    return {"input_ids": (evidence[0], claim), "attention_mask": None}

  def __call__(self, input_ids, attention_mask):
    premise, hypothesis = input_ids
    return torch.tensor([[2.0, 0.0, 1.0]] if (premise, hypothesis) == ("a poodle", "a dog") else [[0.0, 0.0, 1.0]])


def test_nli_from_verifier_premise_hypothesis_order_and_cache():
  model = FakeVerifier()
  nli = nli_from_verifier(model)
  assert nli("a poodle", "a dog") == "entailment" and nli("a dog", "a poodle") == "neutral"
  assert tag("a dog", "a poodle", nli) == Op.FORWARD
  assert model.calls == [("a poodle", "a dog"), ("a dog", "a poodle")]   # premise = evidence, cached afterwards
