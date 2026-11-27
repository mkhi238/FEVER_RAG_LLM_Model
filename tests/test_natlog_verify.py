"""verify_claim end to end (claimscan/neurosymbolic_layer/natlog/verify.py) with stub encode and NLI; real spaCy spans."""
import importlib.util

import pytest

pytest.importorskip("spacy")
if importlib.util.find_spec("en_core_web_sm") is None:
  pytest.skip("spaCy model en_core_web_sm is not installed", allow_module_level=True)

from claimscan.neurosymbolic_layer.natlog.verify import verify_claim
from stub_natlog import bow_encode, stub_nli

NLI = stub_nli(entails=[("is a poodle", "is a dog")], contradicts=[("was not born", "was born")])
CLAIM = "Barack Obama was born in Hawaii."


def test_support():
  r = verify_claim(CLAIM, ["Michelle Obama lives in Chicago.", "Barack Obama was born in Hawaii."], NLI, bow_encode)
  assert r.label == "TRUE" and r.enough and r.used == [1]
  assert r.proof == [("Barack Obama", "Barack Obama", "≡"), ("was born", "was born", "≡"), ("in Hawaii", "in Hawaii", "≡")]


def test_forward_supports():
  r = verify_claim("Rex is a dog.", ["Rex is a poodle."], NLI, bow_encode)
  assert r.label == "TRUE" and [op for _, _, op in r.proof] == ["≡", "⊑"]


def test_refute():
  r = verify_claim(CLAIM, ["Barack Obama was not born in Hawaii."], NLI, bow_encode)
  assert r.label == "FALSE" and r.enough
  assert ("was born", "was not born", "¬") in r.proof


def test_unaligned_span_is_nei():
  r = verify_claim(CLAIM, ["Barack Obama was born."], NLI, bow_encode)
  assert r.label == "NOT ENOUGH INFO" and not r.enough
  assert r.proof[-1] == ("in Hawaii", "", "#") and r.used == [0]


def test_no_evidence():
  r = verify_claim(CLAIM, [], NLI, bow_encode)
  assert r.label == "NOT ENOUGH INFO" and not r.enough and r.used == []
  assert [op for _, _, op in r.proof] == ["#", "#", "#"]

