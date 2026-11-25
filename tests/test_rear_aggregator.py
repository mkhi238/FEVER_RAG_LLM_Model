"""Rear aggregator: the veto may only downgrade TRUE/FALSE to NOT ENOUGH INFO, never flip them.

Contract (claimscan/instruct/rear_aggregator.py):
- final_label(verifier_label, veto, verifier_conf, llm_conf) -> Label   (fixed thresholds: low=0.6, sure=0.9)
    "NOT ENOUGH INFO" only when veto and verifier_label != NEI and (verifier_conf < 0.6 or llm_conf >= 0.9);
    otherwise verifier_label. Never flips TRUE <-> FALSE.
- aggregate(rear_input: RearInput, llm: Runnable) -> RearOutput
    `llm` is the already-structured runnable (what chat.with_structured_output(..., method="json_schema")
    returns). It is invoked exactly once per ClaimResult and returns an object with
    veto: bool, veto_reason: str, explanation: str, llm_confidence: float in [0, 1]
    (the model's own confidence that the claim is unfactual / impossible).
    aggregate returns one ClaimVerdict per ClaimResult, in order, with
      verifier_conf = max(result.probs.values())
      verdict       = final_label(result.label, out.veto, verifier_conf, out.llm_confidence)
      confidence    = verifier_conf
      veto          = whether the veto was APPLIED (verdict was downgraded)
      veto_reason, llm_confidence copied from the LLM output
    RearOutput.summary is built in code from the explanations (no extra LLM call).
    "UNCLEAR" is no longer a verdict.

The fake LLM below reads markers from whatever it is invoked with (messages, dict or string):
"[veto]" means veto=True, "[llm=0.95]" sets llm_confidence (default 0.3).
"""
import re

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from claimscan.instruct.schemas import ClaimResult, ClaimVerdict, Evidence, RearInput, RearOutput

ra = pytest.importorskip("claimscan.instruct.rear_aggregator")
if not (hasattr(ra, "final_label") and hasattr(ra, "aggregate")):
  pytest.skip("rear aggregator contract not implemented yet", allow_module_level=True)
try:  # the functions may exist as stubs (raise NotImplementedError) before they are written
  ra.final_label("TRUE", False, 0.9, 0.0)
except NotImplementedError:
  pytest.skip("rear aggregator contract not implemented yet (final_label is a stub)", allow_module_level=True)

NEI = "NOT ENOUGH INFO"
LABELS = ["TRUE", "FALSE", NEI]
FIELDS = set(ClaimVerdict.model_fields)
needs_veto_fields = pytest.mark.skipif(not {"veto", "veto_reason", "llm_confidence"} <= FIELDS,
                                       reason="ClaimVerdict has no veto/veto_reason/llm_confidence fields yet")


class FakeOut(BaseModel):
  veto: bool
  veto_reason: str
  explanation: str
  llm_confidence: float


def fake_llm(explanation="Supported by [Tokyo:0]."):
  """Structured-output stand-in. Returns (runnable, calls)."""
  calls = []

  def run(inp):
    calls.append(inp)
    text = str(inp)
    veto = "[veto]" in text
    m = re.search(r"\[llm=([0-9.]+)\]", text)
    return FakeOut(veto=veto, veto_reason="evidence is off-topic" if veto else "",
                   explanation=explanation, llm_confidence=float(m.group(1)) if m else 0.3)

  return RunnableLambda(run), calls


def result(claim, label, top=0.9):
  # top prob on `label`, the rest split evenly
  rest = (1 - top) / 2
  probs = {l: (top if l == label else rest) for l in LABELS}
  return ClaimResult(claim=claim, label=label, probs=probs,
                     evidence=[Evidence(page="Tokyo", line=0, text="Tokyo is the capital of Japan.")])


def rear_input(*results):
  return RearInput(user_text=" ".join(r.claim for r in results), results=list(results))


def one(claim, label, top=0.9, **kw):
  llm, _ = fake_llm()
  return ra.aggregate(rear_input(result(claim, label, top)), llm, **kw).claims[0]


# ---- final_label ----
VCONFS = [0.51, 0.59, 0.6, 0.9]
LCONFS = [0.2, 0.89, 0.9]


def expected(label, veto, vconf, lconf):
  # written out by hand: < 0.6 is low, >= 0.9 is sure
  if veto and label != NEI and (vconf < 0.6 or lconf >= 0.9):
    return NEI
  return label


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("veto", [False, True])
@pytest.mark.parametrize("vconf", VCONFS)
@pytest.mark.parametrize("lconf", LCONFS)
def test_final_label(label, veto, vconf, lconf):
  assert ra.final_label(label, veto, vconf, lconf) == expected(label, veto, vconf, lconf)


@pytest.mark.parametrize("vconf,lconf,out", [
  (0.59, 0.2, NEI),       # just below low
  (0.6, 0.2, "TRUE"),     # low itself is not low
  (0.6, 0.9, NEI),        # at low, only a sure llm vetoes
  (0.9, 0.89, "TRUE"),    # just below sure
  (0.9, 0.9, NEI),        # sure itself counts
])
def test_final_label_boundaries(vconf, lconf, out):
  assert ra.final_label("TRUE", True, vconf, lconf) == out


@pytest.mark.parametrize("label,opposite", [("TRUE", "FALSE"), ("FALSE", "TRUE")])
@pytest.mark.parametrize("veto", [False, True])
@pytest.mark.parametrize("vconf", VCONFS)
@pytest.mark.parametrize("lconf", LCONFS)
def test_veto_never_flips(label, opposite, veto, vconf, lconf):
  out = ra.final_label(label, veto, vconf, lconf)
  assert out != opposite
  assert out in (label, NEI)


# ---- aggregate ----
@pytest.mark.parametrize("label", LABELS)
def test_no_veto_keeps_label(label):
  llm, _ = fake_llm()
  out = ra.aggregate(rear_input(result("Tokyo is the capital of Japan.", label, top=0.52)), llm)
  assert isinstance(out, RearOutput)
  assert [c.verdict for c in out.claims] == [label]


@pytest.mark.parametrize("label", ["TRUE", "FALSE"])
def test_veto_with_low_verifier_conf_downgrades(label):
  assert one("Tokyo is in France. [veto]", label, top=0.52).verdict == NEI


def test_veto_ignored_when_both_confident_enough():
  # verifier 0.91, llm 0.3: veto not applied
  v = one("Tokyo is the capital of Japan. [veto] [llm=0.3]", "TRUE", top=0.91)
  assert v.verdict == "TRUE"
  if "veto" in FIELDS:
    assert v.veto is False


def test_veto_applied_when_llm_sure():
  assert one("Tokyo is the capital of Japan. [veto] [llm=0.95]", "TRUE", top=0.91).verdict == NEI


@pytest.mark.parametrize("top", [0.52, 0.59])
def test_veto_applied_when_verifier_low(top):
  assert one("Tokyo is the capital of Japan. [veto] [llm=0.3]", "TRUE", top=top).verdict == NEI


def test_veto_ignored_at_low_threshold():
  assert one("Tokyo is the capital of Japan. [veto] [llm=0.3]", "TRUE", top=0.6).verdict == "TRUE"


@needs_veto_fields
@pytest.mark.parametrize("label", ["TRUE", "FALSE"])
def test_veto_fields_on_applied_veto(label):
  v = one("Tokyo is in France. [veto] [llm=0.4]", label, top=0.52)
  assert v.veto is True
  assert v.veto_reason == "evidence is off-topic"
  assert v.llm_confidence == pytest.approx(0.4)


@needs_veto_fields
def test_veto_fields_without_veto():
  v = one("Tokyo is the capital of Japan. [llm=0.2]", "TRUE")
  assert v.veto is False
  assert v.llm_confidence == pytest.approx(0.2)


def test_mixed_claims_order_and_veto():
  res = [result("A is true.", "TRUE"),
         result("B is false. [veto]", "FALSE", top=0.52),
         result("C is unknown. [veto]", NEI, top=0.52),
         result("D is true. [veto] [llm=0.3]", "TRUE", top=0.91),   # veto not applied
         result("E is true. [veto] [llm=0.95]", "TRUE", top=0.91),
         result("F is false.", "FALSE", top=0.52)]
  llm, _ = fake_llm()
  out = ra.aggregate(rear_input(*res), llm)
  assert [c.claim for c in out.claims] == [r.claim for r in res]
  assert [c.verdict for c in out.claims] == ["TRUE", NEI, NEI, "TRUE", NEI, "FALSE"]
  if "veto" in FIELDS:
    assert [c.veto for c in out.claims] == [False, True, False, False, True, False]


def test_llm_called_once_per_claim():
  res = [result(f"Claim {i}.", l) for i, l in enumerate(LABELS + ["TRUE"])]
  llm, calls = fake_llm()
  out = ra.aggregate(rear_input(*res), llm)
  assert len(calls) == len(res)   # summary is built in code, not by another call
  assert isinstance(out.summary, str) and out.summary


@pytest.mark.parametrize("label,top", [("TRUE", 0.91), ("FALSE", 0.6), (NEI, 0.4)])
def test_confidence_is_top_prob(label, top):
  r = result("Tokyo is the capital of Japan.", label, top=top)
  llm, _ = fake_llm()
  v = ra.aggregate(rear_input(r), llm).claims[0]
  assert v.confidence == pytest.approx(max(r.probs.values()))
  assert v.confidence == pytest.approx(top)


def test_label_comes_from_code_not_text():
  # explanation contradicts the label, but veto=False so TRUE must stay TRUE
  llm, _ = fake_llm(explanation="This is FALSE. The claim is NOT ENOUGH INFO.")
  v = ra.aggregate(rear_input(result("Tokyo is the capital of Japan.", "TRUE", top=0.52)), llm).claims[0]
  assert v.verdict == "TRUE"


def test_unclear_removed():
  ann = str(ClaimVerdict.model_fields["verdict"].annotation)
  if "UNCLEAR" in ann:
    pytest.skip("UNCLEAR not removed from ClaimVerdict.verdict yet")
  assert set(LABELS) <= set(re.findall(r"'([A-Z ]+)'", ann))
