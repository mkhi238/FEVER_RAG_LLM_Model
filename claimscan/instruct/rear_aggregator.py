"""Rear: combine per-claim verdicts + evidence from the verifier into the final answer.

The LLM (Gemma on llama-server) reads one ClaimResult at a time and may ask to VETO the verifier's verdict.
Code (final_label) decides whether the veto is applied, not the LLM:
  - it only ever turns TRUE/FALSE into NOT ENOUGH INFO, never TRUE <-> FALSE;
  - only when the verifier is unsure (conf < 0.6) or the LLM is sure (llm_confidence >= 0.9). Fixed, not configurable.
No llama.cpp or connection code here: `llm` is passed in (chat_clients.build_chat(...).with_structured_output(VetoOutput)),
so tests can pass a fake. Contract: tests/test_rear_aggregator.py. Wired into the API by claimscan/aggregation.py.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from claimscan.instruct.schemas import ClaimResult, ClaimVerdict, Label, RearInput, RearOutput
#Label is Pydantic class in schemas

NEI = "NOT ENOUGH INFO"
# Veto power OFF (2026-11-26): the trained Gemma adapter vetoed 122/300 claims, 84 of them correct verdicts
# (FEVER suite acc 0.65 -> 0.46). With it off the LLM only explains; its veto request stays visible in the output.
VETO_ENABLED = False


class VetoOutput(BaseModel):
  """What the structured LLM returns for one claim (llm.with_structured_output(VetoOutput, method="json_schema"))."""
  veto: bool = Field(description="True if the evidence does not actually support the verdict")
  veto_reason: str = Field(description="Why the verdict should be vetoed; empty if not vetoed")
  explanation: str = Field(description="1-3 sentences, citing evidence as [page:line]")
  llm_confidence: float = Field(ge=0, le=1, description="Your confidence (0-1) that the claim is unfactual or impossible")


SYSTEM_PROMPT = """You check a fact-checker's verdict. You get a claim, the verdict (TRUE, FALSE or NOT ENOUGH INFO)
and the Wikipedia sentences it was based on, each tagged [page:line].
Set veto=true only if the sentences do not actually support the verdict (off-topic, about a different entity,
or too weak). Do not judge the claim from your own knowledge alone.
Write a 1-3 sentence explanation citing sentences as [page:line]. Answer in JSON."""


def build_messages(result: ClaimResult) -> list:
  """One ClaimResult -> chat messages for the LLM."""
  evidence = "\n".join(f"[{e.page}:{e.line}] {e.text}" for e in result.evidence) or "(no evidence)"
  probs = ", ".join(f"{k}={v:.2f}" for k, v in result.probs.items())
  user = f"Claim: {result.claim}\nVerdict: {result.label} ({probs})\nEvidence:\n{evidence}"
  return [SystemMessage(SYSTEM_PROMPT), HumanMessage(user)]


def final_label(verifier_label: Label, veto: bool, verifier_conf, llm_conf):
  """verifier_label while VETO_ENABLED is False. Otherwise: NEI only when veto and verifier_label != NEI and (verifier_conf < 0.6 or llm_conf >= 0.9);
  otherwise verifier_label. Never flips TRUE <-> FALSE."""
  if VETO_ENABLED and veto and verifier_label != NEI and (verifier_conf < 0.6 or llm_conf >= 0.9):
    verifier_label = NEI
    return verifier_label
  else:
    return verifier_label


def aggregate(rear_input: RearInput, llm) -> RearOutput:
  """Call `llm` once per ClaimResult (llm.invoke(build_messages(r)) -> VetoOutput) and return one
  ClaimVerdict per result, in order:
    verifier_conf = max(r.probs.values())
    verdict       = final_label(r.label, out.veto, verifier_conf, out.llm_confidence)
    confidence    = verifier_conf
    veto          = verdict != r.label   (whether the veto was APPLIED)
    veto_reason, llm_confidence, explanation copied from out
  summary: built in code from the verdicts/explanations, no second LLM call."""
  claims = []
  for r in rear_input.results:
    #llm.invoke() is a standard method that sends a single input 
    #(such as a prompt string or messages) to a 
    #language model and returns the complete response after generation finishes
    out = llm.invoke(build_messages(r))
    verifier_conf = max(r.probs.values())
    verdict = final_label(r.label, out.veto, verifier_conf, out.llm_confidence)
    claims.append(ClaimVerdict(
      claim=r.claim,
      verdict=verdict,
      confidence=verifier_conf,
      explanation=out.explanation,
      veto=verdict != r.label,
      veto_reason=out.veto_reason,
      llm_confidence=out.llm_confidence,
    ))
  summary = "\n".join(f"{c.claim} -> {c.verdict} ({c.confidence:.2f}). {c.explanation}" for c in claims)
  return RearOutput(summary=summary or "No claims to check.", claims=claims)
