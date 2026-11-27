"""The natural-logic verifier for one claim: spans -> align -> operators -> automaton (docs/OVERHAUL.md).

  verify_claim(claim, evidence, nli, encode) -> NatlogResult
    nli     nli(premise, hypothesis) -> "entailment" | "contradiction" | "neutral", or the pipeline's HFNLIVerifier
    encode  list[str] -> 2D array (the retriever's bge-small SentenceTransformer.encode)
The whole claim is one predicate for now; FOL decomposition comes later from the Qwen decomposer.
"""
from dataclasses import dataclass

from claimscan.neurosymbolic_layer.natlog.align import align
from claimscan.neurosymbolic_layer.natlog.automaton import Op, run
from claimscan.neurosymbolic_layer.natlog.ops import nli_from_verifier, project, tag
from claimscan.neurosymbolic_layer.natlog.spans import split_spans


@dataclass
class NatlogResult:
  label: str                          # "TRUE" | "FALSE" | "NOT ENOUGH INFO"
  proof: list[tuple[str, str, str]]   # (claim_span, evidence_span or "", operator symbol)
  used: list[int]                     # sorted indices of the evidence sentences that got an aligned span
  enough: bool                        # no # in the proof


def verify_claim(claim: str, evidence: list[str], nli, encode) -> NatlogResult:
  if hasattr(nli, "encode_text"):     # the pipeline's HFNLIVerifier
    nli = nli_from_verifier(nli)
  alignments = align(split_spans(claim), evidence, encode)
  ops = [project(tag(a.claim_span, a.evidence_span, nli), a.claim_span) for a in alignments]
  proof = [(a.claim_span, a.evidence_span or "", op.value) for a, op in zip(alignments, ops)]
  used = sorted({a.evidence_idx for a in alignments if a.evidence_idx is not None})
  return NatlogResult(run(ops).value, proof, used, Op.INDEPENDENCE not in ops)
