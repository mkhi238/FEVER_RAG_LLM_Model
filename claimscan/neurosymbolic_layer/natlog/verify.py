"""The natural-logic verifier for one claim: spans -> align -> operators -> automaton (docs/OVERHAUL.md).

  verify_claim(claim, evidence, nli, encode) -> NatlogResult
    nli     nli(premise, hypothesis) -> "entailment" | "contradiction" | "neutral", or the pipeline's HFNLIVerifier
    encode  list[str] -> 2D array (the retriever's bge-small SentenceTransformer.encode)
The whole claim is one predicate here; claimscan/neurosymbolic_layer/fol/ runs this per FOL unit (fol/check.py).
"""
from dataclasses import dataclass

from claimscan.neurosymbolic_layer.natlog.align import MIN_SIM, align
from claimscan.neurosymbolic_layer.natlog.automaton import Op, run, table_with
from claimscan.neurosymbolic_layer.natlog.ops import nli_from_verifier, project, tag
from claimscan.neurosymbolic_layer.natlog.spans import split_spans


@dataclass
class NatlogResult:
  label: str                          # "TRUE" | "FALSE" | "NOT ENOUGH INFO"
  proof: list[tuple[str, str, str]]   # (claim_span, evidence_span or "", operator symbol)
  used: list[int]                     # sorted indices of the evidence sentences that got an aligned span
  enough: bool                        # no # in the proof


def verify_claim(claim: str, evidence: list[str], nli, encode, min_sim: float = MIN_SIM,
                 forward_supports: bool = True) -> NatlogResult:
  """min_sim: alignment cut-off (align.MIN_SIM). forward_supports: does evidence ⊑ claim span count as support
  (automaton.table_with)."""
  if hasattr(nli, "encode_text"):     # the pipeline's HFNLIVerifier
    nli = nli_from_verifier(nli)
  alignments = align(split_spans(claim), evidence, encode, min_sim)
  ops = [project(tag(a.claim_span, a.evidence_span, nli), a.claim_span) for a in alignments]
  proof = [(a.claim_span, a.evidence_span or "", op.value) for a, op in zip(alignments, ops)]
  used = sorted({a.evidence_idx for a in alignments if a.evidence_idx is not None})
  return NatlogResult(run(ops, table_with(forward_supports)).value, proof, used, Op.INDEPENDENCE not in ops)
