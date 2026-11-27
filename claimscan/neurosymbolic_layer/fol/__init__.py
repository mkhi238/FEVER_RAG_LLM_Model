"""FOL layer over the natural-logic verifier (docs/NATLOG_FOL.md): the FOL form of a claim is the checklist.

  formula.py        AST, JSON schema (constrained decoding), validator (FOLError), FOLIO/MALLS string parser
  decomposition.py  the decomposer's JSON (entities, FOL, queries per claim), parse_output, fallback
  hypothesis.py     FOL unit -> NL hypothesis (node text, else a template); quantifier modes
  check.py          one hypothesis vs the evidence: VerifierChecker (Verify decision as is) | NatlogChecker
  compose.py        Kleene logic over the unit checks -> FOLVerdict (label, reason, units)
  verify.py         verify_fol(claim, evidence, checker, decomposition); settings(); build_checker()
  reward.py         GRPO reward for the decomposer (valid parse + verdict == gold); not used for training yet
  sft_data.py       SFT data for the decomposer: MALLS / FOLIO NL-FOL pairs -> chat records (data prep only)
"""
