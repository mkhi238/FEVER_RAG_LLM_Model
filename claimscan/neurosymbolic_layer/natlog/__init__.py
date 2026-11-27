"""Natural-logic verifier (ClaimScan overhaul, docs/OVERHAUL.md).

  automaton.py  operators, states and the fixed transition table: operator sequence -> verdict
  spans.py      claim -> short spans (spaCy noun phrases, verb groups, prepositions) + multi-granular chunks
  align.py      claim span -> best evidence span (bge-small cosine over spans + chunks of every sentence)
  ops.py        aligned pair -> operator (NLI both ways); project() = the quantifier hook, a no-op for now
  verify.py     verify_claim(claim, evidence, nli, encode): spans -> align -> operators -> automaton
"""
