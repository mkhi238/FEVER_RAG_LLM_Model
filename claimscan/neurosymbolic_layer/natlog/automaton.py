"""The natural-logic automaton: a sequence of operators (one per aligned claim/evidence span pair) -> a verdict.

Operators (MacCartney & Manning 2009; ProoFVer drops "cover", which is rare, mapping it to INDEPENDENCE):
  EQUIVALENCE  ≡  same meaning                       "Obama" / "Barack Obama"
  FORWARD      ⊑  the evidence span entails the claim span (evidence more specific): "a poodle" / "a dog"
  REVERSE      ⊒  the claim span is more specific than the evidence: "a dog" / "a poodle"
  NEGATION     ¬  one is the negation of the other   "was born" / "was not born"
  ALTERNATION  ⇃  they exclude each other            "in Paris" / "in Rome"
  INDEPENDENCE #  unrelated / no information

States: SUPPORT, REFUTE, NEI. Start in SUPPORT. NEI is a trap (once NEI, always NEI).
The table follows the automaton of Angeli & Manning (2014), as used by ProoFVer (Krishna et al., TACL 2022, Fig. 1)
and QA-NatVer (Aly et al., EMNLP 2023, Fig. 1). Written here from the published table, not copied code.
"""
from enum import Enum


class Op(str, Enum):
  EQUIVALENCE = "≡"
  FORWARD = "⊑"
  REVERSE = "⊒"
  NEGATION = "¬"
  ALTERNATION = "⇃"
  INDEPENDENCE = "#"


class State(str, Enum):
  SUPPORT = "TRUE"            # values = the pipeline's label names (LABEL_MAP in claimscan/pipeline.py)
  REFUTE = "FALSE"
  NEI = "NOT ENOUGH INFO"


S, R, N = State.SUPPORT, State.REFUTE, State.NEI
TRANSITIONS: dict[State, dict[Op, State]] = {
  S: {Op.EQUIVALENCE: S, Op.FORWARD: S, Op.REVERSE: N, Op.NEGATION: R, Op.ALTERNATION: R, Op.INDEPENDENCE: N},
  R: {Op.EQUIVALENCE: R, Op.FORWARD: N, Op.REVERSE: R, Op.NEGATION: S, Op.ALTERNATION: N, Op.INDEPENDENCE: N},
  N: {op: N for op in Op},
}


def run(ops: list[Op]) -> State:
  """Walk the table from SUPPORT. An empty proof says nothing, so it is NEI."""
  if not ops:
    return N
  state = S
  for op in ops:
    state = TRANSITIONS[state][Op(op)]
  return state
