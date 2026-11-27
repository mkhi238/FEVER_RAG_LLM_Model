"""The natural-logic automaton: a sequence of operators (one per aligned claim/evidence span pair) -> a verdict.

Operators (5; DeBERTa NLI predicts them from entailment in both directions):
  EQUIVALENCE  ≡  entailment both ways                         "Obama" / "Barack Obama"
  FORWARD      ⊑  evidence entails claim (evidence more specific) "a poodle" / "a dog"
  REVERSE      ⊒  claim entails evidence (claim more specific)    "a dog" / "a poodle"
  NEGATION     ¬  contradiction                                "was not born" / "was born", "in Paris" / "in Rome"
  INDEPENDENCE #  neutral: unrelated / no information
The papers' 7 relations are cut to these 5 (user's choice, 2026-11-26): "alternation" (⇃, e.g. Paris / Rome) is
folded into NEGATION, and "cover" was already dropped by ProoFVer.

States: SUPPORT, REFUTE, NEI. Start in SUPPORT. NEI is a trap (once NEI, always NEI).
The table follows the automaton of Angeli & Manning (2014), as used by ProoFVer (Krishna et al., TACL 2022, Fig. 1)
and QA-NatVer (Aly et al., EMNLP 2023, Fig. 1), written here from the published table (not copied code), with one change:
REFUTE + ¬ stays REFUTE (published: -> SUPPORT, a double negation). With alternation folded into ¬, two wrong facts
("in Rome, in 1985" vs claimed "in Paris, in 1990") would otherwise cancel out into SUPPORT.
"""
from enum import Enum


class Op(str, Enum):
  EQUIVALENCE = "≡"
  FORWARD = "⊑"
  REVERSE = "⊒"
  NEGATION = "¬"
  INDEPENDENCE = "#"


class State(str, Enum):
  SUPPORT = "TRUE"            # values = the pipeline's label names (LABEL_MAP in claimscan/pipeline.py)
  REFUTE = "FALSE"
  NEI = "NOT ENOUGH INFO"


S, R, N = State.SUPPORT, State.REFUTE, State.NEI
TRANSITIONS: dict[State, dict[Op, State]] = {
  S: {Op.EQUIVALENCE: S, Op.FORWARD: S, Op.REVERSE: N, Op.NEGATION: R, Op.INDEPENDENCE: N},
  R: {Op.EQUIVALENCE: R, Op.FORWARD: N, Op.REVERSE: R, Op.NEGATION: R, Op.INDEPENDENCE: N},
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
