"""The natural-logic automaton (claimscan/neurosymbolic_layer/natlog/automaton.py). Pure Python, no models."""
import pytest

from claimscan.neurosymbolic_layer.natlog.automaton import TRANSITIONS, Op, State, run

E, F, Rv, NG, I = Op.EQUIVALENCE, Op.FORWARD, Op.REVERSE, Op.NEGATION, Op.INDEPENDENCE


@pytest.mark.parametrize("ops, label, example", [
  ([E, E], State.SUPPORT, "claim 'Obama was born in Hawaii' / evidence 'Barack Obama was born in Hawaii'"),
  ([E, F], State.SUPPORT, "claim 'a dog barked' / evidence 'a poodle barked': poodle entails dog"),
  ([E, Rv], State.NEI, "claim 'a poodle barked' / evidence 'a dog barked': dog doesn't tell us poodle"),
  ([E, NG], State.REFUTE, "claim 'born in Paris' / evidence 'born in Rome'"),
  ([NG, E], State.REFUTE, "claim 'was born' / evidence 'was not born'"),
  ([E, NG, NG], State.REFUTE, "two wrong facts ('in Rome, in 1985' vs 'in Paris, in 1990') stay refuted"),
  ([NG, Rv], State.REFUTE, "refuted, and a more general evidence span keeps it refuted"),
  ([NG, F], State.NEI, "refuted, then a more specific evidence span: undecided"),
  ([E, I, E], State.NEI, "one unrelated span and the evidence is not enough"),
  ([], State.NEI, "no aligned spans"),
])
def test_run(ops, label, example):
  assert run(ops) == label, example


def test_nei_is_a_trap():
  assert all(nxt is State.NEI for nxt in TRANSITIONS[State.NEI].values())
  assert run([I, NG]) == State.NEI


def test_table_is_complete_and_accepts_symbols():
  assert all(set(row) == set(Op) for row in TRANSITIONS.values())
  assert run(["≡", "¬"]) == State.REFUTE            # operators may come in as their symbols
  assert State.SUPPORT.value == "TRUE"               # same label names as the pipeline


def test_table_with_forward_supports():
  from claimscan.neurosymbolic_layer.natlog.automaton import TRANSITIONS, table_with
  assert table_with(True) is TRANSITIONS
  strict = table_with(False)
  assert run([Op.EQUIVALENCE, Op.FORWARD]) == State.SUPPORT
  assert run([Op.EQUIVALENCE, Op.FORWARD], strict) == State.NEI
  assert TRANSITIONS[State.SUPPORT][Op.FORWARD] == State.SUPPORT     # the published table is untouched
