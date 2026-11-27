"""FOL formulas (claimscan/neurosymbolic_layer/fol/formula.py, decomposition.py, hypothesis.py). No models."""
import json

import pytest

from claimscan.neurosymbolic_layer.fol.decomposition import DECOMPOSER_SCHEMA, fallback, parse_output
from claimscan.neurosymbolic_layer.fol.formula import (And, Atom, Exists, FOLError, ForAll, Implies, Not, Or, atoms,
                                                       from_json, parse, to_json, to_string, validate)
from claimscan.neurosymbolic_layer.fol.hypothesis import distribute, render, render_atom, words

HINTON = {"op": "and", "args": [
  {"op": "atom", "pred": "writer", "args": ["hinton", "fantasy"], "text": "Hinton is a fantasy writer"},
  {"op": "atom", "pred": "nationality", "args": ["hinton", "American"]},
  {"op": "atom", "pred": "born", "args": ["hinton", "1950"], "text": "Hinton was born in 1950"}]}


def test_json_round_trip_and_string():
  f = validate(HINTON)
  assert isinstance(f, And) and len(atoms(f)) == 3
  assert to_json(f) == HINTON
  assert to_string(f) == "writer(hinton, fantasy) ∧ nationality(hinton, American) ∧ born(hinton, 1950)"


@pytest.mark.parametrize("bad, msg", [
  ({"op": "xor", "args": []}, "unknown op"),
  ({"op": "atom", "pred": "1x", "args": []}, "predicate name"),
  ({"op": "atom", "pred": "p", "args": "a"}, "list"),
  ({"op": "atom", "pred": "dog", "args": ["x"]}, "not bound"),
  ({"op": "forall", "var": "dog", "body": {"op": "atom", "pred": "p", "args": []}}, "single lowercase"),
  ({"op": "forall", "var": "x", "body": {"op": "exists", "var": "x", "body": {"op": "atom", "pred": "p",
                                                                                 "args": ["x"]}}}, "already bound"),
  ({"op": "and", "args": []}, "non-empty"),
  ({"op": "not"}, "expected an object"),
  ("writer(h)", "expected an object"),
])
def test_validator_rejects(bad, msg):
  with pytest.raises(FOLError, match=msg):
    validate(bad)


def test_single_arg_and_collapses():
  assert validate({"op": "and", "args": [{"op": "atom", "pred": "p", "args": []}]}) == Atom("p", ())


def test_bound_variable_ok():
  f = validate({"op": "forall", "var": "x", "body": {"op": "implies",
                "left": {"op": "atom", "pred": "dog", "args": ["x"]},
                "right": {"op": "atom", "pred": "animal", "args": ["x"]}}})
  assert f == ForAll("x", Implies(Atom("dog", ("x",)), Atom("animal", ("x",))))


@pytest.mark.parametrize("s, want", [
  ("Dog(rex)", Atom("Dog", ("rex",))),
  ("¬A(rex) ∧ C(rex)", And((Not(Atom("A", ("rex",))), Atom("C", ("rex",))))),
  ("A(rex) ∨ C(rex) ∧ D(rex)", Or((Atom("A", ("rex",)), And((Atom("C", ("rex",)), Atom("D", ("rex",))))))),
  ("∀x (Dog(x) → Animal(x))", ForAll("x", Implies(Atom("Dog", ("x",)), Atom("Animal", ("x",))))),
  ("A(rex) -> C(rex) -> D(rex)", Implies(Atom("A", ("rex",)), Implies(Atom("C", ("rex",)), Atom("D", ("rex",))))),
  ("∃x ∃y (Loves(x, y))", Exists("x", Exists("y", Atom("Loves", ("x", "y"))))),
])
def test_parse(s, want):
  assert parse(s) == want


def test_parse_iff_and_xor_rewritten():
  a, b = Atom("A", ("rex",)), Atom("B", ("rex",))
  assert parse("A(rex) ↔ B(rex)") == And((Implies(a, b), Implies(b, a)))
  assert parse("A(rex) ⊕ B(rex)") == Or((And((a, Not(b))), And((Not(a), b))))
  # MALLS precedence: ∧ binds tighter than →, ⊕ tighter than →
  f = parse("∀x (Person(x) ∧ Librarian(x) → Public(x) ⊕ Academic(x))")
  assert isinstance(f.body, Implies) and isinstance(f.body.left, And) and isinstance(f.body.right, Or)


@pytest.mark.parametrize("s", ["Dog(x)", "Dog(rex", "∀dog (A(dog))", "f(g(a))", "A(rex) B(rex)", "A(rex) ∧", "A(rex) = C"])
def test_parse_rejects(s):
  with pytest.raises(FOLError):
    parse(s)


def test_parse_agrees_with_json():
  f = parse("∀x (Dog(x) → Animal(x))")
  assert from_json(to_json(f)) == f


def test_render():
  ents = {"hinton": "Hinton"}
  assert render_atom(Atom("writer", ("hinton", "fantasy"), "Hinton is a fantasy writer"), ents) == \
    "Hinton is a fantasy writer"
  assert render_atom(Atom("American", ("hinton",)), ents) == "Hinton is American"
  assert render_atom(Atom("BornIn", ("hinton", "1950")), ents) == "Hinton born in 1950"
  assert words("WorkInPublicLibrary") == "work in public library" and words("NASA") == "NASA"
  assert render(parse("∀x (Dog(x) → Animal(x))")) == "Every dog is animal"
  assert render(parse("∃x (Dog(x) ∧ Brown(x))")) == "Some dog is brown"


def test_distribute_splits_forall_over_and():
  f = distribute(parse("∀x (Poodle(x) → Dog(x) ∧ Curly(x))"))
  assert f == And((ForAll("x", Implies(Atom("Poodle", ("x",)), Atom("Dog", ("x",)))),
                   ForAll("x", Implies(Atom("Poodle", ("x",)), Atom("Curly", ("x",))))))


def test_decomposer_output():
  reply = "<think>ok</think>```json\n" + json.dumps({"claims": [
    {"claim": "Hinton is an American fantasy writer born in 1950.", "entities": {"hinton": "Hinton"},
     "fol": HINTON, "queries": ["Hinton writer", " "]}]}) + "\n```"
  [d] = parse_output(reply)
  assert d.entities == {"hinton": "Hinton"} and d.queries == ["Hinton writer"] and not d.fallback
  assert d.subclaims == ["Hinton is a fantasy writer", "Hinton nationality American", "Hinton was born in 1950"]
  assert parse_output({"claims": [d.to_json()]})[0].formula == d.formula


@pytest.mark.parametrize("reply, msg", [
  ("no json here", "no JSON"),
  ('{"claims": []}', "at least one"),
  ('{"claims": [{"claim": "x", "entities": {}, "fol": {"op": "atom", "pred": "p", "args": ["y"]}}]}', "claims\\[0\\]"),
  ('{"claims": [{"claim": "x", "entities": {"h": "Hinton"}, "fol": {"op": "atom", "pred": "p", "args": []}}]}',
   "variables"),
])
def test_decomposer_output_rejects(reply, msg):
  with pytest.raises(FOLError, match=msg):
    parse_output(reply)


def test_fallback_is_whole_claim():
  d = fallback("Rex is a dog.")
  assert d.fallback and d.subclaims == ["Rex is a dog."] and d.queries == ["Rex is a dog."]


def test_schema_is_json_serialisable():
  s = json.dumps(DECOMPOSER_SCHEMA)
  assert '"$ref": "#/$defs/formula"' in s


def test_parse_unicode_names_and_long_arrow():
  assert parse("Plays(świątek, tennis)") == Atom("Plays", ("świątek", "tennis"))
  assert parse("A(rex) ⟷ B(rex)") == parse("A(rex) ↔ B(rex)")
