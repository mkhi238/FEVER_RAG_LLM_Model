"""FOL node -> short natural-language hypothesis, the text DeBERTa checks against the evidence.

A *unit* is a node checked as one hypothesis: every atom, and (depending on the quantifier mode) every quantified
subformula. The logic above the units (∧ ∨ ¬ →) is composed in code (compose.py), never by DeBERTa.

Hypothesis text: the node's own "text" if the decomposer gave one (preferred, the prompt asks for it), else a template:
  writer(hinton, fantasy)  -> "Hinton writer fantasy"      (entity ids resolved through the decomposition's entities)
  American(hinton)    -> "Hinton is American"
  ∀x (dog(x) → animal(x))  -> "Every dog is animal"   ∃x (dog(x) ∧ brown(x)) -> "Some dog is brown"
Templates are a fallback only: they read badly for verbs, so the prompt always asks for "text".

Quantifier modes (an open design question, natlog.quantifier_mode in default.yaml):
  unit        (default) a forall / exists subformula is one unit, checked as a whole by its text
  distribute  like unit, but ∀x (R(x) → B1(x) ∧ ... ∧ Bn(x)) first splits into ∀x (R(x) → B1(x)) ∧ ... (logically
              equivalent), so Define's term(x) → genus(x) ∧ feature(x) checks genus and feature separately
  nei         quantified subformulas are never checked: they are NOT ENOUGH INFO (most conservative)
"""
from __future__ import annotations

import re

from claimscan.neurosymbolic_layer.fol.formula import And, Atom, Exists, ForAll, Formula, Implies, Not, Or

QUANTIFIER_MODES = ("unit", "distribute", "nei")


def words(name: str) -> str:
  """Predicate / constant name -> words: "WorkInLibrary" -> "work in library", "born_in" -> "born in";
  a single word keeps its case ("American")."""
  split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name.replace("_", " ")).strip()
  return split.lower() if split != name.strip() and not name.isupper() else split


def _arg(a: str, entities: dict[str, str]) -> str:
  if re.fullmatch(r"[a-z]", a):
    return "something"
  return entities.get(a, words(a))


def render_atom(atom: Atom, entities: dict[str, str] | None = None) -> str:
  if atom.text:
    return atom.text
  ents = entities or {}
  args = [_arg(a, ents) for a in atom.args]
  pred = words(atom.pred)
  if not args:
    return pred
  if len(args) == 1:
    return f"{args[0]} is {pred}"
  return f"{args[0]} {pred} {' '.join(args[1:])}"


def _unary_pred(f: Formula, var: str) -> str | None:
  """A class name over the variable, lowercased ("Dog" -> "dog") unless an acronym."""
  if not (isinstance(f, Atom) and f.args == (var,)):
    return None
  w = words(f.pred)
  return w if w.isupper() else w.lower()


def render(f: Formula, entities: dict[str, str] | None = None) -> str:
  """Any node -> one sentence (its text, else a template)."""
  if f.text:
    return f.text
  if isinstance(f, Atom):
    return render_atom(f, entities)
  if isinstance(f, (ForAll, Exists)):
    body = f.body
    if isinstance(f, ForAll) and isinstance(body, Implies):
      r, b = _unary_pred(body.left, f.var), _unary_pred(body.right, f.var)
      if r and b:
        return f"Every {r} is {b}"
    if isinstance(f, Exists) and isinstance(body, And) and len(body.args) == 2:
      r, b = _unary_pred(body.args[0], f.var), _unary_pred(body.args[1], f.var)
      if r and b:
        return f"Some {r} is {b}"
    return f"{'for every' if isinstance(f, ForAll) else 'there is'} {f.var}: {render(body, entities)}"
  if isinstance(f, Not):
    return f"it is not true that {render(f.arg, entities)}"
  if isinstance(f, Implies):
    return f"if {render(f.left, entities)}, then {render(f.right, entities)}"
  joiner = " and " if isinstance(f, And) else " or "
  return joiner.join(render(a, entities) for a in f.args)


def distribute(f: Formula) -> Formula:
  """∀x (R → B1 ∧ ... ∧ Bn) -> ∀x (R → B1) ∧ ... ∧ ∀x (R → Bn), everywhere in f (texts of split nodes are dropped:
  they described the whole)."""
  if isinstance(f, ForAll) and isinstance(f.body, Implies) and isinstance(f.body.right, And):
    left = distribute(f.body.left)
    return And(tuple(ForAll(f.var, Implies(left, distribute(b))) for b in f.body.right.args))
  if isinstance(f, Not):
    return Not(distribute(f.arg), f.text)
  if isinstance(f, (And, Or)):
    return type(f)(tuple(distribute(a) for a in f.args), f.text)
  if isinstance(f, Implies):
    return Implies(distribute(f.left), distribute(f.right), f.text)
  if isinstance(f, (ForAll, Exists)):
    return type(f)(f.var, distribute(f.body), f.text)
  return f


def is_unit(f: Formula) -> bool:
  """Checked as one hypothesis: atoms and quantified subformulas."""
  return isinstance(f, (Atom, ForAll, Exists))
