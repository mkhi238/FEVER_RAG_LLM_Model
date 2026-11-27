"""First-order-logic formulas: the AST, the JSON schema the decomposer must emit, a validator, and a string parser.

JSON shape (one node per object, "op" says which; every node may carry "text", its plain-English reading):
  {"op": "atom", "pred": "writer", "args": ["hinton", "fantasy"], "text": "Hinton is a fantasy writer"}
  {"op": "not", "arg": F}
  {"op": "and", "args": [F, F, ...]}          {"op": "or", "args": [F, F, ...]}
  {"op": "implies", "left": F, "right": F}
  {"op": "forall", "var": "x", "body": F}     {"op": "exists", "var": "x", "body": F}
Atom args are constants (entity ids such as "hinton", values such as "1950") or variables. A variable is a single lowercase
letter and must be bound by an enclosing forall / exists; anything else is a constant.

from_json() validates and raises FOLError with a message the decomposer can be shown to repair its output.
parse() reads the FOLIO / MALLS string syntax ("∀x (Dog(x) → Animal(x))"; ¬ ∧ ∨ → ↔ ⊕ and their ASCII forms) into
the same AST, for the SFT data (sft_data.py); ↔ and ⊕ are rewritten into and / or / not / implies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_DEPTH = 8
MAX_ATOMS = 16
MAX_ARGS = 4
_PRED = re.compile(r"^[^\W\d_][\w'\-]*$")       # a letter (any script), then letters / digits / _ ' -
_VAR = re.compile(r"^[a-z]$")


class FOLError(ValueError):
  """Malformed formula; the message says where and why."""


@dataclass(frozen=True)
class Atom:
  pred: str
  args: tuple[str, ...]
  text: str = ""


@dataclass(frozen=True)
class Not:
  arg: "Formula"
  text: str = ""


@dataclass(frozen=True)
class And:
  args: tuple["Formula", ...]
  text: str = ""


@dataclass(frozen=True)
class Or:
  args: tuple["Formula", ...]
  text: str = ""


@dataclass(frozen=True)
class Implies:
  left: "Formula"
  right: "Formula"
  text: str = ""


@dataclass(frozen=True)
class ForAll:
  var: str
  body: "Formula"
  text: str = ""


@dataclass(frozen=True)
class Exists:
  var: str
  body: "Formula"
  text: str = ""


Formula = Atom | Not | And | Or | Implies | ForAll | Exists


# ---- JSON schema (for schema-constrained decoding: llama-server turns it into a grammar) ----
def _node(op: str, props: dict, required: list[str]) -> dict:
  return {"type": "object", "additionalProperties": False, "required": ["op", *required],
          "properties": {"op": {"const": op}, **props, "text": {"type": "string"}}}


_F = {"$ref": "#/$defs/formula"}
FORMULA_DEFS = {"formula": {"anyOf": [
  _node("atom", {"pred": {"type": "string"}, "args": {"type": "array", "items": {"type": "string"},
                                                       "maxItems": MAX_ARGS}}, ["pred", "args"]),
  _node("not", {"arg": _F}, ["arg"]),
  _node("and", {"args": {"type": "array", "items": _F, "minItems": 2}}, ["args"]),
  _node("or", {"args": {"type": "array", "items": _F, "minItems": 2}}, ["args"]),
  _node("implies", {"left": _F, "right": _F}, ["left", "right"]),
  _node("forall", {"var": {"type": "string"}, "body": _F}, ["var", "body"]),
  _node("exists", {"var": {"type": "string"}, "body": _F}, ["var", "body"]),
]}}


# ---- JSON <-> AST ----
def from_json(obj, bound: frozenset = frozenset(), depth: int = 0, path: str = "formula") -> Formula:
  """Validate one JSON node (dict) and build the AST. Lenient on one thing only: an and / or with a single argument
  is that argument (small models emit it for one-atom claims)."""
  if depth > MAX_DEPTH:
    raise FOLError(f"{path}: nested deeper than {MAX_DEPTH}")
  if not isinstance(obj, dict):
    raise FOLError(f"{path}: expected an object, got {type(obj).__name__}")
  op = obj.get("op")
  text = obj.get("text") or ""
  if not isinstance(text, str):
    raise FOLError(f"{path}.text: must be a string")
  sub = lambda key, b=bound: from_json(obj.get(key), b, depth + 1, f"{path}.{key}")
  if op == "atom":
    pred, args = obj.get("pred"), obj.get("args")
    if not isinstance(pred, str) or not _PRED.match(pred):
      raise FOLError(f"{path}.pred: {pred!r} is not a predicate name (a letter, then letters / digits / _)")
    if not isinstance(args, list) or len(args) > MAX_ARGS or not all(isinstance(a, str) and a.strip() for a in args):
      raise FOLError(f"{path}.args: must be a list of up to {MAX_ARGS} non-empty strings")
    for a in args:
      if _VAR.match(a) and a not in bound:
        raise FOLError(f"{path}.args: variable {a!r} is not bound by a forall / exists")
    return Atom(pred, tuple(a.strip() for a in args), text)
  if op == "not":
    return Not(sub("arg"), text)
  if op in ("and", "or"):
    args = obj.get("args")
    if not isinstance(args, list) or not args:
      raise FOLError(f"{path}.args: '{op}' needs a non-empty list")
    parts = tuple(from_json(a, bound, depth + 1, f"{path}.args[{i}]") for i, a in enumerate(args))
    if len(parts) == 1:
      return parts[0]
    return (And if op == "and" else Or)(parts, text)
  if op == "implies":
    return Implies(sub("left"), sub("right"), text)
  if op in ("forall", "exists"):
    var = obj.get("var")
    if not isinstance(var, str) or not _VAR.match(var):
      raise FOLError(f"{path}.var: {var!r} must be a single lowercase letter")
    if var in bound:
      raise FOLError(f"{path}.var: {var!r} is already bound by an outer quantifier")
    return (ForAll if op == "forall" else Exists)(var, sub("body", bound | {var}), text)
  raise FOLError(f"{path}.op: unknown op {op!r} (atom, not, and, or, implies, forall, exists)")


def validate(obj) -> Formula:
  """from_json plus the size limit."""
  f = from_json(obj)
  n = len(atoms(f))
  if n > MAX_ATOMS:
    raise FOLError(f"formula has {n} atoms, more than {MAX_ATOMS}")
  return f


def to_json(f: Formula) -> dict:
  t = {"text": f.text} if f.text else {}
  if isinstance(f, Atom):
    return {"op": "atom", "pred": f.pred, "args": list(f.args), **t}
  if isinstance(f, Not):
    return {"op": "not", "arg": to_json(f.arg), **t}
  if isinstance(f, (And, Or)):
    return {"op": "and" if isinstance(f, And) else "or", "args": [to_json(a) for a in f.args], **t}
  if isinstance(f, Implies):
    return {"op": "implies", "left": to_json(f.left), "right": to_json(f.right), **t}
  return {"op": "forall" if isinstance(f, ForAll) else "exists", "var": f.var, "body": to_json(f.body), **t}


def atoms(f: Formula) -> list[Atom]:
  """Atoms left to right (repeats kept)."""
  if isinstance(f, Atom):
    return [f]
  if isinstance(f, Not):
    return atoms(f.arg)
  if isinstance(f, (And, Or)):
    return [a for g in f.args for a in atoms(g)]
  if isinstance(f, Implies):
    return atoms(f.left) + atoms(f.right)
  return atoms(f.body)


def to_string(f: Formula) -> str:
  """Unicode FOL, e.g. writer(h, fantasy) ∧ ¬born(hinton, 1950)."""
  if isinstance(f, Atom):
    return f"{f.pred}({', '.join(f.args)})" if f.args else f.pred
  if isinstance(f, Not):
    return "¬" + _wrap(f.arg)
  if isinstance(f, (And, Or)):
    return f" {'∧' if isinstance(f, And) else '∨'} ".join(_wrap(a) for a in f.args)
  if isinstance(f, Implies):
    return f"{_wrap(f.left)} → {_wrap(f.right)}"
  return f"{'∀' if isinstance(f, ForAll) else '∃'}{f.var} {_wrap(f.body)}"


def _wrap(f: Formula) -> str:
  s = to_string(f)
  return s if isinstance(f, (Atom, Not, ForAll, Exists)) else f"({s})"


# ---- string syntax (FOLIO / MALLS) -> AST ----
_TOKENS = re.compile(r"\s*(<->|↔|⟷|->|→|⊕|¬|~|!|∧|&|∨|\||∀|∃|\(|\)|,|\w[\w'\-]*)")
_SYMBOL = {"<->": "↔", "⟷": "↔", "->": "→", "~": "¬", "!": "¬", "&": "∧", "|": "∨"}


def _tokenize(s: str) -> list[str]:
  toks, i = [], 0
  s = s.strip()
  while i < len(s):
    m = _TOKENS.match(s, i)
    if not m or m.end() == i:
      raise FOLError(f"cannot read {s[i:i + 12]!r} at position {i}")
    toks.append(_SYMBOL.get(m.group(1), m.group(1)))
    i = m.end()
    while i < len(s) and s[i].isspace():
      i += 1
  return toks


class _Parser:
  def __init__(self, s: str):
    self.t, self.i, self.bound = _tokenize(s), 0, []

  def peek(self):
    return self.t[self.i] if self.i < len(self.t) else None

  def take(self, want=None):
    tok = self.peek()
    if tok is None or (want is not None and tok != want):
      raise FOLError(f"expected {want or 'more input'}, got {tok!r}")
    self.i += 1
    return tok

  def formula(self):                       # precedence, lowest first: ↔  →  ⊕  ∨  ∧  ¬/quantifiers
    left = self.implies()
    while self.peek() == "↔":
      self.take()
      right = self.implies()
      left = And((Implies(left, right), Implies(right, left)))
    return left

  def implies(self):
    left = self.xor()
    if self.peek() == "→":
      self.take()
      return Implies(left, self.implies())  # right-associative
    return left

  def xor(self):
    left = self.nary("∨", Or, self.conj)
    while self.peek() == "⊕":
      self.take()
      right = self.nary("∨", Or, self.conj)
      left = Or((And((left, Not(right))), And((Not(left), right))))
    return left

  def conj(self):
    return self.nary("∧", And, self.unary)

  def nary(self, sym, cls, part):
    args = [part()]
    while self.peek() == sym:
      self.take()
      args.append(part())
    return args[0] if len(args) == 1 else cls(tuple(args))

  def unary(self):
    tok = self.peek()
    if tok == "¬":
      self.take()
      return Not(self.unary())
    if tok in ("∀", "∃"):
      self.take()
      var = self.take()
      if not _VAR.match(var):
        raise FOLError(f"quantified variable {var!r} must be a single lowercase letter")
      self.bound.append(var)
      body = self.unary()
      self.bound.pop()
      return (ForAll if tok == "∀" else Exists)(var, body)
    if tok == "(":
      self.take()
      f = self.formula()
      self.take(")")
      return f
    return self.atom()

  def atom(self):
    pred = self.take()
    if not _PRED.match(pred):
      raise FOLError(f"{pred!r} is not a predicate name")
    args = []
    if self.peek() == "(":
      self.take()
      while True:
        a = self.take()
        if not re.match(r"^\w[\w'\-]*$", a):
          raise FOLError(f"bad argument {a!r} of {pred}")
        if self.peek() == "(":
          raise FOLError(f"function terms such as {a}(...) are not supported")
        if _VAR.match(a) and a not in self.bound:
          raise FOLError(f"variable {a!r} in {pred} is not bound")
        args.append(a)
        if self.peek() == ",":
          self.take()
          continue
        self.take(")")
        break
    if len(args) > MAX_ARGS:
      raise FOLError(f"{pred} has more than {MAX_ARGS} arguments")
    return Atom(pred, tuple(args))


def parse(s: str) -> Formula:
  """FOLIO / MALLS string -> AST (FOLError if malformed). Quantifiers scope over the next unary formula, so write
  ∀x (A(x) → B(x)) with the parentheses, as both datasets do."""
  p = _Parser(s)
  f = p.formula()
  if p.peek() is not None:
    raise FOLError(f"unexpected {p.peek()!r} after a complete formula")
  return f


__all__ = ["Atom", "Not", "And", "Or", "Implies", "ForAll", "Exists", "Formula", "FOLError", "FORMULA_DEFS",
           "from_json", "validate", "to_json", "atoms", "to_string", "parse"]
