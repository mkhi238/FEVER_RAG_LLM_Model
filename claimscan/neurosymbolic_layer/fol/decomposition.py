"""The decomposer's output: per claim, its structure (entities + atomic subclaims), its FOL form and search queries.

JSON the decomposer must emit (DECOMPOSER_SCHEMA, used for schema-constrained decoding):
  {"claims": [{"claim": "Hinton is an American fantasy writer born in 1950.",
               "entities": {"hinton": "Hinton"},
               "fol": {"op": "and", "args": [
                  {"op": "atom", "pred": "writer", "args": ["hinton", "fantasy"], "text": "Hinton is a fantasy writer"},
                  {"op": "atom", "pred": "nationality", "args": ["hinton", "American"], "text": "Hinton is American"},
                  {"op": "atom", "pred": "born", "args": ["hinton", "1950"], "text": "Hinton was born in 1950"}]},
               "queries": ["Hinton fantasy writer", "Hinton born 1950"]}]}
The atomic subclaims are the atoms' "text" (Decomposition.subclaims). parse_output() validates all of it and raises
FOLError (with a message fit for a repair prompt); fallback() is the whole claim as one atom, so verification
degrades to the current single-hypothesis verifier.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from claimscan.neurosymbolic_layer.fol.formula import (FORMULA_DEFS, Atom, FOLError, Formula, atoms, to_json,
                                                       validate)
from claimscan.neurosymbolic_layer.fol.hypothesis import render_atom

MAX_CLAIMS = 8
MAX_QUERIES = 5

DECOMPOSER_SCHEMA = {
  "type": "object", "additionalProperties": False, "required": ["claims"],
  "properties": {"claims": {"type": "array", "minItems": 1, "maxItems": MAX_CLAIMS, "items": {
    "type": "object", "additionalProperties": False, "required": ["claim", "entities", "fol", "queries"],
    "properties": {
      "claim": {"type": "string"},
      "entities": {"type": "object", "additionalProperties": {"type": "string"}},
      "fol": {"$ref": "#/$defs/formula"},
      "queries": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_QUERIES},
    }}}},
  "$defs": FORMULA_DEFS,
}


@dataclass
class Decomposition:
  claim: str
  formula: Formula
  entities: dict[str, str] = field(default_factory=dict)
  queries: list[str] = field(default_factory=list)
  fallback: bool = False                 # True = the decomposer failed; formula is the whole claim as one atom

  @property
  def subclaims(self) -> list[str]:
    return [render_atom(a, self.entities) for a in atoms(self.formula)]

  def to_json(self) -> dict:
    return {"claim": self.claim, "entities": self.entities, "fol": to_json(self.formula), "queries": self.queries}


def fallback(claim: str) -> Decomposition:
  return Decomposition(claim, Atom("claim", (), claim), {}, [claim], fallback=True)


def _loads(text: str):
  """JSON from a model reply: the whole reply, else the outermost {...} (models wrap JSON in prose / code fences,
  and Qwen3 may open with a <think> block)."""
  text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
      raise FOLError("reply contains no JSON object") from None
    try:
      return json.loads(text[a:b + 1])
    except json.JSONDecodeError as e:
      raise FOLError(f"reply is not valid JSON: {e}") from None


def parse_claim(obj, i: int = 0) -> Decomposition:
  where = f"claims[{i}]"
  if not isinstance(obj, dict):
    raise FOLError(f"{where}: expected an object")
  claim = obj.get("claim")
  if not isinstance(claim, str) or not claim.strip():
    raise FOLError(f"{where}.claim: must be a non-empty string")
  ents = obj.get("entities") or {}
  if not isinstance(ents, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in ents.items()):
    raise FOLError(f"{where}.entities: must map ids to names (strings)")
  if any(re.fullmatch(r"[a-z]", k) for k in ents):
    raise FOLError(f"{where}.entities: single lowercase letters are variables, use a longer id such as 'hinton'")
  queries = obj.get("queries") or []
  if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
    raise FOLError(f"{where}.queries: must be a list of strings")
  try:
    formula = validate(obj.get("fol"))
  except FOLError as e:
    raise FOLError(f"{where}.{e}") from None
  return Decomposition(claim.strip(), formula, dict(ents), [q.strip() for q in queries if q.strip()][:MAX_QUERIES])


def parse_output(text_or_obj) -> list[Decomposition]:
  obj = _loads(text_or_obj) if isinstance(text_or_obj, str) else text_or_obj
  claims = obj.get("claims") if isinstance(obj, dict) else None
  if not isinstance(claims, list) or not claims:
    raise FOLError("expected {\"claims\": [...]} with at least one claim")
  if len(claims) > MAX_CLAIMS:
    raise FOLError(f"at most {MAX_CLAIMS} claims")
  return [parse_claim(c, i) for i, c in enumerate(claims)]
