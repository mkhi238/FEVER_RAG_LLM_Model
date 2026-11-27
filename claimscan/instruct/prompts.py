"""Prompts for the decomposer (the local Qwen3-4B): structure the claim, write its FOL form and search queries.

The aggregator's prompt lives in rear_aggregator.py (SYSTEM_PROMPT); no front_facer prompt yet. The output shape is
claimscan/neurosymbolic_layer/fol/decomposition.py DECOMPOSER_SCHEMA; the few-shot answers below must validate
against it (tests/test_front_decomposer.py checks that).
"""
import json

DECOMPOSER_SYSTEM = """You turn text into first-order logic (FOL) so each part can be checked against Wikipedia.

For every checkable claim in the text return:
- "claim": the claim as one self-contained sentence (resolve pronouns).
- "entities": short ids for the named things, e.g. {"hinton": "S. E. Hinton"}. Ids are lowercase words, never a
  single letter (single letters are variables).
- "fol": the claim as a formula. Nodes: {"op":"atom","pred":...,"args":[...],"text":...}, {"op":"not","arg":F},
  {"op":"and","args":[F,F,...]}, {"op":"or","args":[F,F,...]}, {"op":"implies","left":F,"right":F},
  {"op":"forall","var":"x","body":F}, {"op":"exists","var":"x","body":F}.
  Split the claim into the smallest facts and join them with "and". Every atom and every forall / exists needs
  "text": that fact alone as a short plain sentence with the full names (it is checked on its own).
  Use "not" only when the text itself negates. Use forall / exists only for "all", "every", "no", "some", "a ..." over
  a class; a named thing is a constant, not a variable.
- "queries": 1 to 3 short search queries that would find the evidence.
Answer with the JSON object only."""

_HINTON = {"claims": [{
  "claim": "S. E. Hinton is an American fantasy writer born in 1950.",
  "entities": {"hinton": "S. E. Hinton"},
  "fol": {"op": "and", "args": [
    {"op": "atom", "pred": "writer", "args": ["hinton", "fantasy"], "text": "S. E. Hinton is a fantasy writer"},
    {"op": "atom", "pred": "nationality", "args": ["hinton", "American"], "text": "S. E. Hinton is American"},
    {"op": "atom", "pred": "born_in", "args": ["hinton", "1950"], "text": "S. E. Hinton was born in 1950"}]},
  "queries": ["S. E. Hinton", "S. E. Hinton born 1950"]}]}

_OBAMA = {"claims": [{
  "claim": "Barack Obama was not born in Kenya.",
  "entities": {"obama": "Barack Obama"},
  "fol": {"op": "not", "arg": {"op": "atom", "pred": "born_in", "args": ["obama", "Kenya"],
                               "text": "Barack Obama was born in Kenya"}},
  "queries": ["Barack Obama birthplace"]}]}

_COSTER = {"claims": [
  {"claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
   "entities": {"nikolaj": "Nikolaj Coster-Waldau", "fox": "Fox Broadcasting Company"},
   "fol": {"op": "atom", "pred": "worked_with", "args": ["nikolaj", "fox"],
           "text": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company"},
   "queries": ["Nikolaj Coster-Waldau Fox"]},
  {"claim": "Nikolaj Coster-Waldau has won an Oscar.",
   "entities": {"nikolaj": "Nikolaj Coster-Waldau"},
   "fol": {"op": "exists", "var": "x", "text": "Nikolaj Coster-Waldau has won an Oscar", "body": {"op": "and", "args": [
     {"op": "atom", "pred": "oscar", "args": ["x"]},
     {"op": "atom", "pred": "won", "args": ["nikolaj", "x"]}]}},
   "queries": ["Nikolaj Coster-Waldau awards"]}]}

_POODLE = {"claims": [{
  "claim": "A poodle is a dog breed with a curly coat.",
  "entities": {},
  "fol": {"op": "forall", "var": "x", "text": "Every poodle is a dog breed with a curly coat", "body": {
    "op": "implies",
    "left": {"op": "atom", "pred": "poodle", "args": ["x"]},
    "right": {"op": "and", "args": [
      {"op": "atom", "pred": "dog_breed", "args": ["x"], "text": "A poodle is a dog breed"},
      {"op": "atom", "pred": "curly_coat", "args": ["x"], "text": "A poodle has a curly coat"}]}}},
  "queries": ["poodle dog breed", "poodle coat"]}]}

FEW_SHOTS = [
  ("S. E. Hinton is an American fantasy writer born in 1950.", _HINTON),
  ("Barack Obama was not born in Kenya.", _OBAMA),
  ("Nikolaj Coster-Waldau worked with the Fox Broadcasting Company and has won an Oscar.", _COSTER),
  ("A poodle is a dog breed with a curly coat.", _POODLE),
]


def decomposer_messages(text: str, previous: str | None = None, error: str | None = None,
                        few_shots=FEW_SHOTS) -> list[dict]:
  """OpenAI-style messages. With previous + error, the last turn asks the model to repair its invalid output."""
  msgs = [{"role": "system", "content": DECOMPOSER_SYSTEM}]
  for user, answer in few_shots:
    msgs += [{"role": "user", "content": user},
             {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]
  msgs.append({"role": "user", "content": text})
  if previous is not None:
    msgs += [{"role": "assistant", "content": previous},
             {"role": "user", "content": f"That output is invalid: {error}. Answer again with corrected JSON only."}]
  return msgs
