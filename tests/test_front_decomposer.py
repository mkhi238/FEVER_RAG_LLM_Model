"""Decomposer (claimscan/instruct/front_decomposer.py, prompts.py) with a fake client: no model server."""
import json

from claimscan.instruct.front_decomposer import FakeClient, decompose, langchain_client
from claimscan.instruct.prompts import FEW_SHOTS, decomposer_messages
from claimscan.neurosymbolic_layer.fol.decomposition import DECOMPOSER_SCHEMA, parse_output
from claimscan.neurosymbolic_layer.fol.formula import atoms

GOOD = json.dumps(FEW_SHOTS[0][1])


def test_few_shots_validate():
  for text, answer in FEW_SHOTS:
    claims = parse_output(answer)
    assert claims and all(c.queries for c in claims), text
    # every unit the checker will see has its own text (atoms over variables sit inside a quantifier that has one)
    for c in claims:
      assert all(a.text for a in atoms(c.formula) if not any(len(x) == 1 for x in a.args)), text


def test_messages_shape():
  msgs = decomposer_messages("Rex is a dog.")
  assert msgs[0]["role"] == "system" and msgs[-1] == {"role": "user", "content": "Rex is a dog."}
  assert len(msgs) == 2 + 2 * len(FEW_SHOTS)
  repair = decomposer_messages("Rex is a dog.", previous="{}", error="bad")
  assert repair[-2] == {"role": "assistant", "content": "{}"} and "bad" in repair[-1]["content"]


def test_decompose_first_try():
  client = FakeClient([GOOD])
  r = decompose("S. E. Hinton is an American fantasy writer born in 1950.", client)
  assert r.attempts == 1 and not r.fallback and not r.errors
  assert r.claims[0].subclaims == ["S. E. Hinton is a fantasy writer", "S. E. Hinton is American",
                                   "S. E. Hinton was born in 1950"]


def test_decompose_repairs_after_invalid_reply():
  bad = '{"claims": [{"claim": "x", "entities": {}, "fol": {"op": "atom", "pred": "p", "args": ["y"]}, "queries": []}]}'
  client = FakeClient([bad, GOOD])
  r = decompose("text", client, attempts=2)
  assert r.attempts == 2 and not r.fallback and "not bound" in r.errors[0]
  assert "not bound" in client.calls[1][-1]["content"]          # the repair turn shows the validator's message


def test_decompose_falls_back():
  r = decompose("Rex is a dog.", FakeClient(["nonsense"]), attempts=2)
  assert r.fallback and r.attempts == 2 and len(r.errors) == 2
  assert r.claims[0].subclaims == ["Rex is a dog."]


def test_decompose_client_error_fails_open():
  def broken(messages, schema):
    raise ConnectionError("server down")
  r = decompose("Rex is a dog.", broken)
  assert r.fallback and r.attempts == 1 and "ConnectionError" in r.errors[0]


def test_langchain_client_passes_schema():
  class Reply:
    content = GOOD

  class Chat:
    def bind(self, **kw):
      self.kw = kw
      return self

    def invoke(self, messages):
      self.messages = messages
      return Reply()

  chat = Chat()
  assert langchain_client(chat)([{"role": "user", "content": "x"}], DECOMPOSER_SCHEMA) == GOOD
  assert chat.kw["response_format"]["json_schema"]["schema"] is DECOMPOSER_SCHEMA
