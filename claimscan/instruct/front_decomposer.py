"""Decomposer: text -> per claim its structure (entities, atomic subclaims), its FOL form and search queries.

The model is the local Qwen3-4B (role "decomposer" in llamacpp/roles.yaml); the output is checked by
claimscan/neurosymbolic_layer/fol/decomposition.py parse_output. Prompt + few-shot: prompts.py. Not on the API path yet.

The LLM call is pluggable. A client is any callable

    client(messages: list[dict], schema: dict) -> str     (OpenAI-style messages; schema = DECOMPOSER_SCHEMA)

  langchain_client(chat)   wraps a ChatOpenAI (chat_clients.build_chat("decomposer")) with response_format
                           json_schema, so llama-server constrains decoding to the schema
  FakeClient(replies)      canned replies, for tests and runs without a model server

decompose() tries up to `attempts` times; after an invalid reply it shows the model the validator's message and asks
for a repair. If every try fails it returns the fallback (the whole text as one atom), so verification never breaks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from claimscan.instruct.prompts import decomposer_messages
from claimscan.neurosymbolic_layer.fol.decomposition import DECOMPOSER_SCHEMA, Decomposition, fallback, parse_output
from claimscan.neurosymbolic_layer.fol.formula import FOLError


@dataclass
class DecomposeResult:
  claims: list[Decomposition]
  attempts: int                       # model calls made
  errors: list[str] = field(default_factory=list)   # validator / client error per failed attempt

  @property
  def fallback(self) -> bool:
    return any(d.fallback for d in self.claims)


def decompose(text: str, client, attempts: int = 2) -> DecomposeResult:
  previous = error = None
  errors = []
  for i in range(1, attempts + 1):
    try:
      reply = client(decomposer_messages(text, previous, error), DECOMPOSER_SCHEMA)
    except Exception as e:            # server down / timeout: fail open, like the rear aggregator
      errors.append(f"client: {type(e).__name__}: {e}")
      return DecomposeResult([fallback(text)], i, errors)
    try:
      return DecomposeResult(parse_output(reply), i, errors)
    except FOLError as e:
      previous, error = reply, str(e)
      errors.append(error)
  return DecomposeResult([fallback(text)], attempts, errors)


def langchain_client(chat):
  """client() over a LangChain chat model (ChatOpenAI to llama-server). Plain dict messages work with invoke."""
  def client(messages, schema):
    bound = chat.bind(response_format={"type": "json_schema",
                                       "json_schema": {"name": "decomposition", "schema": schema}})
    return bound.invoke(messages).content
  return client


def build_client(role: str = "decomposer", base_url: str | None = None, timeout: float = 60.0):
  """langchain_client over chat_clients.build_chat(role); needs the llama-server for that role running."""
  from claimscan.instruct.chat_clients import build_chat
  return langchain_client(build_chat(role, base_url, timeout))


class FakeClient:
  """Replies in order (the last one repeats), or reply(text) -> str for a callable; records the messages."""
  def __init__(self, replies):
    self.replies, self.calls = replies, []

  def __call__(self, messages, schema):
    self.calls.append(messages)
    if callable(self.replies):
      return self.replies(messages[-1]["content"])
    return self.replies[min(len(self.calls), len(self.replies)) - 1]
