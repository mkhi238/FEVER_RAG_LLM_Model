"""One client per LLM role, configured by llamacpp/roles.yaml.   (YOU: implement)

A role points at local llama.cpp (``backend: llamacpp``, http://127.0.0.1:<port>/v1) or a
hosted OpenAI-compatible API (``backend: hosted``). Callers only ever use the role name,
so moving a role between local and hosted is a roles.yaml edit (ClaimScan plan:
"one config per LLM role").
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoleConfig:
  name: str
  backend: str            # "llamacpp" | "hosted"
  base_url: str           # llamacpp: f"http://127.0.0.1:{port}/v1"
  model: str
  api_key: str | None = None


def load_role(name: str, roles_file: str | None = None) -> RoleConfig:
  """Read one role from llamacpp/roles.yaml.

  TODO: default roles_file to <repo>/llamacpp/roles.yaml; for hosted roles read the key from
  os.environ[api_key_env] (raise a clear error if it's missing). Never log the key.
  """
  raise NotImplementedError


def chat(role: str, messages: list[dict], **params) -> str:
  """POST /v1/chat/completions for ``role`` and return the assistant text.

  TODO: use httpx (already installed); pass temperature/max_tokens through **params;
  for research_agent the grammar is enforced server-side (--grammar-file), so just return text.
  """
  raise NotImplementedError


def parse_action(text: str) -> tuple[str, str | None]:
  """Parse a research_agent turn into ("search", query) or ("stop", None).

  TODO: must accept exactly what llamacpp/grammars/research_action.gbnf allows; raise ValueError
  otherwise (the graph's failure edge handles it).
  """
  raise NotImplementedError


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
  """POST /v1/rerank on the reranker role; return (doc_index, score) sorted by score desc.

  TODO: llama-server needs --reranking; check scores look sane (README step 3) before relying on them.
  """
  raise NotImplementedError
