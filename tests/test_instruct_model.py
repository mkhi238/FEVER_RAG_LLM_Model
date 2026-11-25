"""build_chat (claimscan/instruct/chat_clients.py): one ChatOpenAI client per role in llamacpp/roles.yaml.

Contract (from model.py's docstring):
- build_chat(role, base_url=None, timeout=30.0) -> ChatOpenAI
- roles.yaml keeps its roles under the top-level `roles:` key.
- A llamacpp role is reached at http://127.0.0.1:<port>/v1.
- A role with `shares: <other>` uses the other role's server (port and model).
- base_url, if given, wins over roles.yaml.
- temperature 0, max_retries 0 (the caller fails open), timeout passed through.
No server is needed: building a ChatOpenAI does not connect.
Skipped while build_chat is a stub (raises NotImplementedError or returns None).
"""
import inspect

import pytest
import yaml
from langchain_openai import ChatOpenAI

from claimscan.instruct import chat_clients as model

ROLES = {
  "llama_server": "/unused/llama-server",
  "models_dir": "/unused/models",
  "roles": {
    "front_facer": {"backend": "llamacpp", "model": "front.gguf", "port": 9101},
    "decomposer": {"backend": "llamacpp", "shares": "front_facer"},
    "aggregator": {"backend": "llamacpp", "model": "agg.gguf", "port": 9103},
  },
}


def _stub():
  try:
    return model.build_chat("aggregator") is None
  except NotImplementedError:
    return True
  except Exception:   # a real bug: let the tests below show it
    return False


if _stub():
  pytest.skip("build_chat not written yet", allow_module_level=True)


@pytest.fixture
def roles_file(tmp_path, monkeypatch):
  path = tmp_path / "roles.yaml"
  path.write_text(yaml.safe_dump(ROLES), encoding="utf-8")
  monkeypatch.setattr(model, "ROLES_FILE", str(path))
  return path


def test_returns_chat_openai(roles_file):
  assert isinstance(model.build_chat("aggregator"), ChatOpenAI)


def test_own_port(roles_file):
  assert model.build_chat("aggregator").openai_api_base == "http://127.0.0.1:9103/v1"
  assert model.build_chat("front_facer").openai_api_base == "http://127.0.0.1:9101/v1"


def test_shares_uses_other_roles_server(roles_file):
  chat = model.build_chat("decomposer")
  assert chat.openai_api_base == "http://127.0.0.1:9101/v1"
  assert chat.model_name == "front.gguf"


def test_model_from_role(roles_file):
  assert model.build_chat("aggregator").model_name == "agg.gguf"


def test_base_url_override_wins(roles_file):
  chat = model.build_chat("aggregator", base_url="http://llama:8083/v1")
  assert chat.openai_api_base == "http://llama:8083/v1"


def test_deterministic_no_retries(roles_file):
  chat = model.build_chat("aggregator")
  assert chat.temperature == 0
  assert chat.max_retries == 0


def test_timeout(roles_file):
  assert model.build_chat("aggregator").request_timeout == 30.0
  assert model.build_chat("aggregator", timeout=5.0).request_timeout == 5.0


def test_unknown_role_raises(roles_file):
  with pytest.raises((KeyError, ValueError)):
    model.build_chat("no_such_role")


@pytest.mark.skipif("api_key" not in inspect.signature(model.build_chat).parameters,
                    reason="build_chat takes no api_key (aggregation.build_llm uses its inline client for hosted keys)")
def test_api_key_passed_through(roles_file):
  chat = model.build_chat("aggregator", api_key="sk-test")
  assert chat.openai_api_key.get_secret_value() == "sk-test"


def test_real_roles_yaml():
  """The checked-in llamacpp/roles.yaml: ports must match configs/default.yaml and docker-compose (8083)."""
  assert model.build_chat("aggregator").openai_api_base == "http://127.0.0.1:8083/v1"
  assert model.build_chat("decomposer").openai_api_base == model.build_chat("front_facer").openai_api_base
