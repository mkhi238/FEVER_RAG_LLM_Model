"""Glue between the pipeline/API and the rear aggregator (claimscan/aggregation.py).

The fake structured LLM vetoes when `veto=True`; it works whatever prompt shape aggregate() sends.
Veto gate: applied only if verifier conf < 0.6 OR llm_confidence >= 0.9.
Tests that need the user's aggregate()/final_label() skip until rear_aggregator.py has them;
the disabled and fail-open paths run either way.
"""
import math

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from claimscan import aggregation
from claimscan.aggregation import aggregate_verdict, to_claim_result
from claimscan.api import deps
from claimscan.api.main import app
from claimscan.pipeline import Services, Verdict

try:
  from claimscan.instruct import rear_aggregator as ra
  HAS_CONTRACT = hasattr(ra, "aggregate") and hasattr(ra, "final_label")
  if HAS_CONTRACT:
    try:
      ra.final_label("TRUE", False, 0.9, 0.0)
      from claimscan.instruct.schemas import RearInput
      ra.aggregate(RearInput(user_text="", results=[]), RunnableLambda(lambda x: x))
    except NotImplementedError:   # the user's stubs are in, the bodies are not yet
      HAS_CONTRACT = False
except Exception:
  HAS_CONTRACT = False
needs_contract = pytest.mark.skipif(not HAS_CONTRACT, reason="rear aggregator contract not implemented yet")

NEI = "NOT ENOUGH INFO"
CLAIM = "Tokyo is the capital of Japan"


class FakeOut(BaseModel):
  veto: bool
  veto_reason: str
  llm_confidence: float
  explanation: str


def fake_llm(veto=False, llm_conf=0.95, explanation="Supported by [Tokyo:0]."):
  calls = []

  def run(inp):
    calls.append(inp)
    return FakeOut(veto=veto, veto_reason="evidence is off-topic" if veto else "", llm_confidence=llm_conf,
                   explanation=explanation)
  return RunnableLambda(run), calls


def failing_llm():
  def run(inp):
    raise ConnectionError("llama-server at http://127.0.0.1:8083 refused the connection")
  return RunnableLambda(run)


def verdict(label="TRUE", top=0.9, n_evidence=1):
  rest = (1 - top) / 2
  probs = {l: (top if l == label else rest) for l in ("TRUE", "FALSE", NEI)}
  ev = [(f"sentence {i}", f"Tokyo#{i}") for i in range(n_evidence)]
  return Verdict(label, top, probs, ev)


# ---- adapter ----
def test_to_claim_result_maps_sources_and_keeps_top5():
  r = to_claim_result(CLAIM, verdict(n_evidence=8))
  assert r.label == "TRUE" and r.claim == CLAIM
  assert [(e.page, e.line) for e in r.evidence] == [("Tokyo", i) for i in range(5)]
  assert set(r.probs) == {"TRUE", "FALSE", NEI}


def test_page_with_hash_in_title():
  v = Verdict("TRUE", 0.9, {"TRUE": 0.9, "FALSE": 0.05, NEI: 0.05}, [("x", "C#_(language)#3")])
  e = to_claim_result(CLAIM, v).evidence[0]
  assert (e.page, e.line) == ("C#_(language)", 3)


# ---- aggregate_verdict ----
def test_disabled_returns_verifier_verdict():
  v = verdict()
  a = aggregate_verdict(CLAIM, v, None)
  assert a.label == a.verifier_label == "TRUE"
  assert a.verdict is v and not a.vetoed and a.explanation is None and a.aggregator_error is None


def test_no_evidence_skips_llm():
  llm, calls = fake_llm(veto=True)
  a = aggregate_verdict(CLAIM, Verdict(NEI, 0.0, None, []), llm)
  assert a.label == NEI and calls == [] and a.aggregator_error is None


def test_fail_open_on_exception():
  a = aggregate_verdict(CLAIM, verdict("FALSE"), failing_llm())
  assert a.label == a.verifier_label == "FALSE"
  assert not a.vetoed
  assert a.aggregator_error and "127.0.0.1" not in a.aggregator_error


@needs_contract
@pytest.mark.parametrize("label", ["TRUE", "FALSE"])
def test_veto_becomes_nei(label):
  llm, calls = fake_llm(veto=True)
  a = aggregate_verdict(CLAIM, verdict(label), llm)
  assert a.label == NEI and a.verifier_label == label
  assert a.veto_requested and a.vetoed and a.aggregator_error is None
  assert a.llm_confidence == pytest.approx(0.95)
  assert len(calls) == 1


@needs_contract
@pytest.mark.parametrize("label", ["TRUE", "FALSE"])
@pytest.mark.parametrize("verifier_conf,llm_conf,applied", [
  (0.9, 0.5, False),    # verifier sure, LLM unsure: veto ignored
  (0.6, 0.89, False),   # boundary: conf 0.6 is not < 0.6, llm just under 0.9
  (0.55, 0.1, True),    # verifier unsure: veto applied
  (0.9, 0.9, True),     # LLM sure (== 0.9): veto applied
  (0.95, 0.99, True),
])
def test_veto_gate(label, verifier_conf, llm_conf, applied):
  llm, _ = fake_llm(veto=True, llm_conf=llm_conf)
  a = aggregate_verdict(CLAIM, verdict(label, top=verifier_conf), llm)
  assert a.veto_requested is True
  assert a.vetoed is applied
  assert a.label == (NEI if applied else label)
  assert a.aggregator_error is None


@needs_contract
@pytest.mark.parametrize("label", ["TRUE", "FALSE", NEI])
def test_no_veto_keeps_label(label):
  llm, _ = fake_llm(veto=False)
  a = aggregate_verdict(CLAIM, verdict(label), llm)
  assert a.label == label and not a.vetoed and not a.veto_requested
  assert a.explanation == "Supported by [Tokyo:0]."


@needs_contract
def test_flip_is_rejected(monkeypatch):
  # a buggy aggregate() that flips TRUE -> FALSE must fail open, not reach the user
  real = ra.aggregate
  def flip(rear_input, llm, **kw):
    out = real(rear_input, llm, **kw)
    out.claims[0].verdict = "FALSE"
    return out
  llm, _ = fake_llm()
  monkeypatch.setattr(ra, "aggregate", flip)
  a = aggregate_verdict(CLAIM, verdict("TRUE"), llm)
  assert a.label == "TRUE" and a.aggregator_error


# ---- API (stub services copied from tests/test_api.py) ----
class StubRetriever:
  def search(self, query, k=50):
    return [{"id": "Tokyo", "line": 0, "score": 0.5}]


class StubVerifier:
  def __init__(self, logits):
    self.logits = torch.tensor([logits])

  def encode_text(self, claim, evidence):
    return {"input_ids": torch.zeros(1, 4, dtype=torch.long), "attention_mask": torch.ones(1, 4, dtype=torch.long)}

  def __call__(self, input_ids, attention_mask):
    return self.logits


CORPUS = pd.DataFrame({"id": ["Tokyo"], "line": [0], "text": ["tokyo is the capital of japan"]})


@pytest.fixture
def client_with(monkeypatch):
  def make(logits, llm=None):
    monkeypatch.setattr(deps, "load_services", lambda cfg: Services(StubVerifier(logits), StubRetriever(), CORPUS))
    monkeypatch.setattr(deps, "load_aggregator", lambda cfg: llm)
    return TestClient(app)
  return make


def _post(client):
  r = client.post("/fact-check", json={"claim": CLAIM})
  assert r.status_code == 200
  return r.json()


def test_api_disabled_is_unchanged(client_with):
  with client_with([5.0, 0.0, 0.0]) as client:
    body = _post(client)
  assert body["prediction"] == "TRUE"
  assert body["response"] == "This statement is TRUE. Evidence: tokyo is the capital of japan"
  assert body["verifier_prediction"] is None and body["vetoed"] is False and body["veto_requested"] is False
  assert body["explanation"] is None and body["aggregator_error"] is None


def test_default_config_is_off(monkeypatch):
  from claimscan.config import DEFAULT_CONFIG_FILE, load_config
  monkeypatch.delenv("CLAIMSCAN_AGGREGATOR_ENABLED", raising=False)
  cfg = load_config(DEFAULT_CONFIG_FILE)
  assert cfg.aggregator_enabled is False and deps.load_aggregator(cfg) is None
  assert cfg.aggregator_api_key_env == ""


def test_env_overrides(monkeypatch):
  from claimscan.config import DEFAULT_CONFIG_FILE, load_config
  monkeypatch.setenv("CLAIMSCAN_AGGREGATOR_ENABLED", "1")
  monkeypatch.setenv("CLAIMSCAN_AGGREGATOR_URL", "http://llama:8080/v1")
  cfg = load_config(DEFAULT_CONFIG_FILE)
  assert cfg.aggregator_enabled is True and cfg.aggregator_base_url == "http://llama:8080/v1"


def test_api_fails_open_when_llm_down(client_with):
  with client_with([5.0, 0.0, 0.0], llm=failing_llm()) as client:
    body = _post(client)
  assert body["prediction"] == body["verifier_prediction"] == "TRUE"
  assert body["response"] == "This statement is TRUE. Evidence: tokyo is the capital of japan"
  assert body["aggregator_error"] and "refused" not in body["aggregator_error"]


@needs_contract
def test_api_veto(client_with):
  llm, _ = fake_llm(veto=True, explanation="The evidence does not mention this.")
  with client_with([5.0, 0.0, 0.0], llm=llm) as client:
    body = _post(client)
  assert body["prediction"] == NEI and body["verifier_prediction"] == "TRUE"
  assert body["veto_requested"] is True and body["vetoed"] is True
  assert body["veto_reason"] == "evidence is off-topic" and body["llm_confidence"] == pytest.approx(0.95)
  assert body["response"] == "The evidence does not mention this."
  assert body["confidence"] == max(body["probabilities"].values())


@needs_contract
def test_api_ignored_veto_is_visible(client_with):
  # verifier conf ~0.99, LLM unsure: the veto is requested but not applied
  llm, _ = fake_llm(veto=True, llm_conf=0.3)
  with client_with([5.0, 0.0, 0.0], llm=llm) as client:
    body = _post(client)
  assert body["prediction"] == body["verifier_prediction"] == "TRUE"
  assert body["veto_requested"] is True and body["vetoed"] is False


@needs_contract
def test_api_no_veto(client_with):
  llm, _ = fake_llm(veto=False)
  with client_with([5.0, 0.0, 0.0], llm=llm) as client:
    body = _post(client)
  assert body["prediction"] == body["verifier_prediction"] == "TRUE"
  assert body["vetoed"] is False and body["response"] == "Supported by [Tokyo:0]."


# ---- API key for a hosted aggregator ----
class CapturedChat:
  """Stands in for ChatTogether / ChatOpenAI: records the constructor kwargs."""
  made = []

  def __init__(self, **kw):
    self.kw = kw
    CapturedChat.made.append(kw)

  def with_structured_output(self, schema, method=None):
    return self


@pytest.fixture
def capture_chat(monkeypatch):
  import langchain_together

  from claimscan.instruct import chat_clients
  CapturedChat.made = []
  monkeypatch.setattr(langchain_together, "ChatTogether", CapturedChat)   # hosted path (key set)
  monkeypatch.setattr(chat_clients, "ChatOpenAI", CapturedChat)           # local path (build_chat)
  return CapturedChat.made


def _cfg(**kw):
  import dataclasses

  from claimscan.config import config
  return dataclasses.replace(config, aggregator_enabled=True, **kw)


def test_key_read_from_named_env_var(capture_chat, monkeypatch):
  monkeypatch.setenv("TEST_AGG_KEY", "sk-test-secret-123")
  aggregation.build_llm(_cfg(aggregator_api_key_env="TEST_AGG_KEY", aggregator_model="meta-llama/Llama-3.3-70B-Instruct-Turbo"))
  kw = capture_chat[-1]
  assert kw["api_key"] == "sk-test-secret-123"
  assert kw["model"] == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
  assert "base_url" not in kw   # Together's own endpoint, not the local llama-server URL


def test_default_key_is_not_needed(capture_chat):
  aggregation.build_llm(_cfg(aggregator_api_key_env=""))
  assert capture_chat[-1]["api_key"] == "not-needed"


@pytest.mark.parametrize("value", [None, ""])
def test_missing_key_fails_open(capture_chat, monkeypatch, value):
  if value is None:
    monkeypatch.delenv("TEST_AGG_KEY", raising=False)
  else:
    monkeypatch.setenv("TEST_AGG_KEY", value)
  llm = aggregation.build_llm(_cfg(aggregator_api_key_env="TEST_AGG_KEY"))
  assert capture_chat == []   # no client built
  a = aggregate_verdict(CLAIM, verdict("TRUE"), llm)
  assert a.label == a.verifier_label == "TRUE" and not a.vetoed
  assert "TEST_AGG_KEY" in a.aggregator_error


def test_key_never_in_error_or_log(monkeypatch, caplog):
  monkeypatch.setattr(aggregation, "_secrets", {"sk-test-secret-123"})
  def leaky(inp):
    raise RuntimeError("401 Incorrect API key provided: sk-test-secret-123")
  from claimscan.instruct import rear_aggregator
  monkeypatch.setattr(rear_aggregator, "aggregate", lambda ri, llm, **kw: llm.invoke(ri))
  caplog.set_level("WARNING", logger="claimscan.aggregation")
  a = aggregate_verdict(CLAIM, verdict("TRUE"), RunnableLambda(leaky))
  assert "secret" not in a.aggregator_error
  assert "secret" not in caplog.text and "RuntimeError" in caplog.text


def test_env_overrides_model_and_key_env(monkeypatch):
  from claimscan.config import DEFAULT_CONFIG_FILE, load_config
  monkeypatch.setenv("CLAIMSCAN_AGGREGATOR_MODEL", "gpt-5-nano")
  monkeypatch.setenv("CLAIMSCAN_AGGREGATOR_API_KEY_ENV", "OPENAI_API_KEY")
  cfg = load_config(DEFAULT_CONFIG_FILE)
  assert cfg.aggregator_model == "gpt-5-nano" and cfg.aggregator_api_key_env == "OPENAI_API_KEY"


def test_api_missing_key_fails_open(client_with, monkeypatch):
  monkeypatch.delenv("TEST_AGG_KEY", raising=False)
  llm = aggregation.build_llm(_cfg(aggregator_api_key_env="TEST_AGG_KEY"))
  with client_with([5.0, 0.0, 0.0], llm=llm) as client:
    body = _post(client)
  assert body["prediction"] == "TRUE" and "TEST_AGG_KEY" in body["aggregator_error"]
