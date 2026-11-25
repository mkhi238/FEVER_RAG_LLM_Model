"""The rear aggregator on real LLM-shaped output: the actual ChatOpenAI + with_structured_output(json_schema) stack
that build_llm() makes, with only the network faked (httpx.MockTransport returns OpenAI chat-completion JSON, the
same wire format llama-server and hosted APIs send).

test_aggregation.py / test_rear_aggregator.py feed aggregate() ready-made Python objects. These tests check the step
before that: JSON text -> VetoOutput, what the request asks for, and that every kind of bad output (broken JSON,
out-of-range or missing fields, HTTP errors, refusals) fails open to the verifier's verdict.
Tests that go through aggregate() skip until the user's rear_aggregator.aggregate is written.
"""
import json

import httpx
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from claimscan import aggregation
from claimscan.aggregation import aggregate_verdict
from claimscan.api import deps
from claimscan.api.main import app
from claimscan.pipeline import Services, Verdict

langchain_openai = pytest.importorskip("langchain_openai")

try:
  from claimscan.instruct import rear_aggregator as ra
  ra.aggregate(None, None)
  HAS_AGGREGATE = True
except NotImplementedError:
  HAS_AGGREGATE = False
except Exception:          # written: it fails on the None input instead of raising NotImplementedError
  HAS_AGGREGATE = True
needs_aggregate = pytest.mark.skipif(not HAS_AGGREGATE, reason="rear_aggregator.aggregate not written yet")

NEI = "NOT ENOUGH INFO"
CLAIM = "Tokyo is the capital of Japan"


# ---- fake OpenAI-compatible server ----
def completion(content, finish_reason="stop", refusal=None):
  """One OpenAI /v1/chat/completions response body whose message text is `content`."""
  msg = {"role": "assistant", "content": content}
  if refusal is not None:
    msg["refusal"] = refusal
  return {"id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "gemma",
          "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
          "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}


def veto_json(veto=True, conf=0.95, reason="evidence is about a different city",
              explanation="The sentence is about Kyoto [Kyoto:0], not Tokyo."):
  return json.dumps({"veto": veto, "veto_reason": reason, "explanation": explanation, "llm_confidence": conf})


class FakeServer:
  """Records each request body; answers with `body` (a dict) or an (status, dict) tuple."""
  def __init__(self, body):
    self.body = body
    self.requests = []

  def __call__(self, request: httpx.Request) -> httpx.Response:
    self.requests.append(json.loads(request.content))
    status, body = self.body if isinstance(self.body, tuple) else (200, self.body)
    return httpx.Response(status, json=body)


@pytest.fixture
def llm_from(monkeypatch):
  """llm_from(body) -> (the runnable build_llm() returns, wired to a FakeServer, the server)."""
  from claimscan.instruct import model
  real = langchain_openai.ChatOpenAI

  def make(body):
    server = FakeServer(body)

    def chat(*a, **kw):
      kw.setdefault("max_retries", 0)
      return real(*a, http_client=httpx.Client(transport=httpx.MockTransport(server)), **kw)
    # build_llm uses model.build_chat once the user writes it, else its inline ChatOpenAI fallback: patch both
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", chat)
    monkeypatch.setattr(model, "ChatOpenAI", chat, raising=False)
    import dataclasses

    from claimscan.config import config
    cfg = dataclasses.replace(config, aggregator_enabled=True, aggregator_api_key_env="",
                              aggregator_base_url="http://llama.test/v1")
    return aggregation.build_llm(cfg), server
  return make


def verdict(label="TRUE", top=0.55):
  rest = (1 - top) / 2
  probs = {l: (top if l == label else rest) for l in ("TRUE", "FALSE", NEI)}
  return Verdict(label, top, probs, [("tokyo is the capital of japan", "Tokyo#0")])


# ---- the request ----
@needs_aggregate
def test_request_asks_for_veto_json_schema(llm_from):
  llm, server = llm_from(completion(veto_json(veto=False, conf=0.1)))
  aggregate_verdict(CLAIM, verdict(), llm)
  assert len(server.requests) == 1                    # one LLM call per claim
  req = server.requests[0]
  rf = req["response_format"]
  assert rf["type"] == "json_schema"                  # llama-server turns this into a grammar
  props = rf["json_schema"]["schema"]["properties"]
  assert {"veto", "veto_reason", "explanation", "llm_confidence"} <= set(props)
  assert req.get("temperature", 0) == 0
  user = " ".join(m["content"] for m in req["messages"] if m["role"] == "user")
  assert CLAIM in user and "[Tokyo:0]" in user and "TRUE" in user   # claim, cited evidence, verdict


# ---- well-formed output ----
@needs_aggregate
@pytest.mark.parametrize("label", ["TRUE", "FALSE"])
def test_json_veto_is_parsed_and_applied(llm_from, label):
  llm, _ = llm_from(completion(veto_json(veto=True, conf=0.95)))
  a = aggregate_verdict(CLAIM, verdict(label, top=0.55), llm)
  assert a.aggregator_error is None
  assert a.label == NEI and a.verifier_label == label and a.vetoed and a.veto_requested
  assert a.llm_confidence == pytest.approx(0.95)
  assert a.veto_reason == "evidence is about a different city"
  assert "[Kyoto:0]" in a.explanation


@needs_aggregate
def test_json_veto_ignored_by_gate_stays_visible(llm_from):
  # verifier sure (0.8 >= 0.6) and LLM not sure (0.5 < 0.9): the veto is requested but not applied
  llm, _ = llm_from(completion(veto_json(veto=True, conf=0.5)))
  a = aggregate_verdict(CLAIM, verdict("TRUE", top=0.8), llm)
  assert a.aggregator_error is None
  assert a.label == "TRUE" and a.veto_requested and not a.vetoed


@needs_aggregate
def test_json_no_veto_keeps_label(llm_from):
  llm, _ = llm_from(completion(veto_json(veto=False, conf=0.05, reason="")))
  a = aggregate_verdict(CLAIM, verdict("FALSE", top=0.55), llm)
  assert a.aggregator_error is None
  assert a.label == "FALSE" and not a.vetoed and not a.veto_requested


# ---- bad output: every case keeps the verifier's verdict and sets aggregator_error ----
BAD_OUTPUTS = {
  "truncated_json": completion('{"veto": true, "veto_reason": "off-top', finish_reason="length"),
  "not_json": completion("The verdict looks wrong to me."),
  "empty": completion(""),
  "confidence_above_1": completion(veto_json(conf=1.5)),
  "confidence_below_0": completion(veto_json(conf=-0.2)),
  "missing_field": completion(json.dumps({"veto": True, "veto_reason": "x", "explanation": "y"})),
  "wrong_type": completion(json.dumps({"veto": "maybe", "veto_reason": "", "explanation": "", "llm_confidence": 0.9})),
  "refusal": completion(None, refusal="I can't help with that."),
  "http_500": (500, {"error": {"message": "llama-server crashed", "type": "server_error"}}),
  "http_401": (401, {"error": {"message": "invalid api key", "type": "invalid_request_error"}}),
}


@pytest.mark.parametrize("name", list(BAD_OUTPUTS))
@pytest.mark.parametrize("label", ["TRUE", "FALSE"])
def test_bad_output_fails_open(llm_from, name, label):
  # runs even before aggregate() is written: a stub raising NotImplementedError must fail open too
  llm, _ = llm_from(BAD_OUTPUTS[name])
  a = aggregate_verdict(CLAIM, verdict(label, top=0.55), llm)
  assert a.label == label and a.verifier_label == label
  assert not a.vetoed and a.aggregator_error


# ---- through the API ----
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
def api_with(monkeypatch, llm_from):
  def make(logits, body):
    llm, server = llm_from(body)
    monkeypatch.setattr(deps, "load_services", lambda cfg: Services(StubVerifier(logits), StubRetriever(), CORPUS))
    monkeypatch.setattr(deps, "load_aggregator", lambda cfg: llm)
    return TestClient(app), server
  return make


@needs_aggregate
def test_api_returns_llm_fields(api_with):
  # logits [1, 0, 0] -> TRUE at 0.576: above the pipeline's 0.5 NEI threshold, below the veto's 0.6
  client, _ = api_with([1.0, 0.0, 0.0], completion(veto_json(veto=True, conf=0.95)))
  with client:
    r = client.post("/fact-check", json={"claim": CLAIM})
  assert r.status_code == 200
  body = r.json()
  assert body["prediction"] == NEI and body["verifier_prediction"] == "TRUE"
  assert body["vetoed"] is True and body["veto_requested"] is True
  assert body["llm_confidence"] == pytest.approx(0.95)
  assert body["veto_reason"] and "[Kyoto:0]" in body["explanation"]
  assert body["aggregator_error"] is None


@pytest.mark.parametrize("name", ["not_json", "http_500"])
def test_api_bad_llm_output_fails_open(api_with, name):
  client, _ = api_with([5.0, 0.0, 0.0], BAD_OUTPUTS[name])
  with client:
    r = client.post("/fact-check", json={"claim": CLAIM})
  assert r.status_code == 200
  body = r.json()
  assert body["prediction"] == "TRUE" and body["vetoed"] is False
  assert body["aggregator_error"]
  assert "llama-server crashed" not in json.dumps(body)   # provider error text is not leaked
