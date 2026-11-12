"""Endpoint wiring with stub services (the real FAISS index needs ~26 GB RAM to load)."""
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from claimscan.api import deps
from claimscan.api.main import app
from claimscan.pipeline import Services


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
  def make(logits):
    monkeypatch.setattr(deps, "load_services", lambda cfg: Services(StubVerifier(logits), StubRetriever(), CORPUS))
    return TestClient(app)
  return make


def test_health_and_true(client_with):
  with client_with([5.0, 0.0, 0.0]) as client:
    assert client.get("/health").json() == {"status": "healthy", "models_loaded": True}
    r = client.post("/fact-check", json={"claim": "Tokyo is the capital of Japan"})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "TRUE"
    assert body["response"] == "This statement is TRUE. Evidence: tokyo is the capital of japan"


def test_low_confidence_becomes_nei(client_with):
  # softmax([0.1, 0, 0]) max ~ 0.35 < 0.55 threshold
  with client_with([0.1, 0.0, 0.0]) as client:
    body = client.post("/fact-check", json={"claim": "Tokyo is the capital of Japan"}).json()
    assert body["prediction"] == "NOT ENOUGH INFO"
    assert body["response"] == "Insufficient evidence."


def test_rejects_bad_request(client_with):
  with client_with([5.0, 0.0, 0.0]) as client:
    assert client.post("/fact-check", json={"text": "no claim field"}).status_code == 422
