import json

import httpx
import pandas as pd
import pytest

from claimscan.config import config, load_config
from claimscan.rerank import load_reranker
from claimscan.rerank.base import Reranker, rerank_hits, sort_scores
from claimscan.rerank.cohere import CohereReranker
from claimscan.rerank.cross_encoder import CrossEncoderReranker
from claimscan.rerank.openai_llm import OpenAIReranker, parse_ranking, parse_score

DOCS = ["alpha", "beta", "gamma"]


def test_sort_scores():
  assert sort_scores([0.1, 0.9, 0.5]) == [(1, 0.9), (2, 0.5), (0, 0.1)]
  assert sort_scores([1, 1, 2], top_n=2) == [(2, 2.0), (0, 1.0)]


def test_config_default_is_cohere():
  assert config.rerank_backend == "cohere"
  assert isinstance(load_reranker(config), CohereReranker)   # no network until rerank() is called
  local = load_reranker(config, backend="cross_encoder")   # lazy: the model is only downloaded/loaded on first rerank
  assert isinstance(local, CrossEncoderReranker) and local.model_name == "BAAI/bge-reranker-v2-m3"
  assert load_reranker(config, backend="none") is None
  gpt = load_reranker(config, backend="openai")   # reasoning-model request settings for gpt-5-nano
  assert gpt.temperature is None and gpt.max_tokens_field == "max_completion_tokens"


def test_old_config_without_rerank_section(tmp_path):
  import yaml
  from claimscan.config import DEFAULT_CONFIG_FILE
  raw = yaml.safe_load(open(DEFAULT_CONFIG_FILE))
  raw.pop("rerank")
  f = tmp_path / "old.yaml"
  f.write_text(yaml.safe_dump(raw))
  assert load_config(f).rerank_backend == "none"


def test_load_reranker_backends():
  assert isinstance(load_reranker(backend="cohere"), CohereReranker)
  assert isinstance(load_reranker(backend="openai"), OpenAIReranker)
  assert isinstance(load_reranker(backend="cross_encoder"), CrossEncoderReranker)  # lazy: no download
  with pytest.raises(ValueError):
    load_reranker(backend="bogus")


def test_cohere_request_and_parsing(monkeypatch):
  monkeypatch.setenv("COHERE_API_KEY", "test-key")
  seen = {}

  def handler(request):
    seen["auth"] = request.headers["authorization"]
    seen["url"] = str(request.url)
    seen["body"] = json.loads(request.content)
    return httpx.Response(200, json={"results": [
      {"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4}]})

  rr = CohereReranker("rerank-v3.5", transport=httpx.MockTransport(handler))
  out = rr.rerank("q", DOCS, top_n=2)
  assert out == [(2, 0.9), (0, 0.4)]
  assert seen["auth"] == "Bearer test-key"
  assert seen["url"] == "https://api.cohere.com/v2/rerank"
  assert seen["body"] == {"model": "rerank-v3.5", "query": "q", "documents": DOCS, "top_n": 2}


def test_cohere_missing_key(monkeypatch):
  monkeypatch.delenv("COHERE_API_KEY", raising=False)
  with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
    CohereReranker(transport=httpx.MockTransport(lambda r: httpx.Response(200))).rerank("q", DOCS)


def test_cohere_retries_then_errors(monkeypatch):
  monkeypatch.setenv("COHERE_API_KEY", "k")
  calls = []

  def handler(request):
    calls.append(1)
    return httpx.Response(429, text="slow down")

  rr = CohereReranker(transport=httpx.MockTransport(handler), max_retries=2, sleep=lambda s: None)
  with pytest.raises(RuntimeError, match="429"):
    rr.rerank("q", DOCS)
  assert len(calls) == 3


def _chat_reply(text):
  return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_openai_pointwise(monkeypatch):
  monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
  seen = []

  def handler(request):
    body = json.loads(request.content)
    seen.append((request.headers["authorization"], body))
    sentence = body["messages"][1]["content"]
    return _chat_reply({"alpha": "2", "beta": "9", "gamma": "Score: 5"}[
      next(d for d in DOCS if f"Sentence: {d}" in sentence)])

  rr = OpenAIReranker("m", transport=httpx.MockTransport(handler))
  assert rr.rerank("claim", DOCS) == [(1, 9.0), (2, 5.0), (0, 2.0)]
  assert all(a == "Bearer sk-test" for a, _ in seen)
  assert seen[0][1]["model"] == "m" and seen[0][1]["temperature"] == 0.0
  assert rr.parse_failures == 0


def test_openai_pointwise_malformed(monkeypatch):
  monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
  replies = iter(["I cannot say", "", "42"])
  rr = OpenAIReranker(batch_size=1, transport=httpx.MockTransport(lambda r: _chat_reply(next(replies))))
  out = rr.rerank("claim", DOCS)
  assert out == [(2, 10.0), (0, 0.0), (1, 0.0)]  # 42 clamped to 10; junk -> 0
  assert rr.parse_failures == 2


def test_openai_listwise_with_garbage(monkeypatch):
  monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
  rr = OpenAIReranker(mode="listwise",
                      transport=httpx.MockTransport(lambda r: _chat_reply("[3] > [9] > [3] > [1]")))
  out = rr.rerank("claim", DOCS)
  assert [i for i, _ in out] == [2, 0, 1]  # missing [2] appended; 9 and repeat dropped
  assert rr.parse_failures == 1


def test_openai_missing_key_and_local_server(monkeypatch):
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)
  with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
    OpenAIReranker(transport=httpx.MockTransport(lambda r: _chat_reply("1"))).rerank("q", DOCS)
  local = OpenAIReranker(base_url="http://localhost:8080/v1", batch_size=1,
                         transport=httpx.MockTransport(lambda r: _chat_reply("7")))
  assert local.rerank("q", ["a"]) == [(0, 7.0)]


def test_parsers():
  assert parse_score("Score: 7/10") == 7.0
  assert parse_score(None) is None
  assert parse_ranking("[2] > [1]", 2) == [1, 0]


def test_cross_encoder_stub():
  class Stub:
    def predict(self, pairs, batch_size, show_progress_bar):
      return [len(d) for _, d in pairs]

  rr = CrossEncoderReranker(model_obj=Stub())
  assert rr.rerank("q", ["a", "ccc", "bb"], top_n=2) == [(1, 3.0), (2, 2.0)]


class StubReranker(Reranker):
  def rerank(self, query, docs, top_n=None):
    return sort_scores([len(d) for d in docs], top_n)


def test_rerank_hits_keeps_score_and_orders():
  corpus = pd.DataFrame({"id": ["A", "A", "B"], "line": [0, 1, 0],
                         "text": ["short", "a much longer sentence", "mid size"]})
  corpus = corpus.set_index(["id", "line"]).sort_index()
  hits = [{"id": "A", "line": 0, "score": 0.1}, {"id": "B", "line": 0, "score": 0.2},
          {"id": "A", "line": 1, "score": 0.3}, {"id": "Z", "line": 9, "score": 0.4}]
  out = rerank_hits(StubReranker(), "claim", hits, corpus)
  assert [(h["id"], h["line"]) for h in out] == [("A", 1), ("B", 0), ("A", 0), ("Z", 9)]
  assert [h["score"] for h in out[:3]] == [0.3, 0.2, 0.1]
  assert out[-1]["rerank_score"] is None
  assert "rerank_score" not in hits[0]  # input untouched
  top = rerank_hits(StubReranker(), "claim", hits, corpus, top_n=2)
  assert len(top) == 2 and top[0]["rerank_score"] == len("a much longer sentence")


def test_cohere_waits_between_calls(monkeypatch):
  monkeypatch.setenv("COHERE_API_KEY", "k")
  waits = []
  rr = CohereReranker(sleep=waits.append, transport=httpx.MockTransport(
    lambda r: httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})))
  rr.rerank("q", DOCS)
  rr.rerank("q", DOCS)
  assert len(waits) == 1 and 5.0 < waits[0] <= 6.0   # second call waits out the 6 s trial-key gap


class FailingReranker(Reranker):
  def rerank(self, query, docs, top_n=None):
    raise RuntimeError("COHERE_API_KEY is not set")


@pytest.mark.parametrize("reranker, order", [
  (StubReranker(), ["A#1", "B#0"]),        # reranked: longest sentence first, top k=2
  (FailingReranker(), ["A#0", "B#0"]),     # fails open: retriever order, top k=2
  (None, ["A#0", "B#0"]),                  # no reranker: retriever's k hits
])
def test_pipeline_uses_reranker(monkeypatch, reranker, order):
  from claimscan import pipeline
  corpus = pd.DataFrame({"id": ["A", "A", "B"], "line": [0, 1, 0],
                         "text": ["short", "a much longer sentence", "mid size"]}).set_index(["id", "line"]).sort_index()
  hits = [{"id": "A", "line": 0, "score": 0.1}, {"id": "B", "line": 0, "score": 0.2},
          {"id": "A", "line": 1, "score": 0.3}]
  asked = []

  class Retriever:
    def search(self, q, k):
      asked.append(k)
      return hits[:k]

  monkeypatch.setattr(pipeline, "get_evidence", lambda *a, **kw: [])   # stop before the verifier
  services = pipeline.Services(model=None, retriever=Retriever(), corpus_df=corpus, reranker=reranker)
  v = pipeline.fact_check_detailed("claim", services, k=2)
  assert [h["docid"] for h in v.hits] == order
  assert asked == [config.rerank_candidates if reranker else 2]   # the reranker gets all candidates
