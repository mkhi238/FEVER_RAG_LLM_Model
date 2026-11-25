"""posttrain_store: rows written to sft_examples from stubbed downloads, and the TRL rows load_sft_dataset builds."""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
import torch

from claimscan.data import posttrain_store as P

VITAMINC = pd.DataFrame({"unique_id": ["a", "b"], "claim": ["tokyo is in japan", "tokyo is in france"],
                         "evidence": ["tokyo is the capital of japan"] * 2, "page": ["Tokyo"] * 2,
                         "label": ["SUPPORTS", "REFUTES"]})
SF = pd.DataFrame({"claim": ["a claim"], "evidence": [np.array([["A", "first"], ["B", "second"]], dtype=object)],
                   "label_before": ["SUPPORTS"], "label_after": ["NOT ENOUGH"]})
LIAR = pd.DataFrame({"id": [7], "statement": ["taxes went up"], "speaker": ["someone"],
                     "justification": ["they went down"], "label": [0]})


def fake_hf(repo, file):
  return {"tals/vitaminc": VITAMINC, "copenlu/sufficient_facts": SF, "chengxuphd/liar2": LIAR}[repo].copy()


class Verifier:
  """Always says TRUE with 0.9."""
  def encode_text(self, claim, evidence):
    return {"input_ids": torch.zeros(1, 4, dtype=torch.long), "attention_mask": torch.ones(1, 4, dtype=torch.long)}

  def __call__(self, input_ids, attention_mask):
    return torch.log(torch.tensor([[0.9, 0.05, 0.05]]))


@pytest.fixture
def con(tmp_path, monkeypatch):
  monkeypatch.setattr(P, "DB", tmp_path / "p.sqlite")
  monkeypatch.setattr(P, "_hf", fake_hf)
  monkeypatch.setattr("claimscan.pipeline.load_verifier", lambda *a, **k: Verifier())
  c = P._connect()
  yield c
  c.close()


def rows(con, source):
  return pd.read_sql_query("SELECT * FROM sft_examples WHERE source = ?", con, params=[source])


def test_aggregator_rows_and_veto_target(con):
  P.build_verified(con, 10)
  vc = rows(con, "agg_vitaminc")
  veto = {ref: json.loads(a)["veto"] for ref, a in zip(vc.ref, vc.assistant)}
  assert veto == {"a": False, "b": True}   # the verifier says TRUE every time; b's gold is FALSE
  sf = rows(con, "agg_sf_hover").iloc[0]
  assert "Verdict: TRUE" in sf.user and "[A:0] first" in sf.user and json.loads(sf.assistant)["veto"] is True
  P.build_verified(con, 10)   # rerun replaces, doesn't duplicate
  assert len(rows(con, "agg_vitaminc")) == 2


def test_liar_label_names(con):
  P.build_liar(con, 1)
  r = rows(con, "liar").iloc[0]
  assert r.assistant == "pants-fire" and "Fact-checker summary: they went down" in r.user


def test_load_sft_dataset_rows(con):
  pytest.importorskip("datasets")
  P.build_verified(con, 10)
  con.execute("INSERT INTO sft_examples VALUES ('ultrafeedback', 'x', NULL, 'say hi', 'hi')")
  con.commit()
  ds = P.load_sft_dataset()
  assert len(ds) == 5
  by_user = {r["prompt"][-1]["content"]: r for r in ds}
  assert [m["role"] for m in by_user["say hi"]["prompt"]] == ["user"]   # no system prompt -> user only
  agg = next(r for r in ds if len(r["prompt"]) == 2)
  out = json.loads(agg["completion"][0]["content"])
  assert set(out) == {"veto", "veto_reason", "explanation", "llm_confidence"}
