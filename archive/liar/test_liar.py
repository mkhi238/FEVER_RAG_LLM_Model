"""LIAR loader, label mapping and eval helpers on tiny synthetic TSVs (no downloads, no artifacts)."""
import pytest

from claimscan.data import liar
from claimscan.eval.liar_eval import PrecomputedRetriever, label_metrics, shared_content_words, split_sentences

ROWS = [
  ("1.json", "true", "Texas has the most jobs.", "jobs", "rick-perry", "Governor", "Texas", "republican",
   "1", "2", "3", "4", "5", "a speech"),
  ("2.json", "pants-fire", 'Says "Obama" was born in Kenya.', "obama", "chain-email", "", "", "none",
   "0", "0", "0", "0", "9", "an email"),
  ("3.json", "half-true", "Taxes went up 50 percent.", "taxes", "x", "", "", "democrat",
   "0", "0", "0", "0", "0", "an ad"),
  ("4.json", "mostly-true", "Coal jobs fell.", "energy", "y", "", "", "democrat",
   "0", "0", "0", "0", "0", "a debate"),
  ("5.json", "barely-true", "Crime doubled.", "crime", "z", "", "", "republican",
   "0", "0", "0", "0", "0", "a tweet"),
]


@pytest.fixture()
def data_dir(tmp_path):
  (tmp_path / "raw").mkdir()
  (tmp_path / "liar_plus").mkdir()
  body = "\n".join("\t".join(r) for r in ROWS) + "\n"
  plus = "\n".join("\t".join((str(i),) + r + ("Some justification. It says things.",)) for i, r in enumerate(ROWS[:3])) + "\n"
  for split in liar.SPLIT_FILES:
    (tmp_path / "raw" / liar.SPLIT_FILES[split]).write_text(body, encoding="utf-8")
    (tmp_path / "liar_plus" / liar.PLUS_FILES[split]).write_text(plus, encoding="utf-8")
  return tmp_path


def test_read_tsv_keeps_quotes_and_counts(data_dir):
  df = liar.read_tsv(data_dir / "raw" / "test.tsv")
  assert len(df) == 5
  assert df.loc[1, "claim"] == 'Says "Obama" was born in Kenya.'
  assert df.loc[1, "pants_fire_count"] == 9


@pytest.mark.parametrize("preset,expected", [
  ("binary", {"1.json": "TRUE", "2.json": "FALSE", "4.json": "TRUE"}),
  ("binary_strict", {"1.json": "TRUE", "2.json": "FALSE"}),
  ("three_way", {"1.json": "TRUE", "2.json": "FALSE", "3.json": "NOT ENOUGH INFO",
                 "4.json": "TRUE", "5.json": "NOT ENOUGH INFO"}),
])
def test_label_map_presets(data_dir, preset, expected):
  df = liar.load_liar("test", preset, data_dir=data_dir)
  assert dict(zip(df["json_id"], df["label_text"])) == expected


def test_custom_label_map_and_validation(data_dir):
  df = liar.load_liar("test", {"half-true": "NOT ENOUGH INFO"}, data_dir=data_dir)
  assert df["json_id"].tolist() == ["3.json"]
  with pytest.raises(ValueError):
    liar.resolve_label_map({"half-true": "MAYBE"})
  with pytest.raises(KeyError):
    liar.resolve_label_map("nope")


def test_justification_join(data_dir):
  df = liar.load_liar("validation", "three_way", with_justification=True, data_dir=data_dir)
  just = dict(zip(df["json_id"], df["justification"]))
  assert just["1.json"].startswith("Some justification") and just["4.json"] == ""


def test_agent_claim_pool_keeps_all_labels_dedups(data_dir):
  pool = liar.load_claims_for_agent(data_dir=data_dir)
  assert len(pool) == 5 and set(pool["liar_label"]) == set(r[1] for r in ROWS)


def test_label_metrics_counts_nei_as_wrong():
  m = label_metrics(["TRUE", "FALSE", "TRUE", "FALSE"], ["TRUE", "NOT ENOUGH INFO", "FALSE", "FALSE"])
  assert m["accuracy"] == 0.5 and m["coverage"] == 0.75
  assert m["decided_accuracy"] == pytest.approx(2 / 3)
  assert m["macro_f1"] == pytest.approx((2 / 3 + 1 / 2) / 2)  # TRUE F1 2/3, FALSE F1 1/2


def test_helpers():
  assert shared_content_words("Barack Obama was born in Kenya", "He was born in Honolulu.", "Barack_Obama") == 3
  assert split_sentences("One two three. Four five six! ok.") == ["One two three.", "Four five six! ok."]
  r = PrecomputedRetriever({"c": [{"id": "A", "line": 0, "score": 1.0}] * 5})
  assert len(r.search("c", k=2)) == 2


def test_corpus_for_pages_reads_only_hit_pages(tmp_path, monkeypatch):
  import pandas as pd
  from claimscan.config import config
  from claimscan.eval import liar_eval
  path = tmp_path / "corpus.parquet"
  pd.DataFrame({"id": ["A", "A", "B", "C"], "line": [0, 1, 0, 0], "text": ["a0", "a1", "b0", "c0"]}).to_parquet(path)
  monkeypatch.setattr(config, "corpus", path)
  got = liar_eval._corpus_for_pages({"A", "C"})
  assert sorted(got.index.tolist()) == [("A", 0), ("A", 1), ("C", 0)]
  assert got.loc[("C", 0)]["text"] == "c0"
