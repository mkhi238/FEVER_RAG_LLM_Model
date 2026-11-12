from pathlib import Path

import pytest

from claimscan.config import config, load_config, validate_config


def test_paths_are_absolute_and_linux():
  for name in ["claims_file", "gold_file", "corpus", "evidence", "retriever_index_dir", "verifier_output_dir"]:
    p = getattr(config, name)
    assert isinstance(p, Path) and p.is_absolute(), name
    assert "\\" not in str(p), f"{name} still has a Windows path: {p}"


def test_env_override(monkeypatch, tmp_path):
  monkeypatch.setenv("CLAIMSCAN_ARTIFACTS", str(tmp_path))
  cfg = load_config()
  assert cfg.corpus == tmp_path / "data/processed/fever_corpus_parquet"
  assert not (tmp_path / "verifier").exists(), "loading config must not create directories"


def test_label_mapping_matches_verifier_head():
  assert config.label_mapping == {"TRUE": 0, "FALSE": 1, "NOT ENOUGH INFO": 2}
  assert config.num_labels == 3


@pytest.mark.skipif(not config.artifacts_root.exists(), reason="artifacts drive not mounted")
def test_artifacts_present():
  validate_config()
  assert config.verifier_checkpoint.is_file()
  assert (config.retriever_index_dir / config.retriever_index_file).is_file()
  assert config.verifier_training_data.is_file()
