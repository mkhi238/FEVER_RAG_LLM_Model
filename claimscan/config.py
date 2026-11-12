"""Single config root for ClaimScan.

Values come from ``configs/default.yaml`` (or the file named by ``CLAIMSCAN_CONFIG``).
``CLAIMSCAN_ARTIFACTS`` overrides ``artifacts_root``. Importing this module never
creates directories; scripts that write output create their own.

The attribute names match the old ``rag/Verifier/config.py`` ``VerifierConfig`` so
existing call sites (``config.corpus``, ``config.verifier_output_dir`` ...) keep working.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILE = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class ClaimScanConfig:
  artifacts_root: Path
  hf_home: Path | None

  # data
  data_processed: Path
  claims_file: Path
  gold_file: Path
  corpus: Path
  evidence: Path

  # retriever
  retriever_index_dir: Path
  retriever_model: str
  retriever_index_file: str
  retriever_docids_file: str
  retriever_nprobe: int

  # verifier
  verifier_model: str
  verifier_output_dir: Path
  verifier_training_data: Path
  batch_size: int
  learning_rate: float
  num_epochs: int
  max_length: int
  warmup_steps: int
  negative_samples_per_claim: int
  max_evidence_sentences: int
  eval_batch_size: int
  num_labels: int

  # tracking
  mlflow_experiment_name: str
  mlflow_tracking_uri: str

  #YES, NO, NEI
  label_mapping: dict = field(default_factory=lambda: {
    "TRUE": 0,
    "FALSE": 1,
    "NOT ENOUGH INFO": 2,
  })

  # name -> {"model": str, "dir": Path, "dim": int}; see dense_indexes in default.yaml
  dense_indexes: dict = field(default_factory=dict)
  default_dense_index: str = "bge_small"

  @property
  def verifier_checkpoint(self) -> Path:
    return self.verifier_output_dir / "trained_model" / "model.pt"

  def dense_index(self, name: str | None = None) -> dict:
    name = name or self.default_dense_index
    if name not in self.dense_indexes:
      raise KeyError(f"Unknown dense index {name!r}; choose from {sorted(self.dense_indexes)}")
    return self.dense_indexes[name]


def load_config(path: str | os.PathLike | None = None) -> ClaimScanConfig:
  path = Path(path or os.environ.get("CLAIMSCAN_CONFIG", DEFAULT_CONFIG_FILE))
  with open(path, "r", encoding="utf-8") as f:
    raw = yaml.safe_load(f)

  root = Path(os.environ.get("CLAIMSCAN_ARTIFACTS", raw["artifacts_root"])).expanduser()
  p = {k: root / v for k, v in raw["paths"].items()}
  r, v = raw["retriever"], raw["verifier"]
  hf_home = raw.get("hf_home")

  cfg = ClaimScanConfig(
    artifacts_root=root,
    hf_home=Path(hf_home) if hf_home else None,
    data_processed=p["data_processed"],
    claims_file=p["claims_file"],
    gold_file=p["gold_file"],
    corpus=p["corpus"],
    evidence=p["evidence"],
    retriever_index_dir=p["retriever_index_dir"],
    retriever_model=r["model"],
    retriever_index_file=r["index_file"],
    retriever_docids_file=r["docids_file"],
    retriever_nprobe=int(r["nprobe"]),
    verifier_model=v["model"],
    verifier_output_dir=p["verifier_output_dir"],
    verifier_training_data=p["verifier_training_data"],
    batch_size=int(v["batch_size"]),
    learning_rate=float(v["learning_rate"]),
    num_epochs=int(v["num_epochs"]),
    max_length=int(v["max_length"]),
    warmup_steps=int(v["warmup_steps"]),
    negative_samples_per_claim=int(v["negative_samples_per_claim"]),
    max_evidence_sentences=int(v["max_evidence_sentences"]),
    eval_batch_size=int(v["eval_batch_size"]),
    num_labels=int(v["num_labels"]),
    mlflow_experiment_name=raw["mlflow_experiment_name"],
    mlflow_tracking_uri=str(p["mlflow_tracking_uri"]),
    dense_indexes={
      name: {"model": d["model"], "dir": root / d["dir"], "dim": int(d["dim"])}
      for name, d in raw.get("dense_indexes", {}).items()
    },
    default_dense_index=raw.get("default_dense_index", "bge_small"),
  )

  # Must happen before transformers / sentence_transformers are imported.
  if cfg.hf_home is not None and "HF_HOME" not in os.environ and cfg.hf_home.is_dir():
    os.environ["HF_HOME"] = str(cfg.hf_home)

  return cfg


config = load_config()


def validate_config(cfg: ClaimScanConfig = config):
  required_files = [
    cfg.claims_file,
    cfg.gold_file,
    cfg.corpus,
    cfg.evidence,
  ]

  required_dirs = [
    cfg.retriever_index_dir
  ]

  missing_files = [f for f in required_files if not f.exists()]
  missing_dirs = [d for d in required_dirs if not d.exists()]

  if missing_files:
      raise FileNotFoundError(f"Missing required files: {missing_files}")

  if missing_dirs:
      raise FileNotFoundError(f"Missing required directories: {missing_dirs}")

  else:
     print("Configuration Complete")


if __name__ == "__main__":
    validate_config()
