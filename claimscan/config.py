"""Single config root for ClaimScan.

Values come from ``configs/default.yaml`` (or the file named by ``CLAIMSCAN_CONFIG``).
``CLAIMSCAN_ARTIFACTS`` overrides ``artifacts_root`` and ``CLAIMSCAN_RETRIEVER`` overrides
``retriever.kind``. ``CLAIMSCAN_AGGREGATOR_ENABLED`` / ``_URL`` / ``_MODEL`` / ``_TIMEOUT`` / ``_API_KEY_ENV`` override
``aggregator.*``. Importing this module never creates directories; scripts that write output create their own.

The attribute names match the old ``rag/Verifier/config.py`` ``VerifierConfig`` so
existing call sites (``config.verifier_output_dir`` ...) keep working.
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

  # data (claims, gold evidence and sentences are all in sentence_store, see claimscan/data/fever_db.py)
  data_processed: Path

  # retriever
  retriever_index_dir: Path
  retriever_model: str
  retriever_index_file: str
  retriever_docids_file: str
  retriever_nprobe: int
  retriever_mmap: bool
  retriever_kind: str

  # sparse + hybrid retrieval
  bm25_dir: Path
  bm25_k1: float
  bm25_b: float
  bm25_rm3: bool
  hybrid_rrf_k: int
  hybrid_candidates: int

  # verifier
  verifier_output_dir: Path
  verifier_training_data: Path
  max_length: int

  # tracking
  mlflow_experiment_name: str
  mlflow_tracking_uri: str
  mlflow_artifact_dir: Path
  mlflow_retriever_experiment: str
  eval_dir: Path

  #YES, NO, NEI
  label_mapping: dict = field(default_factory=lambda: {
    "TRUE": 0,
    "FALSE": 1,
    "NOT ENOUGH INFO": 2,
  })

  # name -> {"model": str, "dir": Path, "dim": int}; see dense_indexes in default.yaml
  dense_indexes: dict = field(default_factory=dict)
  default_dense_index: str = "bge_small"

  # reranker (backend "none" = not used); see rerank: in default.yaml
  rerank_backend: str = "none"
  rerank_top_n: int = 10
  rerank_candidates: int = 100
  rerank_batch_size: int = 32
  rerank_device: str = "cpu"
  rerank_cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
  rerank_cohere_model: str = "rerank-v3.5"
  rerank_cohere_base_url: str = "https://api.cohere.com"
  rerank_openai_model: str = "gpt-5-nano"
  rerank_openai_base_url: str = "https://api.openai.com/v1"
  rerank_openai_mode: str = "pointwise"

  # rear aggregator (LLM veto over the verifier's verdict); see aggregator: in default.yaml
  aggregator_enabled: bool = False
  aggregator_base_url: str | None = None   # None = the aggregator port in llamacpp/roles.yaml; set by CLAIMSCAN_AGGREGATOR_URL
  aggregator_model: str = "gemma"
  aggregator_timeout: float = 30.0
  aggregator_api_key_env: str = ""   # NAME of the env var holding the key; "" = local llama-server

  # verifier: a HuggingFace NLI model name (claimscan/verifier/hf_nli.py)
  verifier_inference_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
  # TRUE/FALSE -> NEI when the verifier's P(NEI) >= this (None = off)
  verifier_nei_prob_threshold: float | None = None

  @property
  def bm25_index_dir(self) -> Path:
    return self.bm25_dir / "index"

  @property
  def bm25_jsonl_dir(self) -> Path:
    return self.bm25_dir / "jsonl"

  @property
  def sentence_store(self) -> Path:
    """The one FEVER SQLite file: sentences, claims, evidence (claimscan/data/fever_db.py builds it)."""
    return self.data_processed / "fever_sentences.sqlite"

  def dense_index(self, name: str | None = None) -> dict:
    name = name or self.default_dense_index
    if name not in self.dense_indexes:
      raise KeyError(f"Unknown dense index {name!r}; choose from {sorted(self.dense_indexes)}")
    return self.dense_indexes[name]


def _env_bool(name: str, default: bool) -> bool:
  v = os.environ.get(name)
  return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def load_config(path: str | os.PathLike | None = None) -> ClaimScanConfig:
  path = Path(path or os.environ.get("CLAIMSCAN_CONFIG", DEFAULT_CONFIG_FILE))
  with open(path, "r", encoding="utf-8") as f:
    raw = yaml.safe_load(f)

  root = Path(os.environ.get("CLAIMSCAN_ARTIFACTS", raw["artifacts_root"])).expanduser()
  p = {k: root / v for k, v in raw["paths"].items()}
  r, v = raw["retriever"], raw["verifier"]
  bm25, hybrid = raw.get("bm25") or {}, raw.get("hybrid") or {}
  rr = raw.get("rerank") or {}
  ag = raw.get("aggregator") or {}
  hf_home = raw.get("hf_home")

  cfg = ClaimScanConfig(
    artifacts_root=root,
    hf_home=Path(hf_home) if hf_home else None,
    data_processed=p["data_processed"],
    retriever_index_dir=p["retriever_index_dir"],
    retriever_model=r["model"],
    retriever_index_file=r["index_file"],
    retriever_docids_file=r["docids_file"],
    retriever_nprobe=int(r["nprobe"]),
    retriever_mmap=bool(r.get("mmap", False)),
    retriever_kind=str(os.environ.get("CLAIMSCAN_RETRIEVER", r.get("kind", "faiss"))),
    bm25_dir=root / bm25.get("dir", "sparse/bm25_sentences"),
    bm25_k1=float(bm25.get("k1", 1.2)),
    bm25_b=float(bm25.get("b", 0.4)),
    bm25_rm3=bool(bm25.get("rm3", True)),
    hybrid_rrf_k=int(hybrid.get("rrf_k", 60)),
    hybrid_candidates=int(hybrid.get("candidates", 100)),
    verifier_output_dir=p["verifier_output_dir"],
    verifier_training_data=p["verifier_training_data"],
    max_length=int(v["max_length"]),
    mlflow_experiment_name=raw["mlflow_experiment_name"],
    mlflow_tracking_uri=str(p["mlflow_tracking_uri"]),
    mlflow_artifact_dir=p.get("mlflow_artifact_dir", root / "mlflow/artifacts"),
    mlflow_retriever_experiment=raw.get("mlflow_retriever_experiment", "retriever_eval"),
    eval_dir=p.get("eval_dir", root / "eval"),
    dense_indexes={
      name: {"model": d["model"], "dir": root / d["dir"], "dim": int(d["dim"])}
      for name, d in raw.get("dense_indexes", {}).items()
    },
    default_dense_index=raw.get("default_dense_index", "bge_small"),
    rerank_backend=str(rr.get("backend", "none")),
    rerank_top_n=int(rr.get("top_n", 10)),
    rerank_candidates=int(rr.get("candidates", 100)),
    rerank_batch_size=int(rr.get("batch_size", 32)),
    rerank_device=str(rr.get("device", "cpu")),
    rerank_cross_encoder_model=str(rr.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
    rerank_cohere_model=str(rr.get("cohere_model", "rerank-v3.5")),
    rerank_cohere_base_url=str(rr.get("cohere_base_url", "https://api.cohere.com")),
    rerank_openai_model=str(rr.get("openai_model", "gpt-5-nano")),
    rerank_openai_base_url=str(rr.get("openai_base_url", "https://api.openai.com/v1")),
    rerank_openai_mode=str(rr.get("openai_mode", "pointwise")),
    aggregator_enabled=_env_bool("CLAIMSCAN_AGGREGATOR_ENABLED", bool(ag.get("enabled", False))),
    aggregator_base_url=os.environ.get("CLAIMSCAN_AGGREGATOR_URL") or None,
    aggregator_model=str(os.environ.get("CLAIMSCAN_AGGREGATOR_MODEL", ag.get("model", "gemma"))),
    aggregator_timeout=float(os.environ.get("CLAIMSCAN_AGGREGATOR_TIMEOUT", ag.get("timeout", 30.0))),
    aggregator_api_key_env=str(os.environ.get("CLAIMSCAN_AGGREGATOR_API_KEY_ENV", ag.get("api_key_env") or "")),
    verifier_inference_model=str(v.get("inference_model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")),
    verifier_nei_prob_threshold=(None if v.get("nei_prob_threshold") is None else float(v["nei_prob_threshold"])),
  )

  # Must happen before transformers / sentence_transformers are imported.
  if cfg.hf_home is not None and "HF_HOME" not in os.environ and cfg.hf_home.is_dir():
    os.environ["HF_HOME"] = str(cfg.hf_home)

  return cfg


config = load_config()


def validate_config(cfg: ClaimScanConfig = config):
  required_files = [
    cfg.sentence_store,
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
