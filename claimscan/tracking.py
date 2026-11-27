"""MLflow setup shared by every script that logs runs.

MLflow 3.x refuses the old ``./mlruns`` file store, so runs go to a SQLite database under
artifacts_root (``paths.mlflow_tracking_uri``) with artifacts next to it
(``paths.mlflow_artifact_dir``). ``MLFLOW_TRACKING_URI`` in the environment still wins.

Browse runs:  mlflow ui --backend-store-uri sqlite:////mnt/d/crisis-claim-analysis/artifacts/mlflow/mlflow.db
"""
from __future__ import annotations

import os
from pathlib import Path

from claimscan.config import ClaimScanConfig, config


def tracking_uri(cfg: ClaimScanConfig = config) -> str:
  """The MLflow tracking URI: $MLFLOW_TRACKING_URI, else sqlite:///<artifacts_root>/mlflow/mlflow.db."""
  if os.environ.get("MLFLOW_TRACKING_URI"):
    return os.environ["MLFLOW_TRACKING_URI"]
  uri = str(cfg.mlflow_tracking_uri)
  if "://" in uri:
    return uri
  path = Path(uri)
  path.parent.mkdir(parents=True, exist_ok=True)
  return f"sqlite:///{path.resolve()}"


def setup_mlflow(experiment: str | None = None, cfg: ClaimScanConfig = config,
                 uri: str | None = None, artifact_dir: str | os.PathLike | None = None):
  """Point MLflow at the project store and select (creating if needed) ``experiment``.

  Returns the mlflow module, so callers can do ``mlflow = setup_mlflow("retriever_eval")``.
  """
  os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
  import mlflow

  mlflow.set_tracking_uri(uri or tracking_uri(cfg))
  name = experiment or cfg.mlflow_experiment_name
  if mlflow.get_experiment_by_name(name) is None:
    art = Path(artifact_dir or cfg.mlflow_artifact_dir) / name
    art.mkdir(parents=True, exist_ok=True)
    mlflow.create_experiment(name, artifact_location=art.resolve().as_uri())
  mlflow.set_experiment(name)
  return mlflow
