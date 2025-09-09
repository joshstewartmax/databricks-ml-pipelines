from __future__ import annotations

import os

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
import tempfile
import pandas as pd

def begin_pipeline_run(cfg: DictConfig) -> str:
    exp = mlflow.get_experiment_by_name(cfg.experiment.name)
    client = MlflowClient()
    run = client.create_run(
        experiment_id=exp.experiment_id,
        run_name="pipeline_run",
        tags={},
    )

    return run.info.run_id


def end_pipeline_run(parent_run_id: str, status: str = "FINISHED") -> None:
    """Mark the MLflow parent run as terminated with the given status."""
    client = MlflowClient()
    client.set_terminated(run_id=parent_run_id, status=status)


def save_dataframe_as_artifact(df: pd.DataFrame, filename: str, artifact_subdir: str) -> None:
    """Save a DataFrame to a temp parquet and log as an MLflow artifact under subdir."""
    tmpdir = tempfile.mkdtemp()
    local_path = os.path.join(tmpdir, filename)
    df.to_parquet(local_path, index=False)
    mlflow.log_artifact(local_path, artifact_path=artifact_subdir)


def load_parquet_artifact_as_df(run_id: str, artifact_rel_path: str) -> pd.DataFrame:
    """Download a parquet artifact and load it as a DataFrame."""
    client = MlflowClient()
    artifact_path = client.download_artifacts(run_id, artifact_rel_path)
    # If artifact_rel_path is a file path, download_artifact may return the file path; handle both dir/file
    if os.path.isdir(artifact_path):
        # take first parquet file in directory
        for name in os.listdir(artifact_path):
            if name.endswith(".parquet"):
                return pd.read_parquet(os.path.join(artifact_path, name))
        raise FileNotFoundError(f"No parquet file found in {artifact_path}")
    return pd.read_parquet(artifact_path)

