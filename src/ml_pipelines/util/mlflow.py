from __future__ import annotations

import os

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
import tempfile
import pandas as pd

def begin_pipeline_run(cfg: DictConfig) -> str:
    exp = mlflow.get_experiment_by_name(cfg.pipeline.experiment_name)
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
    df = pd.read_parquet(artifact_path)
    df.attrs["source_artifact"] = artifact_rel_path
    return df


def log_input_dataset(df: pd.DataFrame, name: str) -> None:
    source_artifact = df.attrs.get("source_artifact")

    dataset_name = str(source_artifact) if source_artifact else name
    
    # Upcast only integer-like columns to float64 to avoid mlflow warnings
    int_to_float_map = {
        col: "float64"
        for col in df.columns
        if pd.api.types.is_integer_dtype(df[col])
    }
    if int_to_float_map:
        df_for_logging = df.astype(int_to_float_map, copy=False)
    else:
        df_for_logging = df

    dataset = mlflow.data.from_pandas(df_for_logging, name=dataset_name)

    mlflow.log_input(dataset)

