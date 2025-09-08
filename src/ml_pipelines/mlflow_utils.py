from __future__ import annotations

import os
import uuid
from typing import Optional, Dict

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
import tempfile
import pandas as pd

# these will be available in databricks runtime
from pyspark.sql import SparkSession  # type: ignore
from pyspark.dbutils import DBUtils  # type: ignore


def set_experiment_from_cfg(cfg: DictConfig) -> str:
    """Set the MLflow experiment using cfg.

    Prefers cfg.experiment.path (Databricks), falls back to cfg.experiment.name (local).

    Returns the experiment identifier used (path or name).
    """
    # Allow environment override for explicit path selection
    env_path = os.environ.get("MLFLOW_EXPERIMENT_PATH")
    exp_path: Optional[str] = env_path or getattr(cfg.experiment, "path", None)
    exp_identifier = exp_path or cfg.experiment.name
    mlflow.set_experiment(exp_identifier)
    return exp_identifier


def _get_dbutils():
    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    return DBUtils(spark)


def resolve_parent_run_id(cfg: DictConfig) -> Optional[str]:  # noqa: ARG001 - cfg present for uniform API
    """Resolve parent run id strictly from Databricks task values.

    Reads the value produced by the `initialize` task.
    """
    try:
        dbutils = _get_dbutils()
        return dbutils.jobs.taskValues.get(key="parent_run_id", taskKey="initialize")
    except Exception:
        return None


## Removed legacy parent run resolution in favor of resolve_parent_run_id

def get_databricks_run_identifiers() -> Dict[str, Optional[str]]:
    """Collect Databricks job/task identifiers from env and context when available."""
    identifiers: Dict[str, Optional[str]] = {
        "job_id": os.environ.get("DATABRICKS_JOB_ID"),
        "job_run_id": os.environ.get("DATABRICKS_JOB_RUN_ID"),
        "task_run_id": os.environ.get("DATABRICKS_TASK_RUN_ID"),
        "cluster_id": os.environ.get("DATABRICKS_CLUSTER_ID"),
    }
    try:
        dbutils = _get_dbutils()
        if dbutils is not None:
            # taskContext APIs may not be available in all runtimes; guard carefully
            ctx = dbutils.jobs.taskContext()
            identifiers.setdefault("job_run_id", getattr(ctx, "runId", lambda: None)())
            identifiers.setdefault("task_run_id", getattr(ctx, "taskRunId", lambda: None)())
    except Exception:
        pass
    return identifiers


def create_parent_run(cfg: DictConfig, tags: Optional[Dict[str, str]] = None) -> str:
    """Create a parent MLflow run in the configured experiment and return its run_id.

    Does not end the run. Intended for use across multiple job tasks.
    """
    exp_identifier = set_experiment_from_cfg(cfg)

    # Ensure experiment exists and get its ID
    exp = mlflow.get_experiment_by_name(exp_identifier)
    if exp is None:
        # set_experiment should have created it, but double-check
        mlflow.set_experiment(exp_identifier)
        exp = mlflow.get_experiment_by_name(exp_identifier)
    assert exp is not None, "Failed to resolve MLflow experiment"

    client = MlflowClient()
    all_tags: Dict[str, str] = {}
    if tags:
        all_tags.update(tags)
    # Provide a readable run name
    run = client.create_run(experiment_id=exp.experiment_id, run_name="pipeline_run", tags=all_tags)
    return run.info.run_id


def terminate_run(run_id: str, status: str = "FINISHED") -> None:
    """Terminate a run by ID (idempotent best-effort)."""
    try:
        client = MlflowClient()
        client.set_terminated(run_id=run_id, status=status)
    except Exception:
        # Do not raise if the run is already terminated or inaccessible
        pass


def generate_pipeline_run_id() -> str:
    return str(uuid.uuid4())


## Removed unused is_databricks_job_context


## Removed thin wrapper get_task_value_from_task


def download_artifact(run_id: str, artifact_path: str, dst_dir: Optional[str] = None) -> str:
    """Download an MLflow artifact to a local directory and return the local path."""
    client = MlflowClient()
    return client.download_artifacts(run_id, artifact_path, dst_dir or ".")


def save_dataframe_as_artifact(df: pd.DataFrame, filename: str, artifact_subdir: str) -> None:
    """Save a DataFrame to a temp parquet and log as an MLflow artifact under subdir."""
    tmpdir = tempfile.mkdtemp()
    local_path = os.path.join(tmpdir, filename)
    df.to_parquet(local_path, index=False)
    mlflow.log_artifact(local_path, artifact_path=artifact_subdir)


def load_parquet_artifact_as_df(run_id: str, artifact_rel_path: str) -> pd.DataFrame:
    """Download a parquet artifact and load it as a DataFrame."""
    local_dir = download_artifact(run_id, artifact_rel_path)
    # If artifact_rel_path is a file path, download_artifact may return the file path; handle both dir/file
    if os.path.isdir(local_dir):
        # take first parquet file in directory
        for name in os.listdir(local_dir):
            if name.endswith(".parquet"):
                return pd.read_parquet(os.path.join(local_dir, name))
        raise FileNotFoundError(f"No parquet file found in {local_dir}")
    return pd.read_parquet(local_dir)

