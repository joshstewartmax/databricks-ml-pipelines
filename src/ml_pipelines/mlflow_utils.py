from __future__ import annotations

import os
import uuid
from typing import Optional, Dict

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
import tempfile
import pandas as pd


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
    """Access Databricks dbutils, with explicit debug logging (no silent failures)."""
    print("[dbutils] Attempting to acquire dbutils...")
    # 1) Global injected dbutils (e.g., notebooks)
    injected = globals().get("dbutils", None)
    if injected is not None:
        print("[dbutils] Using injected global dbutils")
        return injected
    # 2) Construct via Spark session for Jobs / wheel tasks
    from pyspark.sql import SparkSession  # type: ignore
    from pyspark.dbutils import DBUtils  # type: ignore

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    dbu = DBUtils(spark)
    print("[dbutils] Constructed DBUtils from Spark session")
    return dbu


def set_task_value(key: str, value: str) -> None:
    """Set a Databricks jobs task value with debug logging."""
    print(f"[taskValues.set] key={key} value={value}")
    dbutils = _get_dbutils()
    dbutils.jobs.taskValues.set(key=key, value=str(value))
    print("[taskValues.set] success")


def get_task_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a Databricks jobs task value with debug logging.

    Note: In wheel tasks, get() may not accept a default argument; call with key only.
    """
    print(f"[taskValues.get] key={key} default={default}")
    dbutils = _get_dbutils()
    try:
        val = dbutils.jobs.taskValues.get(key=key)
    except TypeError:
        # Fallback to possible alternative signature
        val = dbutils.jobs.taskValues.get(key)
    print(f"[taskValues.get] -> {val}")
    return val


def get_parent_run_id() -> Optional[str]:
    """Best-effort retrieval of parent run id across tasks.

    Order:
    1) MLFLOW_PARENT_RUN_ID env var
    2) Databricks task value from initialize task
    3) Databricks task value in current task
    """
    # 1) Env var
    env_val = os.environ.get("MLFLOW_PARENT_RUN_ID")
    if env_val:
        return env_val
    # 2) Databricks task value from initialize task
    val_other = get_task_value_from_task("parent_run_id", task_key="initialize", default=None)
    if val_other:
        return val_other
    # 3) Current task scope
    return get_task_value("parent_run_id", default=None)


def search_parent_run_id_in_experiment(cfg: DictConfig) -> Optional[str]:
    """Search the experiment for the parent run using stable tags.

    Priority:
    1) tags.orchestrator_run_id == <Databricks job run id>
    2) tags.pipeline_run_id == <pipeline_run_id from task values>
    """
    identifiers = get_databricks_run_identifiers()
    job_run_id = identifiers.get("job_run_id")
    pipeline_uuid = get_task_value("pipeline_run_id")

    exp_identifier = set_experiment_from_cfg(cfg)
    exp = mlflow.get_experiment_by_name(exp_identifier)
    if exp is None:
        return None

    client = MlflowClient()
    candidates = []
    if job_run_id:
        filter_str = f"tags.orchestrator_run_id = '{job_run_id}' and attributes.status = 'RUNNING'"
        candidates = client.search_runs([exp.experiment_id], filter_string=filter_str, max_results=1, order_by=["attributes.start_time DESC"])
    if not candidates and pipeline_uuid:
        filter_str = f"tags.pipeline_run_id = '{pipeline_uuid}'"
        candidates = client.search_runs([exp.experiment_id], filter_string=filter_str, max_results=1, order_by=["attributes.start_time DESC"])
    if not candidates:
        return None
    return candidates[0].info.run_id


def resolve_parent_run_id(cfg: DictConfig) -> Optional[str]:
    """Resolve parent run id using multiple strategies.

    Order:
    1) MLFLOW_PARENT_RUN_ID env
    2) Task value from initialize task
    3) Task value in current task
    4) Search MLflow in current experiment by Databricks job run id tag
    """
    env_val = os.environ.get("MLFLOW_PARENT_RUN_ID")
    if env_val:
        return env_val
    val_other = get_task_value_from_task("parent_run_id", task_key="initialize", default=None)
    if val_other:
        return val_other
    local_val = get_task_value("parent_run_id")
    if local_val:
        return local_val
    return search_parent_run_id_in_experiment(cfg)


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


def is_databricks_job_context() -> bool:
    """Heuristic to detect Databricks Jobs runtime."""
    if os.environ.get("DATABRICKS_JOB_RUN_ID"):
        return True
    try:
        return _get_dbutils() is not None
    except Exception:
        return False


def get_task_value_from_task(key: str, task_key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve a task value produced by another task, with debug logging.

    Note: In wheel tasks, get() may not accept a default argument; call with key and taskKey only.
    """
    print(f"[taskValues.get(other)] key={key} taskKey={task_key} default={default}")
    dbutils = _get_dbutils()
    try:
        val = dbutils.jobs.taskValues.get(key=key, taskKey=task_key)
    except TypeError:
        # Fallback to positional signature
        val = dbutils.jobs.taskValues.get(key, task_key)
    print(f"[taskValues.get(other)] -> {val}")
    return val


def download_artifact(run_id: str, artifact_path: str, dst_dir: Optional[str] = None) -> str:
    """Download an MLflow artifact to a local directory and return the local path."""
    client = MlflowClient()
    return client.download_artifacts(run_id, artifact_path, dst_dir or ".")


def search_step_run_id(cfg: DictConfig, step_name: str) -> Optional[str]:
    """Find the most recent run in the current experiment with tags.step == step_name."""
    exp_identifier = set_experiment_from_cfg(cfg)
    exp = mlflow.get_experiment_by_name(exp_identifier)
    if exp is None:
        return None
    client = MlflowClient()
    filter_str = f"tags.step = '{step_name}'"
    runs = client.search_runs([exp.experiment_id], filter_string=filter_str, max_results=1, order_by=["attributes.start_time DESC"])
    if not runs:
        return None
    return runs[0].info.run_id


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

