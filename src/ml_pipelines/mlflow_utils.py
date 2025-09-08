from __future__ import annotations

import os
import uuid
from typing import Optional, Dict

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig


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
    """Best-effort access to Databricks dbutils if available. Returns None if not accessible."""
    try:
        # In Databricks jobs, a global `dbutils` is injected.
        return globals().get("dbutils", None)
    except Exception:
        return None


def set_task_value(key: str, value: str) -> None:
    """Set a Databricks jobs task value if available; otherwise no-op."""
    try:
        dbutils = _get_dbutils()
        if dbutils is not None:
            dbutils.jobs.taskValues.set(key=key, value=value)
    except Exception:
        # Silently ignore if not on Databricks
        pass


def get_task_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a Databricks jobs task value if available; otherwise return default."""
    try:
        dbutils = _get_dbutils()
        if dbutils is not None:
            return dbutils.jobs.taskValues.get(key=key, defaultValue=default)
    except Exception:
        pass
    return default


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


