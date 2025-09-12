from __future__ import annotations

from typing import Optional, Dict
import warnings

import mlflow
from mlflow.data import load_delta
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

def begin_pipeline_run(cfg: DictConfig) -> str:
    exp = mlflow.get_experiment_by_name(cfg.pipeline.experiment_name)
    client = MlflowClient()
    run = client.create_run(
        experiment_id=exp.experiment_id,
        run_name=cfg.pipeline.pipeline_run_name,
        tags={},
    )

    return run.info.run_id


def end_pipeline_run(parent_run_id: str, status: str = "FINISHED") -> None:
    """Mark the MLflow parent run as terminated with the given status."""
    client = MlflowClient()
    client.set_terminated(run_id=parent_run_id, status=status)


def log_delta_input(
    *,
    path: Optional[str] = None,
    table_name: Optional[str] = None,
    name: str,
    version: Optional[int] = None,
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """Create a Dataset from a Delta table and log it as an MLflow input.

    Args:
        path: Filesystem/UC Volumes path to the Delta directory.
        table_name: Optional fully-qualified table name (catalog.schema.table).
        name: Display name or context shown in the Inputs tab.
        version: Optional Delta version/snapshot to bind the dataset to.
        tags: Optional extra tags to attach to the run for discoverability.
    """
    if path is None and table_name is None:
        raise ValueError("Provide either path or table_name")

    # we probably want to handle this in a better way when we migrate the real pipeline
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Hint: Inferred schema contains integer column",
            category=UserWarning,
            module="mlflow.types.utils",
        )
        dataset = load_delta(path=path, table_name=table_name, version=version)

    # Helpful tags for discoverability in the run page
    mlflow.set_tag("dataset_path", path or table_name or "")
    if hasattr(dataset, "version") and getattr(dataset, "version") is not None:
        mlflow.set_tag("dataset_version", str(getattr(dataset, "version")))
    if tags:
        mlflow.set_tags(tags)

    # we probably want to handle this in a better way when we migrate the real pipeline
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Hint: Inferred schema contains integer column",
            category=UserWarning,
            module="mlflow.types.utils",
        )
        try:
            mlflow.log_input(dataset, name=name)
        except TypeError:
            mlflow.log_input(dataset, context=name)