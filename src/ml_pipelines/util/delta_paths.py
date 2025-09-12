from __future__ import annotations

from typing import Optional

import mlflow
from omegaconf import DictConfig


def build_delta_path(
    cfg: DictConfig,
    step_key: str,
    dataset_name: str,
) -> str:
    """Construct a Delta directory path for a dataset produced by a step.

    Path format:
      "{prefix}/{env_name}/{experiment_name}/{pipeline_run_name_and_id}/{step_run_name_and_id}/{dataset_name}"
    """
    env_name = getattr(cfg.pipeline, "env_name", "local")
    experiment_name: str = cfg.pipeline.experiment_name

    # Choose prefix based on environment
    if env_name == "local":
        prefix: str = cfg.data.delta_prefix.local
        path_parts = [
            prefix.rstrip("/"),
            env_name,
            experiment_name.strip("/"),
        ]
    else:
        # Databricks Volumes layout: /Volumes/<catalog>/<schema>/<volume>/<...>
        # We need: /Volumes/ml_artifacts/<env>/<volume_name>/<experiment_name>/...
        prefix = cfg.data.delta_prefix.databricks
        volume_name: str = cfg.data.get("volume_name", "testing")
        path_parts = [
            prefix.rstrip("/"),
            env_name,
            volume_name,
            experiment_name.strip("/"),
        ]

    # Resolve current step run id and parent (pipeline) run id from MLflow
    current_run = mlflow.active_run()
    if current_run is None:
        raise RuntimeError("No active MLflow run. build_delta_path must be called inside a step run.")
    current_run_id = current_run.info.run_id
    run_info = mlflow.get_run(current_run_id)
    parent_run_id: Optional[str] = run_info.data.tags.get("mlflow.parentRunId")
    pipeline_run_id = parent_run_id or current_run_id

    pipeline_segment = f"pipeline_run_{pipeline_run_id}"

    step_run_id_short = current_run_id[:8]
    step_name: str = getattr(getattr(cfg.steps, step_key), "step_name")
    step_segment = f"{step_name}_{step_run_id_short}"

    path_parts.extend([pipeline_segment, step_segment, dataset_name])
    return "/".join(path_parts)


