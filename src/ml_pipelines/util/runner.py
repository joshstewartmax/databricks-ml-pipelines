from __future__ import annotations

from typing import Callable, Dict, Optional, TypeVar, Any

import mlflow
from omegaconf import DictConfig, OmegaConf

from .task_store import TaskStore


T = TypeVar("T")


def run_step(
    cfg: DictConfig,
    step_key: str,
    task_store: TaskStore,
    step_func: Callable[..., T],
    parent_run_id: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    step_inputs: Optional[Dict[str, Any]] = None,
) -> T:
    """Run a step function inside an MLflow run and record the run id.

    - Opens an MLflow run (nested if parent_run_id provided)
    - Sets a standard step tag
    - Logs the step's resolved config as config.json
    - Calls the provided function with cfg and kwargs
    - Writes the run id to the task_store under key "{step_key}_run_id"
    """

    step_cfg = getattr(cfg.steps, step_key)
    step_inputs = step_inputs or {}

    with mlflow.start_run(
        run_name=step_cfg.step_name,
        nested=parent_run_id is not None,
        parent_run_id=parent_run_id,
    ):
        mlflow.set_tag("step", step_key)

        if tags:
            mlflow.set_tags(tags)

        step_cfg_dict = OmegaConf.to_container(step_cfg, resolve=True)  # type: ignore[arg-type]
        mlflow.log_dict(step_cfg_dict, "config.json")

        result: T = step_func(cfg, **step_inputs)

        current_run = mlflow.active_run()
        if current_run is not None:
            task_store.set(key=f"{step_key}_run_id", value=current_run.info.run_id)
            
        return result


