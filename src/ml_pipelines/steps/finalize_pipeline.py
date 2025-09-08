from __future__ import annotations

import os

from omegaconf import DictConfig
import hydra

from ml_pipelines.mlflow_utils import (
    terminate_run,
    get_task_value_from_task,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):  # noqa: ARG001 - cfg reserved for future use
    parent_run_id = get_task_value_from_task("parent_run_id", task_key="initialize", default=None) or os.environ.get("MLFLOW_PARENT_RUN_ID")
    print(f"[finalize] resolved parent_run_id={parent_run_id}")
    if parent_run_id:
        terminate_run(parent_run_id)


if __name__ == "__main__":  # pragma: no cover
    main()


