from __future__ import annotations

import os
from typing import Dict

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd

from ml_pipelines.mlflow_utils import (
    set_experiment_from_cfg,
    resolve_parent_run_id,
    get_task_value_from_task,
)

def run(cfg: DictConfig, model_data: Dict | None = None, data: Dict[str, pd.DataFrame] | None = None):
    exp_used = set_experiment_from_cfg(cfg)
    parent_run_id = resolve_parent_run_id(cfg)
    if parent_run_id:
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id
    print(f"[model_qa] experiment={exp_used} tracking_uri={mlflow.get_tracking_uri()} parent_run_id={parent_run_id}")
    start_params = {"run_name": "05_model_qa", "nested": True}
    if parent_run_id:
        start_params["tags"] = {"mlflow.parentRunId": parent_run_id}
    with mlflow.start_run(**start_params):
        mlflow.set_tag("step", "model_qa")
        mlflow.log_param("prod_model_uri", cfg.steps.model_qa.prod_model_uri)
        # Load train run model if needed
        if model_data is None:
            train_run_id = get_task_value_from_task("train_run_id", task_key="train", default=None)
            if train_run_id:
                _ = mlflow.sklearn.load_model(f"runs:/{train_run_id}/model")
        mlflow.log_metric("qa_placeholder", 1.0)
        return {}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run(cfg, None, None)


if __name__ == "__main__":  # pragma: no cover
    main()
