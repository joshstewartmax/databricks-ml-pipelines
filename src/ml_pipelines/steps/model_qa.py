from __future__ import annotations

import os
from typing import Dict

import mlflow
from omegaconf import DictConfig
import hydra
from pyspark.sql import SparkSession  # type: ignore
from pyspark.dbutils import DBUtils  # type: ignore

from ml_pipelines.mlflow_utils import (
    set_experiment_from_cfg,
    resolve_parent_run_id,
)

def init_mlflow_experiment_and_run_config(cfg: DictConfig):
    exp_used = set_experiment_from_cfg(cfg)
    parent_run_id = resolve_parent_run_id(cfg)
    if parent_run_id:
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id
    print(f"[model_qa:init] experiment={exp_used} tracking_uri={mlflow.get_tracking_uri()} parent_run_id={parent_run_id}")
    return exp_used, parent_run_id


def load_previous_artifacts_for_model_qa(cfg: DictConfig, model_data: Dict | None):
    if model_data is None:
        dbutils = DBUtils(SparkSession.getActiveSession() or SparkSession.builder.getOrCreate())
        train_run_id = dbutils.jobs.taskValues.get(key="train_run_id", taskKey="train")
        if train_run_id:
            _ = mlflow.sklearn.load_model(f"runs:/{train_run_id}/model")
    else:
        _ = model_data.get("model")
    return None


def model_qa(cfg: DictConfig, exp_used: str, parent_run_id: str | None):
    start_params = {"run_name": "05_model_qa", "nested": True}
    if parent_run_id:
        start_params["tags"] = {"mlflow.parentRunId": parent_run_id}
    with mlflow.start_run(**start_params):
        mlflow.set_tag("step", "model_qa")
        mlflow.log_param("prod_model_uri", cfg.steps.model_qa.prod_model_uri)
        mlflow.log_metric("qa_placeholder", 1.0)
        return {}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    exp_used, parent_run_id = init_mlflow_experiment_and_run_config(cfg)
    _ = load_previous_artifacts_for_model_qa(cfg, None)
    model_qa(cfg, exp_used, parent_run_id)


if __name__ == "__main__":  # pragma: no cover
    main()
