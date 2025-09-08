from __future__ import annotations

import os
from typing import Dict

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd
from sklearn.inspection import permutation_importance
from pyspark.sql import SparkSession  # type: ignore
from pyspark.dbutils import DBUtils  # type: ignore

from ml_pipelines.mlflow_utils import (
    set_experiment_from_cfg,
    resolve_parent_run_id,
    load_parquet_artifact_as_df,
)

def init_mlflow_experiment_and_run_config(cfg: DictConfig):
    exp_used = set_experiment_from_cfg(cfg)
    parent_run_id = resolve_parent_run_id(cfg)
    if parent_run_id:
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id
    print(f"[feature_importance:init] experiment={exp_used} tracking_uri={mlflow.get_tracking_uri()} parent_run_id={parent_run_id}")
    return exp_used, parent_run_id


def load_previous_artifacts_for_feature_importance(cfg: DictConfig, model_data: Dict | None, data: Dict[str, pd.DataFrame] | None):
    if model_data is None or data is None:
        dbutils = DBUtils(SparkSession.getActiveSession() or SparkSession.builder.getOrCreate())
        train_run_id = dbutils.jobs.taskValues.get(key="train_run_id", taskKey="train")
        if not train_run_id:
            raise RuntimeError("Could not locate train_run_id from task values")
        model = mlflow.sklearn.load_model(f"runs:/{train_run_id}/model")
        X_train = load_parquet_artifact_as_df(train_run_id, "train/X_train.parquet")
        y_train = load_parquet_artifact_as_df(train_run_id, "train/y_train.parquet")
    else:
        model = model_data["model"]
        X_train = model_data["X_train"]
        y_train = model_data["y_train"]
    return model, X_train, y_train


def feature_importance(cfg: DictConfig, exp_used: str, parent_run_id: str | None, model, X_train: pd.DataFrame, y_train: pd.DataFrame):
    start_params = {"run_name": "04_feature_importance", "nested": True}
    if parent_run_id:
        start_params["tags"] = {"mlflow.parentRunId": parent_run_id}
    with mlflow.start_run(**start_params):
        mlflow.set_tag("step", "feature_importance")
        result = permutation_importance(
            model,
            X_train,
            y_train,
            n_repeats=cfg.steps.feature_importance.n_repeats,
            random_state=cfg.seed,
        )
        importances = {col: imp for col, imp in zip(X_train.columns, result.importances_mean)}
        mlflow.log_dict(importances, "feature_importance.json")
        return importances


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    exp_used, parent_run_id = init_mlflow_experiment_and_run_config(cfg)
    model, X_train, y_train = load_previous_artifacts_for_feature_importance(cfg, None, None)
    feature_importance(cfg, exp_used, parent_run_id, model, X_train, y_train)


if __name__ == "__main__":  # pragma: no cover
    main()
