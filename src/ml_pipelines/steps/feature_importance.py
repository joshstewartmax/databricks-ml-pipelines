from __future__ import annotations

import os
from typing import Dict

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd
from sklearn.inspection import permutation_importance

from ml_pipelines.mlflow_utils import (
    set_experiment_from_cfg,
    resolve_parent_run_id,
    get_task_value_from_task,
    load_parquet_artifact_as_df,
)

def run(cfg: DictConfig, model_data: Dict | None = None, data: Dict[str, pd.DataFrame] | None = None):
    exp_used = set_experiment_from_cfg(cfg)
    parent_run_id = resolve_parent_run_id(cfg)
    if parent_run_id:
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id
    print(f"[feature_importance] experiment={exp_used} tracking_uri={mlflow.get_tracking_uri()} parent_run_id={parent_run_id}")
    start_params = {"run_name": "04_feature_importance", "nested": True}
    if parent_run_id:
        start_params["tags"] = {"mlflow.parentRunId": parent_run_id}
    with mlflow.start_run(**start_params):
        mlflow.set_tag("step", "feature_importance")
        if model_data is None or data is None:
            train_run_id = get_task_value_from_task("train_run_id", task_key="train", default=None)
            if not train_run_id:
                raise RuntimeError("Could not locate train_run_id from task values")
            model = mlflow.sklearn.load_model(f"runs:/{train_run_id}/model")
            X_train = load_parquet_artifact_as_df(train_run_id, "train/X_train.parquet")
            y_train = load_parquet_artifact_as_df(train_run_id, "train/y_train.parquet")
        else:
            model = model_data["model"]
            X_train = model_data["X_train"]
            y_train = model_data["y_train"]
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
    run(cfg, None, None)


if __name__ == "__main__":  # pragma: no cover
    main()
