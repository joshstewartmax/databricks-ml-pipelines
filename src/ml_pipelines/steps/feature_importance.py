from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd
from sklearn.inspection import permutation_importance

from ml_pipelines.util.mlflow import load_parquet_artifact_as_df
from ml_pipelines.util.task_values import TaskValues, DatabricksTaskValues
from ml_pipelines.util.runner import run_step


def run(cfg: DictConfig, model, X_train: pd.DataFrame, y_train: pd.DataFrame):
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


def get_step_inputs(task_values: TaskValues, cfg: DictConfig):
    train_run_id = task_values.get(key="train_run_id", task_key="train")
    if train_run_id is None:
        train_run_id = task_values.get(key="train_run_id")
    model = mlflow.sklearn.load_model(f"runs:/{train_run_id}/model")
    X_train = load_parquet_artifact_as_df(train_run_id, "train/X_train.parquet")
    y_train = load_parquet_artifact_as_df(train_run_id, "train/y_train.parquet")
    return {"model": model, "X_train": X_train, "y_train": y_train}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.experiment.name)

    task_values = DatabricksTaskValues()
    pipeline_run_id = task_values.get(key="pipeline_run_id", task_key="prepare_data")
    
    step_inputs = get_step_inputs(task_values, cfg)
    run_step(
        cfg,
        step_key="feature_importance",
        task_values=task_values,
        step_func=run,
        parent_run_id=pipeline_run_id,
        step_inputs=step_inputs,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
