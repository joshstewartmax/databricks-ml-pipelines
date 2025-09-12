from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra
import polars as pl
from sklearn.inspection import permutation_importance

from ml_pipelines.util.task_values import TaskValues, DatabricksTaskValues
from ml_pipelines.util.runner import run_step
from ml_pipelines.util.mlflow_datasets import log_delta_input


def run(cfg: DictConfig, task_values: TaskValues, model, train_uri: str):
    if cfg.mlflow.log_datasets:
        log_delta_input(path=train_uri, name="prepare_data.train")
        
    train_pl = pl.scan_delta(train_uri).collect()
    X_pl = train_pl.drop("label")
    y_pl = train_pl["label"]
    X_train = X_pl.to_numpy()
    y_train = y_pl.to_numpy()

    result = permutation_importance(
        model,
        X_train,
        y_train,
        n_repeats=cfg.steps.feature_importance.n_repeats,
        random_state=cfg.seed,
    )
    importances = {col: imp for col, imp in zip(X_pl.columns, result.importances_mean)}
    mlflow.log_dict(importances, "feature_importance.json")
    task_values.set(
        key=cfg.steps.feature_importance.outputs.feature_importance_logged.key,
        value=True,
        task_key=cfg.steps.feature_importance.outputs.feature_importance_logged.task_key,
    )
    return importances


def get_step_inputs(task_values: TaskValues, cfg: DictConfig):
    train_run_id = task_values.get(
        key=cfg.steps.feature_importance.inputs.train_run_id.key,
        task_key=cfg.steps.feature_importance.inputs.train_run_id.task_key,
    )
    model = mlflow.sklearn.load_model(f"runs:/{train_run_id}/model")
    train_uri = task_values.get(
        key=cfg.steps.feature_importance.inputs.train_uri.key,
        task_key=cfg.steps.feature_importance.inputs.train_uri.task_key,
    )
    return {"model": model, "train_uri": train_uri}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.pipeline.experiment_name)

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
