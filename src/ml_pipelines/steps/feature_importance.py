from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra
import polars as pl
from sklearn.inspection import permutation_importance

from ml_pipelines.util.task_values import TaskValues, DatabricksTaskValues
from ml_pipelines.runner import run_step
from ml_pipelines.util.mlflow import log_delta_input


def run(cfg: DictConfig, task_values: TaskValues, train_run_id: str, test_path: str):
    if cfg.mlflow.log_datasets:
        log_delta_input(path=test_path, name="prepare_data.test")
        
    test_df = pl.scan_delta(test_path).collect()
    X_test = test_df.drop("label")
    y_test = test_df.select("label")

    model = mlflow.sklearn.load_model(f"runs:/{train_run_id}/model")

    result = permutation_importance(
        model,
        X_test.to_numpy(),
        y_test.to_numpy(),
        n_repeats=cfg.steps.feature_importance.n_repeats,
        random_state=cfg.seed,
    )
    importances = {col: imp for col, imp in zip(X_test.columns, result.importances_mean)}
    mlflow.log_dict(importances, "feature_importance.json")
    return importances


def get_step_inputs(task_values: TaskValues, cfg: DictConfig):
    train_run_id = task_values.get(
        key=cfg.steps.feature_importance.inputs.train_run_id.key,
        task_key=cfg.steps.feature_importance.inputs.train_run_id.source_step,
    )
    
    test_path = task_values.get(
        key=cfg.steps.feature_importance.inputs.train_path.key,
        task_key=cfg.steps.feature_importance.inputs.train_path.source_step,
    )
    return {"train_run_id": train_run_id, "test_path": test_path}


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
