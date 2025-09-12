from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra
import polars as pl
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

from ml_pipelines.util.task_values import TaskValues, DatabricksTaskValues
from ml_pipelines.runner import run_step
from ml_pipelines.util.mlflow import log_delta_input


def run(cfg: DictConfig, task_values: TaskValues, train_run_id: str, test_path: str):
    if cfg.mlflow.log_datasets:
        log_delta_input(path=test_path, name="prepare_data.test")

    # I've just used mlflow.models.evaluate to get out-of-the-box evals, 
    # but the below would work for when we do custom stuff
    # model = mlflow.sklearn.load_model(f"runs:/{train_run_id}/model")

    test_df = pl.scan_delta(test_path)

    X_test = test_df.drop("label").collect().to_numpy()
    y_test = test_df.select("label").collect().to_numpy().ravel()

    result = mlflow.models.evaluate(
        model=f"runs:/{train_run_id}/model",
        data=X_test,
        targets=y_test,
        model_type="classifier",
        evaluators=["default"],
    )

    return {"eval_result": result}


def get_step_inputs(task_values: TaskValues, cfg: DictConfig):
    test_path = task_values.get(
        key=cfg.steps.evaluate.inputs.test_path.key,
        task_key=cfg.steps.evaluate.inputs.test_path.source_step,
    )
    train_run_id = task_values.get(
        key=cfg.steps.evaluate.inputs.train_run_id.key,
        task_key=cfg.steps.evaluate.inputs.train_run_id.source_step,
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
        step_key="evaluate",
        task_values=task_values,
        step_func=run,
        parent_run_id=pipeline_run_id,
        step_inputs=step_inputs,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
