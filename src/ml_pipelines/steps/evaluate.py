from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra
import polars as pl
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

from ml_pipelines.util.task_values import TaskValues, DatabricksTaskValues
from ml_pipelines.util.runner import run_step


def run(cfg: DictConfig, task_values: TaskValues, model, test_uri: str):
    test_pl = pl.scan_delta(test_uri).collect()
    X_pl = test_pl.drop("label")
    y_pl = test_pl["label"]
    X_test = X_pl.to_numpy()
    y_test = y_pl.to_numpy()
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    mlflow.log_metric("test_auc", auc)
    disp = RocCurveDisplay.from_predictions(y_test, probs)
    fig = disp.figure_
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)
    task_values.set(
        key=cfg.steps.evaluate.outputs.test_auc.key,
        value=auc,
        task_key=cfg.steps.evaluate.outputs.test_auc.task_key,
    )
    return {"test_auc": auc}


def get_step_inputs(task_values: TaskValues, cfg: DictConfig):
    test_uri = task_values.get(
        key=cfg.steps.evaluate.inputs.test_uri.key,
        task_key=cfg.steps.evaluate.inputs.test_uri.task_key,
    )
    train_run_id = task_values.get(
        key=cfg.steps.evaluate.inputs.train_run_id.key,
        task_key=cfg.steps.evaluate.inputs.train_run_id.task_key,
    )
    model = mlflow.sklearn.load_model(f"runs:/{train_run_id}/model")
    return {"model": model, "test_uri": test_uri}


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
