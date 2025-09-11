from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra
import polars as pl
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

from ml_pipelines.util.task_values import TaskValues, DatabricksTaskValues
from ml_pipelines.util.runner import run_step


def run(cfg: DictConfig, model, test_uri: str):
    test_pl = pl.scan_delta(test_uri).collect()
    test_df = test_pl.to_pandas()
    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    mlflow.log_metric("test_auc", auc)
    disp = RocCurveDisplay.from_predictions(y_test, probs)
    fig = disp.figure_
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)
    return {"test_auc": auc}


def get_step_inputs(task_values: TaskValues, cfg: DictConfig):
    test_uri = task_values.get(key="test_uri", task_key="prepare_data")
    if test_uri is None:
        test_uri = task_values.get(key="test_uri")

    train_run_id = task_values.get(key="train_run_id", task_key="train")
    if train_run_id is None:
        train_run_id = task_values.get(key="train_run_id")
    model = mlflow.sklearn.load_model(f"runs:/{train_run_id}/model")

    return {"model": model, "test_uri": test_uri}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.experiment.name)

    task_values = DatabricksTaskValues()
    pipeline_run_id = task_values.get(key="pipeline_run_id", task_key="prepare_data")
    
    step_inputs = get_step_inputs(task_values, cfg)
    result = run_step(
        cfg,
        step_key="evaluate",
        task_values=task_values,
        step_func=run,
        parent_run_id=pipeline_run_id,
        step_inputs=step_inputs,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
