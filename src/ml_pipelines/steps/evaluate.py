from __future__ import annotations

import os
from typing import Dict

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

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
    print(f"[evaluate] experiment={exp_used} tracking_uri={mlflow.get_tracking_uri()} parent_run_id={parent_run_id}")
    start_params = {"run_name": "03_evaluate", "nested": True}
    if parent_run_id:
        start_params["tags"] = {"mlflow.parentRunId": parent_run_id}
    with mlflow.start_run(**start_params):
        mlflow.set_tag("step", "evaluate")
        if data is None:
            prep_run_id = get_task_value_from_task("prepare_run_id", task_key="prepare_data", default=None)
            if not prep_run_id:
                raise RuntimeError("Could not locate prepare_run_id from task values")
            test_df = load_parquet_artifact_as_df(prep_run_id, "prepare_data/test.parquet")
        else:
            test_df = data["test"]
        X_test = test_df.drop("label", axis=1)
        y_test = test_df["label"]
        if model_data is None:
            # Load model from train run
            train_run_id = get_task_value_from_task("train_run_id", task_key="train", default=None)
            if not train_run_id:
                raise RuntimeError("Could not locate train_run_id from task values")
            model = mlflow.sklearn.load_model(f"runs:/{train_run_id}/model")
        else:
            model = model_data["model"]
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        mlflow.log_metric("test_auc", auc)
        disp = RocCurveDisplay.from_predictions(y_test, probs)
        fig = disp.figure_
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)
        return {"test_auc": auc}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run(cfg, None, None)


if __name__ == "__main__":  # pragma: no cover
    main()
