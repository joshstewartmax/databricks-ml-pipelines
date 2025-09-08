from __future__ import annotations

import os
from typing import Dict

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
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
    print(f"[evaluate:init] experiment={exp_used} tracking_uri={mlflow.get_tracking_uri()} parent_run_id={parent_run_id}")
    return exp_used, parent_run_id


def load_previous_artifacts_for_evaluate(cfg: DictConfig, model_data: Dict | None, data: Dict[str, pd.DataFrame] | None):
    if data is None:
        dbutils = DBUtils(SparkSession.getActiveSession() or SparkSession.builder.getOrCreate())
        prep_run_id = dbutils.jobs.taskValues.get(key="prepare_run_id", taskKey="prepare_data")
        if not prep_run_id:
            raise RuntimeError("Could not locate prepare_run_id from task values")
        test_df = load_parquet_artifact_as_df(prep_run_id, "prepare_data/test.parquet")
    else:
        test_df = data["test"]
    if model_data is None:
        dbutils = DBUtils(SparkSession.getActiveSession() or SparkSession.builder.getOrCreate())
        train_run_id = dbutils.jobs.taskValues.get(key="train_run_id", taskKey="train")
        if not train_run_id:
            raise RuntimeError("Could not locate train_run_id from task values")
        model = mlflow.sklearn.load_model(f"runs:/{train_run_id}/model")
    else:
        model = model_data["model"]
    return model, test_df


def evaluate(cfg: DictConfig, exp_used: str, parent_run_id: str | None, model, test_df: pd.DataFrame):
    start_params = {"run_name": "03_evaluate", "nested": True}
    if parent_run_id:
        start_params["tags"] = {"mlflow.parentRunId": parent_run_id}
    with mlflow.start_run(**start_params):
        mlflow.set_tag("step", "evaluate")
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    exp_used, parent_run_id = init_mlflow_experiment_and_run_config(cfg)
    model, test_df = load_previous_artifacts_for_evaluate(cfg, None, None)
    evaluate(cfg, exp_used, parent_run_id, model, test_df)


if __name__ == "__main__":  # pragma: no cover
    main()
