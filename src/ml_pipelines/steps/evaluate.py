from __future__ import annotations

from typing import Dict

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

from ml_pipelines.util.databricks import get_dbutils
from ml_pipelines.util.mlflow import load_parquet_artifact_as_df


dbutils = get_dbutils()

def load_previous_artifacts_for_evaluate(model_data: Dict | None, data: Dict[str, pd.DataFrame] | None):
    if data is None:
        prep_run_id = dbutils.jobs.taskValues.get(key="prepare_run_id", taskKey="prepare_data")
        test_df = load_parquet_artifact_as_df(prep_run_id, "prepare_data/test.parquet")
    else:
        test_df = data["test"]
    if model_data is None:
        train_run_id = dbutils.jobs.taskValues.get(key="train_run_id", taskKey="train")
        model = mlflow.sklearn.load_model(f"runs:/{train_run_id}/model")
    else:
        model = model_data["model"]
    return model, test_df


def evaluate(cfg: DictConfig, pipeline_run_id: str, model, test_df: pd.DataFrame):
    with mlflow.start_run(
        run_name=cfg.steps.evaluate.step_name,
        nested=True,
        parent_run_id=pipeline_run_id,
    ):
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
    mlflow.set_experiment(cfg.experiment.name)
    pipeline_run_id = dbutils.jobs.taskValues.get(key="pipeline_run_id", taskKey="prepare_data")
    model, test_df = load_previous_artifacts_for_evaluate(None, None)
    evaluate(cfg, pipeline_run_id, model, test_df)


if __name__ == "__main__":  # pragma: no cover
    main()
