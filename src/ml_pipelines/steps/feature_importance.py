from __future__ import annotations

from typing import Dict

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd
from sklearn.inspection import permutation_importance

from ml_pipelines.util.databricks import get_dbutils
from ml_pipelines.util.mlflow import load_parquet_artifact_as_df


dbutils = get_dbutils()

def load_previous_artifacts_for_feature_importance(model_data: Dict | None, data: Dict[str, pd.DataFrame] | None):
    if model_data is None or data is None:
        train_run_id = dbutils.jobs.taskValues.get(key="train_run_id", taskKey="train")
        model = mlflow.sklearn.load_model(f"runs:/{train_run_id}/model")
        X_train = load_parquet_artifact_as_df(train_run_id, "train/X_train.parquet")
        y_train = load_parquet_artifact_as_df(train_run_id, "train/y_train.parquet")
    else:
        model = model_data["model"]
        X_train = model_data["X_train"]
        y_train = model_data["y_train"]
    return model, X_train, y_train


def feature_importance(cfg: DictConfig, parent_run_id: str, model, X_train: pd.DataFrame, y_train: pd.DataFrame):
    with mlflow.start_run(
        run_name=cfg.steps.feature_importance.step_name,
        nested=True,
        parent_run_id=parent_run_id,
    ):
        mlflow.set_tag("step", "feature_importance")
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
    mlflow.set_experiment(cfg.experiment.name)
    parent_run_id = dbutils.jobs.taskValues.get(key="parent_run_id", taskKey="initialize")
    model, X_train, y_train = load_previous_artifacts_for_feature_importance(None, None)
    feature_importance(cfg, parent_run_id, model, X_train, y_train)


if __name__ == "__main__":  # pragma: no cover
    main()
