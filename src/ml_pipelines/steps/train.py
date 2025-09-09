from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from ml_pipelines.util.mlflow import (
    load_parquet_artifact_as_df,
    save_dataframe_as_artifact,
    log_input_dataset,
)
from ml_pipelines.util.task_values import TaskValues, DatabricksTaskValues
from ml_pipelines.util.runner import run_step


def run(cfg: DictConfig, train_df: pd.DataFrame):
    log_input_dataset(train_df, name="train_df")

    X = train_df.drop("label", axis=1)
    y = train_df["label"]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg.steps.train.val_size, random_state=cfg.seed
    )

    param_grid = {"n_estimators": [50, 100], "max_depth": [None, 5]}
    search = GridSearchCV(
        RandomForestClassifier(random_state=cfg.seed),
        param_grid,
        cv=3,
        n_jobs=-1,
    )

    search.fit(X_tr, y_tr)
    best_model = search.best_estimator_
    val_probs = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    mlflow.log_metric("val_auc", val_auc)
    mlflow.log_params(search.best_params_)

    input_example = X_tr.head(5).astype("float64")
    signature = infer_signature(model_input=input_example, model_output=best_model.predict(input_example))
    mlflow.sklearn.log_model(
        best_model,
        name="model",
        input_example=input_example,
        signature=signature,
    )

    save_dataframe_as_artifact(X_tr.reset_index(drop=True), "X_train.parquet", artifact_subdir="train")
    save_dataframe_as_artifact(y_tr.to_frame(name="label").reset_index(drop=True), "y_train.parquet", artifact_subdir="train")

    return {"model": best_model, "X_train": X_tr, "y_train": y_tr}


def get_step_inputs(task_values: TaskValues, cfg: DictConfig):
    prep_run_id = task_values.get(key="prepare_data_run_id", task_key="prepare_data")
    if prep_run_id is None:
        prep_run_id = task_values.get(key="prepare_data_run_id")
    train_df = load_parquet_artifact_as_df(prep_run_id, "prepare_data/train.parquet")
    return {"train_df": train_df}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.experiment.name)
    
    task_values = DatabricksTaskValues()
    pipeline_run_id = task_values.get(key="pipeline_run_id", task_key="prepare_data")

    step_inputs = get_step_inputs(task_values, cfg)
    run_step(
        cfg,
        step_key="train",
        task_values=task_values,
        step_func=run,
        parent_run_id=pipeline_run_id,
        step_inputs=step_inputs,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
