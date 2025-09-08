from __future__ import annotations

import os
from typing import Dict

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from ml_pipelines.mlflow_utils import (
    set_experiment_from_cfg,
    resolve_parent_run_id,
    load_parquet_artifact_as_df,
    get_task_value_from_task,
    set_task_value,
)

def run(cfg: DictConfig, data: Dict[str, pd.DataFrame] | None = None):
    exp_used = set_experiment_from_cfg(cfg)
    parent_run_id = resolve_parent_run_id(cfg)
    if parent_run_id:
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id
    print(f"[train] experiment={exp_used} tracking_uri={mlflow.get_tracking_uri()} parent_run_id={parent_run_id}")
    start_params = {"run_name": "02_train", "nested": True}
    if parent_run_id:
        start_params["tags"] = {"mlflow.parentRunId": parent_run_id}
    with mlflow.start_run(**start_params):
        mlflow.set_tag("step", "train")
        if data is None:
            prep_run_id = get_task_value_from_task("prepare_run_id", task_key="prepare_data", default=None)
            if not prep_run_id:
                raise RuntimeError("Could not locate prepare_run_id from task values; ensure permissions and depends_on")
            train_df = load_parquet_artifact_as_df(prep_run_id, "prepare_data/train.parquet")
        else:
            train_df = data["train"]
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
        # Ensure float inputs to avoid integer column warnings in schema
        input_example = X_tr.head(5).astype("float64")
        signature = infer_signature(model_input=input_example, model_output=best_model.predict(input_example))


        mlflow.sklearn.log_model(
            best_model,
            name="model",
            input_example=input_example,
            signature=signature,
        )
        # Log train matrices as artifacts for downstream steps
        X_tr.to_parquet("X_train.parquet", index=False)
        y_tr.to_frame(name="label").to_parquet("y_train.parquet", index=False)
        mlflow.log_artifact("X_train.parquet", artifact_path="train")
        mlflow.log_artifact("y_train.parquet", artifact_path="train")
        # Expose this run id for downstream
        current_run_id = mlflow.active_run().info.run_id
        print(f"[train] setting task value train_run_id={current_run_id}")
        set_task_value("train_run_id", current_run_id)
        return {"model": best_model, "X_train": X_tr, "y_train": y_tr}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run(cfg, None)


if __name__ == "__main__":  # pragma: no cover
    main()
