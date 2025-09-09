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
)
from ml_pipelines.util.databricks import get_dbutils


dbutils = get_dbutils()

def train(cfg: DictConfig, parent_run_id: str, train_df: pd.DataFrame):
    with mlflow.start_run(
        run_name=cfg.steps.train.step_name,
        nested=True,
        parent_run_id=parent_run_id,
    ):
        mlflow.set_tag("step", "train")

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

        X_tr.to_parquet("X_train.parquet", index=False)
        y_tr.to_frame(name="label").to_parquet("y_train.parquet", index=False)
        mlflow.log_artifact("X_train.parquet", artifact_path="train")
        mlflow.log_artifact("y_train.parquet", artifact_path="train")

        current_run_id = mlflow.active_run().info.run_id
        dbutils.jobs.taskValues.set(key="train_run_id", value=current_run_id)

        return {"model": best_model, "X_train": X_tr, "y_train": y_tr}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.experiment.name)

    parent_run_id = dbutils.jobs.taskValues.get(key="parent_run_id", taskKey="initialize")
    prep_run_id = dbutils.jobs.taskValues.get(key="prepare_run_id", taskKey="prepare_data")
    
    train_df = load_parquet_artifact_as_df(prep_run_id, "prepare_data/train.parquet")

    train(cfg, parent_run_id, train_df)


if __name__ == "__main__":  # pragma: no cover
    main()
