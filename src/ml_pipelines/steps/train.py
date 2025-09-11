from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra
import polars as pl
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from ml_pipelines.util.task_values import TaskValues, DatabricksTaskValues
from ml_pipelines.util.runner import run_step
from ml_pipelines.util.delta_paths import build_delta_path


def run(cfg: DictConfig, task_values: TaskValues, train_uri: str):
    train_pl = pl.scan_delta(train_uri).collect()
    train_df = train_pl.to_pandas()

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

    # write derived training splits as Delta using Polars
    X_uri = build_delta_path(cfg, "train", "X_train")
    y_uri = build_delta_path(cfg, "train", "y_train")

    print(X_uri)
    print(y_uri)
    env_name = getattr(cfg.experiment, "env_name", "local")
    if env_name == "local":
        pl.from_pandas(X_tr.reset_index(drop=True)).write_delta(X_uri, mode="overwrite")
        pl.from_pandas(y_tr.to_frame(name="label").reset_index(drop=True)).write_delta(y_uri, mode="overwrite")
    else:
        # Use Spark to write to Unity Catalog Volumes to avoid delta-rs rename limitations
        from pyspark.sql import SparkSession  # type: ignore
        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        spark.createDataFrame(X_tr.reset_index(drop=True)).write.format("delta").mode("overwrite").option("delta.enableDeletionVectors", "false").save(X_uri)
        spark.createDataFrame(y_tr.to_frame(name="label").reset_index(drop=True)).write.format("delta").mode("overwrite").option("delta.enableDeletionVectors", "false").save(y_uri)

    # write task values for downstream steps and access
    current_run = mlflow.active_run()
    if current_run is not None:
        task_values.set(key="train_run_id", value=current_run.info.run_id, task_key="train")

    task_values.set(key="X_train_uri", value=X_uri, task_key="train")
    task_values.set(key="y_train_uri", value=y_uri, task_key="train")

    return {"model": best_model, "X_train_uri": X_uri, "y_train_uri": y_uri}


def get_step_inputs(task_values: TaskValues, cfg: DictConfig):
    train_uri = task_values.get(key="train_uri", task_key="prepare_data")
    if train_uri is None:
        train_uri = task_values.get(key="train_uri")
    return {"train_uri": train_uri}


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
