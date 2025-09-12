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
from ml_pipelines.util.mlflow_datasets import log_delta_input


def run(cfg: DictConfig, task_values: TaskValues, train_uri: str):
    if cfg.mlflow.log_datasets:
        log_delta_input(path=train_uri, name="prepare_data.train")
        
    train_pl = pl.scan_delta(train_uri).collect()
    X_pl = train_pl.drop("label")
    y_pl = train_pl["label"]

    X = X_pl.to_numpy()
    y = y_pl.to_numpy()

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

    input_example = X_tr[:5].astype("float64")
    signature = infer_signature(model_input=input_example, model_output=best_model.predict(input_example))
    mlflow.sklearn.log_model(
        best_model,
        name="model",
        input_example=input_example,
        signature=signature,
    )

    # write train_run_id via config-defined outputs mapping
    current_run = mlflow.active_run()
    task_values.set(
        key=cfg.steps.train.outputs.train_run_id.key,
        value=current_run.info.run_id,
        task_key=cfg.steps.train.outputs.train_run_id.task_key,
    )

    return {"model": best_model}


def get_step_inputs(task_values: TaskValues, cfg: DictConfig):
    train_uri = task_values.get(
        key=cfg.steps.train.inputs.train_uri.key,
        task_key=cfg.steps.train.inputs.train_uri.task_key,
    )
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
