from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra
import polars as pl
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier

from ml_pipelines.util.task_values import TaskValues, DatabricksTaskValues
from ml_pipelines.runner import run_step
from ml_pipelines.util.mlflow import log_delta_input


def run(cfg: DictConfig, task_values: TaskValues, train_path: str):
    if cfg.mlflow.log_datasets:
        log_delta_input(path=train_path, name="prepare_data.train")
    
    # scan is lazy
    train_df = pl.scan_delta(train_path)

    # in practice we should only call .collect() after we've run all our transformations since
    # polars does query optimisation similarly to spark
    X_train = train_df.drop("label").collect().to_numpy()
    y_train = train_df.select("label").collect().to_numpy().ravel()

    rf_model = RandomForestClassifier()

    rf_model.fit(X_train, y_train)

    input_example = X_train[:5].astype("float64")
    signature = infer_signature(model_input=input_example, model_output=rf_model.predict(input_example))
    mlflow.sklearn.log_model(
        rf_model,
        name="model",
        input_example=input_example,
        signature=signature,
    )

    # write train_run_id via config-defined outputs mapping
    current_run = mlflow.active_run()
    task_values.set(
        key=cfg.steps.train.outputs.train_run_id.key,
        value=current_run.info.run_id,
        task_key=cfg.steps.train.step_name,
    )

    return {"model": rf_model}


def get_step_inputs(task_values: TaskValues, cfg: DictConfig):
    train_path = task_values.get(
        key=cfg.steps.train.inputs.train_path.key,
        task_key=cfg.steps.train.inputs.train_path.source_step,
    )
    return {"train_path": train_path}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.pipeline.experiment_name)
    
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
