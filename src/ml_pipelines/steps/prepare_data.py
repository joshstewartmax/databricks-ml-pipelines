from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from omegaconf import DictConfig
import hydra

from ml_pipelines.util.mlflow import save_dataframe_as_artifact, begin_pipeline_run, log_input_dataset
from ml_pipelines.util.task_values import DatabricksTaskValues
from ml_pipelines.util.runner import run_step


def run(cfg: DictConfig):
    data = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "label": np.random.randint(0, 2, 100),
    })

    log_input_dataset(data, name="data")

    train_df, test_df = train_test_split(
        data,
        test_size=cfg.steps.prepare_data.test_size,
        random_state=cfg.seed,
    )
    
    mlflow.log_params({"train_rows": len(train_df), "test_rows": len(test_df)})
    save_dataframe_as_artifact(
        train_df.reset_index(drop=True), 
        "train.parquet", 
        artifact_subdir="prepare_data"
    )
    save_dataframe_as_artifact(
        test_df.reset_index(drop=True), 
        "test.parquet", 
        artifact_subdir="prepare_data"
    )
    return {"train": train_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.experiment.name)
    
    task_values = DatabricksTaskValues()
    pipeline_run_id = begin_pipeline_run(cfg)
    task_values.set(key="pipeline_run_id", value=pipeline_run_id)

    run_step(
        cfg,
        step_key="prepare_data",
        task_values=task_values,
        step_func=run,
        parent_run_id=pipeline_run_id,
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
