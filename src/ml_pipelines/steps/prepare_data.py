from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from omegaconf import DictConfig
import hydra

from ml_pipelines.util.databricks import get_dbutils
from ml_pipelines.util.mlflow import save_dataframe_as_artifact


dbutils = get_dbutils()

def prepare_data(cfg: DictConfig, parent_run_id: str):
    with mlflow.start_run(
        run_name=cfg.steps.prepare_data.step_name,
        nested=True,
        parent_run_id=parent_run_id,
    ):
        mlflow.set_tag("step", "prepare_data")
        data = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "label": np.random.randint(0, 2, 100),
        })

        train_df, test_df = train_test_split(
            data,
            test_size=cfg.steps.prepare_data.test_size,
            random_state=cfg.seed,
        )
        mlflow.log_params({"train_rows": len(train_df), "test_rows": len(test_df)})
        save_dataframe_as_artifact(train_df.reset_index(drop=True), "train.parquet", artifact_subdir="prepare_data")
        save_dataframe_as_artifact(test_df.reset_index(drop=True), "test.parquet", artifact_subdir="prepare_data")
        current_run_id = mlflow.active_run().info.run_id
        dbutils.jobs.taskValues.set(key="prepare_run_id", value=current_run_id)
        return {"train": train_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.experiment.name)
    parent_run_id = dbutils.jobs.taskValues.get(key="parent_run_id", taskKey="initialize")
    prepare_data(cfg, parent_run_id)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
