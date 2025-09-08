from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from omegaconf import DictConfig
import hydra
from pyspark.sql import SparkSession  # type: ignore
from pyspark.dbutils import DBUtils  # type: ignore

from ml_pipelines.mlflow_utils import (
    set_experiment_from_cfg,
    resolve_parent_run_id,
    save_dataframe_as_artifact,
)

def init_mlflow_experiment_and_run_config(cfg: DictConfig):
    exp_used = set_experiment_from_cfg(cfg)
    parent_run_id = resolve_parent_run_id(cfg)
    if parent_run_id:
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id
    print(
        f"[prepare_data:init] experiment={exp_used} tracking_uri={mlflow.get_tracking_uri()} parent_run_id={parent_run_id}"
    )
    return exp_used, parent_run_id


def prepare_data(cfg: DictConfig, exp_used: str, parent_run_id: str | None):
    parent_params = {"run_name": "01_prepare_data"}
    if parent_run_id:
        parent_params["tags"] = {"mlflow.parentRunId": parent_run_id}
        parent_params["nested"] = True
    with mlflow.start_run(**parent_params):
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
        print(f"[prepare_data] setting task value prepare_run_id={current_run_id}")
        dbutils = DBUtils(SparkSession.getActiveSession() or SparkSession.builder.getOrCreate())
        dbutils.jobs.taskValues.set(key="prepare_run_id", value=current_run_id)
        return {"train": train_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    exp_used, parent_run_id = init_mlflow_experiment_and_run_config(cfg)
    prepare_data(cfg, exp_used, parent_run_id)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
