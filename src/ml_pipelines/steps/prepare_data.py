from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from omegaconf import DictConfig
import hydra

from ml_pipelines.mlflow_utils import (
    set_experiment_from_cfg,
    resolve_parent_run_id,
    save_dataframe_as_artifact,
    set_task_value,
)

def run(cfg: DictConfig):
    """Prepare data for training (pure function wrapper for job task use)."""
    # Ensure experiment exists/selected when running as a job task
    exp_used = set_experiment_from_cfg(cfg)
    parent_run_id = resolve_parent_run_id(cfg)
    if parent_run_id:
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id
    print(f"[prepare_data] experiment={exp_used} tracking_uri={mlflow.get_tracking_uri()} parent_run_id={parent_run_id}")
    parent_params = {"run_name": "01_prepare_data"}
    if parent_run_id:
        parent_params["tags"] = {"mlflow.parentRunId": parent_run_id}
        parent_params["nested"] = True
    with mlflow.start_run(**parent_params):
        mlflow.set_tag("step", "prepare_data")
        data = _prepare_data_frame()
        train_df, test_df = train_test_split(
            data,
            test_size=cfg.steps.prepare_data.test_size,
            random_state=cfg.seed,
        )
        mlflow.log_params({"train_rows": len(train_df), "test_rows": len(test_df)})
        save_dataframe_as_artifact(train_df.reset_index(drop=True), "train.parquet", artifact_subdir="prepare_data")
        save_dataframe_as_artifact(test_df.reset_index(drop=True), "test.parquet", artifact_subdir="prepare_data")
        # Expose this run id to downstream tasks (requires job permission)
        current_run_id = mlflow.active_run().info.run_id
        print(f"[prepare_data] setting task value prepare_run_id={current_run_id}")
        set_task_value("prepare_run_id", current_run_id)
        return {"train": train_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}


def _prepare_data_frame() -> pd.DataFrame:
    """Create the synthetic dataset used by prepare_data."""
    return pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "label": np.random.randint(0, 2, 100),
        }
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
