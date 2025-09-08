from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from omegaconf import DictConfig
import hydra

from ml_pipelines.mlflow_utils import set_experiment_from_cfg, get_task_value

def run(cfg: DictConfig):
    """Prepare data for training.

    In prod/dev this would load from Spark tables. For the local
    workflow, we generate a small random dataset that mimics the
    expected schema.
    """
    # Ensure experiment exists/selected when running as a job task
    set_experiment_from_cfg(cfg)
    parent_run_id = get_task_value("parent_run_id")
    parent_params = {"run_name": "01_prepare_data"}
    if parent_run_id:
        parent_params["tags"] = {"mlflow.parentRunId": parent_run_id}
    with mlflow.start_run(**parent_params):
        mlflow.set_tag("step", "prepare_data")
        # stub dataset: two numerical features and a binary label
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "label": np.random.randint(0, 2, 100),
            }
        )
        train_df, test_df = train_test_split(
            data,
            test_size=cfg.steps.prepare_data.test_size,
            random_state=cfg.seed,
        )
        mlflow.log_params({"train_rows": len(train_df), "test_rows": len(test_df)})
        return {"train": train_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
