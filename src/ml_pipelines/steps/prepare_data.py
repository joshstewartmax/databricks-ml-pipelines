# Databricks notebook source
from __future__ import annotations

# COMMAND ----------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from omegaconf import DictConfig
import hydra

# COMMAND ----------
def run(cfg: DictConfig):
    """Prepare data for training.

    In prod/dev this would load from Spark tables. For the local
    workflow, we generate a small random dataset that mimics the
    expected schema.
    """
    with mlflow.start_run(run_name="01_prepare_data", nested=True) as run:
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

# COMMAND ----------
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run(cfg)

# COMMAND ----------
if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
