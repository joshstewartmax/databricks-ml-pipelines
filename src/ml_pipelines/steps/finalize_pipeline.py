from __future__ import annotations

import os

from omegaconf import DictConfig
import hydra
from pyspark.sql import SparkSession  # type: ignore
from pyspark.dbutils import DBUtils  # type: ignore

from ml_pipelines.mlflow_utils import terminate_run, resolve_parent_run_id


def init_mlflow_experiment_and_run_config(cfg: DictConfig):  # noqa: ARG001 - nothing to init
    return None


def load_previous_artifacts_for_finalize_pipeline(cfg: DictConfig):
    parent_run_id = resolve_parent_run_id(cfg)
    print(f"[finalize:load] resolved parent_run_id={parent_run_id}")
    return parent_run_id


def finalize_pipeline(cfg: DictConfig, parent_run_id: str | None):
    if parent_run_id:
        terminate_run(parent_run_id)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):  # noqa: ARG001 - cfg reserved for future use
    _ = init_mlflow_experiment_and_run_config(cfg)
    parent_run_id = load_previous_artifacts_for_finalize_pipeline(cfg)
    finalize_pipeline(cfg, parent_run_id)


if __name__ == "__main__":  # pragma: no cover
    main()


