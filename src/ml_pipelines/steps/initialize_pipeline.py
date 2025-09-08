from __future__ import annotations

import os

from omegaconf import DictConfig
import hydra
from pyspark.sql import SparkSession  # type: ignore
from pyspark.dbutils import DBUtils  # type: ignore

from ml_pipelines.mlflow_utils import (
    set_experiment_from_cfg,
    create_parent_run,
    get_databricks_run_identifiers,
    generate_pipeline_run_id,
)


def init_mlflow_experiment_and_run_config(cfg: DictConfig):
    exp_used = set_experiment_from_cfg(cfg)
    print(f"[initialize:init] Experiment selected: {exp_used}")
    return exp_used


def initialize_pipeline(cfg: DictConfig, exp_used: str):
    identifiers = get_databricks_run_identifiers()
    pipeline_run_id = generate_pipeline_run_id()
    tags = {
        "pipeline_run_id": pipeline_run_id,
        "git_sha": os.environ.get("GIT_SHA", "<GIT_SHA_PLACEHOLDER>"),
        "pipeline_version": os.environ.get("PIPELINE_VERSION", "0.1.0"),
        "orchestrator_run_id": identifiers.get("job_run_id") or "databricks_job",
        "data_version": os.environ.get("DATA_VERSION", "stub"),
    }
    parent_run_id = create_parent_run(cfg, tags=tags)
    print(f"[initialize] Created parent run_id: {parent_run_id}")
    dbutils = DBUtils(SparkSession.getActiveSession() or SparkSession.builder.getOrCreate())
    dbutils.jobs.taskValues.set(key="parent_run_id", value=parent_run_id)
    dbutils.jobs.taskValues.set(key="pipeline_run_id", value=pipeline_run_id)
    print("[initialize] Stored task values: parent_run_id and pipeline_run_id")
    os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id
    os.environ["PIPELINE_RUN_ID"] = pipeline_run_id


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    exp_used = init_mlflow_experiment_and_run_config(cfg)
    initialize_pipeline(cfg, exp_used)


if __name__ == "__main__":  # pragma: no cover
    main()


