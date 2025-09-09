from __future__ import annotations

import os
import uuid

from omegaconf import DictConfig
import hydra
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict

from ml_pipelines.util.databricks import get_databricks_run_identifiers, get_dbutils


dbutils = get_dbutils()

def initialize_pipeline(cfg: DictConfig):
    mlflow.set_experiment(cfg.experiment.name)
    identifiers = get_databricks_run_identifiers()
    pipeline_run_id = str(uuid.uuid4())
    tags = {
        "pipeline_run_id": pipeline_run_id,
        "orchestrator_run_id": identifiers.get("job_run_id") or "databricks_job",
    }
    exp = mlflow.get_experiment_by_name(cfg.experiment.name)
    
    client = MlflowClient()
    all_tags: Dict[str, str] = {}
    if tags:
        all_tags.update(tags)

    # Provide a readable run name
    run = client.create_run(experiment_id=exp.experiment_id, run_name="pipeline_run", tags=all_tags)
    dbutils.jobs.taskValues.set(key="parent_run_id", value=run.info.run_id)
    dbutils.jobs.taskValues.set(key="pipeline_run_id", value=pipeline_run_id)
    os.environ["MLFLOW_PARENT_RUN_ID"] = run.info.run_id
    os.environ["PIPELINE_RUN_ID"] = pipeline_run_id


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    initialize_pipeline(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()


