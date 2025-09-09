from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra

from ml_pipelines.util.databricks import get_dbutils
from ml_pipelines.util.mlflow import end_pipeline_run


dbutils = get_dbutils()

def model_qa(cfg: DictConfig, pipeline_run_id: str):
    with mlflow.start_run(
        run_name=cfg.steps.model_qa.step_name,
        nested=True,
        parent_run_id=pipeline_run_id,
    ):
        mlflow.set_tag("step", "model_qa")
        mlflow.log_param("prod_model_uri", cfg.steps.model_qa.prod_model_uri)
        mlflow.log_metric("qa_placeholder", 1.0)
        return {}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.experiment.name)
    pipeline_run_id = dbutils.jobs.taskValues.get(key="pipeline_run_id", taskKey="prepare_data")
    model_qa(cfg, pipeline_run_id)
    end_pipeline_run(pipeline_run_id, status="FINISHED")


if __name__ == "__main__":  # pragma: no cover
    main()
