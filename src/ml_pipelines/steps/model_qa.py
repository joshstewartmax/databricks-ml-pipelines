from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra

from ml_pipelines.util.databricks import get_dbutils


dbutils = get_dbutils()

def model_qa(cfg: DictConfig, parent_run_id: str):
    with mlflow.start_run(
        run_name=cfg.steps.model_qa.step_name,
        nested=True,
        parent_run_id=parent_run_id,
    ):
        mlflow.set_tag("step", "model_qa")
        mlflow.log_param("prod_model_uri", cfg.steps.model_qa.prod_model_uri)
        mlflow.log_metric("qa_placeholder", 1.0)
        return {}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.experiment.name)
    parent_run_id = dbutils.jobs.taskValues.get(key="parent_run_id", taskKey="initialize")
    model_qa(cfg, parent_run_id)


if __name__ == "__main__":  # pragma: no cover
    main()
