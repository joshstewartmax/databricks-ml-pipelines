from __future__ import annotations

from omegaconf import DictConfig
from mlflow import MlflowClient
import hydra

from ml_pipelines.util.databricks import get_dbutils


dbutils = get_dbutils()

def finalize_pipeline(cfg: DictConfig):  # noqa: ARG001 - cfg reserved for future use
    parent_run_id = dbutils.jobs.taskValues.get(key="parent_run_id", taskKey="initialize")
    
    client = MlflowClient()
    client.set_terminated(run_id=parent_run_id, status="FINISHED")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):  # noqa: ARG001 - cfg reserved for future use
    finalize_pipeline(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()


