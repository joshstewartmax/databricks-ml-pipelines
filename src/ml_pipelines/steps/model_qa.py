from __future__ import annotations

from typing import Dict

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd

from ml_pipelines.mlflow_utils import set_experiment_from_cfg, get_task_value

def run(cfg: DictConfig, model_data: Dict, data: Dict[str, pd.DataFrame]):
    set_experiment_from_cfg(cfg)
    parent_run_id = get_task_value("parent_run_id")
    start_params = {"run_name": "05_model_qa"}
    if parent_run_id:
        start_params["tags"] = {"mlflow.parentRunId": parent_run_id}
    with mlflow.start_run(**start_params):
        mlflow.set_tag("step", "model_qa")
        mlflow.log_param("prod_model_uri", cfg.steps.model_qa.prod_model_uri)
        # Placeholder: here you would load the production model from
        # the MLflow Model Registry and compare metrics.
        # For local execution, we simply log that this step executed.
        mlflow.log_metric("qa_placeholder", 1.0)
        return {}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    from ml_pipelines.steps import prepare_data, train

    data = prepare_data.run(cfg)
    model_data = train.run(cfg, data)
    run(cfg, model_data, data)


if __name__ == "__main__":  # pragma: no cover
    main()
