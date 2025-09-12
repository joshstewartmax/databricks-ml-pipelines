from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra

from ml_pipelines.util.mlflow import end_pipeline_run
from ml_pipelines.util.task_values import TaskValues, DatabricksTaskValues
from ml_pipelines.runner import run_step


def run(cfg: DictConfig, task_values: TaskValues):
    mlflow.log_param("prod_model_uri", cfg.steps.model_qa.prod_model_uri)
    mlflow.log_metric("qa_placeholder", 1.0)
    task_values.set(
        key=cfg.steps.model_qa.outputs.qa_complete.key,
        value=True,
        task_key=cfg.steps.model_qa.outputs.qa_complete.task_key,
    )
    return {}


def get_step_inputs(task_values: TaskValues, cfg: DictConfig):
    return {}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.pipeline.experiment_name)

    task_values = DatabricksTaskValues()
    pipeline_run_id = task_values.get(key="pipeline_run_id", task_key="prepare_data")
    
    step_inputs = get_step_inputs(task_values, cfg)
    run_step(
        cfg,
        step_key="model_qa",
        task_values=task_values,
        step_func=run,
        parent_run_id=pipeline_run_id,
        step_inputs=step_inputs,
    )
    end_pipeline_run(pipeline_run_id, status="FINISHED")


if __name__ == "__main__":  # pragma: no cover
    main()
