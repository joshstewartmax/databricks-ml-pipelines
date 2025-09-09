from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra

from ml_pipelines.util.mlflow import end_pipeline_run
from ml_pipelines.util.task_store import TaskStore, DatabricksTaskStore
from ml_pipelines.util.runner import run_step


def run(cfg: DictConfig):
    mlflow.log_param("prod_model_uri", cfg.steps.model_qa.prod_model_uri)
    mlflow.log_metric("qa_placeholder", 1.0)
    return {}


def get_step_inputs(store: TaskStore, cfg: DictConfig):
    return {}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.experiment.name)

    store = DatabricksTaskStore()
    pipeline_run_id = store.get(key="pipeline_run_id", task_key="prepare_data")
    
    step_inputs = get_step_inputs(store, cfg)
    run_step(
        cfg,
        step_key="model_qa",
        task_store=store,
        step_func=run,
        parent_run_id=pipeline_run_id,
        step_inputs=step_inputs,
    )
    end_pipeline_run(pipeline_run_id, status="FINISHED")


if __name__ == "__main__":  # pragma: no cover
    main()
