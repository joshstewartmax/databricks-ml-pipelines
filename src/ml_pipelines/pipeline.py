from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra

from ml_pipelines.steps import prepare_data, train, evaluate, feature_importance, model_qa
from ml_pipelines.util.runner import run_step
from ml_pipelines.util.task_values import LocalTaskValues
from ml_pipelines.util.mlflow import begin_pipeline_run


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Run the full ML pipeline locally."""
    mlflow.set_experiment(cfg.pipeline.experiment_name)
    
    # this replicates the functionality of Databricks task values
    task_values = LocalTaskValues()

    pipeline_run_id = begin_pipeline_run(cfg)
    with mlflow.start_run(run_id=pipeline_run_id):
        task_values.set(
            key=cfg.steps.prepare_data.outputs.pipeline_run_id.key,
            value=pipeline_run_id,
            task_key=cfg.steps.prepare_data.outputs.pipeline_run_id.task_key,
        )

        run_step(
            cfg,
            step_key="prepare_data",
            task_values=task_values,
            step_func=prepare_data.run,
            parent_run_id=pipeline_run_id,
        )

        train_inputs = train.get_step_inputs(task_values, cfg)
        run_step(
            cfg,
            step_key="train",
            task_values=task_values,
            step_func=train.run,
            parent_run_id=pipeline_run_id,
            step_inputs=train_inputs,
        )
        evaluate_inputs = evaluate.get_step_inputs(task_values, cfg)
        run_step(
            cfg,
            step_key="evaluate",
            task_values=task_values,
            step_func=evaluate.run,
            parent_run_id=pipeline_run_id,
            step_inputs=evaluate_inputs,
        )

        feature_importance_inputs = feature_importance.get_step_inputs(task_values, cfg)
        run_step(
            cfg,
            step_key="feature_importance",
            task_values=task_values,
            step_func=feature_importance.run,
            parent_run_id=pipeline_run_id,
            step_inputs=feature_importance_inputs,
        )

        model_qa_inputs = model_qa.get_step_inputs(task_values, cfg)
        run_step(
            cfg,
            step_key="model_qa",
            task_values=task_values,
            step_func=model_qa.run,
            parent_run_id=pipeline_run_id,
            step_inputs=model_qa_inputs,
        )



if __name__ == "__main__":  # pragma: no cover
    main()
