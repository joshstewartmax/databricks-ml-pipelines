from __future__ import annotations


import logging
import mlflow
from hydra import initialize, compose

from ml_pipelines.steps import prepare_data, train, evaluate, feature_importance, model_qa
from ml_pipelines.runner import run_step
from ml_pipelines.util.task_values import LocalTaskValues
from ml_pipelines.util.mlflow import begin_pipeline_run


logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

def main():
    """Run the full ML pipeline locally."""
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config", overrides=["pipeline=local"])

    mlflow.set_experiment(cfg.pipeline.experiment_name)
    logger.info("Starting local ML pipeline: %s", cfg.pipeline.experiment_name)
    
    # this replicates the functionality of Databricks task values
    task_values = LocalTaskValues()

    pipeline_run_id = begin_pipeline_run(cfg)
    with mlflow.start_run(run_id=pipeline_run_id):
        task_values.set(
            key=cfg.steps.prepare_data.outputs.pipeline_run_id.key,
            value=pipeline_run_id,
            task_key=cfg.steps.prepare_data.outputs.pipeline_run_id.task_key,
        )

        logger.info("Starting step: prepare_data")
        run_step(
            cfg,
            step_key="prepare_data",
            task_values=task_values,
            step_func=prepare_data.run,
            parent_run_id=pipeline_run_id,
        )
        logger.info("Finished step: prepare_data")

        train_inputs = train.get_step_inputs(task_values, cfg)
        logger.info("Starting step: train")
        run_step(
            cfg,
            step_key="train",
            task_values=task_values,
            step_func=train.run,
            parent_run_id=pipeline_run_id,
            step_inputs=train_inputs,
        )
        logger.info("Finished step: train")
        evaluate_inputs = evaluate.get_step_inputs(task_values, cfg)
        logger.info("Starting step: evaluate")
        run_step(
            cfg,
            step_key="evaluate",
            task_values=task_values,
            step_func=evaluate.run,
            parent_run_id=pipeline_run_id,
            step_inputs=evaluate_inputs,
        )
        logger.info("Finished step: evaluate")

        feature_importance_inputs = feature_importance.get_step_inputs(task_values, cfg)
        logger.info("Starting step: feature_importance")
        run_step(
            cfg,
            step_key="feature_importance",
            task_values=task_values,
            step_func=feature_importance.run,
            parent_run_id=pipeline_run_id,
            step_inputs=feature_importance_inputs,
        )
        logger.info("Finished step: feature_importance")

        model_qa_inputs = model_qa.get_step_inputs(task_values, cfg)
        logger.info("Starting step: model_qa")
        run_step(
            cfg,
            step_key="model_qa",
            task_values=task_values,
            step_func=model_qa.run,
            parent_run_id=pipeline_run_id,
            step_inputs=model_qa_inputs,
        )
        logger.info("Finished step: model_qa")

    logger.info("Local ML pipeline finished: %s", cfg.pipeline.experiment_name)



if __name__ == "__main__":  # pragma: no cover
    main()
