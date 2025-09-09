from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra

from .steps import prepare_data, train, evaluate, feature_importance, model_qa
from .util.runner import run_step
from .util.task_store import LocalTaskStore
from .util.mlflow import begin_pipeline_run


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Run the full ML pipeline locally."""
    mlflow.set_experiment(cfg.experiment.name)
    
    store = LocalTaskStore()

    pipeline_run_id = begin_pipeline_run(cfg)
    with mlflow.start_run(run_id=pipeline_run_id):
        store.set(key="pipeline_run_id", value=pipeline_run_id, task_key="prepare_data")

        run_step(
            cfg,
            step_key="prepare_data",
            task_store=store,
            step_func=prepare_data.run,
            parent_run_id=pipeline_run_id,
        )

        train_inputs = train.get_step_inputs(store, cfg)
        run_step(
            cfg,
            step_key="train",
            task_store=store,
            step_func=train.run,
            parent_run_id=pipeline_run_id,
            step_inputs=train_inputs,
        )
        evaluate_inputs = evaluate.get_step_inputs(store, cfg)
        run_step(
            cfg,
            step_key="evaluate",
            task_store=store,
            step_func=evaluate.run,
            parent_run_id=pipeline_run_id,
            step_inputs=evaluate_inputs,
        )

        feature_importance_inputs = feature_importance.get_step_inputs(store, cfg)
        run_step(
            cfg,
            step_key="feature_importance",
            task_store=store,
            step_func=feature_importance.run,
            parent_run_id=pipeline_run_id,
            step_inputs=feature_importance_inputs,
        )

        model_qa_inputs = model_qa.get_step_inputs(store, cfg)
        run_step(
            cfg,
            step_key="model_qa",
            task_store=store,
            step_func=model_qa.run,
            parent_run_id=pipeline_run_id,
            step_inputs=model_qa_inputs,
        )



if __name__ == "__main__":  # pragma: no cover
    main()
