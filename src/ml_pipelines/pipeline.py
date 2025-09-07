from __future__ import annotations

import uuid

import mlflow
from omegaconf import DictConfig
import hydra

from .steps import prepare_data, train, evaluate, feature_importance, model_qa


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Run the full ML pipeline locally."""
    mlflow.set_experiment(cfg.experiment.name)
    pipeline_run_id = str(uuid.uuid4())
    with mlflow.start_run(run_name="pipeline_run") as parent_run:
        mlflow.set_tags(
            {
                "pipeline_run_id": pipeline_run_id,
                "git_sha": "<GIT_SHA_PLACEHOLDER>",
                "pipeline_version": "0.1.0",
                "orchestrator_run_id": "local",
                "data_version": "stub",
            }
        )
        data = prepare_data.run(cfg)
        model_data = train.run(cfg, data)
        eval_metrics = evaluate.run(cfg, model_data, data)
        feature_importance.run(cfg, model_data, data)
        model_qa.run(cfg, model_data, data)
        mlflow.log_metrics({"test_auc": eval_metrics["test_auc"]})


if __name__ == "__main__":  # pragma: no cover
    main()
