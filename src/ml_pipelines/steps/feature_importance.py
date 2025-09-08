from __future__ import annotations

from typing import Dict

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd
from sklearn.inspection import permutation_importance

from ml_pipelines.mlflow_utils import set_experiment_from_cfg, get_task_value

def run(cfg: DictConfig, model_data: Dict, data: Dict[str, pd.DataFrame]):
    set_experiment_from_cfg(cfg)
    parent_run_id = get_task_value("parent_run_id")
    start_params = {"run_name": "04_feature_importance"}
    if parent_run_id:
        start_params["tags"] = {"mlflow.parentRunId": parent_run_id}
    with mlflow.start_run(**start_params):
        mlflow.set_tag("step", "feature_importance")
        result = permutation_importance(
            model_data["model"],
            model_data["X_train"],
            model_data["y_train"],
            n_repeats=cfg.steps.feature_importance.n_repeats,
            random_state=cfg.seed,
        )
        importances = {
            col: imp for col, imp in zip(model_data["X_train"].columns, result.importances_mean)
        }
        mlflow.log_dict(importances, "feature_importance.json")
        return importances


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    from ml_pipelines.steps import prepare_data, train

    data = prepare_data.run(cfg)
    model_data = train.run(cfg, data)
    run(cfg, model_data, data)


if __name__ == "__main__":  # pragma: no cover
    main()
