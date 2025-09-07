from __future__ import annotations

from typing import Dict

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd
from sklearn.inspection import permutation_importance


def run(cfg: DictConfig, model_data: Dict, data: Dict[str, pd.DataFrame]):
    with mlflow.start_run(run_name="04_feature_importance", nested=True):
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
    from . import prepare_data, train

    data = prepare_data.run(cfg)
    model_data = train.run(cfg, data)
    run(cfg, model_data, data)


if __name__ == "__main__":  # pragma: no cover
    main()
