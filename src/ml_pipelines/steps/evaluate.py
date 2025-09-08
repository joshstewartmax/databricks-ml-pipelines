from __future__ import annotations

from typing import Dict

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

from ml_pipelines.mlflow_utils import set_experiment_from_cfg, get_task_value

def run(cfg: DictConfig, model_data: Dict, data: Dict[str, pd.DataFrame]):
    set_experiment_from_cfg(cfg)
    parent_run_id = get_task_value("parent_run_id")
    start_params = {"run_name": "03_evaluate"}
    if parent_run_id:
        start_params["tags"] = {"mlflow.parentRunId": parent_run_id}
    with mlflow.start_run(**start_params):
        mlflow.set_tag("step", "evaluate")
        test_df = data["test"]
        X_test = test_df.drop("label", axis=1)
        y_test = test_df["label"]
        probs = model_data["model"].predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        mlflow.log_metric("test_auc", auc)
        disp = RocCurveDisplay.from_predictions(y_test, probs)
        fig = disp.figure_
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)
        return {"test_auc": auc}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    from ml_pipelines.steps import prepare_data, train

    data = prepare_data.run(cfg)
    model_data = train.run(cfg, data)
    run(cfg, model_data, data)


if __name__ == "__main__":  # pragma: no cover
    main()
