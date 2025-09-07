from __future__ import annotations

from typing import Dict

import mlflow
from omegaconf import DictConfig
import hydra
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def run(cfg: DictConfig, data: Dict[str, pd.DataFrame]):
    with mlflow.start_run(run_name="02_train", nested=True):
        mlflow.set_tag("step", "train")
        train_df = data["train"]
        X = train_df.drop("label", axis=1)
        y = train_df["label"]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=cfg.steps.train.val_size, random_state=cfg.seed
        )
        param_grid = {"n_estimators": [50, 100], "max_depth": [None, 5]}
        search = GridSearchCV(
            RandomForestClassifier(random_state=cfg.seed),
            param_grid,
            cv=3,
            n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        best_model = search.best_estimator_
        val_probs = best_model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_params(search.best_params_)
        # Ensure float inputs to avoid integer column warnings in schema
        input_example = X_tr.head(5).astype("float64")
        signature = infer_signature(model_input=input_example, model_output=best_model.predict(input_example))


        mlflow.sklearn.log_model(
            best_model,
            name="model",
            input_example=input_example,
            signature=signature,
        )
        return {"model": best_model, "X_train": X_tr, "y_train": y_tr}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    from . import prepare_data

    data = prepare_data.run(cfg)
    run(cfg, data)


if __name__ == "__main__":  # pragma: no cover
    main()
