from __future__ import annotations

import mlflow
from omegaconf import DictConfig
import hydra
import pyspark.sql.functions as F

from ml_pipelines.util.mlflow import begin_pipeline_run
from ml_pipelines.util.task_values import DatabricksTaskValues, TaskValues
from ml_pipelines.runner import run_step
from ml_pipelines.util.delta_paths import build_delta_path
from ml_pipelines.util.spark import get_spark_session


def run(cfg: DictConfig, task_values: TaskValues):
    spark = get_spark_session(cfg)

    # synthesize dataset directly in Spark
    n = 100
    df = spark.range(0, n).select(
        F.randn(seed=cfg.seed).alias("feature1"),
        F.randn(seed=cfg.seed + 1).alias("feature2"),
        (F.rand(seed=cfg.seed + 2) > 0.5).cast("int").alias("label"),
    )

    # train/test split
    test_size = float(cfg.steps.prepare_data.test_size)
    train_size = 1.0 - test_size
    train_df, test_df = df.randomSplit([train_size, test_size], seed=int(cfg.seed))

    # write to delta
    train_uri = build_delta_path(cfg, "prepare_data", "train")
    test_uri = build_delta_path(cfg, "prepare_data", "test")
    train_df.write.format("delta").mode("overwrite").option("delta.enableDeletionVectors", "false").save(train_uri)
    test_df.write.format("delta").mode("overwrite").option("delta.enableDeletionVectors", "false").save(test_uri)

    # persist URIs via config-defined outputs mapping
    task_values.set(
        key=cfg.steps.prepare_data.outputs.train_uri.key,
        value=train_uri,
        task_key=cfg.steps.prepare_data.outputs.train_uri.task_key,
    )
    task_values.set(
        key=cfg.steps.prepare_data.outputs.test_uri.key,
        value=test_uri,
        task_key=cfg.steps.prepare_data.outputs.test_uri.task_key,
    )
    return {"train_uri": train_uri, "test_uri": test_uri}



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.pipeline.experiment_name)
    
    task_values = DatabricksTaskValues()
    pipeline_run_id = begin_pipeline_run(cfg)
    task_values.set(
        key=cfg.steps.prepare_data.outputs.pipeline_run_id.key,
        value=pipeline_run_id,
        task_key=cfg.steps.prepare_data.outputs.pipeline_run_id.task_key,
    )

    run_step(
        cfg,
        step_key="prepare_data",
        task_values=task_values,
        step_func=run,
        parent_run_id=pipeline_run_id,
    )

if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
