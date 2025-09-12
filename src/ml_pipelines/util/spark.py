from __future__ import annotations

from omegaconf import DictConfig
from pyspark.sql import SparkSession


def get_spark_session(cfg: DictConfig) -> SparkSession:
    """Create or retrieve a SparkSession.

    For local runs, configures Spark with Delta Lake via delta-spark. For
    non-local environments, returns the active session or creates one.

    Parameters
    ----------
    cfg: DictConfig
        Hydra/OmegaConf config expected to contain `pipeline.env_name`.

    Returns
    -------
    SparkSession
    """
    env_name = getattr(cfg.pipeline, "env_name", "local")

    if env_name == "local":
        # Configure Spark with Delta Lake for local runs
        from delta import configure_spark_with_delta_pip  # type: ignore

        builder = SparkSession.builder.appName("local-spark")
        builder = (
            builder
            .master("local[*]")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        return spark

    # On Databricks or other environments, use the active session or create one
    return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()


