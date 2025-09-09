import os
from typing import Dict, Optional

# these will be available in databricks runtime
from pyspark.sql import SparkSession  # type: ignore
from pyspark.dbutils import DBUtils  # type: ignore


def get_dbutils():
    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    return DBUtils(spark)

def get_databricks_run_identifiers() -> Dict[str, Optional[str]]:
    """Collect Databricks job/task identifiers from env and context when available."""
    identifiers: Dict[str, Optional[str]] = {
        "job_id": os.environ.get("DATABRICKS_JOB_ID"),
        "job_run_id": os.environ.get("DATABRICKS_JOB_RUN_ID"),
        "task_run_id": os.environ.get("DATABRICKS_TASK_RUN_ID"),
        "cluster_id": os.environ.get("DATABRICKS_CLUSTER_ID"),
    }
    try:
        dbutils = get_dbutils()
        if dbutils is not None:
            # taskContext APIs may not be available in all runtimes; guard carefully
            ctx = dbutils.jobs.taskContext()
            identifiers.setdefault("job_run_id", getattr(ctx, "runId", lambda: None)())
            identifiers.setdefault("task_run_id", getattr(ctx, "taskRunId", lambda: None)())
    except Exception:
        pass
    return identifiers