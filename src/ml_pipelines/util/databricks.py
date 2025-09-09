# these will be available in databricks runtime
from pyspark.sql import SparkSession  # type: ignore
from pyspark.dbutils import DBUtils  # type: ignore


def get_dbutils():
    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    return DBUtils(spark)