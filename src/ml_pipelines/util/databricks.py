from pyspark.sql import SparkSession

# this will only be available in databricks runtime
from pyspark.dbutils import DBUtils  # type: ignore


def get_dbutils():
    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    return DBUtils(spark)