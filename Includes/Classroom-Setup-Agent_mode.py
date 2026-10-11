# Databricks notebook source
# This notebook performs feature engineering on tables created in Classroom-Setup-1.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

# %run ./Classroom-Setup-1

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.set_printoptions(precision=2)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, when

from databricks.feature_engineering import FeatureEngineeringClient

# COMMAND ----------

def fqtn(catalog: str, schema: str, table: str) -> str:
    return f"{catalog}.{schema}.{table}"

def _table_exists(fq_table_name: str) -> bool:
    # Prefer helper if your _common defines it
    try:
        return uc_table_exists(fq_table_name)  # noqa: F821 (defined in _common)
    except Exception:
        # Fallback to Spark catalog
        try:
            return spark.catalog.tableExists(fq_table_name)
        except Exception:
            # Extra fallback for some workspaces
            parts = fq_table_name.split(".")
            if len(parts) == 3:
                cat, sch, tbl = parts
                return spark._jsparkSession.catalog().tableExists(f"{cat}.{sch}.{tbl}")
            return False

# COMMAND ----------

@DBAcademyHelper.add_init
def create_demo_table(self):
    my_catalog = DA.catalog_name
    my_schema  = DA.schema_name
    table_name = "wine_quality_table"

    spark.sql(f"USE CATALOG {my_catalog}")
    spark.sql(f"USE SCHEMA {my_schema}")

    fq_table = fqtn(my_catalog, my_schema, table_name)

    if _table_exists(fq_table):
        print(f"Delta table already exists: {fq_table}")
        return

    data_path = f"{DA.paths.datasets.wine_quality}/data"

    df = (
        spark.read.format("delta")
        .load(data_path)
        .withColumn("wine_id", F.monotonically_increasing_id().cast("long"))
        .select(
            "wine_id",
            "fixed_acidity",
            "volatile_acidity",
            "citric_acid",
            "residual_sugar",
            "chlorides",
            "free_sulfur_dioxide",
            "total_sulfur_dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
            "quality",
        )
    )

    df.write.format("delta").mode("overwrite").saveAsTable(fq_table)
    print(f"Created Delta table: {fq_table}")

# COMMAND ----------

@DBAcademyHelper.add_init
def create_feature_table(self):
    my_catalog = DA.catalog_name
    my_schema  = DA.schema_name

    source_table  = fqtn(my_catalog, my_schema, "wine_quality_table")
    feature_table = fqtn(my_catalog, my_schema, "wine_quality_features")

    spark.sql(f"USE CATALOG {my_catalog}")
    spark.sql(f"USE SCHEMA {my_schema}")

    fs = FeatureEngineeringClient()

    # If a table exists, verify it's a valid feature table; if not valid, drop it.
    if _table_exists(feature_table):
        try:
            _ = fs.read_table(name=feature_table)
            print(f"Valid feature table already exists: {feature_table}")
            return
        except Exception as e:
            msg = str(e).lower()
            # Most common issue: missing PK constraint -> not a valid feature table
            print(
                f"A table named `{feature_table}` exists but cannot be read as a feature table.\n"
                f"Reason: {e}\n"
                f"Dropping and recreating it..."
            )
            spark.sql(f"DROP TABLE {feature_table}")

    print(f"Creating feature table: {feature_table}")

    df = spark.table(source_table)

    # Quantiles for pH category (0.0 = exact; OK for this small dataset)
    q1, q3 = df.approxQuantile("pH", [0.25, 0.75], 0.0)

    features_df = (
        df.withColumn(
            "pHCategory",
            when(col("pH") <= F.lit(q1), F.lit("Low"))
            .when((col("pH") > F.lit(q1)) & (col("pH") < F.lit(q3)), F.lit("Average"))
            .otherwise(F.lit("High")),
        )
        .select(
            "wine_id",
            "fixed_acidity",
            "volatile_acidity",
            "citric_acid",
            "pH",
            "sulphates",
            "alcohol",
            "pHCategory",
            "quality",
        )
    )

    fs.create_table(
        name=feature_table,
        primary_keys=["wine_id"],
        df=features_df,
        description="Wine quality feature table for demos",
        tags={"source": "wine_quality_table", "format": "delta"},
    )

    # Confirm feature table is usable
    _ = fs.read_table(name=feature_table)
    print(f"Feature table created and verified: {feature_table}")

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()