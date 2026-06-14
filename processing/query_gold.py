"""Batch-read the Silver and Gold tables to confirm processing + exceptions.

Run:
    python processing/query_gold.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from pyspark.sql import SparkSession
from pyspark.sql.functions import count, countDistinct, col


def build_spark():
    return (
        SparkSession.builder
        .appName("query-gold")
        .config("spark.jars.packages", config.DELTA_PKG)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


def safe_read(spark, path, label):
    try:
        return spark.read.format("delta").load(path)
    except Exception as e:
        print(f"[!] could not read {label} at {path}: {e}")
        return None


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    silver = safe_read(spark, config.SILVER_PATH, "Silver")
    if silver is not None:
        print("\n=== Silver summary ===")
        silver.select(
            count("*").alias("rows"),
            countDistinct("shipment_id").alias("shipments"),
        ).show(truncate=False)
        print("=== Silver sample (with geo enrichment) ===")
        (silver.select("shipment_id", "route_id", "progress_pct", "is_moving",
                       "dist_to_route_km", "at_facility", "event_time")
               .orderBy(col("event_time").desc()).show(10, truncate=False))

    status = safe_read(spark, config.GOLD_STATUS_PATH, "Gold/status")
    if status is not None:
        print("=== Gold STATUS events by type ===")
        status.groupBy("event_type").agg(count("*").alias("events")).show(truncate=False)
        print("=== Sample status events ===")
        (status.select("shipment_id", "event_type", "detail", "lat", "lon", "event_time")
               .orderBy(col("event_time").desc()).show(10, truncate=False))

    exc = safe_read(spark, config.GOLD_EXCEPTIONS_PATH, "Gold/exceptions")
    if exc is not None:
        print("=== Gold EXCEPTIONS by type ===")
        exc.groupBy("event_type").agg(count("*").alias("events")).show(truncate=False)
        print("=== Sample exceptions (windowed) ===")
        (exc.select("shipment_id", "event_type", "window_start", "window_end",
                    "moved_km", "max_speed", "at_facility")
            .orderBy(col("window_start").desc()).show(10, truncate=False))

    spark.stop()


if __name__ == "__main__":
    main()
