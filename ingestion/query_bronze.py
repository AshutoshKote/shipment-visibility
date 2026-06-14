"""Batch-read the Bronze Delta table to confirm pings landed.

Run:
    python ingestion/query_bronze.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from pyspark.sql import SparkSession
from pyspark.sql.functions import count, countDistinct, min as smin, max as smax


def main():
    spark = (
        SparkSession.builder
        .appName("query-bronze")
        .config("spark.jars.packages", config.DELTA_PKG)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.format("delta").load(config.BRONZE_PATH)

    print("\n=== Bronze summary ===")
    df.select(
        count("*").alias("total_pings"),
        countDistinct("shipment_id").alias("shipments"),
        countDistinct("route_id").alias("routes"),
        smin("event_time").alias("earliest_event"),
        smax("event_time").alias("latest_event"),
    ).show(truncate=False)

    print("=== Pings per route ===")
    df.groupBy("route_id").agg(count("*").alias("pings")).orderBy("route_id").show(truncate=False)

    print("=== Sample rows ===")
    (df.select("shipment_id", "route_id", "lat", "lon", "speed_kmh",
               "seq", "event_time", "kafka_partition", "ingest_time")
        .orderBy("ingest_time")
        .show(10, truncate=False))

    spark.stop()


if __name__ == "__main__":
    main()
