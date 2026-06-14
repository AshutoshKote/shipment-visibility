"""Gold layer: Silver -> exception events -> Delta.

Two streaming queries in one Spark session:

1) STATUS (per-ping, stateless) -> gold/status
   - OFF_ROUTE   : distance from planned route > OFF_ROUTE_KM
   - AT_FACILITY : ping falls inside a facility geofence

2) EXCEPTIONS (windowed, stateful) -> gold/exceptions
   - STOPPED / DWELL : within a STOP_WINDOW the shipment barely moved and
     speed stayed low. DWELL if it happened at a facility, else STOPPED.
   This query uses a watermark + tumbling window, so it demonstrates how
   late / out-of-order pings are handled before a window is finalised.

Run:
    python processing/gold_exceptions.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, when, window, max as smax, min as smin, first as sfirst,
)


def build_spark():
    return (
        SparkSession.builder
        .appName("gold-exceptions")
        .master("local[2]")
        .config("spark.jars.packages", config.DELTA_PKG)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def status_query(spark):
    """Per-ping status events: OFF_ROUTE and AT_FACILITY."""
    silver = spark.readStream.format("delta").load(config.SILVER_PATH)

    off_route = (
        silver.where(col("dist_to_route_km") > lit(config.OFF_ROUTE_KM))
        .withColumn("event_type", lit("OFF_ROUTE"))
        .withColumn("detail", col("dist_to_route_km").cast("string"))
    )
    at_facility = (
        silver.where(col("at_facility").isNotNull())
        .withColumn("event_type", lit("AT_FACILITY"))
        .withColumn("detail", col("at_facility"))
    )

    cols = ["shipment_id", "route_id", "lat", "lon", "speed_kmh",
            "event_time", "event_type", "detail"]
    events = off_route.select(*cols).unionByName(at_facility.select(*cols))

    return (
        events.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", config.GOLD_STATUS_CHECKPOINT)
        .option("mergeSchema", "true")
        .start(config.GOLD_STATUS_PATH)
    )


def exceptions_query(spark):
    """Windowed STOPPED / DWELL detection with watermark."""
    silver = spark.readStream.format("delta").load(config.SILVER_PATH)

    windowed = (
        silver
        .withWatermark("event_time", config.WATERMARK)
        .groupBy(col("shipment_id"), window(col("event_time"), config.STOP_WINDOW))
        .agg(
            smax("speed_kmh").alias("max_speed"),
            smax("dist_km").alias("max_dist"),
            smin("dist_km").alias("min_dist"),
            smax("at_facility").alias("at_facility"),
            sfirst("route_id").alias("route_id"),
        )
        .withColumn("moved_km", col("max_dist") - col("min_dist"))
        .where((col("moved_km") < lit(config.STOP_MOVED_KM))
               & (col("max_speed") < lit(config.STOP_MAX_SPEED)))
        .withColumn("event_type",
                    when(col("at_facility").isNotNull(), lit("DWELL")).otherwise(lit("STOPPED")))
        .select(
            col("shipment_id"), col("route_id"),
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("max_speed"), col("moved_km"), col("at_facility"), col("event_type"),
        )
    )

    return (
        windowed.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", config.GOLD_EXCEPTIONS_CHECKPOINT)
        .option("mergeSchema", "true")
        .start(config.GOLD_EXCEPTIONS_PATH)
    )


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    status_query(spark)
    exceptions_query(spark)

    print(f"[gold] status -> {config.GOLD_STATUS_PATH}")
    print(f"[gold] exceptions -> {config.GOLD_EXCEPTIONS_PATH}  (Ctrl-C to stop)")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
