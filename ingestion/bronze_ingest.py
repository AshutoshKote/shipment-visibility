"""Bronze ingestion: Kafka -> Spark Structured Streaming -> Delta (raw).

Reads the raw GPS ping stream, parses the JSON payload, attaches Kafka
metadata (partition/offset) and an ingest timestamp, and appends to a
Delta table. Bronze stays faithful to the source: no dedup, no filtering.
Watermarking and aggregation come in the Silver/Gold layers (Day 2).

Run:
    python ingestion/bronze_ingest.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, current_timestamp
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, LongType,
)

PING_SCHEMA = StructType([
    StructField("shipment_id", StringType()),
    StructField("vehicle_id", StringType()),
    StructField("route_id", StringType()),
    StructField("lat", DoubleType()),
    StructField("lon", DoubleType()),
    StructField("speed_kmh", DoubleType()),
    StructField("heading", DoubleType()),
    StructField("dist_km", DoubleType()),
    StructField("total_km", DoubleType()),
    StructField("seq", LongType()),
    StructField("event_time", StringType()),
])


def build_spark():
    return (
        SparkSession.builder
        .appName("bronze-ingest")
        .config("spark.jars.packages", f"{config.SPARK_KAFKA_PKG},{config.DELTA_PKG}")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP)
        .option("subscribe", config.TOPIC_PINGS)
        .option("startingOffsets", "earliest")
        .load()
    )

    bronze = (
        raw.select(
            col("key").cast("string").alias("kafka_key"),
            from_json(col("value").cast("string"), PING_SCHEMA).alias("d"),
            col("partition").alias("kafka_partition"),
            col("offset").alias("kafka_offset"),
            col("timestamp").alias("kafka_ts"),
        )
        .select("kafka_key", "d.*", "kafka_partition", "kafka_offset", "kafka_ts")
        .withColumn("event_time", to_timestamp(col("event_time")))
        .withColumn("ingest_time", current_timestamp())
    )

    query = (
        bronze.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", config.BRONZE_CHECKPOINT)
        .option("mergeSchema", "true")
        .start(config.BRONZE_PATH)
    )

    print(f"[bronze] streaming to {config.BRONZE_PATH}  (Ctrl-C to stop)")
    query.awaitTermination()


if __name__ == "__main__":
    main()
