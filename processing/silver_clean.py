"""Silver layer: Bronze -> cleaned, deduped, enriched stream -> Delta.

- validates lat/lon/ids
- applies a watermark and drops duplicate (shipment_id, seq) pings
  (handles the late / re-delivered pings the simulator injects)
- enriches each ping with progress %, movement flag, distance-to-route,
  and nearest-facility (geofence) -- so Gold can just threshold these

Run:
    python processing/silver_clean.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from processing import geo

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, lit, round as sround
from pyspark.sql.types import DoubleType, StringType

dist_to_route_udf = udf(lambda lat, lon, rid: geo.dist_to_route_km(lat, lon, rid), DoubleType())
nearest_facility_udf = udf(lambda lat, lon: geo.nearest_facility(lat, lon), StringType())


def build_spark():
    return (
        SparkSession.builder
        .appName("silver-clean")
        .master("local[2]")
        .config("spark.jars.packages", config.DELTA_PKG)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    bronze = spark.readStream.format("delta").load(config.BRONZE_PATH)

    silver = (
        bronze
        .where(
            col("shipment_id").isNotNull()
            & col("lat").isNotNull() & col("lon").isNotNull()
            & (col("lat").between(-90, 90)) & (col("lon").between(-180, 180))
        )
        .withWatermark("event_time", config.WATERMARK)
        .dropDuplicates(["shipment_id", "seq"])
        .withColumn("progress_pct",
                    sround(when(col("total_km") > 0, col("dist_km") / col("total_km") * 100)
                           .otherwise(lit(0.0)), 1))
        .withColumn("is_moving", col("speed_kmh") >= lit(3.0))
        .withColumn("dist_to_route_km",
                    sround(dist_to_route_udf(col("lat"), col("lon"), col("route_id")), 3))
        .withColumn("at_facility", nearest_facility_udf(col("lat"), col("lon")))
    )

    query = (
        silver.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", config.SILVER_CHECKPOINT)
        .option("mergeSchema", "true")
        .start(config.SILVER_PATH)
    )

    print(f"[silver] streaming to {config.SILVER_PATH}  (Ctrl-C to stop)")
    query.awaitTermination()


if __name__ == "__main__":
    main()
