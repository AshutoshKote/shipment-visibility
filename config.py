"""Shared configuration for the shipment-visibility pipeline."""

# --- Kafka ---
KAFKA_BOOTSTRAP = "localhost:9092"   # host-facing listener (see docker-compose.yml)
TOPIC_PINGS = "gps.pings"
NUM_PARTITIONS = 6                    # partitioned by shipment_id -> per-shipment ordering

# --- Delta lake paths (local, gitignored) ---
BRONZE_PATH = "data/bronze/pings"
BRONZE_CHECKPOINT = "data/checkpoints/bronze"

# --- Silver / Gold Delta paths ---
SILVER_PATH = "data/silver/pings"
SILVER_CHECKPOINT = "data/checkpoints/silver"

GOLD_STATUS_PATH = "data/gold/status"          # per-ping status events (OFF_ROUTE, AT_FACILITY)
GOLD_STATUS_CHECKPOINT = "data/checkpoints/gold_status"

GOLD_EXCEPTIONS_PATH = "data/gold/exceptions"  # windowed exceptions (STOPPED, DWELL)
GOLD_EXCEPTIONS_CHECKPOINT = "data/checkpoints/gold_exceptions"

# --- Stream / detection tuning ---
WATERMARK = "2 minutes"          # bounds state + tolerates late pings
STOP_WINDOW = "30 seconds"       # window for stopped/dwell detection
OFF_ROUTE_KM = 1.5               # distance from planned route to flag OFF_ROUTE
FACILITY_RADIUS_KM = 0.8         # geofence radius around a facility
STOP_MOVED_KM = 0.08             # < this moved in a window => not really moving
STOP_MAX_SPEED = 5.0             # km/h, combined with moved_km => stopped

# --- ML (Day 3) ---
TRAINING_DATA_PATH = "data/ml/training.parquet"
FEATURE_COLUMNS_PATH = "data/ml/feature_columns.json"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"   # sqlite backend enables the model registry

# --- Serving + monitoring (Day 4) ---
PREDICTION_LOG_PATH = "data/serving/predictions.jsonl"
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_URI = "models:/eta-predictor@production"
PSI_WARN = 0.1        # 0.1-0.2 = moderate shift
PSI_ALERT = 0.2       # > 0.2 = significant drift -> consider retrain
FALLBACK_SPEED_KMH = 25.0   # assumed speed when the model is unavailable / vehicle stopped

# --- Spark dependency coordinates (downloaded on first run) ---
SPARK_KAFKA_PKG = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3"
DELTA_PKG = "io.delta:delta-spark_2.12:3.2.0"
