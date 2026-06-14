# Real-Time Shipment Visibility + ETA

A working prototype of a real-time transportation-visibility platform: streaming
GPS ingestion, a Medallion stream-processing layer with exception detection, an
ETA model with a full MLflow lifecycle, an API serving layer, and a live
dashboard. Built locally; architected to map onto Azure Event Hubs, Databricks,
and Azure ML.

```
GPS simulator -> Kafka -> Spark Structured Streaming (Bronze/Silver/Gold Delta)
              -> ETA model (MLflow) -> FastAPI -> Streamlit dashboard
```

## Day 5 scope: live dashboard

A Streamlit dashboard that ties the whole pipeline together on one screen:
a live map of shipments colour-coded by status, model ETAs pulled from the
serving API, summary metrics, and a recent-exceptions feed. Reads current
state from Silver and exceptions from Gold (via delta-rs), and calls the API
for ETAs.

```bash
make dashboard       # http://localhost:8501
```

For a live, moving demo, have the streams and the API running:
`make simulate`, `make bronze`, `make silver`, `make gold`, `make serve`,
then `make dashboard`. If the laptop is strained, run the streams for a
couple of minutes to populate Silver/Gold, stop the Spark jobs, and run just
`make serve` + `make dashboard` against the latest snapshot.

**Definition of done (Day 5):** the dashboard shows shipments on the map
colour-coded by status, per-shipment ETAs, and recent exceptions.

## Day 4 scope: serving + drift monitoring + retraining

Puts the model behind a live API, with graceful fallback, drift monitoring,
and a champion/challenger retrain gate. Pure Python.

```bash
make serve            # FastAPI ETA service on http://localhost:8000
# in another terminal:
make predict          # send a batch of predictions (healthy traffic)
make drift            # PSI drift report -> "stable"
make predict-shift    # send drifted traffic
make drift            # PSI drift report -> "SIGNIFICANT DRIFT"
make retrain          # train challenger; promote only if it beats the champion
```

Endpoints (see http://localhost:8000/docs):
- `GET /health` — liveness + which model version is loaded
- `POST /predict` — predict from explicit current state (always available)
- `GET /eta/{shipment_id}` — predict from the shipment's latest Silver ping
  (read via delta-rs, no Spark dependency)

Key behaviours:
- **Fallback** — if the model can't load, every prediction degrades to a naive
  distance/speed estimate and reports `"method": "fallback"`.
- **Drift** — `monitoring/drift_monitor.py` computes PSI of live serving inputs
  vs the training distribution (PSI > 0.2 = significant drift).
- **Champion/challenger** — `ml/retrain.py` promotes a retrained model to the
  `production` alias only if it beats the current champion on a held-out set.

**Definition of done (Day 4):** `GET /health` shows the model loaded; `POST
/predict` returns a model ETA; `make drift` reads "stable" for normal traffic
and "SIGNIFICANT DRIFT" after `make predict-shift`; `make retrain` makes a
keep-or-promote decision.

## Day 3 scope: ETA model + MLflow lifecycle

Trains a gradient-boosted ETA model and tracks the full lifecycle in MLflow.
Pure Python (pandas + scikit-learn + MLflow) — no Spark/Kafka/Docker needed.

```bash
make gen-data    # generate the labelled training dataset (data/ml/training.parquet)
make train       # train + log to MLflow; registers 'eta-predictor' -> alias 'production'
make mlflow-ui   # browse runs at http://localhost:5000
```

- `generate_training_data.py` plays out completed trips on the route network
  and labels each observation with its actual `minutes_to_arrival`. (In
  production this dataset would be the completed trips accumulated in the lake.)
- `train_eta.py` trains a `HistGradientBoostingRegressor` (LightGBM-class, no
  native deps), logs params + metrics (MAE / RMSE / R2) + a permutation-
  importance artifact, registers the model, and promotes it to the
  `production` alias so the serving layer can load `models:/eta-predictor@production`.

**Definition of done (Day 3):** `make train` prints MAE/RMSE/R2 and registers
the model; `make mlflow-ui` shows the run with metrics and the registered model.

## Day 2 scope: Silver + Gold (clean, enrich, detect exceptions)

**Silver** reads the Bronze stream, validates and de-duplicates pings
(watermark + `dropDuplicates` on `shipment_id, seq` — this handles the late /
re-delivered pings), and enriches each with `progress_pct`, `is_moving`,
`dist_to_route_km`, and `at_facility` (geofence).

**Gold** turns the enriched stream into exception events:
- `OFF_ROUTE` — distance from the planned route exceeds the threshold
- `AT_FACILITY` — ping inside a facility geofence
- `STOPPED` / `DWELL` — windowed: within a time window the shipment barely
  moved and speed stayed low (DWELL if at a facility, else STOPPED). This query
  uses a watermark + tumbling window, so it shows how late pings are absorbed
  before a window finalises.

The simulator now injects realistic anomalies (stops, off-route detours, GPS
noise) so these detectors have something to catch.

### Run it (keep all four streams running together)

```bash
make up                        # Kafka (if not already up)
# terminal A:
make simulate                  # pings + injected anomalies -> Kafka
# terminal B:
make bronze                    # Kafka -> Bronze
# terminal C:
make silver                    # Bronze -> Silver (clean + enrich)
# terminal D:
make gold                      # Silver -> Gold (status + windowed exceptions)
# terminal E (after ~2 min so windows finalise):
make query-gold                # inspect Silver + Gold tables
```

Keep the simulator running while you check Gold — the windowed `STOPPED`/`DWELL`
detector only finalises a window once event-time advances past it, which needs
fresh pings flowing.

**Definition of done (Day 2):** `make query-gold` shows enriched Silver rows
(with `dist_to_route_km` / `at_facility`), a count of `OFF_ROUTE` and
`AT_FACILITY` status events, and at least one windowed `STOPPED`/`DWELL`
exception.

**Laptop running hot?** Three Spark streams + Kafka is the heaviest moment of
the build. If it gets sluggish, run a lighter simulator: `python
simulator/gps_simulator.py --shipments 8`.

### Day 1 scope: streaming ingestion

GPS pings flow from the simulator through Kafka into a raw **Bronze** Delta table
via Spark Structured Streaming. Pings are keyed by `shipment_id`, so all pings
for a shipment land on the same partition (per-shipment ordering). The simulator
can inject late / out-of-order pings to exercise watermarking in later layers.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Make sure Docker Desktop is running and `JAVA_HOME` points at Java 17
(`echo $JAVA_HOME`).

## Run it

```bash
make up          # 1. start Kafka + Kafka UI
make topic       # 2. create the gps.pings topic (6 partitions)
make ui          # 3. (optional) open http://localhost:8080 to watch the topic

# in terminal A:
make simulate    # 4. start streaming GPS pings into Kafka
# in terminal B:
make verify      # 5. quick check that pings are flowing
make bronze      # 6. Spark: Kafka -> Bronze Delta (first run downloads jars, ~1 min)
# in terminal C (after bronze has run a bit):
make query       # 7. confirm rows landed in the Bronze Delta table
```

`make dry` runs the simulator without Kafka (prints pings) — handy before
containers are up.

## Definition of done (Day 1)

- `make up` brings up Kafka; `make topic` shows 6 partitions
- `make simulate` streams pings; `make verify` prints them with partition/offset
- `make bronze` writes to `data/bronze/pings`
- `make query` reports total pings, distinct shipments, and sample rows

## Troubleshooting

**Spark fails on Java 17 with `InaccessibleObjectException`** — add these JVM
flags before launching (Spark 3.5 usually injects them in local mode, but if not):

```bash
export SPARK_SUBMIT_OPTS="--add-opens=java.base/sun.nio.ch=ALL-UNNAMED \
  --add-opens=java.base/java.nio=ALL-UNNAMED \
  --add-opens=java.base/java.lang=ALL-UNNAMED \
  --add-opens=java.base/java.util=ALL-UNNAMED"
```

**First `make bronze` is slow / "resolving dependencies"** — Spark is downloading
the Kafka + Delta connector jars from Maven Central. One-time; needs internet.

**`make verify` sees nothing** — confirm the simulator (`make simulate`) is
running in another terminal and the topic exists (`make topic`).

**Kafka won't start** — check `make logs`; ensure ports 9092 and 8080 are free.

## Layout

```
config.py                 shared constants
docker-compose.yml        Kafka (KRaft) + Kafka UI
simulator/
  routes.py               Singapore-Johor route waypoints
  gps_simulator.py        ping generator -> Kafka
ingestion/
  verify_consumer.py      pure-Python Kafka sanity check
  bronze_ingest.py        Spark Structured Streaming -> Bronze Delta
  query_bronze.py         batch read of the Bronze table
```

## Architecture

```
  GPS simulator (anomalies: stops, detours, GPS noise, late pings)
        │  keyed by shipment_id
        ▼
  Kafka (KRaft, Docker)
        ▼
  Spark Structured Streaming  ── Medallion ──┐
        ├─ Bronze : raw pings (Delta)         │
        ├─ Silver : dedup + watermark + geo enrichment (Delta)
        └─ Gold   : OFF_ROUTE / AT_FACILITY (per-ping) + STOPPED / DWELL (windowed)
        ▼
  ETA model  ── HistGradientBoosting, full MLflow lifecycle
        │        (tracking, registry, 'production' alias)
        ▼
  FastAPI  ── GET /eta/{id} (reads Silver via delta-rs) + POST /predict
        │     fallback when model down · logs every prediction
        ├─► PSI drift monitor (live inputs vs training) → retrain trigger
        ├─► champion/challenger retrain (promote only if better)
        └─► Streamlit dashboard (live map + ETAs + exceptions)
```

## Production next steps (what changes at scale)

- **Cloud-native swap:** Kafka → Azure Event Hubs (Kafka-compatible); local
  Spark → Databricks jobs; Delta paths → Unity Catalog tables; MLflow → Azure
  ML or Databricks MLflow; the same code shape carries over.
- **Stateful serving features:** replace per-request Silver reads with an online
  feature store (e.g. Redis / Feast) to cut latency and avoid train/serve skew.
- **Exactly-once + scale:** partition Kafka by region, autoscale Spark, and use
  idempotent Delta writes; add schema enforcement on Bronze.
- **Real labels:** ground-truth arrival times feed continuous evaluation, so
  drift detection is backed by live error (concept drift), not just input PSI.
- **Ops:** containerise the API (Docker/K8s), add auth + rate limiting, wire
  metrics/traces (OpenTelemetry), and gate model promotion behind a CI check.
