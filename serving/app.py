"""ETA serving API.

Endpoints:
  GET  /health                 - liveness + which model is loaded
  POST /predict                - predict from explicit current state (always works)
  GET  /eta/{shipment_id}      - predict from the shipment's latest Silver ping

If the model can't be loaded, every prediction falls back to a naive
distance/speed estimate (graceful degradation) and says so. Every prediction
is appended to a JSONL log for the drift monitor.

Run:
    uvicorn serving.app:app --reload --port 8000
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from serving.features import build_feature_row, fallback_eta_min, load_feature_columns

app = FastAPI(title="Shipment ETA API", version="1.0")

STATE = {"model": None, "model_version": None, "feature_cols": None}


def _load_model():
    """Try to load the production model; on failure leave model=None (fallback mode)."""
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.client import MlflowClient
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        STATE["model"] = mlflow.sklearn.load_model(config.MODEL_URI)
        STATE["feature_cols"] = load_feature_columns()
        try:
            mv = MlflowClient().get_model_version_by_alias("eta-predictor", "production")
            STATE["model_version"] = mv.version
        except Exception:
            STATE["model_version"] = "unknown"
        print(f"[api] model loaded: eta-predictor v{STATE['model_version']}")
    except Exception as e:
        STATE["model"] = None
        print(f"[api] model load FAILED -> fallback mode. ({e})")


@app.on_event("startup")
def startup():
    _load_model()


def _log_prediction(record):
    os.makedirs(os.path.dirname(config.PREDICTION_LOG_PATH), exist_ok=True)
    record["ts"] = datetime.now(timezone.utc).isoformat()
    with open(config.PREDICTION_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _predict(state):
    """Return (eta_min, method). Uses the model if available, else fallback."""
    t0 = time.perf_counter()
    if STATE["model"] is not None:
        try:
            X = build_feature_row(state, STATE["feature_cols"])
            eta = round(float(STATE["model"].predict(X)[0]), 1)
            method = "model"
        except Exception:
            eta = fallback_eta_min(state["dist_remaining_km"], state.get("avg_speed_kmh"))
            method = "fallback"
    else:
        eta = fallback_eta_min(state["dist_remaining_km"], state.get("avg_speed_kmh"))
        method = "fallback"
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return eta, method, latency_ms


class PredictRequest(BaseModel):
    route_id: str
    dist_remaining_km: float
    progress_pct: float = 0.0
    speed_kmh: float = 0.0
    avg_speed_kmh: float | None = None
    dist_to_route_km: float = 0.0
    hour: int | None = None
    dow: int | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": STATE["model"] is not None,
        "model_version": STATE["model_version"],
        "mode": "model" if STATE["model"] is not None else "fallback",
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if req.dist_remaining_km < 0:
        raise HTTPException(422, "dist_remaining_km must be >= 0")
    now = datetime.now(timezone.utc)
    state = req.model_dump()
    state["hour"] = req.hour if req.hour is not None else now.hour
    state["dow"] = req.dow if req.dow is not None else now.weekday()
    state["is_stopped"] = int(req.speed_kmh < 3)
    if state["avg_speed_kmh"] is None:
        state["avg_speed_kmh"] = req.speed_kmh

    eta, method, latency_ms = _predict(state)
    record = {"route_id": req.route_id, "dist_remaining_km": req.dist_remaining_km,
              "speed_kmh": req.speed_kmh, "predicted_eta_min": eta, "method": method}
    _log_prediction(record)
    return {"predicted_eta_min": eta, "method": method,
            "model_version": STATE["model_version"], "latency_ms": latency_ms}


@app.get("/eta/{shipment_id}")
def eta_for_shipment(shipment_id: str):
    """Predict from the shipment's latest ping in the Silver Delta table."""
    try:
        from deltalake import DeltaTable
        df = DeltaTable(config.SILVER_PATH).to_pandas(
            columns=["shipment_id", "route_id", "dist_km", "total_km", "progress_pct",
                     "speed_kmh", "dist_to_route_km", "event_time"])
    except Exception:
        raise HTTPException(503, "Silver table unavailable. Run the streams, or use POST /predict.")

    sub = df[df["shipment_id"] == shipment_id].sort_values("event_time")
    if sub.empty:
        raise HTTPException(404, f"no pings found for shipment '{shipment_id}'")

    last = sub.iloc[-1]
    avg_speed = float(sub["speed_kmh"].tail(5).mean())
    et = last["event_time"]
    state = {
        "route_id": last["route_id"],
        "dist_remaining_km": max(0.0, float(last["total_km"]) - float(last["dist_km"])),
        "progress_pct": float(last["progress_pct"]),
        "speed_kmh": float(last["speed_kmh"]),
        "avg_speed_kmh": avg_speed,
        "dist_to_route_km": float(last["dist_to_route_km"] or 0.0),
        "hour": int(et.hour), "dow": int(et.weekday()),
        "is_stopped": int(float(last["speed_kmh"]) < 3),
    }
    eta, method, latency_ms = _predict(state)
    record = {"shipment_id": shipment_id, "route_id": state["route_id"],
              "dist_remaining_km": state["dist_remaining_km"], "speed_kmh": state["speed_kmh"],
              "predicted_eta_min": eta, "method": method}
    _log_prediction(record)
    return {"shipment_id": shipment_id, "predicted_eta_min": eta, "method": method,
            "progress_pct": state["progress_pct"], "model_version": STATE["model_version"],
            "latency_ms": latency_ms}
