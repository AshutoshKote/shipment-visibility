"""Shared feature assembly + fallback ETA, used by the API and retraining.

Keeps the serving-time feature vector identical to training (same columns,
same one-hot encoding) by loading the persisted feature_columns.json.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

NUMERIC = ["hour", "dow", "dist_remaining_km", "progress_pct",
           "speed_kmh", "avg_speed_kmh", "dist_to_route_km", "is_stopped"]


def load_feature_columns():
    with open(config.FEATURE_COLUMNS_PATH) as f:
        return json.load(f)


def build_feature_row(state, feature_cols):
    """Build a single-row DataFrame matching the trained feature columns.

    `state` is a dict with the NUMERIC keys plus 'route_id'.
    """
    row = {c: 0 for c in feature_cols}
    for k in NUMERIC:
        if k in state and state[k] is not None:
            row[k] = state[k]
    route_col = f"route_id_{state.get('route_id')}"
    if route_col in row:
        row[route_col] = 1
    return pd.DataFrame([row])[feature_cols]


def fallback_eta_min(dist_remaining_km, avg_speed_kmh=None):
    """Naive physics-based ETA used when the model is unavailable."""
    speed = avg_speed_kmh if (avg_speed_kmh and avg_speed_kmh > 5) else config.FALLBACK_SPEED_KMH
    return round((dist_remaining_km / speed) * 60.0, 1)
