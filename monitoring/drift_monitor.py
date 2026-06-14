"""Drift monitor.

Compares the distribution of live serving inputs (from the prediction log)
against the training data (reference) using the Population Stability Index
(PSI). PSI is a standard data-drift metric:

    PSI < 0.1   : no significant shift
    0.1 - 0.2   : moderate shift  (watch)
    > 0.2       : significant drift (consider retraining)

Run:
    python monitoring/drift_monitor.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

FEATURES = ["dist_remaining_km", "speed_kmh"]


def psi(reference, live, bins=10):
    """Population Stability Index between reference and live samples."""
    cuts = np.quantile(reference, np.linspace(0, 1, bins + 1))
    cuts[0], cuts[-1] = -np.inf, np.inf
    ref = np.histogram(reference, cuts)[0] / len(reference)
    cur = np.histogram(live, cuts)[0] / len(live)
    ref = np.clip(ref, 1e-4, None)
    cur = np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def classify(v):
    if v > config.PSI_ALERT:
        return "SIGNIFICANT DRIFT"
    if v > config.PSI_WARN:
        return "moderate shift"
    return "stable"


def main():
    if not os.path.exists(config.PREDICTION_LOG_PATH):
        print("[drift] no prediction log yet. Run serving/load_test.py first.")
        return
    live = pd.read_json(config.PREDICTION_LOG_PATH, lines=True)
    ref = pd.read_parquet(config.TRAINING_DATA_PATH)

    print(f"[drift] reference rows: {len(ref):,} | live predictions: {len(live):,}\n")
    print(f"{'feature':<20}{'PSI':>8}   status")
    print("-" * 48)
    alert = False
    for f in FEATURES:
        if f in live and f in ref:
            v = psi(ref[f].values, live[f].values)
            status = classify(v)
            alert = alert or v > config.PSI_ALERT
            print(f"{f:<20}{v:>8.3f}   {status}")

    # method mix (how often we fell back)
    if "method" in live:
        mix = live["method"].value_counts().to_dict()
        print(f"\n[drift] prediction methods: {mix}")

    print()
    if alert:
        print(">>> SIGNIFICANT DRIFT detected -> recommend retraining (python ml/retrain.py)")
    else:
        print(">>> inputs stable vs training distribution -> no retrain needed")


if __name__ == "__main__":
    main()
