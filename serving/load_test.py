"""Send a batch of predictions to the API to populate the prediction log.

    python serving/load_test.py --n 200
    python serving/load_test.py --n 200 --shift   # send drifted inputs

Normal mode samples real values from the training distribution (so a healthy
service reads as stable). `--shift` biases inputs toward short distances and
low speeds (congestion-like), so the drift monitor has something to detect.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
import requests

API = "http://localhost:8000/predict"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shift", action="store_true", help="send drifted inputs")
    args = ap.parse_args()

    ref = pd.read_parquet(config.TRAINING_DATA_PATH)
    sample = ref.sample(args.n, replace=True).reset_index(drop=True)

    ok = 0
    for _, row in sample.iterrows():
        dist = float(row["dist_remaining_km"])
        speed = float(row["speed_kmh"])
        if args.shift:
            dist *= random.uniform(0.1, 0.3)     # much closer to arrival
            speed *= random.uniform(0.3, 0.5)    # congested / slow
        body = {
            "route_id": row["route_id"],
            "dist_remaining_km": round(dist, 2),
            "progress_pct": float(row["progress_pct"]),
            "speed_kmh": round(speed, 1),
            "dist_to_route_km": 0.0,
        }
        try:
            r = requests.post(API, json=body, timeout=5)
            ok += int(r.ok)
        except Exception as e:
            print(f"request failed (is the API running? make serve): {e}")
            break

    print(f"[load] sent {ok}/{args.n} predictions {'(SHIFTED)' if args.shift else ''}")


if __name__ == "__main__":
    main()
