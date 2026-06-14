"""Generate a historical trip dataset with ETA labels for model training.

In production the ETA model would train on completed trips accumulated in
the lake (Gold/Silver). For the prototype we generate a representative
history using the same route network and movement model, so there's a
rich, labelled dataset to train on.

Each row is one observation along a trip:
  features -> route_id, hour, dow, dist_remaining_km, progress_pct,
              speed_kmh, avg_speed_kmh, dist_to_route_km, is_stopped
  label    -> minutes_to_arrival  (the actual remaining time for that trip)

Run:
    python ml/generate_training_data.py --trips 1200
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from simulator.routes import ROUTES, ROUTE_IDS

EARTH_KM = 6371.0088
RUSH_HOURS = {7, 8, 9, 17, 18, 19}
DT_SECONDS = 30          # simulated time step
STEP_SAMPLE = 0.5        # keep ~50% of steps to bound dataset size


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(h))


def route_length_km(route_id):
    w = ROUTES[route_id]["waypoints"]
    return sum(haversine_km(w[i - 1], w[i]) for i in range(1, len(w)))


def simulate_trip(route_id):
    """Play out one full trip; return list of per-step feature dicts (label added later)."""
    total_km = route_length_km(route_id)
    base_speed = random.uniform(30, 80)
    hour = random.randint(0, 23)
    dow = random.randint(0, 6)
    rush_factor = 0.6 if hour in RUSH_HOURS else 1.0

    dist = 0.0
    elapsed_s = 0.0
    speeds = []
    steps = []
    stop_left = 0
    detour_left = 0
    guard = 0

    while dist < total_km and guard < 5000:
        guard += 1
        # occasional anomalies
        if stop_left == 0 and detour_left == 0:
            r = random.random()
            if r < 0.03:
                stop_left = random.randint(2, 6)        # stopped for a few steps
            elif r < 0.06:
                detour_left = random.randint(2, 5)

        if stop_left > 0:
            speed = 0.0
            stop_left -= 1
            dist_to_route = 0.0
        else:
            speed = max(5.0, (base_speed + random.uniform(-8, 8)) * rush_factor)
            dist += speed * (DT_SECONDS / 3600.0)
            dist_to_route = 0.0
            if detour_left > 0:
                dist_to_route = random.uniform(1.5, 4.0)
                detour_left -= 1

        speeds.append(speed)
        avg_speed = sum(speeds[-5:]) / len(speeds[-5:])
        dist_remaining = max(0.0, total_km - dist)

        steps.append({
            "route_id": route_id,
            "hour": hour,
            "dow": dow,
            "dist_remaining_km": round(dist_remaining, 3),
            "progress_pct": round(min(100.0, dist / total_km * 100), 1),
            "speed_kmh": round(speed, 1),
            "avg_speed_kmh": round(avg_speed, 1),
            "dist_to_route_km": round(dist_to_route, 3),
            "is_stopped": int(speed == 0.0),
            "_elapsed_s": elapsed_s,
        })
        elapsed_s += DT_SECONDS

    total_time_s = elapsed_s
    rows = []
    for s in steps:
        s["minutes_to_arrival"] = round((total_time_s - s["_elapsed_s"]) / 60.0, 2)
        del s["_elapsed_s"]
        if random.random() < STEP_SAMPLE:
            rows.append(s)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=config.TRAINING_DATA_PATH)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    all_rows = []
    for i in range(args.trips):
        route_id = ROUTE_IDS[i % len(ROUTE_IDS)]
        all_rows.extend(simulate_trip(route_id))

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"[gen] {len(df):,} rows from {args.trips} trips -> {args.out}")
    print(f"[gen] routes: {df['route_id'].nunique()} | "
          f"label minutes_to_arrival: min={df.minutes_to_arrival.min():.1f} "
          f"mean={df.minutes_to_arrival.mean():.1f} max={df.minutes_to_arrival.max():.1f}")


if __name__ == "__main__":
    main()
