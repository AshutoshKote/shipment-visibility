"""GPS ping simulator.

Simulates N shipments moving along predefined routes and emits GPS pings
to Kafka, keyed by shipment_id so all pings for one shipment land on the
same partition (per-shipment ordering). Supports injecting late /
out-of-order pings to exercise watermarking downstream.

Run:
    python simulator/gps_simulator.py --shipments 20 --interval 2 --late-prob 0.1
    python simulator/gps_simulator.py --dry-run        # print, don't send
"""

import argparse
import bisect
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from simulator.routes import ROUTES, ROUTE_IDS

EARTH_KM = 6371.0088


def haversine_km(a, b):
    """Great-circle distance in km between (lat, lon) points a and b."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(h))


def cumulative_distances(waypoints):
    """Cumulative km distance at each waypoint (first = 0)."""
    cum = [0.0]
    for i in range(1, len(waypoints)):
        cum.append(cum[-1] + haversine_km(waypoints[i - 1], waypoints[i]))
    return cum


def bearing_deg(a, b):
    """Initial compass bearing from a to b in degrees."""
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlon = math.radians(b[1] - a[1])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def offset_point(lat, lon, dx_km, dy_km):
    """Shift a point by dx_km east and dy_km north."""
    dlat = dy_km / 110.574
    dlon = dx_km / (111.320 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def point_at_distance(waypoints, cum, dist_km):
    """Interpolate a (lat, lon) point at dist_km along the route."""
    if dist_km <= 0:
        return waypoints[0], bearing_deg(waypoints[0], waypoints[1])
    if dist_km >= cum[-1]:
        return waypoints[-1], bearing_deg(waypoints[-2], waypoints[-1])
    i = bisect.bisect_right(cum, dist_km) - 1
    seg_len = cum[i + 1] - cum[i]
    t = (dist_km - cum[i]) / seg_len if seg_len > 0 else 0.0
    lat = waypoints[i][0] + (waypoints[i + 1][0] - waypoints[i][0]) * t
    lon = waypoints[i][1] + (waypoints[i + 1][1] - waypoints[i][1]) * t
    return (lat, lon), bearing_deg(waypoints[i], waypoints[i + 1])


class Shipment:
    """Tracks one shipment's progress along its route, with injected anomalies."""

    def __init__(self, idx):
        self.shipment_id = f"SHP-{idx:05d}"
        self.vehicle_id = f"VEH-{idx % 12:03d}"
        self.route_id = random.choice(ROUTE_IDS)
        self.waypoints = ROUTES[self.route_id]["waypoints"]
        self.cum = cumulative_distances(self.waypoints)
        self.total_km = self.cum[-1]
        self.dist_km = random.uniform(0, self.total_km * 0.3)  # staggered starts
        self.base_speed = random.uniform(35, 75)               # km/h
        self.seq = 0
        self.done = False
        # anomaly state
        self.stop_rounds = 0       # >0 => currently stopped
        self.detour_rounds = 0     # >0 => currently off-route
        self.detour_offset = 0.0   # km lateral offset during a detour

    def advance(self, dt_seconds, noise=0.0003, stop_prob=0.02, detour_prob=0.02):
        """Advance one tick; inject stops/detours/noise; return a ping dict."""
        self.seq += 1

        # maybe start a new anomaly (only when running normally)
        if self.stop_rounds == 0 and self.detour_rounds == 0:
            r = random.random()
            if r < stop_prob:
                self.stop_rounds = random.randint(20, 45)        # ~40-90s parked
            elif r < stop_prob + detour_prob:
                self.detour_rounds = random.randint(6, 12)
                self.detour_offset = random.uniform(2.0, 4.0)    # km off-route

        if self.stop_rounds > 0:
            speed = 0.0                                          # frozen in place
            self.stop_rounds -= 1
        else:
            speed = max(0.0, self.base_speed + random.uniform(-8, 8))
            self.dist_km = min(self.total_km, self.dist_km + speed * (dt_seconds / 3600.0))

        (lat, lon), heading = point_at_distance(self.waypoints, self.cum, self.dist_km)

        # detour: push the reported position laterally off the route
        if self.detour_rounds > 0:
            perp = heading + 90.0
            dx = self.detour_offset * math.sin(math.radians(perp))
            dy = self.detour_offset * math.cos(math.radians(perp))
            lat, lon = offset_point(lat, lon, dx, dy)
            self.detour_rounds -= 1

        # always add a little GPS noise
        lat += random.gauss(0, noise)
        lon += random.gauss(0, noise)

        if self.dist_km >= self.total_km:
            self.done = True

        return {
            "shipment_id": self.shipment_id,
            "vehicle_id": self.vehicle_id,
            "route_id": self.route_id,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "speed_kmh": round(speed, 1),
            "heading": round(heading, 1),
            "dist_km": round(self.dist_km, 3),
            "total_km": round(self.total_km, 3),
            "seq": self.seq,
            "event_time": datetime.now(timezone.utc).isoformat(),
        }


def make_producer(bootstrap):
    from confluent_kafka import Producer
    return Producer({"bootstrap.servers": bootstrap, "linger.ms": 50})


def main():
    ap = argparse.ArgumentParser(description="GPS ping simulator")
    ap.add_argument("--shipments", type=int, default=20)
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between ping rounds")
    ap.add_argument("--duration", type=float, default=0, help="seconds to run (0 = forever)")
    ap.add_argument("--late-prob", type=float, default=0.1, help="prob a ping is delayed/out-of-order")
    ap.add_argument("--stop-prob", type=float, default=0.02, help="per-round prob a shipment stops")
    ap.add_argument("--detour-prob", type=float, default=0.02, help="per-round prob a shipment detours off-route")
    ap.add_argument("--gps-noise", type=float, default=0.0003, help="stddev of GPS noise in degrees")
    ap.add_argument("--bootstrap", default=config.KAFKA_BOOTSTRAP)
    ap.add_argument("--dry-run", action="store_true", help="print pings instead of producing")
    args = ap.parse_args()

    random.seed()
    fleet = [Shipment(i + 1) for i in range(args.shipments)]
    producer = None if args.dry_run else make_producer(args.bootstrap)
    held = []  # (release_round, ping) for late delivery

    print(f"[sim] {args.shipments} shipments | interval {args.interval}s | "
          f"late-prob {args.late_prob} | {'DRY RUN' if args.dry_run else args.bootstrap}")

    start = time.time()
    round_no = 0
    sent = 0
    try:
        while True:
            round_no += 1

            # release any held (late) pings whose time has come -> arrive out of order
            batch = [p for (r, p) in held if round_no >= r]
            held[:] = [(r, p) for (r, p) in held if round_no < r]

            for s in fleet:
                if s.done:
                    continue
                ping = s.advance(args.interval, noise=args.gps_noise,
                                 stop_prob=args.stop_prob, detour_prob=args.detour_prob)
                if random.random() < args.late_prob:
                    held.append((round_no + random.randint(1, 3), ping))  # deliver later
                else:
                    batch.append(ping)

            for ping in batch:
                payload = json.dumps(ping).encode("utf-8")
                key = ping["shipment_id"].encode("utf-8")
                if args.dry_run:
                    print(payload.decode())
                else:
                    producer.produce(config.TOPIC_PINGS, key=key, value=payload)
                sent += 1

            if producer:
                producer.poll(0)

            active = sum(1 for s in fleet if not s.done)
            print(f"[sim] round {round_no:>4} | sent {sent:>6} | active {active:>3} | held {len(held):>3}")

            if active == 0 and not held:
                print("[sim] all shipments delivered.")
                break
            if args.duration and (time.time() - start) >= args.duration:
                print("[sim] duration reached.")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[sim] interrupted.")
    finally:
        if producer:
            producer.flush(10)
            print(f"[sim] flushed. total produced: {sent}")


if __name__ == "__main__":
    main()
