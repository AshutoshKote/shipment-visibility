"""Geo helpers for the Silver/Gold layers.

- dist_to_route_km: shortest distance from a point to a route's polyline
- nearest_facility: name of a facility if the point is within its radius

Uses a local equirectangular projection (accurate at city scale around
Singapore) for point-to-segment distance.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from simulator.routes import ROUTES, FACILITIES

EARTH_KM = 6371.0088
LAT0 = 1.35  # reference latitude (Singapore) for the local projection


def _to_xy(lat, lon):
    """Project (lat, lon) to local (x, y) km using equirectangular approx."""
    x = math.radians(lon) * math.cos(math.radians(LAT0)) * EARTH_KM
    y = math.radians(lat) * EARTH_KM
    return x, y


def _point_seg_km(p, a, b):
    """Distance in km from point p to segment a-b (all in xy km)."""
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def dist_to_route_km(lat, lon, route_id):
    """Shortest distance (km) from (lat, lon) to the route's polyline."""
    if lat is None or lon is None or route_id not in ROUTES:
        return None
    wpts = ROUTES[route_id]["waypoints"]
    p = _to_xy(lat, lon)
    xy = [_to_xy(la, lo) for (la, lo) in wpts]
    best = min(_point_seg_km(p, xy[i], xy[i + 1]) for i in range(len(xy) - 1))
    return float(best)


def haversine_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = la2 - la1, lo2 - lo1
    h = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(h))


def nearest_facility(lat, lon):
    """Return facility name if within FACILITY_RADIUS_KM, else None."""
    if lat is None or lon is None:
        return None
    best_name, best_d = None, None
    for name, (flat, flon) in FACILITIES.items():
        d = haversine_km(lat, lon, flat, flon)
        if best_d is None or d < best_d:
            best_name, best_d = name, d
    if best_d is not None and best_d <= config.FACILITY_RADIUS_KM:
        return best_name
    return None
