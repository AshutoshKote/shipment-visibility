"""Live shipment-visibility dashboard.

Reads current shipment state from the Silver lake and recent exceptions from
the Gold lake (both via delta-rs, no Spark), and pulls ETAs from the serving
API. Shows a live map colour-coded by status, per-shipment ETAs, and an
exceptions feed.

Run:
    streamlit run dashboard/app.py
(have the streams + the API running for live data)
"""

import os
import sys

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from simulator.routes import FACILITIES

API = f"http://localhost:{config.API_PORT}"

STATUS_COLORS = {
    "IN_TRANSIT":  [38, 166, 91],    # green
    "OFF_ROUTE":   [217, 48, 37],    # red
    "STOPPED":     [241, 156, 24],   # orange
    "AT_FACILITY": [41, 98, 255],    # blue
}


def classify(row):
    if row.get("dist_to_route_km") and row["dist_to_route_km"] > config.OFF_ROUTE_KM:
        return "OFF_ROUTE"
    if row.get("at_facility"):
        return "AT_FACILITY"
    if row.get("speed_kmh", 0) < 3:
        return "STOPPED"
    return "IN_TRANSIT"


@st.cache_data(ttl=4)
def load_current_shipments():
    from deltalake import DeltaTable
    cols = ["shipment_id", "route_id", "lat", "lon", "speed_kmh", "progress_pct",
            "dist_to_route_km", "at_facility", "dist_km", "total_km", "event_time"]
    df = DeltaTable(config.SILVER_PATH).to_pandas(columns=cols)
    if df.empty:
        return df
    df = df.sort_values("event_time").groupby("shipment_id", as_index=False).last()
    df["status"] = df.apply(classify, axis=1)
    df["dist_remaining_km"] = (df["total_km"] - df["dist_km"]).clip(lower=0).round(2)
    return df


@st.cache_data(ttl=4)
def load_exceptions(limit=25):
    from deltalake import DeltaTable
    frames = []
    try:
        s = DeltaTable(config.GOLD_STATUS_PATH).to_pandas(
            columns=["shipment_id", "event_type", "detail", "event_time"])
        frames.append(s.rename(columns={"detail": "info", "event_time": "ts"}))
    except Exception:
        pass
    try:
        e = DeltaTable(config.GOLD_EXCEPTIONS_PATH).to_pandas(
            columns=["shipment_id", "event_type", "at_facility", "window_end"])
        frames.append(e.rename(columns={"at_facility": "info", "window_end": "ts"}))
    except Exception:
        pass
    if not frames:
        return pd.DataFrame(columns=["shipment_id", "event_type", "info", "ts"])
    out = pd.concat(frames, ignore_index=True).sort_values("ts", ascending=False)
    return out.head(limit)


def get_eta(row):
    body = {
        "route_id": row["route_id"],
        "dist_remaining_km": float(row["dist_remaining_km"]),
        "progress_pct": float(row["progress_pct"]),
        "speed_kmh": float(row["speed_kmh"]),
        "avg_speed_kmh": float(row["speed_kmh"]),
        "dist_to_route_km": float(row.get("dist_to_route_km") or 0.0),
    }
    try:
        r = requests.post(f"{API}/predict", json=body, timeout=2)
        if r.ok:
            return r.json()["predicted_eta_min"]
    except Exception:
        return None
    return None


# ---------------- UI ----------------
st.set_page_config(page_title="Shipment Visibility", layout="wide")
st.title("🚚 Real-Time Shipment Visibility")

with st.sidebar:
    st.header("Controls")
    auto = st.checkbox("Auto-refresh (5s)", value=True)
    if auto:
        st.markdown('<meta http-equiv="refresh" content="5">', unsafe_allow_html=True)
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()
    try:
        h = requests.get(f"{API}/health", timeout=2).json()
        st.success(f"API: {h['mode']} (model v{h['model_version']})")
    except Exception:
        st.error("API offline — ETAs unavailable")

try:
    ships = load_current_shipments()
except Exception:
    st.warning("No Silver data yet. Start the streams (make simulate / bronze / silver / gold).")
    st.stop()

if ships.empty:
    st.info("No active shipments yet — give the simulator a moment.")
    st.stop()

ships["eta_min"] = ships.apply(get_eta, axis=1)
ships["color"] = ships["status"].map(STATUS_COLORS)

# summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Active shipments", len(ships))
c2.metric("Off-route", int((ships["status"] == "OFF_ROUTE").sum()))
c3.metric("Stopped", int((ships["status"] == "STOPPED").sum()))
valid_eta = ships["eta_min"].dropna()
c4.metric("Avg ETA (min)", round(valid_eta.mean(), 1) if len(valid_eta) else "—")

# map
fac = pd.DataFrame([{"name": n, "lat": la, "lon": lo} for n, (la, lo) in FACILITIES.items()])
layers = [
    pdk.Layer("ScatterplotLayer", data=fac, get_position="[lon, lat]",
              get_fill_color="[120,120,120]", get_radius=400, opacity=0.3),
    pdk.Layer("ScatterplotLayer", data=ships, get_position="[lon, lat]",
              get_fill_color="color", get_radius=600, pickable=True),
]
view = pdk.ViewState(latitude=1.35, longitude=103.82, zoom=10)
tooltip = {"text": "{shipment_id}\n{status}\nETA: {eta_min} min"}
st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip,
                         map_style="light"))

# legend
st.caption("🟢 in transit   🔴 off-route   🟠 stopped   🔵 at facility   ⚫ facility")

col_a, col_b = st.columns([3, 2])
with col_a:
    st.subheader("Shipments")
    st.dataframe(
        ships[["shipment_id", "route_id", "status", "progress_pct",
               "dist_remaining_km", "speed_kmh", "eta_min"]]
        .sort_values("eta_min", na_position="last"),
        use_container_width=True, hide_index=True,
    )
with col_b:
    st.subheader("Recent exceptions")
    exc = load_exceptions()
    if exc.empty:
        st.write("None yet.")
    else:
        st.dataframe(exc, use_container_width=True, hide_index=True)
