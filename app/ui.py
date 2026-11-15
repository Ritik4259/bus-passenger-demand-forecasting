"""Streamlit Interactive Dispatcher Interface for LPU & Punjab Transit Network.

Grounded in the Phagwara - Jalandhar - LPU NH-44 transit corridor.
Includes:
- Interactive 24-hour demand curve with capacity threshold overlay (Plotly)
- Real-time geospatial route corridor mapping (Pydeck)
- Visual seat occupancy distribution (Seated vs Overflow vs Empty)
- What-if vehicle sizing simulator
- Raw FastAPI JSON payload inspector
"""

import os
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st

from src.app import PredictRequest, extract_inference_features

st.set_page_config(
    page_title="LPU & Punjab Transit Dispatcher Decision Support",
    page_icon="🚌",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MODEL_PATH = Path("models/bus_demand_model.joblib")

# Real Phagwara - Jalandhar - LPU corridor geography
ROUTE_WAYPOINTS = {
    "LPU_JAL_101": {
        "name": "LPU Main Gate → Jalandhar City Bus Stand (via Rama Mandi)",
        "default_cap": 50,
        "description": "Primary student and faculty commute shuttle along NH-44.",
        "stops": [
            {"name": "LPU Gate 1", "lat": 31.2536, "lon": 75.7037, "type": "Origin"},
            {"name": "Rama Mandi Chowk", "lat": 31.3090, "lon": 75.6420, "type": "Interchange"},
            {"name": "PAP Chowk", "lat": 31.3185, "lon": 75.6180, "type": "Transit Stop"},
            {"name": "Jalandhar City Bus Stand", "lat": 31.3260, "lon": 75.5762, "type": "Destination"},
        ],
    },
    "PHG_LPU_202": {
        "name": "Phagwara Junction Railway Station → LPU Campus",
        "default_cap": 40,
        "description": "Express feeder route for students arriving by long-distance trains (Surges Sun/Mon).",
        "stops": [
            {"name": "Phagwara Railway Station", "lat": 31.2223, "lon": 75.7725, "type": "Origin"},
            {"name": "Goal Chowk Phagwara", "lat": 31.2260, "lon": 75.7680, "type": "Transit Stop"},
            {"name": "Law Gate Road", "lat": 31.2480, "lon": 75.7150, "type": "Transit Stop"},
            {"name": "LPU Uni-Mall & Hostels", "lat": 31.2540, "lon": 75.7040, "type": "Destination"},
        ],
    },
    "LPU_MDL_303": {
        "name": "LPU Campus → Model Town, Jalandhar",
        "default_cap": 45,
        "description": "Weekend & Friday evening shopping, cafes, and dining transit corridor.",
        "stops": [
            {"name": "LPU Campus Gate", "lat": 31.2536, "lon": 75.7037, "type": "Origin"},
            {"name": "Cantt Road Crossing", "lat": 31.2980, "lon": 75.6150, "type": "Transit Stop"},
            {"name": "Geeta Mandir", "lat": 31.3080, "lon": 75.5900, "type": "Transit Stop"},
            {"name": "Model Town Market Jalandhar", "lat": 31.3117, "lon": 75.5843, "type": "Destination"},
        ],
    },
    "INT_CAMPUS_404": {
        "name": "LPU Main Gate → Block 34 / Uni-Hospital (Campus Loop)",
        "default_cap": 60,
        "description": "High-frequency internal university shuttle with sharp class-transition surges.",
        "stops": [
            {"name": "LPU Main Gate 1", "lat": 31.2536, "lon": 75.7037, "type": "Origin"},
            {"name": "Block 14 / Central Library", "lat": 31.2545, "lon": 75.7060, "type": "Transit Stop"},
            {"name": "Uni-Hospital", "lat": 31.2552, "lon": 75.7075, "type": "Transit Stop"},
            {"name": "Block 34 Hostels & Labs", "lat": 31.2558, "lon": 75.7085, "type": "Destination"},
        ],
    },
    "JAL_ASR_505": {
        "name": "Jalandhar Cantt → Phagwara GT Road Feeder",
        "default_cap": 35,
        "description": "Sparse suburban feeder connecting Grand Trunk Road villages.",
        "stops": [
            {"name": "Jalandhar Cantt Station", "lat": 31.2941, "lon": 75.6172, "type": "Origin"},
            {"name": "Chaheru Village", "lat": 31.2680, "lon": 75.6700, "type": "Transit Stop"},
            {"name": "Chiheru Bridge", "lat": 31.2590, "lon": 75.6920, "type": "Transit Stop"},
            {"name": "Phagwara GT Road", "lat": 31.2290, "lon": 75.7620, "type": "Destination"},
        ],
    },
}


@st.cache_resource
def load_inprocess_model():
    """Load local model artifact as an instant fallback if FastAPI is offline."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def run_prediction_api_or_fallback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Query FastAPI /predict with transparent fallback to in-memory model."""
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=2.0)
        if resp.status_code == 200:
            res = resp.json()
            res["source"] = "FastAPI Service (Live REST API)"
            return res
    except Exception:
        pass

    bundle = load_inprocess_model()
    if bundle is None:
        raise RuntimeError("Neither FastAPI nor local model artifact is available.")

    req = PredictRequest(
        route_id=payload["route_id"],
        departure_datetime=datetime.fromisoformat(payload["departure_datetime"]),
        bus_capacity=payload["bus_capacity"],
        is_raining=payload.get("is_raining", 0),
        temperature_c=payload.get("temperature_c", 22.0),
        is_special_event=payload.get("is_special_event", 0),
    )
    X = extract_inference_features(req, bundle["lag_lookup_dict"])
    pred = float(np.maximum(0.0, bundle["pipeline"].predict(X)[0]))
    pred = round(pred, 1)
    occ = round((pred / req.bus_capacity) * 100.0, 1)

    if occ < 50.0:
        flag = "UNDERUTILIZED"
        rec = "Low occupancy forecasted. Consider assigning a smaller minibus or consolidating departures."
    elif occ <= 90.0:
        flag = "OPTIMAL"
        rec = "Balanced student/passenger load. Assigned vehicle capacity safely matches expected demand."
    else:
        flag = "OVERFLOW_RISK"
        rec = "High risk of passenger overflow & stranded students. Dispatch a supplemental relief coach."

    return {
        "route_id": req.route_id,
        "departure_datetime": req.departure_datetime.isoformat(),
        "bus_capacity": req.bus_capacity,
        "predicted_passenger_demand": pred,
        "predicted_occupancy_percentage": occ,
        "action_flag": flag,
        "operational_recommendation": rec,
        "model_name": bundle.get("model_name", "LightGBM"),
        "model_version": bundle.get("model_version", "1.0.0"),
        "limitations": (
            "Model assumes normal operating conditions; does not account for real-time weather shocks "
            "or novel events. Predictions represent unconstrained passenger demand 24 hours prior to departure."
        ),
        "source": "In-Process Engine (Local Fallback)",
    }


# =========================================================
# HEADER
# =========================================================
st.title("🚌 LPU & Punjab Regional Bus Dispatch Decision Support")
st.markdown(
    "**Operational Decision-Support System** for the Phagwara – Jalandhar – Lovely Professional University (NH-44) corridor. "
    "Predicts **unconstrained student & commuter demand 24 hours in advance** to prevent passenger stranding."
)
st.write("---")

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("🕹️ Dispatch Controls")

route_choice = st.sidebar.selectbox(
    "Select Transit Corridor",
    options=list(ROUTE_WAYPOINTS.keys()),
    format_func=lambda x: f"{x} - {ROUTE_WAYPOINTS[x]['name']}",
)
route_data = ROUTE_WAYPOINTS[route_choice]
st.sidebar.caption(route_data["description"])

st.sidebar.subheader("Scheduled Departure")

# Quick scenario presets
scenario_preset = st.sidebar.selectbox(
    "Quick Scenario Presets",
    options=[
        "Custom Selection",
        "🌅 Morning Rush (08:00 AM - LPU Commute)",
        "🏫 Lunch Class Dismissal (01:00 PM - Campus Loop)",
        "🌆 Evening Commute (05:30 PM - LPU -> Jalandhar)",
        "🚆 Sunday Train Arrival (09:00 PM - Phagwara Jn)",
        "🌙 Late Night Minimum (11:00 PM - Feeder Route)",
    ],
    index=0,
)

if scenario_preset == "🌅 Morning Rush (08:00 AM - LPU Commute)":
    default_hour, default_min, default_cap = 8, 0, 50
elif scenario_preset == "🏫 Lunch Class Dismissal (01:00 PM - Campus Loop)":
    default_hour, default_min, default_cap = 13, 0, 60
elif scenario_preset == "🌆 Evening Commute (05:30 PM - LPU -> Jalandhar)":
    default_hour, default_min, default_cap = 17, 30, 50
elif scenario_preset == "🚆 Sunday Train Arrival (09:00 PM - Phagwara Jn)":
    default_hour, default_min, default_cap = 21, 0, 40
elif scenario_preset == "🌙 Late Night Minimum (11:00 PM - Feeder Route)":
    default_hour, default_min, default_cap = 23, 0, 35
else:
    default_hour, default_min, default_cap = 8, 0, route_data["default_cap"]

dep_date = st.sidebar.date_input("Scheduled Departure Date", value=datetime(2026, 8, 10).date())

# Hour selectbox dropdown (0 to 23)
HOUR_OPTIONS = list(range(24))
def format_hour(h: int) -> str:
    period = "AM" if h < 12 else "PM"
    display_h = 12 if h in (0, 12) else h % 12
    tag = ""
    if h in (8, 9):
        tag = " - Morning Peak"
    elif h in (13, 17):
        tag = " - Class Dismissal"
    elif h == 18:
        tag = " - Evening Peak"
    elif h >= 22 or h <= 4:
        tag = " - Late Night"
    return f"{h:02d}:00 ({display_h:02d}:00 {period}){tag}"

dep_hour = st.sidebar.selectbox(
    "Departure Hour (0 to 23)",
    options=HOUR_OPTIONS,
    index=default_hour,
    format_func=format_hour,
    help="Select departure hour from full 24-hour schedule (0:00 to 23:00).",
)
dep_minute = st.sidebar.selectbox(
    "Departure Minute",
    options=[0, 15, 30, 45],
    index=[0, 15, 30, 45].index(default_min) if default_min in [0, 15, 30, 45] else 0,
)
departure_dt = datetime.combine(dep_date, time(hour=dep_hour, minute=dep_minute))

st.sidebar.subheader("Fleet Capacity Assignment")
capacity_input = st.sidebar.slider(
    "Assigned Bus Capacity (Seats)",
    min_value=15,
    max_value=120,
    value=default_cap,
    step=5,
    help="Adjust vehicle size to observe how seat occupancy and decision flags respond dynamically.",
)

st.sidebar.subheader("Environmental Shocks")
is_raining = st.sidebar.checkbox("Monsoon Rain / Heavy Downpour", value=False)
temperature = st.sidebar.slider("Ambient Temperature (°C)", min_value=5.0, max_value=45.0, value=28.0, step=0.5)
is_special_event = st.sidebar.checkbox("University Youth Fest / Exam Rush", value=False)

query_payload = {
    "route_id": route_choice,
    "departure_datetime": departure_dt.isoformat(),
    "bus_capacity": int(capacity_input),
    "is_raining": int(is_raining),
    "temperature_c": float(temperature),
    "is_special_event": int(is_special_event),
}

# =========================================================
# DASHBOARD BODY
# =========================================================
try:
    result = run_prediction_api_or_fallback(query_payload)
    pred_demand = result["predicted_passenger_demand"]
    occupancy = result["predicted_occupancy_percentage"]
    action_flag = result["action_flag"]
    recommendation = result["operational_recommendation"]
    backend_source = result.get("source", "Backend")

    # 1. TOP METRIC CARDS
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Predicted Demand",
            value=f"{pred_demand:.1f} pax",
            delta=f"{pred_demand - capacity_input:+.1f} vs. capacity",
            delta_color="inverse",
        )
    with m2:
        st.metric(
            label="Assigned Bus Capacity",
            value=f"{capacity_input} seats",
        )
    with m3:
        st.metric(
            label="Seat Occupancy",
            value=f"{occupancy:.1f}%",
            delta="Target: 50%–90%",
            delta_color="off",
        )
    with m4:
        st.metric(
            label="Serving Layer",
            value=result.get("model_name", "LightGBM"),
            delta=backend_source,
            delta_color="off",
        )

    st.write("")

    # 2. ACTION FLAG CALLOUT
    if action_flag == "OVERFLOW_RISK":
        st.error(
            f"### 🚨 ACTION FLAG: {action_flag} ({occupancy:.1f}% Occupancy)\n\n"
            f"**Operational Decision**: {recommendation}\n\n"
            f"> **Stranded Risk Warning**: True student demand ({pred_demand:.1f}) exceeds vehicle capacity ({capacity_input}). Dispatch a relief coach from the depot immediately."
        )
    elif action_flag == "OPTIMAL":
        st.success(
            f"### ✅ ACTION FLAG: {action_flag} ({occupancy:.1f}% Occupancy)\n\n"
            f"**Operational Decision**: {recommendation}\n\n"
            f"> **Balanced Fleet Utilization**: Passenger demand is safely absorbed with minimal empty seat waste."
        )
    else:  # UNDERUTILIZED
        st.info(
            f"### ℹ️ ACTION FLAG: {action_flag} ({occupancy:.1f}% Occupancy)\n\n"
            f"**Operational Decision**: {recommendation}\n\n"
            f"> **Cost Optimization Opportunity**: Over half the bus is running empty. Consider assigning a smaller minibus or merging with an adjacent departure."
        )

    # 3. INTERACTIVE 24-HOUR DEMONSTRATION TIMELINE (PLOTLY)
    st.write("---")
    st.subheader("📈 24-Hour Continuous Passenger Demand Forecast Curve")
    st.markdown(
        f"Hourly predicted demand on **{route_data['name']}** on **{dep_date.strftime('%A, %d %B %Y')}** with capacity threshold overlay:"
    )

    hours = list(range(24))
    hourly_demands = []
    hourly_rows = []
    for h in hours:
        h_dt = datetime.combine(dep_date, time(hour=h, minute=0))
        h_payload = {
            "route_id": route_choice,
            "departure_datetime": h_dt.isoformat(),
            "bus_capacity": int(capacity_input),
            "is_raining": int(is_raining),
            "temperature_c": float(temperature),
            "is_special_event": int(is_special_event),
        }
        res_h = run_prediction_api_or_fallback(h_payload)
        demand_val = res_h["predicted_passenger_demand"]
        occ_val = res_h["predicted_occupancy_percentage"]
        flag_val = res_h["action_flag"]
        hourly_demands.append(demand_val)
        hourly_rows.append(
            {
                "Date": dep_date.isoformat(),
                "Hour": f"{h:02d}:00",
                "Route_ID": route_choice,
                "Route_Name": route_data["name"],
                "Capacity": capacity_input,
                "Predicted_Demand": demand_val,
                "Occupancy_Pct": occ_val,
                "Action_Flag": flag_val,
            }
        )

    fig = go.Figure()

    # Demand line
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=hourly_demands,
            mode="lines+markers",
            name="Forecasted Passenger Demand",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=7),
            hovertemplate="Hour %{x}:00<br>Forecasted: %{y:.1f} pax<extra></extra>",
        )
    )

    # Capacity threshold line
    fig.add_trace(
        go.Scatter(
            x=[hours[0], hours[-1]],
            y=[capacity_input, capacity_input],
            mode="lines",
            name=f"Assigned Bus Capacity ({capacity_input} seats)",
            line=dict(color="#d62728", width=2.5, dash="dash"),
            hovertemplate=f"Bus Capacity: {capacity_input} seats<extra></extra>",
        )
    )

    # Highlight currently selected departure hour
    fig.add_trace(
        go.Scatter(
            x=[dep_hour],
            y=[pred_demand],
            mode="markers",
            name=f"Selected Trip ({dep_hour:02d}:{dep_minute:02d})",
            marker=dict(color="#ff7f0e", size=14, symbol="star"),
            hovertemplate=f"Current Selection (%{{x}}:{dep_minute:02d}): %{{y:.1f}} pax<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis=dict(
            title="Scheduled Departure Hour (Full 24-Hour Schedule)",
            tickmode="linear",
            tick0=0,
            dtick=1,
            gridcolor="#f0f2f6",
        ),
        yaxis=dict(title="Passenger Headcount (Unconstrained Demand)", gridcolor="#f0f2f6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # CSV Download Button for Dispatchers
    daily_schedule_df = pd.DataFrame(hourly_rows)
    csv_bytes = daily_schedule_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Export 24-Hour Schedule Forecast to CSV ({route_choice} - {dep_date})",
        data=csv_bytes,
        file_name=f"dispatch_forecast_{route_choice}_{dep_date}.csv",
        mime="text/csv",
        help="Download complete hourly schedule predictions with capacity action flags for transit operations.",
    )

    # 4. GEOSPATIAL CORRIDOR MAP & SEAT OCCUPANCY BREAKDOWN
    st.write("---")
    col_map, col_occ = st.columns([1.2, 1.0])

    with col_map:
        st.subheader("🗺️ Phagwara – Jalandhar Transit Corridor Map")
        st.caption(f"Real-world stop coordinates along **{route_data['name']}**")

        stops_df = pd.DataFrame(route_data["stops"])
        mid_lat = stops_df["lat"].mean()
        mid_lon = stops_df["lon"].mean()

        # Pydeck Scatterplot layer
        stop_layer = pdk.Layer(
            "ScatterplotLayer",
            data=stops_df,
            get_position=["lon", "lat"],
            get_color="[type == 'Origin' ? 46 : (type == 'Destination' ? 220 : 255), type == 'Origin' ? 139 : (type == 'Destination' ? 53 : 165), type == 'Origin' ? 87 : (type == 'Destination' ? 69 : 0), 220]",
            get_radius=350,
            pickable=True,
        )

        # Pydeck Line layer connecting the stops
        line_records = []
        for i in range(len(stops_df) - 1):
            line_records.append(
                {
                    "from_lon": stops_df.iloc[i]["lon"],
                    "from_lat": stops_df.iloc[i]["lat"],
                    "to_lon": stops_df.iloc[i + 1]["lon"],
                    "to_lat": stops_df.iloc[i + 1]["lat"],
                }
            )

        line_layer = pdk.Layer(
            "LineLayer",
            data=pd.DataFrame(line_records),
            get_source_position=["from_lon", "from_lat"],
            get_target_position=["to_lon", "to_lat"],
            get_color=[31, 119, 180, 200],
            get_width=5,
        )

        deck = pdk.Deck(
            layers=[line_layer, stop_layer],
            initial_view_state=pdk.ViewState(
                latitude=mid_lat,
                longitude=mid_lon,
                zoom=11.2,
                pitch=20,
            ),
            tooltip={"text": "{name}\nType: {type}"},
            map_style="light",
        )
        st.pydeck_chart(deck)

    with col_occ:
        st.subheader("💺 Vehicle Seat Occupancy Breakdown")
        st.caption(f"Capacity distribution for assigned **{capacity_input}-seater** bus:")

        seated = min(pred_demand, float(capacity_input))
        overflow = max(0.0, pred_demand - capacity_input)
        empty = max(0.0, capacity_input - pred_demand)

        breakdown_df = pd.DataFrame(
            [
                {"Category": "Seated Passengers", "Count": seated, "Color": "#2ca02c"},
                {"Category": "Overflow / Standing", "Count": overflow, "Color": "#d62728"},
                {"Category": "Unoccupied Empty Seats", "Count": empty, "Color": "#1f77b4"},
            ]
        )

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=breakdown_df["Category"],
                    values=breakdown_df["Count"],
                    hole=0.55,
                    marker=dict(colors=["#2ca02c", "#d62728", "#aec7e8"]),
                    textinfo="label+value",
                    hoverinfo="label+value+percent",
                )
            ]
        )
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown(
            f"- **Seated Passengers**: `{seated:.1f}`\n"
            f"- **Overflow / Stranded Risk**: `{overflow:.1f}`\n"
            f"- **Empty Seats Available**: `{empty:.1f}`"
        )

    # 5. WHAT-IF CAPACITY SIMULATOR
    st.write("---")
    st.subheader("📊 What-If Fleet Allocation Matrix (Exogenous Demand Consistency)")
    st.markdown(
        "Compare how alternative fleet vehicles perform for this exact trip without altering underlying student demand:"
    )

    simulation_capacities = [25, 35, 40, 50, 60, 75]
    sim_rows = []
    for cap in simulation_capacities:
        sim_occ = round((pred_demand / cap) * 100.0, 1)
        if sim_occ < 50.0:
            sim_flag = "UNDERUTILIZED ℹ️"
        elif sim_occ <= 90.0:
            sim_flag = "OPTIMAL ✅"
        else:
            sim_flag = "OVERFLOW_RISK 🚨"

        sim_rows.append(
            {
                "Vehicle Fleet Type": f"{cap}-Seat Class",
                "Seat Capacity": cap,
                "Predicted Demand": f"{pred_demand:.1f} pax",
                "Simulated Occupancy": f"{sim_occ}%",
                "Action Flag": sim_flag,
            }
        )
    st.table(pd.DataFrame(sim_rows))

    # 6. RAW FASTAPI REST API PAYLOAD INSPECTOR
    st.write("---")
    with st.expander("🔍 Inspect Raw FastAPI REST API Payload & Response (Reviewer Inspector)"):
        st.markdown("Verify the real production JSON contract served by `/predict`:")
        c_req, c_res = st.columns(2)
        with c_req:
            st.markdown("**Request Payload (`PredictRequest`)**")
            st.json(query_payload)
        with c_res:
            st.markdown("**Response Payload (`PredictResponse`)**")
            st.json(result)

except Exception as ex:
    st.error(f"Inference error: {ex}")
