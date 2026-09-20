"""GRIDPOINT - Warehouse Location Optimization Platform.

Start the app with: streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from optimizer import (
    assign_nearest,
    assign_with_limits,
    evaluate_network,
    optimize_warehouses,
    suggested_existing_sites,
    tradeoff_curve,
)


APP_DIR = Path(__file__).parent
REQUIRED_COLUMNS = {"name", "lat", "lon", "orders"}
PALETTE = [
    [64, 99, 255], [16, 185, 129], [244, 114, 182], [245, 158, 11],
    [139, 92, 246], [6, 182, 212], [239, 68, 68], [132, 204, 22],
]


st.set_page_config(page_title="GRIDPOINT | Warehouse Planner", page_icon="📦", layout="wide")
st.markdown(
    """
    <style>
      .block-container { max-width: 1500px; padding-top: 2.2rem; padding-bottom: 3rem; }
      [data-testid="stMetricValue"] { font-size: 1.65rem; }
      .hero { padding: 1.5rem 1.7rem; border-radius: 18px; color: white;
              background: linear-gradient(125deg, #102a43 0%, #176b87 55%, #12a594 100%); }
      .hero h1 { margin: 0; font-size: 2.4rem; }
      .hero p { margin: .5rem 0 0; font-size: 1.05rem; opacity: .92; }
      .caption-card { border-left: 4px solid #12a594; padding: .5rem 1rem; background: #f0fdfa; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_data(raw_data: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """Validate the input format and return clean, optimization-ready data."""
    data = raw_data.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        return None, "Missing column(s): " + ", ".join(sorted(missing))
    data = data[["name", "lat", "lon", "orders"]].copy()
    data["name"] = data["name"].fillna("").astype(str).str.strip()
    for column in ("lat", "lon", "orders"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    invalid = (
        data["name"].eq("")
        | data[["lat", "lon", "orders"]].isna().any(axis=1)
        | ~data["lat"].between(-90, 90)
        | ~data["lon"].between(-180, 180)
        | (data["orders"] <= 0)
    )
    data = data.loc[~invalid].drop_duplicates(subset="name", keep="first").reset_index(drop=True)
    if len(data) < 2:
        return None, "Enter at least two valid, uniquely named neighbourhoods with positive daily orders."
    return data, None


def make_map(points: np.ndarray, names: pd.Series, orders: np.ndarray, warehouses: np.ndarray, assignment: np.ndarray) -> pdk.Deck:
    """Create one readable map with demand points, allocation lines, and warehouses."""
    rows = []
    for index, (lat, lon) in enumerate(points):
        warehouse_idx = int(assignment[index])
        color = PALETTE[warehouse_idx % len(PALETTE)] if warehouse_idx >= 0 else [130, 130, 130]
        rows.append(
            {
                "name": names.iloc[index], "lat": lat, "lon": lon, "orders": int(orders[index]),
                "assignment": f"Warehouse {warehouse_idx + 1}" if warehouse_idx >= 0 else "Unserved",
                "color": color, "radius": 350 + np.sqrt(orders[index]) * 75,
                "warehouse_lat": warehouses[warehouse_idx, 0] if warehouse_idx >= 0 else lat,
                "warehouse_lon": warehouses[warehouse_idx, 1] if warehouse_idx >= 0 else lon,
            }
        )
    demand = pd.DataFrame(rows)
    warehouse_data = pd.DataFrame(
        [
            {"name": f"Warehouse {index + 1}", "lat": lat, "lon": lon, "color": PALETTE[index % len(PALETTE)]}
            for index, (lat, lon) in enumerate(warehouses)
        ]
    )
    layers = [
        pdk.Layer("LineLayer", demand, get_source_position="[lon, lat]", get_target_position="[warehouse_lon, warehouse_lat]", get_color="color", get_width=2, pickable=False),
        pdk.Layer("ScatterplotLayer", demand, get_position="[lon, lat]", get_fill_color="color", get_radius="radius", opacity=0.78, pickable=True),
        pdk.Layer("ScatterplotLayer", warehouse_data, get_position="[lon, lat]", get_fill_color="color", get_line_color=[255, 255, 255], stroked=True, line_width_min_pixels=3, get_radius=1200, pickable=True),
    ]
    view = pdk.ViewState(latitude=float(points[:, 0].mean()), longitude=float(points[:, 1].mean()), zoom=10)
    return pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text": "{name}\nOrders/day: {orders}\nAssigned: {assignment}"})


def warehouse_summary(data: pd.DataFrame, warehouses: np.ndarray, assignment: np.ndarray, capacity: float | None) -> pd.DataFrame:
    rows = []
    for warehouse_idx, (lat, lon) in enumerate(warehouses):
        served = assignment == warehouse_idx
        daily_orders = float(data.loc[served, "orders"].sum())
        rows.append(
            {
                "warehouse": f"Warehouse {warehouse_idx + 1}",
                "latitude": round(float(lat), 5),
                "longitude": round(float(lon), 5),
                "neighbourhoods": int(served.sum()),
                "orders/day": int(daily_orders),
                "capacity used": f"{daily_orders / capacity:.0%}" if capacity else "No limit",
            }
        )
    return pd.DataFrame(rows)


st.markdown(
    """<div class="hero"><h1>📦 GRIDPOINT</h1><p>Turn neighbourhood demand into smarter warehouse locations.</p></div>""",
    unsafe_allow_html=True,
)
st.caption("An educational planning prototype. Distances are straight-line estimates; final business decisions need road, traffic, rent, and service data.")

with st.sidebar:
    st.header("1. Demand data")
    data_source = st.radio("Choose a starting point", ["Bengaluru demo data", "Upload my CSV"], label_visibility="collapsed")
    uploaded = st.file_uploader("CSV columns: name, lat, lon, orders", type="csv") if data_source == "Upload my CSV" else None
    try:
        input_data = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(APP_DIR / "sample_data.csv")
    except Exception as error:
        st.error(f"Could not read that CSV: {error}")
        st.stop()

with st.expander("✏️ View or edit neighbourhood data", expanded=False):
    st.caption("Add rows or update daily order counts before running the recommendation.")
    edited_data = st.data_editor(input_data, num_rows="dynamic", use_container_width=True, hide_index=True, key="demand_editor")
data, error_message = clean_data(edited_data)
if error_message:
    st.error(error_message)
    st.stop()

points = data[["lat", "lon"]].to_numpy(dtype=float)
orders = data["orders"].to_numpy(dtype=float)

with st.sidebar:
    st.header("2. Planning settings")
    warehouse_count = st.slider("Warehouses to plan", 1, min(8, len(data)), min(3, len(data)))
    cost_rate = st.number_input("Delivery cost per km per order (₹)", min_value=0.0, value=8.0, step=1.0)
    suggested_indices = suggested_existing_sites(points, orders, warehouse_count)
    suggested_names = data.iloc[suggested_indices]["name"].tolist()
    existing_names = st.multiselect(
        "Current warehouse sites (for comparison)",
        data["name"].tolist(),
        default=suggested_names,
        max_selections=warehouse_count,
        key=f"existing_sites_{warehouse_count}_{hash(tuple(data['name']))}",
        help="Choose the neighbourhoods closest to your company's current warehouse sites. The preselected sites are a neutral, spread-out demo baseline.",
    )

    st.header("3. Service rules (optional)")
    enforce_capacity = st.checkbox("Limit daily capacity")
    capacity = st.number_input("Orders per warehouse per day", min_value=100, max_value=100000, value=1500, step=100) if enforce_capacity else None
    enforce_radius = st.checkbox("Limit delivery radius")
    service_radius = st.number_input("Maximum delivery radius (km)", min_value=1.0, max_value=200.0, value=15.0, step=1.0) if enforce_radius else None
    run = st.button("Find best locations", type="primary", use_container_width=True)

tab_overview, tab_solution, tab_tradeoff, tab_explain = st.tabs(["🗺️ Demand overview", "✨ Recommended network", "⚖️ Warehouse trade-off", "🧠 Explain it simply"])

with tab_overview:
    st.subheader("Where is demand today?")
    st.map(data.rename(columns={"lat": "latitude", "lon": "longitude"}), latitude="latitude", longitude="longitude", size="orders", color="#12a594")
    left, middle, right = st.columns(3)
    left.metric("Neighbourhoods", len(data))
    middle.metric("Daily orders", f"{int(orders.sum()):,}")
    right.metric("Highest-demand area", data.loc[data["orders"].idxmax(), "name"])
    st.dataframe(data.sort_values("orders", ascending=False), use_container_width=True, hide_index=True)

with tab_solution:
    if not run:
        st.info("Set your assumptions in the sidebar, then select **Find best locations**.")
        st.markdown("<div class='caption-card'>Tip: Start with the demo data, choose three warehouses, and use the preset comparison sites. That gives you a presentation-ready first scenario in seconds.</div>", unsafe_allow_html=True)
    elif len(existing_names) != warehouse_count:
        st.warning(f"Choose exactly {warehouse_count} current warehouse site(s) to make a fair comparison.")
    else:
        current_rows = data.set_index("name").loc[existing_names]
        current_warehouses = current_rows[["lat", "lon"]].to_numpy(dtype=float)
        current_assignment = assign_nearest(points, current_warehouses)
        current_result = evaluate_network(points, orders, current_warehouses, current_assignment, cost_rate)

        optimized_warehouses, _ = optimize_warehouses(points, orders, warehouse_count)
        optimized_assignment, unserved = assign_with_limits(points, orders, optimized_warehouses, capacity, service_radius)
        optimized_result = evaluate_network(points, orders, optimized_warehouses, optimized_assignment, cost_rate)

        potential_saving = current_result.delivery_cost - optimized_result.delivery_cost
        coverage_is_complete = len(unserved) == 0
        st.subheader("A network shaped by demand, not guesswork")
        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Current delivery cost / day", f"₹{current_result.delivery_cost:,.0f}")
        metric_2.metric("Optimized cost / day", f"₹{optimized_result.delivery_cost:,.0f}")
        metric_3.metric("Average distance / order", f"{optimized_result.average_distance_km:.2f} km", f"{optimized_result.average_distance_km - current_result.average_distance_km:+.2f} km", delta_color="inverse")
        metric_4.metric("Demand covered", f"{optimized_result.coverage_percent:.0f}%", f"{optimized_result.served_orders:,.0f} / {optimized_result.total_orders:,.0f} orders")

        if coverage_is_complete:
            percent_saving = 100 * potential_saving / current_result.delivery_cost if current_result.delivery_cost else 0
            st.success(f"Estimated delivery saving: **₹{potential_saving:,.0f}/day ({percent_saving:.1f}%)** while serving every listed order.")
        else:
            unserved_names = ", ".join(data.iloc[unserved]["name"].tolist())
            st.warning(f"The selected service rules leave {len(unserved)} neighbourhood(s) unserved: {unserved_names}. Do not treat the lower delivery cost as a true saving until coverage is restored.")

        before, after = st.columns(2)
        with before:
            st.markdown("#### Current network")
            st.pydeck_chart(make_map(points, data["name"], orders, current_warehouses, current_assignment), use_container_width=True)
        with after:
            st.markdown("#### Recommended network")
            st.pydeck_chart(make_map(points, data["name"], orders, optimized_warehouses, optimized_assignment), use_container_width=True)

        st.markdown("#### Recommended warehouse locations")
        st.dataframe(warehouse_summary(data, optimized_warehouses, optimized_assignment, capacity), use_container_width=True, hide_index=True)

        assignment_table = data.copy()
        assignment_table["recommended_warehouse"] = [f"Warehouse {value + 1}" if value >= 0 else "UNSERVED" for value in optimized_assignment]
        assignment_table["estimated_distance_km"] = optimized_result.distances_km.round(2)
        assignment_table["estimated_delivery_cost_₹"] = (optimized_result.distances_km * assignment_table["orders"] * cost_rate).round(0)
        st.markdown("#### Neighbourhood assignment")
        st.dataframe(assignment_table, use_container_width=True, hide_index=True)
        st.download_button("Download recommended assignments", assignment_table.to_csv(index=False).encode("utf-8"), "gridpoint_assignments.csv", "text/csv")

with tab_tradeoff:
    st.subheader("What is the right number of warehouses?")
    daily_facility_cost = st.number_input("Daily operating cost per warehouse (₹)", min_value=0.0, value=20_000.0, step=1_000.0, help="Use a daily estimate (rent, staffing, utilities), so it can be compared fairly against daily delivery cost.")
    curve = tradeoff_curve(points, orders, min(8, len(data)), cost_rate, daily_facility_cost)
    best_row = curve.loc[curve["total_daily_cost"].idxmin()]
    st.success(f"At these assumptions, **{int(best_row['warehouses'])} warehouse(s)** has the lowest estimated total daily cost: ₹{best_row['total_daily_cost']:,.0f}.")
    st.line_chart(curve.set_index("warehouses")[["delivery_cost", "facility_cost", "total_daily_cost"]], color=["#ef4444", "#f59e0b", "#0f766e"])
    st.dataframe(curve.round(0), use_container_width=True, hide_index=True)

with tab_explain:
    st.subheader("How to explain GRIDPOINT to a judge")
    st.markdown(
        """
        **The problem:** Delivering from a warehouse far away is more expensive, and a high-order neighbourhood matters more than a low-order one.

        **Our idea:** GRIDPOINT repeatedly groups each neighbourhood with its nearest warehouse, then moves that warehouse toward the centre of the demand it serves. High-order areas pull the location more strongly. It tests several starting layouts and keeps the lowest-cost result.

        **The result:** Teams can compare their current network with a demand-aware layout, see every assignment on a map, and test whether capacity, radius, and facility costs change the recommendation.

        **Honest limitation:** We use straight-line geographical distance as a fast planning estimate. A real rollout would add road routes, traffic, delivery time, property cost, and customer service data.
        """
    )
    st.markdown("#### 45-second demo order")
    st.markdown("1. Show the demand map and point out the largest order bubbles.\n2. Choose three warehouses and press **Find best locations**.\n3. Compare the before/after maps and savings metric.\n4. Turn on a tight capacity or radius limit to show the real-world constraint warning.\n5. Open the trade-off tab and explain the delivery-cost vs facility-cost decision.")
