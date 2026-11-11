"""
app.py
======
Section B(c) - Smart Campus Energy & EV Fleet Management dashboard.

A multi-page Streamlit application for campus facility managers:

    1. Executive Overview      - KPI header + grid / revenue / risk charts
    2. Alert Center            - temperature & overstay thresholds, alerts
    3. Data Viewer & Export    - filtered table + CSV download
    4. Hub Sales Analytics     - accessory / service sales by campus hub

The sidebar filters (date range, minimum energy, charger type, user type)
are applied once and shared by every page, so all visuals update together.

Run:  streamlit run app.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_pipeline import CLEAN_EV_FILE, CLEAN_SALES_FILE, run_pipeline

st.set_page_config(page_title="Smart Campus EV Platform", page_icon="⚡",
                   layout="wide")

CHARGER_COLORS = {"AC": "#1f77b4", "DC": "#ff7f0e"}
USER_ORDER = ["Student", "Staff", "Visitor"]
DEFAULT_TEMP_LIMIT = 45.0
DEFAULT_OVERSTAY_LIMIT = 20


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading cleaned campus data...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned files; rebuild them with the pipeline if missing.

    Returns:
        (energy_sessions, hub_sales) DataFrames with parsed dates.
    """
    try:
        if not (Path(CLEAN_EV_FILE).exists() and Path(CLEAN_SALES_FILE).exists()):
            run_pipeline()
        energy = pd.read_csv(CLEAN_EV_FILE, parse_dates=["Session_Date"])
        sales = pd.read_csv(CLEAN_SALES_FILE, parse_dates=["Transaction_Date"])
    except (OSError, pd.errors.ParserError, SystemExit) as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()
    return energy, sales


def get_thresholds() -> tuple[float, int]:
    """Current alert thresholds (kept in session_state across pages)."""
    return (st.session_state.get("temp_limit", DEFAULT_TEMP_LIMIT),
            st.session_state.get("overstay_limit", DEFAULT_OVERSTAY_LIMIT))


def high_risk_mask(df: pd.DataFrame, temp_limit: float,
                   overstay_limit: int) -> pd.Series:
    """Session is high risk when BOTH limits are exceeded."""
    return (df["Battery_Temp_C"] > temp_limit) & \
        (df["Overstay_Mins"] > overstay_limit)


# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
def sidebar_filters(energy: pd.DataFrame) -> dict:
    """Draw the sidebar widgets and return the chosen filter values."""
    sb = st.sidebar
    sb.header("🔎 Dashboard Filters")
    min_d = energy["Session_Date"].min().date()
    max_d = energy["Session_Date"].max().date()
    picked = sb.date_input("Session date range", value=(min_d, max_d),
                           min_value=min_d, max_value=max_d)
    # While the user is still choosing, date_input returns one date only.
    start, end = (picked if isinstance(picked, tuple) and len(picked) == 2
                  else (min_d, max_d))
    min_energy = sb.slider("Minimum energy delivered (kWh)", 0.0,
                           float(np.ceil(energy["Energy_kWh"].max())), 0.0,
                           step=0.5)
    chargers = sb.multiselect("Charger type", ["AC", "DC"],
                              default=["AC", "DC"])
    users = sb.multiselect("User type", USER_ORDER, default=USER_ORDER)
    sb.caption("Filters apply to every page and chart in real time.")
    return {"start": start, "end": end, "min_energy": min_energy,
            "chargers": chargers, "users": users}


def apply_filters(energy: pd.DataFrame, f: dict) -> pd.DataFrame:
    """Return only the sessions that satisfy every sidebar filter."""
    mask = (energy["Session_Date"].dt.date.between(f["start"], f["end"])
            & (energy["Energy_kWh"] >= f["min_energy"])
            & energy["Charger_Type"].isin(f["chargers"])
            & energy["User_Type"].isin(f["users"]))
    return energy.loc[mask]


# --------------------------------------------------------------------------
# Executive KPI header (shown on every page)
# --------------------------------------------------------------------------
def split_halves(df: pd.DataFrame, start: date, end: date
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the selected window into an earlier and a later half."""
    mid = start + (end - start) / 2
    dates = df["Session_Date"].dt.date
    return df[dates <= mid], df[dates > mid]


def render_kpis(df: pd.DataFrame, f: dict) -> None:
    """Four st.metric cards; delta = later half vs earlier half of range."""
    temp_limit, overstay_limit = get_thresholds()
    early, late = split_halves(df, f["start"], f["end"])
    metrics = [
        ("⚡ Total Energy Delivered", "Energy_kWh", "{:,.0f} kWh", "normal"),
        ("🌱 Carbon Offset", "Carbon_Savings_kgCO2", "{:,.0f} kg CO₂",
         "normal"),
        ("💰 Total Revenue", "Net_Billable_MYR", "RM {:,.2f}", "normal"),
    ]
    cols = st.columns(4)
    for col, (label, field, fmt, colour) in zip(cols, metrics):
        delta = late[field].sum() - early[field].sum()
        col.metric(label, fmt.format(df[field].sum()),
                   delta=fmt.format(delta).replace("RM -", "-RM "),
                   delta_color=colour, border=True,
                   help="Delta = second half of the selected dates minus "
                        "the first half.")
    alerts = int(high_risk_mask(df, temp_limit, overstay_limit).sum())
    alert_delta = int(high_risk_mask(late, temp_limit, overstay_limit).sum()
                      - high_risk_mask(early, temp_limit, overstay_limit).sum())
    cols[3].metric("🚨 Active Alert Count", alerts, delta=alert_delta,
                   delta_color="inverse", border=True,
                   help=f"Sessions with battery > {temp_limit:.0f} °C AND "
                        f"overstay > {overstay_limit} min.")


# --------------------------------------------------------------------------
# Page 1 - Executive Overview
# --------------------------------------------------------------------------
def page_overview() -> None:
    """Grid load, revenue and risk charts for directors."""
    df = st.session_state["filtered"]
    st.subheader("Executive Overview")
    if df.empty:
        st.info("No sessions match the current filters.")
        return

    left, right = st.columns(2)
    hourly = (df.groupby("Start_Hour")
              .agg(Grid_Peak=("Grid_Load_kW", "max"),
                   Solar=("Solar_Generation_kW", "mean"))
              .reindex(range(24)))
    fig = go.Figure()
    fig.add_vrect(x0=11.5, x1=17.5, fillcolor="red", opacity=0.07,
                  line_width=0, annotation_text="Peak tariff")
    fig.add_scatter(x=hourly.index, y=hourly["Grid_Peak"], name="Grid peak",
                    mode="lines+markers", line={"color": "#1b2a49",
                                                "width": 3},
                    connectgaps=True)
    fig.add_scatter(x=hourly.index, y=hourly["Solar"], name="Solar (mean)",
                    fill="tozeroy", line={"color": "#f2b705", "width": 3},
                    connectgaps=True)
    fig.update_layout(title="Hourly grid demand peak vs solar generation",
                      xaxis_title="Hour of day", yaxis_title="kW",
                      height=380, legend={"orientation": "h", "y": -0.25})
    left.plotly_chart(fig, width="stretch")

    rev = (df.groupby(["Start_Hour", "Charger_Type"], observed=True)
           ["Net_Billable_MYR"].sum().reset_index())
    fig = px.bar(rev, x="Start_Hour", y="Net_Billable_MYR",
                 color="Charger_Type", barmode="group",
                 color_discrete_map=CHARGER_COLORS,
                 labels={"Net_Billable_MYR": "Revenue (RM)",
                         "Start_Hour": "Start hour",
                         "Charger_Type": "Charger"},
                 title="Hourly charging revenue: AC vs DC", height=380)
    right.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    fig = px.box(df, x="User_Type", y="Net_Billable_MYR", color="User_Type",
                 category_orders={"User_Type": USER_ORDER}, points="outliers",
                 labels={"Net_Billable_MYR": "Net billable (RM)",
                         "User_Type": "User type"},
                 title="Session cost by user type", height=380)
    fig.update_layout(showlegend=False)
    left.plotly_chart(fig, width="stretch")

    fig = px.scatter(df, x="Battery_Temp_C", y="Energy_kWh",
                     color="Charger_Type", opacity=0.55,
                     color_discrete_map=CHARGER_COLORS,
                     labels={"Battery_Temp_C": "Battery temp (°C)",
                             "Energy_kWh": "Energy (kWh)"},
                     title="Battery temperature vs energy (with trendlines)",
                     height=380)
    for charger, sub in df.groupby("Charger_Type", observed=True):
        if len(sub) > 1:
            slope, icpt = np.polyfit(sub["Battery_Temp_C"], sub["Energy_kWh"], 1)
            xs = np.linspace(sub["Battery_Temp_C"].min(),
                             sub["Battery_Temp_C"].max(), 20)
            fig.add_scatter(x=xs, y=slope * xs + icpt, mode="lines",
                            name=f"{charger} trend",
                            line={"color": CHARGER_COLORS[charger],
                                  "dash": "dash"})
    fig.add_vline(x=get_thresholds()[0], line_dash="dot", line_color="red")
    right.plotly_chart(fig, width="stretch")

    daily = (df.groupby(df["Session_Date"].dt.date)
             .agg(Revenue=("Net_Billable_MYR", "sum"),
                  Energy=("Energy_kWh", "sum")).reset_index())
    fig = px.line(daily, x="Session_Date", y=["Revenue", "Energy"],
                  markers=True, title="Daily revenue (RM) and energy (kWh)",
                  labels={"value": "Value", "Session_Date": "Date",
                          "variable": "Metric"}, height=320)
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------
# Page 2 - High-Risk Overstay & Alert Center
# --------------------------------------------------------------------------
def page_alerts() -> None:
    """Threshold sliders that trigger warning / error banners."""
    df = st.session_state["filtered"]
    st.subheader("High-Risk Overstay & Alert Center")
    c1, c2 = st.columns(2)
    temp_limit = c1.slider("Battery temperature threshold (°C)", 30.0, 55.0,
                           float(get_thresholds()[0]), 0.5)
    overstay_limit = c2.slider("Overstay threshold (minutes)", 0, 60,
                               int(get_thresholds()[1]))
    st.session_state["temp_limit"] = temp_limit
    st.session_state["overstay_limit"] = overstay_limit

    hot = df["Battery_Temp_C"] > temp_limit
    risky = high_risk_mask(df, temp_limit, overstay_limit)
    hot_only = hot & ~risky

    if risky.any():
        st.error(f"🚨 {risky.sum()} HIGH-RISK session(s): battery above "
                 f"{temp_limit:.1f} °C AND overstay above {overstay_limit} "
                 f"min. Dispatch a technician and move the vehicles off "
                 f"the bay.")
    if hot_only.any():
        st.warning(f"⚠️ {hot_only.sum()} session(s) are overheating "
                   f"(> {temp_limit:.1f} °C) but left on time - monitor "
                   f"the charger cooling.")
    if not hot.any():
        st.success("✅ No sessions exceed the selected temperature threshold.")

    peak_grid = df["Grid_Load_kW"].max() if not df.empty else 0
    if peak_grid > 1000:
        st.warning(f"⚡ Grid demand reached {peak_grid:,.0f} kW in the "
                   "selected period - consider shifting DC charging out of "
                   "12:00-17:00.")

    k1, k2, k3 = st.columns(3)
    k1.metric("High-risk sessions", int(risky.sum()), border=True)
    k2.metric("Overheating only", int(hot_only.sum()), border=True)
    k3.metric("15% surcharge on high-risk (RM)",
              f"{(df.loc[risky, 'Base_Fee_MYR'] * 0.15).sum():,.2f}",
              border=True)

    flagged = risky | hot_only
    view = df.loc[flagged].assign(
        Risk_Level=np.where(risky[flagged], "HIGH RISK", "OVERHEATING"),
        Session_Date=df.loc[flagged, "Session_Date"].dt.strftime("%Y-%m-%d"))
    cols = ["Session_ID", "Session_Date", "Start_Hour", "Charger_Type",
            "User_Type", "Battery_Temp_C", "Overstay_Mins",
            "Net_Billable_MYR", "Risk_Level"]
    if view.empty:
        return
    left, right = st.columns([3, 2])
    left.dataframe(
        view[cols].sort_values(["Risk_Level", "Battery_Temp_C"],
                               ascending=[True, False])
        .style.apply(lambda r: ["background-color: #ffd6d6"
                                if r["Risk_Level"] == "HIGH RISK"
                                else "background-color: #fff3cd"] * len(r),
                     axis=1)
        .format({"Battery_Temp_C": "{:.1f}", "Net_Billable_MYR": "{:.2f}"}),
        hide_index=True, width="stretch", height=360)
    fig = px.scatter(df, x="Overstay_Mins", y="Battery_Temp_C",
                     color=np.where(risky, "High risk",
                                    np.where(hot, "Overheating", "Normal")),
                     color_discrete_map={"High risk": "#d62728",
                                         "Overheating": "#ffbf00",
                                         "Normal": "#9fb3c8"},
                     labels={"color": "Status", "Overstay_Mins":
                             "Overstay (min)", "Battery_Temp_C":
                             "Battery temp (°C)"},
                     title="Risk map", height=360)
    fig.add_hline(y=temp_limit, line_dash="dot", line_color="red")
    fig.add_vline(x=overstay_limit, line_dash="dot", line_color="red")
    right.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------
# Page 3 - Filtered Data Viewer & CSV Downloader
# --------------------------------------------------------------------------
def page_data() -> None:
    """Interactive table and CSV export of the filtered sessions."""
    df = st.session_state["filtered"]
    st.subheader("Filtered Data Viewer & CSV Export")
    default_cols = ["Session_ID", "Session_Date", "Start_Hour",
                    "Charger_Type", "User_Type", "Energy_kWh",
                    "Power_Rate_kW", "Carbon_Savings_kgCO2",
                    "Battery_Temp_C", "Overstay_Mins", "Net_Billable_MYR"]
    chosen = st.multiselect("Columns to display", list(df.columns),
                            default=default_cols)
    st.caption(f"Showing {len(df):,} of "
               f"{len(st.session_state['energy']):,} sessions.")
    st.dataframe(
        df[chosen or default_cols], hide_index=True, width="stretch",
        height=430,
        column_config={
            "Session_Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Power_Rate_kW": st.column_config.NumberColumn(format="%.2f"),
            "Duration_Hours": st.column_config.NumberColumn(format="%.2f"),
            "Base_Fee_MYR": st.column_config.NumberColumn(format="%.2f"),
            "Peak_Surge_MYR": st.column_config.NumberColumn(format="%.2f"),
        })
    csv_bytes = df.assign(Session_Date=df["Session_Date"].dt.date) \
        .to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data (CSV)", data=csv_bytes,
                       file_name="filtered_energy_sessions.csv",
                       mime="text/csv", type="primary")


# --------------------------------------------------------------------------
# Page 4 - Hub Sales Analytics (sales_data.csv)
# --------------------------------------------------------------------------
def page_sales() -> None:
    """Sales of services and accessories per hub (date filter applies)."""
    f = st.session_state["filters"]
    sales = st.session_state["sales"]
    sales = sales[sales["Transaction_Date"].dt.date.between(f["start"],
                                                            f["end"])]
    st.subheader("Hub Sales Analytics")
    st.caption("Source: sales_data.csv - the date filter applies here; "
               "charger/user filters relate to charging sessions only.")
    if sales.empty:
        st.info("No sales in the selected dates.")
        return
    a, b, c = st.columns(3)
    a.metric("Sales revenue", f"RM {sales['Total_Revenue_MYR'].sum():,.2f}",
             border=True)
    b.metric("Units sold", f"{sales['Units_Sold'].sum():,}", border=True)
    c.metric("Transactions", f"{len(sales):,}", border=True)
    left, right = st.columns(2)
    by_hub = (sales.groupby(["Hub_Location", "Category"])["Total_Revenue_MYR"]
              .sum().reset_index())
    left.plotly_chart(px.bar(by_hub, x="Hub_Location", y="Total_Revenue_MYR",
                             color="Category", title="Revenue by hub",
                             labels={"Total_Revenue_MYR": "Revenue (RM)",
                                     "Hub_Location": "Hub"}, height=380),
                      width="stretch")
    right.plotly_chart(px.pie(sales, names="Payment_Method",
                              values="Total_Revenue_MYR", hole=0.45,
                              title="Payment method share", height=380),
                       width="stretch")


# --------------------------------------------------------------------------
# App entry point
# --------------------------------------------------------------------------
def main() -> None:
    """Load data, apply shared filters, draw header and run the page."""
    energy, sales = load_data()
    filters = sidebar_filters(energy)
    filtered = apply_filters(energy, filters)
    st.session_state.update(energy=energy, sales=sales, filters=filters,
                            filtered=filtered)

    st.title("⚡ Smart Campus Energy & EV Fleet Platform")
    st.caption("ShopEasy Logistics × Taylor's Smart Campus Division · "
               f"{filters['start']:%d %b} – {filters['end']:%d %b %Y} · "
               f"{len(filtered):,} sessions")
    header = st.container()        # filled last so slider changes show now

    pages = st.navigation([
        st.Page(page_overview, title="Executive Overview", icon="📊",
                url_path="overview", default=True),
        st.Page(page_alerts, title="Alert Center", icon="🚨",
                url_path="alerts"),
        st.Page(page_data, title="Data Viewer & Export", icon="🗂️",
                url_path="data"),
        st.Page(page_sales, title="Hub Sales Analytics", icon="🏪",
                url_path="sales"),
    ])
    pages.run()
    with header:
        render_kpis(filtered, filters)
        st.divider()


if __name__ == "__main__":
    main()
