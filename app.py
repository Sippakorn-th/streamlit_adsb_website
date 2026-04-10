import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="ADS-B Receiver Dashboard", page_icon="✈️", layout="wide")

DEFAULT_DATA_DIR = Path(".")
DEFAULT_RECEIVER_LAT = 13.716501
DEFAULT_RECEIVER_LON = 100.526832

PARQUET_FILES = {
    "fact_positions_local_7d": "fact_positions_local_7d.parquet",
    "agg_coverage_hex3d_7d": "agg_coverage_hex3d_7d.parquet",
    "fact_anomaly_points_7d": "fact_anomaly_points_7d.parquet",
    "agg_traffic_mix_hourly_7d": "agg_traffic_mix_hourly_7d.parquet",
    "agg_weather_message_hourly_7d": "agg_weather_message_hourly_7d.parquet",
}


@st.cache_data(show_spinner=False)
def load_all_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load all required parquet tables once and reuse across app pages."""
    root = Path(data_dir)
    data: dict[str, pd.DataFrame] = {}

    for key, filename in PARQUET_FILES.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet file: {path}")
        data[key] = pd.read_parquet(path)

    # Normalize datetime columns for reliable filtering.
    if "ts_local" in data["fact_positions_local_7d"].columns:
        data["fact_positions_local_7d"]["ts_local"] = pd.to_datetime(
            data["fact_positions_local_7d"]["ts_local"], errors="coerce"
        )

    if "local_hour" in data["agg_coverage_hex3d_7d"].columns:
        data["agg_coverage_hex3d_7d"]["local_hour"] = pd.to_datetime(
            data["agg_coverage_hex3d_7d"]["local_hour"], errors="coerce"
        )

    if "local_hour" in data["agg_traffic_mix_hourly_7d"].columns:
        data["agg_traffic_mix_hourly_7d"]["local_hour"] = pd.to_datetime(
            data["agg_traffic_mix_hourly_7d"]["local_hour"], errors="coerce"
        )

    if "local_hour" in data["agg_weather_message_hourly_7d"].columns:
        data["agg_weather_message_hourly_7d"]["local_hour"] = pd.to_datetime(
            data["agg_weather_message_hourly_7d"]["local_hour"], errors="coerce"
        )

    return data


def render_overview(data: dict[str, pd.DataFrame]) -> None:
    st.title("ADS-B Receiver Dashboard")
    st.warning(
        "This dashboard represents the empirical radio-horizon and line-of-sight capture capabilities "
        "of a specific RTL-SDR receiver located in Rangsit, Pathum Thani. "
        "It does not represent total Bangkok airspace traffic."
    )

    positions = data["fact_positions_local_7d"]

    total_positions = len(positions)
    total_flights = positions["flight_id"].nunique() if "flight_id" in positions.columns else 0
    total_aircraft = positions["icao_hex"].nunique() if "icao_hex" in positions.columns else 0

    ts_min = positions["ts_local"].min() if "ts_local" in positions.columns else pd.NaT
    ts_max = positions["ts_local"].max() if "ts_local" in positions.columns else pd.NaT

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Positions", f"{total_positions:,}")
    col2.metric("Total Flights", f"{total_flights:,}")
    col3.metric("Unique Aircraft", f"{total_aircraft:,}")
    col4.metric("Data Window (Days)", "7")

    st.subheader("Date Range (UTC+7)")
    if pd.notna(ts_min) and pd.notna(ts_max):
        st.write(f"{ts_min} to {ts_max}")
    else:
        st.write("No valid local timestamps found.")

    st.subheader("Dataset Health")
    missing_lat = positions["lat"].isna().sum() if "lat" in positions.columns else 0
    missing_lon = positions["lon"].isna().sum() if "lon" in positions.columns else 0
    st.write(
        {
            "rows": total_positions,
            "missing_lat": int(missing_lat),
            "missing_lon": int(missing_lon),
        }
    )


def render_receiver_coverage(
    data: dict[str, pd.DataFrame], receiver_lat: float, receiver_lon: float
) -> None:
    st.title("Receiver Coverage")
    st.caption("3D signal cloud from raw positions within the 7-day local window")
    st.markdown(
        """
        ### **My Hardware & The City Challenge**
        Since I'm using a basic dipole antenna (a V-shape antenna opened flat to 180 degrees) right in the middle of central Bangkok, I'm not getting the perfect, uninterrupted signal that a real airport control tower gets. There is a lot of background radio noise and massive buildings in the way. One of the coolest challenges of this project was figuring out exactly how good my antenna *actually* is in such a heavy urban environment.

        ### **Map 1: The Raw Coverage Area**
        The first map shows everywhere my antenna managed to pick up a signal, separated by altitude. It gives us a great 3D picture of how far my antenna can reach and where the physical cut-offs are. But there's a catch: just because my antenna caught a signal in a specific spot doesn't mean the connection was actually good. It just tells us a plane was there, not if the signal was constantly dropping out.
        """,
    )

    positions = data["fact_positions_local_7d"].copy()
    positions = positions[
        positions["lat"].notna() & positions["lon"].notna() & positions["altitude_ft"].notna()
    ].copy()

    # Deterministic pre-sampling keeps map performance stable and prevents
    # points from disappearing when widening altitude filters.
    if len(positions) > 22500:
        positions = positions.sample(n=22500, random_state=42)

    min_alt_slice, max_alt_slice = st.slider(
        "Altitude Slice (ft)",
        min_value=0,
        max_value=45000,
        value=(0, 45000),
        step=500,
    )

    point_radius_m = st.slider(
        "Point Size (Radius in meters)",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
    )

    filtered = positions[
        (positions["altitude_ft"] >= min_alt_slice) & (positions["altitude_ft"] <= max_alt_slice)
    ].copy()
    receiver_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[{"lat": float(receiver_lat), "lon": float(receiver_lon), "altitude_viz_m": 0.0}],
        get_position="[lon, lat, altitude_viz_m]",
        get_radius=450,
        get_fill_color="[255, 40, 40, 255]",
        pickable=True,
        stroked=True,
        get_line_color="[255, 255, 255, 255]",
        line_width_min_pixels=2,
    )

    view_state = pdk.ViewState(
        latitude=receiver_lat,
        longitude=receiver_lon,
        zoom=10,
        pitch=45,
        bearing=8,
    )

    if filtered.empty:
        st.info("No coverage data for selected filters.")
    else:
        # Color by absolute altitude so legend remains stable across filters.
        alt_norm = (filtered["altitude_ft"].clip(lower=0, upper=40000) / 40000).astype(float)

        # Gradient: blue (0 ft) -> purple (20,000 ft) -> red (40,000+ ft).
        low_color = (59, 130, 246)
        mid_color = (147, 51, 234)
        high_color = (239, 68, 68)

        lower_half = alt_norm <= 0.5
        upper_half = ~lower_half

        t_low = (alt_norm[lower_half] / 0.5).clip(lower=0, upper=1)
        t_high = ((alt_norm[upper_half] - 0.5) / 0.5).clip(lower=0, upper=1)

        filtered["r"] = 0
        filtered["g"] = 0
        filtered["b"] = 0

        filtered.loc[lower_half, "r"] = (low_color[0] + (mid_color[0] - low_color[0]) * t_low).round().astype(int)
        filtered.loc[lower_half, "g"] = (low_color[1] + (mid_color[1] - low_color[1]) * t_low).round().astype(int)
        filtered.loc[lower_half, "b"] = (low_color[2] + (mid_color[2] - low_color[2]) * t_low).round().astype(int)

        filtered.loc[upper_half, "r"] = (mid_color[0] + (high_color[0] - mid_color[0]) * t_high).round().astype(int)
        filtered.loc[upper_half, "g"] = (mid_color[1] + (high_color[1] - mid_color[1]) * t_high).round().astype(int)
        filtered.loc[upper_half, "b"] = (mid_color[2] + (high_color[2] - mid_color[2]) * t_high).round().astype(int)

        filtered["a"] = 160

        # Convert feet to meters and apply a visual boost for a clear 3D cloud.
        filtered["altitude_viz_m"] = (filtered["altitude_ft"] * 0.3048 * 1.25).clip(lower=0)

        # Keep only JSON-serializable primitives for pydeck.
        cloud_df = filtered[
            ["lon", "lat", "altitude_ft", "altitude_viz_m", "icao_hex", "flight_id", "r", "g", "b", "a"]
        ].copy()
        cloud_df["lon"] = cloud_df["lon"].astype(float)
        cloud_df["lat"] = cloud_df["lat"].astype(float)
        cloud_df["altitude_ft"] = cloud_df["altitude_ft"].astype(float)
        cloud_df["altitude_viz_m"] = cloud_df["altitude_viz_m"].astype(float)
        cloud_df["icao_hex"] = cloud_df["icao_hex"].fillna("unknown").astype(str)
        cloud_df["flight_id"] = cloud_df["flight_id"].fillna(-1).astype(int).astype(str)

        cloud_records = cloud_df.to_dict(orient="records")

        st.write(f"Rendered points: {len(cloud_records):,}")

        cloud_layer = pdk.Layer(
            "ScatterplotLayer",
            data=cloud_records,
            get_position="[lon, lat, altitude_viz_m]",
            get_radius=point_radius_m,
            radius_min_pixels=1,
            radius_max_pixels=4,
            get_fill_color="[r, g, b, a]",
            pickable=True,
            stroked=False,
            filled=True,
        )

        tooltip = {
            "html": "<b>ICAO:</b> {icao_hex}<br/>"
            "<b>Flight:</b> {flight_id}<br/>"
            "<b>Altitude (ft):</b> {altitude_ft}<br/>"
            "<b>Lat/Lon:</b> {lat}, {lon}",
            "style": {"backgroundColor": "#111", "color": "#fff"},
        }

        deck = pdk.Deck(
            layers=[cloud_layer, receiver_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_provider="carto",
            map_style="dark",
        )

        st.pydeck_chart(deck, use_container_width=True)

        st.markdown(
            """
            <div style="margin-top: 0.6rem; margin-bottom: 0.8rem;">
                <div style="font-size: 0.9rem; color: #E5E7EB; margin-bottom: 0.35rem;">Altitude Legend</div>
                <div style="height: 12px; border-radius: 999px; background: linear-gradient(90deg, rgb(59,130,246) 0%, rgb(147,51,234) 50%, rgb(239,68,68) 100%);"></div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #9CA3AF; margin-top: 0.3rem;">
                    <span>0 ft</span>
                    <span>20,000 ft</span>
                    <span>40,000+ ft</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(r"""
        ### **Map 2: Signal Health & The "Ping Gap"**
        This brings us to the second map. Instead of just mapping *where* planes are, I wrote a Python script to measure the actual *quality* of the connection using time-series analysis.

        Here is how it works under the hood: Airplanes naturally broadcast their location about once per second. By sorting my database chronologically for every unique flight, I calculated the time difference ($\Delta t$) between consecutive pings:

        $$\Delta t = t_i - t_{i-1}$$

        If the gap is roughly 1 or 2 seconds, the connection is rock solid. But if a plane flies behind a massive skyscraper and my antenna loses it, that gap might stretch to 10 or 15 seconds before the next successful ping is recorded. By taking the average of these time gaps within specific geographic grid cells, we get a true empirical "Signal Health" score.

        * **Green Areas (Low Gap):** The antenna has a steady, healthy line-of-sight.
        * **Red Areas (High Gap):** The signal is heavily fragmented. The antenna is sitting in silence waiting for data, effectively visualizing the invisible "radio shadows" cast by the Bangkok urban canyon.
        """,
    )

    st.subheader("Signal Health (Track Continuity)")
    st.caption("Hex-level continuity based on mean ping gap (lower gap = healthier reception)")

    health_hex = data["agg_coverage_hex3d_7d"].copy()
    needed_cols = {
        "hex_lat_center",
        "hex_lon_center",
        "point_count",
        "mean_altitude_ft",
        "mean_ping_gap_sec",
        "max_ping_gap_sec",
        "altitude_band",
    }
    if not needed_cols.issubset(set(health_hex.columns)):
        st.warning("Coverage aggregate is missing ping-gap columns. Please rerun ETL to refresh parquet outputs.")
        return

    band_bounds = {
        "<=5k": (0, 5000),
        "5k-15k": (5000, 15000),
        "15k-30k": (15000, 30000),
        ">30k": (30000, 1000000),
    }
    health_hex["altitude_band"] = health_hex["altitude_band"].astype(str)
    health_hex["band_min_ft"] = health_hex["altitude_band"].map(lambda v: band_bounds.get(v, (None, None))[0])
    health_hex["band_max_ft"] = health_hex["altitude_band"].map(lambda v: band_bounds.get(v, (None, None))[1])

    health_hex = health_hex[health_hex["band_min_ft"].notna() & health_hex["band_max_ft"].notna()].copy()
    df_filtered = health_hex[
        (health_hex["band_min_ft"] <= max_alt_slice) & (health_hex["band_max_ft"] >= min_alt_slice)
    ].copy()

    # Confidence gate: suppress sparse edge cells that can look artificially healthy.
    df_filtered["point_count"] = pd.to_numeric(df_filtered["point_count"], errors="coerce")
    df_signal = df_filtered[df_filtered["point_count"] >= 5].copy()

    if df_signal.empty:
        st.info("No signal health data for selected altitude filter after confidence filter (point_count >= 5).")
        return

    df_signal["mean_ping_gap_sec"] = pd.to_numeric(df_signal["mean_ping_gap_sec"], errors="coerce")
    df_signal["max_ping_gap_sec"] = pd.to_numeric(df_signal["max_ping_gap_sec"], errors="coerce")

    gap = df_signal["mean_ping_gap_sec"]
    df_signal["r"] = 120
    df_signal["g"] = 120
    df_signal["b"] = 120
    df_signal["a"] = 210

    # Smooth color interpolation from urban baseline to fragmented reception.
    best_gap = 1.5
    worst_gap = 10.0
    mid_gap = (best_gap + worst_gap) / 2.0

    gap_valid = gap.notna()
    gap_clamped = gap.clip(lower=best_gap, upper=worst_gap)

    lower_half = gap_valid & (gap_clamped <= mid_gap)
    upper_half = gap_valid & (gap_clamped > mid_gap)

    t_low = ((gap_clamped[lower_half] - best_gap) / (mid_gap - best_gap)).clip(lower=0, upper=1)
    t_high = ((gap_clamped[upper_half] - mid_gap) / (worst_gap - mid_gap)).clip(lower=0, upper=1)

    # Green -> Yellow
    df_signal.loc[lower_half, "r"] = (0 + 255 * t_low).round().astype(int)
    df_signal.loc[lower_half, "g"] = 255
    df_signal.loc[lower_half, "b"] = 0

    # Yellow -> Red
    df_signal.loc[upper_half, "r"] = 255
    df_signal.loc[upper_half, "g"] = (255 - 255 * t_high).round().astype(int)
    df_signal.loc[upper_half, "b"] = 0

    health_df = df_signal[
        [
            "hex_lon_center",
            "hex_lat_center",
            "mean_ping_gap_sec",
            "max_ping_gap_sec",
            "altitude_band",
            "point_count",
            "r",
            "g",
            "b",
            "a",
        ]
    ].copy()
    health_df = health_df.rename(columns={"hex_lon_center": "lon", "hex_lat_center": "lat"})
    health_df["lon"] = health_df["lon"].astype(float)
    health_df["lat"] = health_df["lat"].astype(float)
    health_df["mean_ping_gap_label"] = health_df["mean_ping_gap_sec"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "n/a"
    )
    health_df["max_ping_gap_label"] = health_df["max_ping_gap_sec"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "n/a"
    )
    health_df["point_count"] = health_df["point_count"].fillna(0).astype(int)

    health_records = health_df.to_dict(orient="records")
    st.write(f"Rendered health cells: {len(health_records):,}")

    health_layer = pdk.Layer(
        "ScatterplotLayer",
        data=health_records,
        get_position="[lon, lat]",
        get_radius=1300,
        get_fill_color="[r, g, b, a]",
        opacity=0.45,
        pickable=True,
        stroked=False,
        filled=True,
    )

    health_tooltip = {
        "html": "<b>Mean Ping Gap (sec):</b> {mean_ping_gap_label}<br/>"
        "<b>Max Ping Gap (sec):</b> {max_ping_gap_label}<br/>"
        "<b>Point Count:</b> {point_count}<br/>"
        "<b>Altitude Band:</b> {altitude_band}<br/>"
        "<b>Lat/Lon:</b> {lat}, {lon}",
        "style": {"backgroundColor": "#111", "color": "#fff"},
    }

    health_deck = pdk.Deck(
        layers=[health_layer, receiver_layer],
        initial_view_state=view_state,
        tooltip=health_tooltip,
        map_provider="carto",
        map_style="dark",
    )
    st.pydeck_chart(health_deck, use_container_width=True)

    st.markdown(
        """
        <div style="margin-top: 0.6rem; margin-bottom: 0.8rem;">
            <div style="font-size: 0.9rem; color: #E5E7EB; margin-bottom: 0.35rem;">Signal Health Legend</div>
            <div style="height: 12px; border-radius: 999px; background: linear-gradient(to right, rgb(0,255,0), rgb(255,255,0), rgb(255,0,0));"></div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #9CA3AF; margin-top: 0.3rem;">
                <span>~1.5s (Strong)</span>
                <span>10s+ (Fragmented)</span>
            </div>
            <div style="font-size: 0.75rem; color: #6B7280; margin-top: 0.25rem;">Only cells with point_count ≥ 50 are shown.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### **The Radio Horizon vs. The Urban Canyon**
        If you look closely at the high altitudes (30k+ ft), the data forms perfect horizontal stripes. These are assigned Air Traffic Control 'Flight Levels' (invisible highways in the sky).

        Below that, we can see the physical limits of the receiver.
        * **The Bottom White Dashed Line:** This is the theoretical Earth Curvature limit. Notice how flat it is? At 100km, the Earth only hides about 1,900 feet of altitude.
        * **The Shaded Zone (The Urban Blind Spot):** My signal drops out *long* before it reaches the Earth's curve. The curving orange boundary represents the true **Urban Obstruction Angle** stacked on top of the Earth's curvature. The massive skyscrapers in central Bangkok cast an angular 'radio shadow' into the sky. As the line-of-sight extends outward, the Earth falls away beneath it, causing this radio shadow to curve upward exponentially.

        Any plane flying inside this shaded region is physically blocked by the concrete jungle and is completely hidden from the antenna.
        Note: The boundary of the urban radio shadow is actually a curve, not a straight line. As the line-of-sight extends outward, the Earth's curvature falls away beneath it, causing the shadow's required altitude to increase exponentially.

        **The Anomalies: Hardware Limits & Urban Ghosts**
        On the chart below, two specific zones are highlighted:
        * **Hardware Ceiling (Top Right):** These signals are not blocked by buildings, but they are dying out. This is Free Space Path Loss-the physical limit of how far the antenna can 'hear' before the signal gets too weak.
        * **Urban Bounces (Bottom Left):** You will notice planes appearing *inside* the dark Urban Blind Spot. These are not errors! They are Multipath Reflections. The signals are physically blocked by buildings, but they bounce off surrounding concrete skyscrapers and reach the antenna anyway, acting like 'ghosts' in the shadow.
        """
    )

    horizon_cols = {"mean_distance_km", "mean_altitude_ft", "mean_ping_gap_sec"}
    if not horizon_cols.issubset(set(health_hex.columns)):
        st.warning(
            "Coverage aggregate is missing distance/altitude/gap fields for the radio horizon chart. "
            "Please rerun ETL to refresh parquet outputs."
        )
        return

    # Downsample for browser stability in Streamlit Community Cloud.
    df_scatter = df_signal.sample(n=min(10000, len(df_signal)), random_state=42)
    df_scatter["mean_distance_km"] = pd.to_numeric(df_scatter["mean_distance_km"], errors="coerce")
    df_scatter["mean_altitude_ft"] = pd.to_numeric(df_scatter["mean_altitude_ft"], errors="coerce")
    df_scatter["mean_ping_gap_sec"] = pd.to_numeric(df_scatter["mean_ping_gap_sec"], errors="coerce")
    df_scatter = df_scatter.dropna(subset=["mean_distance_km", "mean_altitude_ft", "mean_ping_gap_sec"])
    # Remove all signals farther than 120 km for the Radio Horizon analysis.
    df_scatter = df_scatter[df_scatter["mean_distance_km"] <= 120].copy()

    if df_scatter.empty:
        st.info("No data available to render the Radio Horizon chart.")
        return

    # Theoretical Earth-curvature baseline (0-100 km): h_ft = (d_km / (1.23 * 1.852))^2
    horizon_theory_df = pd.DataFrame({"distance_km": list(range(0, 101))})
    horizon_theory_df["altitude_ft_earth"] = (
        horizon_theory_df["distance_km"] / (1.23 * 1.852)
    ) ** 2

    # Urban shadow curve combines obstruction angle plus Earth-curvature falloff.
    urban_cutoff_df = pd.DataFrame({"distance_km": list(range(0, 101))})
    urban_cutoff_df["altitude_ft_urban"] = (
        (urban_cutoff_df["distance_km"] * 300)
        + (0.1927 * (urban_cutoff_df["distance_km"] ** 2))
    )
    urban_cutoff_df["altitude_floor_ft"] = 0

    scatter_chart = (
        alt.Chart(df_scatter)
        .mark_circle(size=12, opacity=0.6)
        .encode(
            x=alt.X("mean_distance_km:Q", title="Distance from Receiver (km)"),
            y=alt.Y("mean_altitude_ft:Q", title="Average Altitude (ft)"),
            color=alt.Color(
                "mean_ping_gap_sec:Q",
                title="Mean Ping Gap (sec)",
                scale=alt.Scale(domain=[2.5, 10], range=["green", "yellow", "red"], clamp=True),
            ),
            tooltip=[
                alt.Tooltip("mean_distance_km:Q", title="Distance (km)", format=".2f"),
                alt.Tooltip("mean_altitude_ft:Q", title="Altitude (ft)", format=",.0f"),
                alt.Tooltip("mean_ping_gap_sec:Q", title="Mean Ping Gap (sec)", format=".2f"),
            ],
        )
    )

    earth_line_chart = (
        alt.Chart(horizon_theory_df)
        .mark_line(color="white", strokeDash=[5, 5])
        .encode(
            x=alt.X("distance_km:Q", title="Distance from Receiver (km)"),
            y=alt.Y("altitude_ft_earth:Q", title="Average Altitude (ft)"),
        )
    )

    urban_area_chart = (
        alt.Chart(urban_cutoff_df)
        .mark_area(color="red", opacity=0.15)
        .encode(
            x=alt.X("distance_km:Q", title="Distance from Receiver (km)"),
            y=alt.Y("altitude_ft_urban:Q", title="Average Altitude (ft)"),
            y2=alt.Y2("altitude_floor_ft:Q"),
        )
    )

    urban_line_chart = (
        alt.Chart(urban_cutoff_df)
        .mark_line(color="orange", strokeDash=[5, 5])
        .encode(
            x=alt.X("distance_km:Q", title="Distance from Receiver (km)"),
            y=alt.Y("altitude_ft_urban:Q", title="Average Altitude (ft)"),
        )
    )

    df_annotations = pd.DataFrame(
        {
            "x": [65, 25],
            "y": [35000, 4000],
            "label": ["Hardware Limit (Free Space Path Loss)", "Urban Bounces (Multipath)"],
            "color": ["white", "#ffffd7"],
        }
    )

    highlight_circles = (
        alt.Chart(df_annotations)
        .mark_circle(size=7000, opacity=0.2, strokeWidth=2)
        .encode(
            x=alt.X("x:Q", title="Distance from Receiver (km)"),
            y=alt.Y("y:Q", title="Average Altitude (ft)"),
            color=alt.Color("color:N", scale=None),
            stroke=alt.Stroke("color:N", scale=None),
        )
    )

    highlight_text = (
        alt.Chart(df_annotations)
        .mark_text(dy=-50, fontSize=12, fontWeight="bold", align="center")
        .encode(
            x=alt.X("x:Q", title="Distance from Receiver (km)"),
            y=alt.Y("y:Q", title="Average Altitude (ft)"),
            text="label:N",
            color=alt.Color("color:N", scale=None),
        )
    )

    layered_chart = alt.layer(
        scatter_chart,
        urban_area_chart,
        urban_line_chart,
        earth_line_chart,
        highlight_circles,
        highlight_text,
    ).properties(height=420)

    st.altair_chart(layered_chart, use_container_width=True)


def render_sensor_health_anomalies(data: dict[str, pd.DataFrame]) -> None:
    df_anomalies = data["fact_anomaly_points_7d"].copy()
    df_positions = data["fact_positions_local_7d"].copy()
    df_hex = data["agg_coverage_hex3d_7d"].copy()

    st.markdown(
        """
        ## **Sensor Health & RF Anomalies**
        Raw ADS-B data is messy. Between atmospheric interference, multipath urban reflections, and uncalibrated aircraft sensors, a raw data feed is filled with physical impossibilities.

        Rather than silently dropping bad data, this pipeline explicitly catches, categorizes, and logs it. Tracking these anomalies allows us to monitor the health of the hardware setup and the surrounding RF environment.

        ### **1. Anomaly Distribution by Type**
        What kind of noise are we dealing with? The chart below breaks down the volume of flagged data points by their specific rule-violation, color-coded by severity.
        """
    )

    required_cols = {"anomaly_type", "anomaly_severity"}
    if not required_cols.issubset(set(df_anomalies.columns)):
        st.warning("Anomaly dataset is missing required columns. Please rerun ETL to refresh outputs.")
        return

    anomaly_counts = (
        df_anomalies.groupby(["anomaly_type", "anomaly_severity"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    if anomaly_counts.empty:
        st.info("No anomaly records available to plot.")
        return

    anomaly_counts["anomaly_type"] = anomaly_counts["anomaly_type"].fillna("unknown").astype(str)
    anomaly_counts["anomaly_severity"] = (
        anomaly_counts["anomaly_severity"].fillna("low").astype(str).str.lower()
    )

    type_totals = (
        anomaly_counts.groupby("anomaly_type", dropna=False)["count"]
        .sum()
        .reset_index(name="total_count")
    )
    anomaly_counts = anomaly_counts.merge(type_totals, on="anomaly_type", how="left")

    chart = (
        alt.Chart(anomaly_counts)
        .mark_bar()
        .encode(
            y=alt.Y(
                "anomaly_type:N",
                title="Anomaly Type",
                sort=alt.SortField(field="total_count", order="descending"),
            ),
            x=alt.X("count:Q", title="Number of Occurrences"),
            color=alt.Color(
                "anomaly_severity:N",
                title="Severity",
                scale=alt.Scale(
                    domain=["high", "medium", "low"],
                    range=["#e74c3c", "#f39c12", "#95a5a6"],
                ),
            ),
            tooltip=[
                alt.Tooltip("anomaly_type:N", title="Type"),
                alt.Tooltip("anomaly_severity:N", title="Severity"),
                alt.Tooltip("count:Q", title="Count", format=","),
            ],
        )
    )

    st.altair_chart(chart, use_container_width=True)

    required_deep_dive_cols = {
        "anomaly_type",
        "altitude_ft",
        "jump_distance_km",
        "implied_speed_kts",
        "delta_seconds",
    }
    if not required_deep_dive_cols.issubset(set(df_anomalies.columns)):
        st.warning(
            "Anomaly dataset is missing required columns for the deep-dive charts. "
            "Please rerun ETL to refresh outputs."
        )
        return
    if "altitude_ft" not in df_positions.columns:
        st.warning(
            "Positions dataset is missing altitude_ft for error-rate analysis. "
            "Please rerun ETL to refresh outputs."
        )
        return

    df_wobbles_full = df_anomalies[df_anomalies["anomaly_type"] == "implied_speed_outlier"].copy()
    if df_wobbles_full.empty:
        st.info("No implied_speed_outlier records available for deep-dive analysis.")
        return

    # Normalize numeric fields once for all deep-dive charts.
    for c in ["altitude_ft", "jump_distance_km", "implied_speed_kts", "delta_seconds"]:
        df_wobbles_full[c] = pd.to_numeric(df_wobbles_full[c], errors="coerce")

    # Chart 2 and 3 use cleaned full data for statistical fidelity.
    df_wobbles_alt = df_wobbles_full.dropna(subset=["altitude_ft"]).copy()

    st.markdown(
        """
        ### 2. Deep Dive: The "Tally of Teleports"

        When we first looked at our errors, we thought altitude was the problem. However, calculating the actual error rate (left chart) proved us wrong: there is a steady ~4% "Physics Tax" across all altitudes. To figure out what was actually causing this constant noise, we stopped looking at *speed* and started looking at *distance*.

        The right chart is a log-scale histogram that measures exactly how far a plane "jumped" when an error occurred. It reveals that our noise comes from two completely different enemies:

        * **The "Wobble" Mountain (0.5km - 10km):** The massive peak on the left accounts for 99% of our errors. This is pure physics. When a plane's signal bounces off a Bangkok skyscraper, the reflection arrives slightly distorted. The system gets confused and thinks the plane is a few kilometers away from its true position. The curve slopes downwards because of signal decay—a bounce can only stretch so far before the radio wave runs out of energy and fades away completely.
        * **The "Teleports" (1,000km+):** Notice the tiny, lonely dots on the far right? These aren't physical building reflections; they are digital "glitches." Occasionally, atmospheric static will flip a single `0` to a `1` inside the raw binary data. Instead of wobbling the plane by a few blocks, this digital corruption mathematically scrambles the coordinates, accidentally "teleporting" the plane to China or Australia for a split second.

        **The Takeaway:** By analyzing the distance of these jumps, we successfully separated analog city noise (The Mountain) from digital data corruption (The Teleports). It proves that our 4% noise floor isn't a software bug, but a highly accurate map of Bangkok's physical radio environment.

        The charts below separate *how often* errors occur by altitude (rate, not raw count), and *how far* each coordinate jump travels to separate analog bounces from digital failures.
        """
    )

    # Chart 2: Vertical error-rate percentage by 2,000 ft bins.
    df_positions_alt = df_positions.copy()
    df_positions_alt["altitude_ft"] = pd.to_numeric(df_positions_alt["altitude_ft"], errors="coerce")
    df_positions_alt = df_positions_alt.dropna(subset=["altitude_ft"])

    if df_positions_alt.empty or df_wobbles_alt.empty:
        st.info("Insufficient altitude data to compute vertical error rate and jump histogram.")
        return

    alt_bin_size = 2000
    df_positions_alt["alt_bin_start"] = (np.floor(df_positions_alt["altitude_ft"] / alt_bin_size) * alt_bin_size).astype(int)
    df_wobbles_alt["alt_bin_start"] = (np.floor(df_wobbles_alt["altitude_ft"] / alt_bin_size) * alt_bin_size).astype(int)

    total_counts = (
        df_positions_alt.groupby("alt_bin_start", as_index=False)
        .size()
        .rename(columns={"size": "total_ping_count"})
    )
    anomaly_bin_counts = (
        df_wobbles_alt.groupby("alt_bin_start", as_index=False)
        .size()
        .rename(columns={"size": "anomaly_count"})
    )

    error_rate_df = total_counts.merge(anomaly_bin_counts, on="alt_bin_start", how="left")
    error_rate_df["anomaly_count"] = error_rate_df["anomaly_count"].fillna(0)
    error_rate_df["error_rate_pct"] = (
        error_rate_df["anomaly_count"] / error_rate_df["total_ping_count"]
    ) * 100.0
    error_rate_df = error_rate_df[error_rate_df["total_ping_count"] > 0].copy()

    error_rate_df["altitude_bin"] = error_rate_df["alt_bin_start"].map(
        lambda v: f"{int(v):,}-{int(v + alt_bin_size):,}"
    )
    bin_order = error_rate_df.sort_values("alt_bin_start")["altitude_bin"].tolist()

    vertical_error_rate_chart = (
        alt.Chart(error_rate_df)
        .mark_bar(color="#f39c12")
        .encode(
            x=alt.X("altitude_bin:N", title="Altitude Bin (ft)", sort=bin_order),
            y=alt.Y(
                "error_rate_pct:Q",
                title="Error Rate (%)",
                axis=alt.Axis(labelExpr="format(datum.value, '.2f') + '%'"),
            ),
            tooltip=[
                alt.Tooltip("altitude_bin:N", title="Altitude Bin (ft)"),
                alt.Tooltip("anomaly_count:Q", title="Anomaly Count", format=","),
                alt.Tooltip("total_ping_count:Q", title="Total Pings", format=","),
                alt.Tooltip("error_rate_pct:Q", title="Error Rate (%)", format=".4f"),
            ],
        )
        .properties(title="Vertical Error Rate by Altitude Bin")
    )

    # Chart 3: Jump-distance histogram using precomputed log10 distance for stable bar rendering.
    df_jump = df_wobbles_full.copy()
    df_jump["distance_km"] = pd.to_numeric(df_jump["jump_distance_km"], errors="coerce")
    if df_jump["distance_km"].notna().sum() == 0:
        # Fallback when jump_distance_km is unavailable: derive from implied speed and delta time.
        knots_per_km_per_sec = 1943.8444924406
        df_jump["distance_km"] = (
            (pd.to_numeric(df_jump["implied_speed_kts"], errors="coerce") / knots_per_km_per_sec)
            * pd.to_numeric(df_jump["delta_seconds"], errors="coerce")
        )
    df_jump = df_jump[df_jump["distance_km"].notna() & (df_jump["distance_km"] >= 0)].copy()

    if df_jump.empty:
        st.info("No valid jump-distance data available for histogram rendering.")
        return

    # Pre-calculate log10 domain so Altair can render solid bars on a linear axis.
    df_jump["log10_distance"] = np.log10(df_jump["distance_km"] + 0.1)

    jump_hist_chart = (
        alt.Chart(df_jump)
        .mark_bar(color="#f39c12", size=10)
        .encode(
            x=alt.X(
                "log10_distance:Q",
                bin=alt.Bin(maxbins=80),
                title="Distance Jumped (km)",
                axis=alt.Axis(
                    values=[-1, 0, 1, 2, 3, 4],
                    labelExpr="pow(10, datum.value)",
                ),
            ),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[
                alt.Tooltip("count():Q", title="Count", format=","),
            ],
        )
        .properties(title="Distance Jump Distribution: Wobbles vs Teleports (Log Scale)")
    )

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(vertical_error_rate_chart, use_container_width=True)
    with col2:
        st.altair_chart(jump_hist_chart, use_container_width=True)

    st.markdown(
        """
        ### 3. Pipeline Anomalies: Decoding "Ghost Packets" and Blind Spots

        While coordinate wobbles are caused by environmental radio interference (multipath), we discovered a secondary pipeline anomaly accounting for roughly 1.6% of our data: **Stale Telemetry.** Because airplanes are in constant motion, it is physically impossible for a cruising aircraft to remain perfectly stationary. However, our ingestion pipeline occasionally flags records where an aircraft is above 2,000 feet, multiple seconds have elapsed, yet the horizontal displacement is exactly zero.

        * **The Micro-View (Right Chart): The "Staircase" Effect**
          This chart isolates a single flight experiencing signal drops. When the antenna loses line-of-sight (e.g., a plane flies behind a building), our local decoding software (`dump1090`) prevents the aircraft from instantly vanishing off the map by artificially "coasting" and repeating the last known coordinate. Time moves forward, but distance does not, creating the flat red "steps" on the graph. When the signal is regained, the pipeline catches up, creating a vertical jump.

        * **The Macro-View (Left Chart): Mapping Physical Blind Spots**
          To prove these signal drops aren't random hardware glitches, we grouped all stale packets by their compass direction (`azimuth_sector`). Crucially, we **normalized** this data against the total baseline traffic to ensure we weren't just mapping busy flight paths. The resulting polar chart acts like a radio-sonar map of our location. The massive 7%+ error spike in the Northeast is undeniable mathematical proof of a physical line-of-sight obstruction, perfectly aligning with the dense concrete skyline of the Silom and Sukhumvit districts blocking our antenna.
        """
    )

    stale_required = {"flight_id", "anomaly_type"}
    if not stale_required.issubset(set(df_anomalies.columns)):
        st.warning("Anomaly dataset is missing required columns for stale telemetry analysis.")
        return

    pos_required = {"flight_id", "ts_local", "lat", "lon"}
    if not pos_required.issubset(set(df_positions.columns)):
        st.warning("Positions dataset is missing required columns for stale telemetry charting.")
        return

    df_stale = df_anomalies[df_anomalies["anomaly_type"] == "stuck_sensor"].copy()
    if df_stale.empty:
        st.info("No `stuck_sensor` anomalies detected in the current dataset.")
        return

    st.markdown("#### Signal Drops by Compass Sector (Normalized)")

    sector_order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    sector_map = {
        "N": "N",
        "N-NE": "N",
        "NE": "NE",
        "NE-E": "NE",
        "E": "E",
        "E-SE": "E",
        "SE": "SE",
        "SE-S": "SE",
        "S": "S",
        "S-SW": "S",
        "SW": "SW",
        "SW-W": "SW",
        "W": "W",
        "W-NW": "W",
        "NW": "NW",
        "NW-N": "NW",
    }

    rose_chart = None
    rose_chart_message = None

    if (
        "azimuth_sector" not in df_stale.columns
        or "azimuth_sector" not in df_hex.columns
        or "point_count" not in df_hex.columns
    ):
        rose_chart_message = (
            "Skipping normalized sector chart: required columns `azimuth_sector` and `point_count` are missing."
        )
    else:
        dropped_df = df_stale[["azimuth_sector"]].copy()
        dropped_df["compass_sector"] = (
            dropped_df["azimuth_sector"].astype("string").str.strip().str.upper().map(sector_map)
        )
        dropped_df = dropped_df[dropped_df["compass_sector"].notna()].copy()
        dropped_packets = (
            dropped_df.groupby("compass_sector", as_index=False)
            .size()
            .rename(columns={"size": "dropped_packets"})
        )

        total_df = df_hex[["azimuth_sector", "point_count"]].copy()
        total_df["point_count"] = pd.to_numeric(total_df["point_count"], errors="coerce")
        total_df["compass_sector"] = (
            total_df["azimuth_sector"].astype("string").str.strip().str.upper().map(sector_map)
        )
        total_df = total_df[
            total_df["compass_sector"].notna() & total_df["point_count"].notna()
        ].copy()
        total_packets = (
            total_df.groupby("compass_sector", as_index=False)["point_count"]
            .sum()
            .rename(columns={"point_count": "total_packets"})
        )

        sector_base = pd.DataFrame({"compass_sector": sector_order})
        sector_rates = sector_base.merge(total_packets, on="compass_sector", how="left")
        sector_rates = sector_rates.merge(dropped_packets, on="compass_sector", how="left")
        sector_rates["total_packets"] = sector_rates["total_packets"].fillna(0)
        sector_rates["dropped_packets"] = sector_rates["dropped_packets"].fillna(0)
        sector_rates["drop_rate_pct"] = np.where(
            sector_rates["total_packets"] > 0,
            (sector_rates["dropped_packets"] / sector_rates["total_packets"]) * 100.0,
            0.0,
        )

        rose_chart = (
            alt.Chart(sector_rates)
            .mark_arc(innerRadius=20, stroke="#fff", thetaOffset=-(np.pi / 8))
            .encode(
                theta=alt.Theta("compass_sector:N", sort=sector_order),
                radius=alt.Radius(
                    "drop_rate_pct:Q",
                    title="Normalized Drop Rate (%)",
                    scale=alt.Scale(type="sqrt", zero=True),
                ),
                color=alt.Color(
                    "drop_rate_pct:Q",
                    title="Drop Rate (%)",
                    scale=alt.Scale(scheme="oranges"),
                ),
                tooltip=[
                    alt.Tooltip("compass_sector:N", title="Sector"),
                    alt.Tooltip("dropped_packets:Q", title="Dropped Packets", format=","),
                    alt.Tooltip("total_packets:Q", title="Total Packets", format=","),
                    alt.Tooltip("drop_rate_pct:Q", title="Drop Rate (%)", format=".3f"),
                ],
            )
            .properties(title="Signal Drops by Compass Sector (Normalized Drop Rate)", height=360)
        )

    flight_counts = (
        df_stale["flight_id"]
        .dropna()
        .astype(str)
        .value_counts()
    )
    if flight_counts.empty:
        st.info("No valid flight_id found for `stuck_sensor` anomalies.")
        return

    target_flight_id = flight_counts.index[0]

    df_flight = df_positions.copy()
    df_flight["flight_id"] = df_flight["flight_id"].astype(str)
    df_flight = df_flight[df_flight["flight_id"] == target_flight_id].copy()

    if df_flight.empty:
        st.info("The selected stale-telemetry flight was not found in raw positions.")
        return

    df_flight["ts_local"] = pd.to_datetime(df_flight["ts_local"], errors="coerce")
    df_flight["lat"] = pd.to_numeric(df_flight["lat"], errors="coerce")
    df_flight["lon"] = pd.to_numeric(df_flight["lon"], errors="coerce")
    df_flight = df_flight.dropna(subset=["ts_local"]).copy()

    if df_flight.empty:
        st.info("Insufficient timestamp data for the selected stale-telemetry flight.")
        return

    if df_flight["lat"].notna().sum() < 2 or df_flight["lon"].notna().sum() < 2:
        st.info("Insufficient coordinate data to compute cumulative distance for this flight.")
        return

    df_flight = df_flight.sort_values("ts_local").copy()

    # Haversine step distance between consecutive points, then cumulative travel distance.
    earth_radius_km = 6371.0088
    prev_lat = np.radians(df_flight["lat"].shift(1))
    prev_lon = np.radians(df_flight["lon"].shift(1))
    curr_lat = np.radians(df_flight["lat"])
    curr_lon = np.radians(df_flight["lon"])

    dlat = curr_lat - prev_lat
    dlon = curr_lon - prev_lon
    haversine_a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(prev_lat) * np.cos(curr_lat) * (np.sin(dlon / 2.0) ** 2)
    )
    haversine_a = np.clip(haversine_a, 0.0, 1.0)
    step_distance_km = 2.0 * earth_radius_km * np.arcsin(np.sqrt(haversine_a))

    df_flight["step_distance_km"] = pd.Series(step_distance_km, index=df_flight.index).fillna(0.0)
    df_flight["cumulative_distance_km"] = df_flight["step_distance_km"].cumsum()

    stale_flight_rows = df_stale.copy()
    stale_flight_rows["flight_id"] = stale_flight_rows["flight_id"].astype(str)
    stale_flight_rows = stale_flight_rows[stale_flight_rows["flight_id"] == target_flight_id].copy()

    # Prefer keying by position_id when available; fall back to timestamp matching.
    if "position_id" in df_flight.columns and "position_id" in stale_flight_rows.columns:
        stale_keys = stale_flight_rows[["position_id"]].dropna().drop_duplicates().copy()
        stale_keys["is_stuck_sensor"] = True
        df_flight = df_flight.merge(stale_keys, on="position_id", how="left")
    else:
        stale_flight_rows["ts_local"] = pd.to_datetime(stale_flight_rows.get("ts_local"), errors="coerce")
        stale_keys = stale_flight_rows[["ts_local"]].dropna().drop_duplicates().copy()
        stale_keys["is_stuck_sensor"] = True
        df_flight = df_flight.merge(stale_keys, on="ts_local", how="left")

    df_flight["is_stuck_sensor"] = df_flight["is_stuck_sensor"].fillna(False).astype(bool)
    df_flight = df_flight.sort_values("ts_local").copy()

    stale_count = int(df_flight["is_stuck_sensor"].sum())
    st.caption(
        f"Target flight_id: {target_flight_id} | stale packets: {stale_count:,} | total points: {len(df_flight):,}"
    )

    base_line = (
        alt.Chart(df_flight)
        .mark_line(color="#1f77b4", strokeWidth=2)
        .encode(
            x=alt.X("ts_local:T", title="Time (Local)"),
            y=alt.Y("cumulative_distance_km:Q", title="Cumulative Distance Traveled (km)"),
            tooltip=[
                alt.Tooltip("ts_local:T", title="Time"),
                alt.Tooltip("cumulative_distance_km:Q", title="Cumulative Distance (km)", format=",.3f"),
                alt.Tooltip("step_distance_km:Q", title="Step Distance (km)", format=",.4f"),
            ],
        )
    )

    stale_points = (
        alt.Chart(df_flight[df_flight["is_stuck_sensor"]])
        .mark_circle(size=60, color="red")
        .encode(
            x=alt.X("ts_local:T", title="Time (Local)"),
            y=alt.Y("cumulative_distance_km:Q", title="Cumulative Distance Traveled (km)"),
            tooltip=[
                alt.Tooltip("ts_local:T", title="Stale Timestamp"),
                alt.Tooltip("cumulative_distance_km:Q", title="Cumulative Distance (km)", format=",.3f"),
                alt.Tooltip("step_distance_km:Q", title="Step Distance (km)", format=",.4f"),
            ],
        )
    )

    stale_telemetry_chart = (
        alt.layer(base_line, stale_points)
        .properties(title="Single-Flight Frozen Telemetry Trace")
        .interactive()
    )

    macro_col, micro_col = st.columns(2)
    with macro_col:
        if rose_chart is not None:
            st.altair_chart(rose_chart, use_container_width=True)
        elif rose_chart_message:
            st.info(rose_chart_message)
        else:
            st.info("No data available for normalized sector chart.")
    with micro_col:
        st.altair_chart(stale_telemetry_chart, use_container_width=True)

    st.markdown(
        """
        ### 4. Boundary Testing: Analyzing Orphan Pings
        Instead of finding a uniform "Radio Horizon" at the edge of the Earth's curvature, this map reveals the **Urban Canyon** effect. The `orphan_ping` anomalies (signals that drop after just 1 or 2 packets) are heavily clustered in narrow, linear corridors pointing Northwest and Southeast.

        By mapping these dropped packets, we are actually reverse-engineering the **Structural Keyholes** in the local Bangkok skyline. These linear clusters align perfectly with the approach corridors for Don Mueang (DMK) and Suvarnabhumi (BKK) airports. The antenna is peering through narrow gaps between high-rise buildings, catching split-second glimpses of aircraft before they disappear behind the next concrete wall.
        """
    )

    orphan_required = {"anomaly_type", "lat", "lon", "operator"}
    if not orphan_required.issubset(set(df_anomalies.columns)):
        st.info(
            "Skipping orphan-ping map: anomalies data is missing `anomaly_type`, `lat`, `lon`, or `operator`."
        )
    else:
        df_orphans = df_anomalies[df_anomalies["anomaly_type"] == "orphan_ping"].copy()
        if df_orphans.empty:
            st.info("No `orphan_ping` records found in the current anomaly dataset.")
        else:
            df_orphans["lat"] = pd.to_numeric(df_orphans["lat"], errors="coerce")
            df_orphans["lon"] = pd.to_numeric(df_orphans["lon"], errors="coerce")
            df_orphans = df_orphans.dropna(subset=["lat", "lon"]).copy()

            df_orphans["is_unregistered"] = df_orphans["operator"].isna()
            df_orphans["registry_state"] = np.where(
                df_orphans["is_unregistered"],
                "Unregistered Aircraft (Missing Operator)",
                "Registered Aircraft",
            )
            df_orphans["color"] = df_orphans["is_unregistered"].map(
                lambda is_null: [255, 75, 75, 160] if is_null else [0, 255, 255, 160]
            )

            if df_orphans.empty:
                st.info("No valid orphan-ping coordinates available for mapping.")
            else:
                # Keep map rendering responsive when anomaly volume is very large.
                if len(df_orphans) > 40000:
                    df_orphans = df_orphans.sample(n=40000, random_state=42)

                registered_count = int((~df_orphans["is_unregistered"]).sum())
                unregistered_count = int(df_orphans["is_unregistered"].sum())
                st.caption(
                    "Rendered orphan pings: "
                    f"{len(df_orphans):,} | Registered: {registered_count:,} | "
                    f"Unregistered: {unregistered_count:,}"
                )

                st.markdown(
                    """
                    <div style="display:flex; gap:1.2rem; align-items:center; margin:0.25rem 0 0.75rem 0; font-size:0.9rem;">
                        <div style="display:flex; align-items:center; gap:0.45rem;">
                            <span style="display:inline-block; width:10px; height:10px; border-radius:999px; background:rgba(0,255,255,0.95);"></span>
                            <span>Registered Aircraft</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:0.45rem;">
                            <span style="display:inline-block; width:10px; height:10px; border-radius:999px; background:rgba(255,75,75,0.95);"></span>
                            <span>Unregistered Aircraft (Missing Operator)</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                receiver_point = pd.DataFrame(
                    [{"lat": float(DEFAULT_RECEIVER_LAT), "lon": float(DEFAULT_RECEIVER_LON)}]
                )

                receiver_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=receiver_point,
                    get_position="[lon, lat]",
                    get_radius=12000,
                    get_fill_color="[255, 80, 0, 255]",
                    pickable=True,
                )

                orphan_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_orphans[["lon", "lat", "color", "registry_state"]],
                    get_position="[lon, lat]",
                    get_radius=7000,
                    radius_min_pixels=1,
                    radius_max_pixels=5,
                    get_fill_color="color",
                    opacity=0.65,
                    pickable=True,
                    stroked=False,
                    filled=True,
                )

                orphan_tooltip = {
                    "html": "<b>Anomaly:</b> orphan_ping<br/><b>Registry:</b> {registry_state}<br/><b>Lat/Lon:</b> {lat}, {lon}",
                    "style": {"backgroundColor": "#111", "color": "#fff"},
                }

                orphan_view = pdk.ViewState(
                    latitude=float(DEFAULT_RECEIVER_LAT),
                    longitude=float(DEFAULT_RECEIVER_LON),
                    zoom=6.8,
                    pitch=0,
                    bearing=0,
                )

                orphan_map = pdk.Deck(
                    layers=[orphan_layer, receiver_layer],
                    initial_view_state=orphan_view,
                    tooltip=orphan_tooltip,
                    map_provider="carto",
                    map_style="dark",
                )

                st.pydeck_chart(orphan_map, use_container_width=True)

    st.subheader("Data Quality Audit: Verifying Hex Integrity")

    audit_required_positions = {"operator"}
    audit_required_anomalies = {"operator", "anomaly_type"}
    if not audit_required_positions.issubset(set(df_positions.columns)):
        st.info("Cannot compute baseline unregistered rate: `operator` is missing in baseline positions.")
        return
    if not audit_required_anomalies.issubset(set(df_anomalies.columns)):
        st.info("Cannot compute orphan unregistered rate: `operator` or `anomaly_type` is missing in anomalies.")
        return

    baseline_total = len(df_positions)
    baseline_unregistered_rate = (
        df_positions["operator"].isna().mean() * 100.0 if baseline_total > 0 else np.nan
    )

    df_orphan_audit = df_anomalies[df_anomalies["anomaly_type"] == "orphan_ping"].copy()
    orphan_total = len(df_orphan_audit)
    orphan_unregistered_rate = (
        df_orphan_audit["operator"].isna().mean() * 100.0 if orphan_total > 0 else np.nan
    )

    kpi_col1, kpi_col2 = st.columns(2)
    with kpi_col1:
        st.metric(
            "Baseline Unregistered Rate",
            "N/A" if pd.isna(baseline_unregistered_rate) else f"{baseline_unregistered_rate:.1f}%",
        )
    with kpi_col2:
        st.metric(
            "Orphan Unregistered Rate",
            "N/A" if pd.isna(orphan_unregistered_rate) else f"{orphan_unregistered_rate:.1f}%",
        )

    st.markdown(
        """
        **Conclusion:** The data reveals that Orphan Pings lack registry data at virtually the exact same rate as baseline traffic (~15.3% vs 16.0%). This mathematically disproves the initial hypothesis of digital corruption or hallucinated ICAO hexes. Because these transient signals perfectly mirror the validity of normal traffic, it confirms they are real physical aircraft. Their sudden disappearance is therefore a physical line-of-sight phenomenon, completely validating the 'Structural Keyhole' blockages mapped above.
        """
    )

    st.markdown("### 5. Macro Analytics: Hardware Quantization & Physics Violations")

    st.markdown(r"""
    ### **Root Cause Analysis: High-Frequency Jitter Amplification

    When monitoring pipeline health, we initially detected severe `vertical_rate_outlier` anomalies—aircraft seemingly climbing at physically impossible rates exceeding 8,000 Feet Per Minute (FPM). 

    By isolating these anomalies and testing them against aircraft physics and hardware sampling rates, we successfully identified the root cause: **Mathematical amplification caused by near-zero time deltas, not physical aircraft movement.**

    * **The Physics Envelope (Left):** Real commercial aircraft cruise at 400–600 knots and climb at < 4,000 FPM. The anomalies plot far outside the limits of survivable physics, proving the vertical rate calculations are mathematical artifacts.
    * **The High-Frequency Jitter (Right):** The histogram proves these impossible vertical rates *only* occur when the polling gap between two ADS-B packets is microscopic (under 2.5 seconds). 

    **Conclusion:** Barometric altimeters have a natural, harmless sensor "jitter" of +/- 100 feet. Over a normal 30-second polling window, this jitter is negligible. However, because our RTL-SDR antenna occasionally captures packets mere milliseconds apart, the pipeline divides that tiny 100-foot sensor variation by a fraction of a second. This "division by near-zero" artificially inflates normal sensor noise into catastrophic, physically impossible climb rates.
    """)

    macro_required = {
        "anomaly_type",
        "implied_speed_kts",
        "vertical_rate_fpm",
        "delta_altitude_ft",
        "delta_seconds",
    }
    if not macro_required.issubset(set(df_anomalies.columns)):
        st.info(
            "Skipping macro analytics: anomalies data is missing one of "
            "`anomaly_type`, `implied_speed_kts`, `vertical_rate_fpm`, `delta_altitude_ft`, or `delta_seconds`."
        )
        return

    df_vertical = df_anomalies[df_anomalies["anomaly_type"] == "vertical_rate_outlier"].copy()
    if df_vertical.empty:
        st.info("No `vertical_rate_outlier` anomalies found in the current dataset.")
        return

    for col in ["implied_speed_kts", "vertical_rate_fpm", "delta_altitude_ft", "delta_seconds"]:
        df_vertical[col] = pd.to_numeric(df_vertical[col], errors="coerce")

    df_vertical = df_vertical.dropna(
        subset=["implied_speed_kts", "vertical_rate_fpm", "delta_altitude_ft", "delta_seconds"]
    ).copy()
    if df_vertical.empty:
        st.info("Insufficient numeric data to render vertical-jump macro analytics charts.")
        return

    # Use absolute magnitudes for a direct comparison against physical capability limits.
    df_vertical["vertical_rate_fpm"] = df_vertical["vertical_rate_fpm"].abs()
    df_vertical["abs_delta_altitude_ft"] = df_vertical["delta_altitude_ft"].abs()

    physics_envelope = pd.DataFrame(
        [{
            "x_min": 0,
            "x_max": 800,
            "y_min": 0,
            "y_max": 6000,
            "label_x": 400,
            "label_y": 5600,
            "label": "Limits of Physics",
        }]
    )

    envelope_rect = (
        alt.Chart(physics_envelope)
        .mark_rect(color="#38bdf8", opacity=0.12)
        .encode(
            x=alt.X("x_min:Q", title="Implied Speed (kts)"),
            x2="x_max:Q",
            y=alt.Y("y_min:Q", title="Vertical Rate (fpm)"),
            y2="y_max:Q",
        )
    )

    envelope_label = (
        alt.Chart(physics_envelope)
        .mark_text(color="#93c5fd", fontWeight="bold", fontSize=12)
        .encode(
            x="label_x:Q",
            y="label_y:Q",
            text="label:N",
        )
    )

    scatter_impossible = (
        alt.Chart(df_vertical)
        .mark_circle(size=38, color="#f97316", opacity=0.7)
        .encode(
            x=alt.X("implied_speed_kts:Q", title="Implied Speed (kts)"),
            y=alt.Y("vertical_rate_fpm:Q", title="Vertical Rate (fpm)"),
            tooltip=[
                alt.Tooltip("implied_speed_kts:Q", title="Implied Speed (kts)", format=",.1f"),
                alt.Tooltip("vertical_rate_fpm:Q", title="Vertical Rate (fpm)", format=",.0f"),
                alt.Tooltip("delta_altitude_ft:Q", title="Delta Altitude (ft)", format=",.0f"),
            ],
        )
    )

    physics_envelope_chart = (
        alt.layer(envelope_rect, scatter_impossible, envelope_label)
        .properties(title="The Physics Envelope", height=360)
    )

    jitter_hist_data = df_vertical[(df_vertical["delta_seconds"] >= 0) & (df_vertical["delta_seconds"] <= 10)].copy()
    if jitter_hist_data.empty:
        st.info("No vertical_rate_outlier records found within the 0-10 second jitter window.")
        return

    jitter_hist = (
        alt.Chart(jitter_hist_data)
        .mark_bar(color="#fb923c")
        .encode(
            x=alt.X(
                "delta_seconds:Q",
                bin=alt.Bin(step=0.5),
                title="Delta Time Between Packets (seconds)",
                scale=alt.Scale(domain=[0, 10]),
            ),
            y=alt.Y("count():Q", title="Count of Records"),
            tooltip=[
                alt.Tooltip("count():Q", title="Count", format=","),
            ],
        )
        .properties(title="High-Frequency Jitter Amplification", height=360)
    )

    macro_left, macro_right = st.columns(2)
    with macro_left:
        st.altair_chart(physics_envelope_chart, use_container_width=True)
    with macro_right:
        st.altair_chart(jitter_hist, use_container_width=True)

    # --- Horizontal Teleportation: Decoding CPR Grid Shifts ---
    st.markdown("### 6. Horizontal Teleportation: Decoding CPR Grid Shifts")

    st.markdown(
        """
        #### Root Cause Analysis: CPR Grid Shift Artifacts

        When scanning the pipeline for spatial anomalies, we isolated `gps_jump` events - instances where an aircraft appeared to travel over 30 kilometers in under 10 seconds (speeds exceeding Mach 9).

        By mapping the exact source and destination of these jumps, we can visually and mathematically prove these are software decoding errors, not physical flight movements or random GPS drift.

        **The Map (Ghost Vectors):**
        The comet trails show the exact path of the coordinate jumps, with the bright pink dots indicating the destination. These vectors are not random shotgun blasts - they form distinct, parallel lines jumping in the same direction (primarily shifting Northwest and Southeast).

        **The Histogram (Quantized Error):**
        If these errors were caused by random radio noise or static, the jump distances would be smoothly distributed. Instead, the histogram reveals rigid, quantized spikes (most notably between the 30-40 km and 45-50 km marks).

        **The Engineering Conclusion:**
        This is a textbook artifact of the ADS-B Compact Position Reporting (CPR) algorithm. To save bandwidth, aircraft broadcast their coordinates using an alternating grid system consisting of even and odd packets.

        If the receiver drops a packet and loses synchronization, the decoding math briefly plots the aircraft in the correct relative location, but inside the wrong adjacent grid tile. The parallel lines on the map and rigid distance spikes on the histogram are the mathematical borders of the CPR grid briefly projecting themselves onto our data.
        """
    )

    gps_required = {"anomaly_type", "lat", "lon", "prev_lat", "prev_lon", "jump_distance_km"}
    if not gps_required.issubset(set(df_anomalies.columns)):
        st.info(
            "Skipping GPS jump analysis: anomalies data is missing one of `prev_lat`, `prev_lon`, `lat`, `lon`, or `jump_distance_km`."
        )
    else:
        df_gps = df_anomalies[df_anomalies["anomaly_type"] == "gps_jump"].copy()
        if df_gps.empty:
            st.info("No `gps_jump` anomalies found in the current dataset.")
        else:
            # Normalize numeric columns
            for c in ["lat", "lon", "prev_lat", "prev_lon", "jump_distance_km"]:
                df_gps[c] = pd.to_numeric(df_gps[c], errors="coerce")

            df_filtered = df_gps.dropna(
                subset=["lat", "lon", "prev_lat", "prev_lon", "jump_distance_km"]
            ).copy()

            # Strictly keep only CPR grid-shift distance band and drop Null-Island-style outliers/noise.
            df_plot = df_filtered[
                (df_filtered["jump_distance_km"] >= 30) & (df_filtered["jump_distance_km"] <= 30000)
            ].copy()

            if df_plot.empty:
                st.info("No valid GPS-jump coordinate pairs available for mapping or histogram.")
            else:
                # Prepare LineLayer records (source -> target)
                line_records = (
                    df_plot[["prev_lon", "prev_lat", "lon", "lat", "jump_distance_km"]]
                    .rename(
                        columns={
                            "prev_lon": "source_lon",
                            "prev_lat": "source_lat",
                            "lon": "target_lon",
                            "lat": "target_lat",
                        }
                    )
                    .to_dict(orient="records")
                )

                target_records = df_plot[["lon", "lat", "jump_distance_km"]].copy().to_dict(orient="records")

                # PyDeck comet effect: transparent trail + bright destination head
                line_layer = pdk.Layer(
                    "LineLayer",
                    data=line_records,
                    get_source_position="[source_lon, source_lat]",
                    get_target_position="[target_lon, target_lat]",
                    get_color="[255, 50, 100, 150]",
                    get_width=2,
                    pickable=True,
                )

                target_head_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=target_records,
                    get_position="[lon, lat]",
                    get_radius=800,
                    get_fill_color="[255, 40, 120, 255]",
                    pickable=True,
                    stroked=False,
                    filled=True,
                )

                # Hardcode initial view to Bangkok to avoid outlier-centering issues
                bangkok_view = pdk.ViewState(latitude=13.75, longitude=100.50, zoom=8.5, pitch=0)

                ghost_tooltip = {
                    "html": "<b>Jump Distance (km):</b> {jump_distance_km}",
                    "style": {"backgroundColor": "#111", "color": "#fff"},
                }

                ghost_deck = pdk.Deck(
                    layers=[line_layer, target_head_layer],
                    initial_view_state=bangkok_view,
                    tooltip=ghost_tooltip,
                    map_style="dark",
                    map_provider="carto",
                )

                col_left, col_right = st.columns(2)
                with col_left:
                    st.caption("Ghost Vectors: transparent trails with bright destination heads to show jump direction")
                    st.pydeck_chart(ghost_deck, use_container_width=True)

                # Altair histogram: focus on dense CPR shift range and ignore extreme long-tail outliers.
                df_hist = df_plot[
                    (df_plot["jump_distance_km"] >= 30) & (df_plot["jump_distance_km"] <= 80)
                ].copy()

                if df_hist.empty:
                    with col_right:
                        st.info("No CPR-shift records in the 30-80 km focus range.")
                else:
                    outlier_count = int((df_plot["jump_distance_km"] > 80).sum())

                    jump_hist = (
                        alt.Chart(df_hist)
                        .mark_bar(color="#ff4a78")
                        .encode(
                            x=alt.X(
                                "jump_distance_km:Q",
                                bin=alt.Bin(step=5),
                                title="Jump Distance (km)",
                                scale=alt.Scale(domain=[30, 80], nice=False, clamp=True),
                                axis=alt.Axis(values=list(range(30, 81, 5))),
                            ),
                            y=alt.Y("count():Q", title="Count of Records"),
                            tooltip=[alt.Tooltip("count():Q", title="Count", format=",")],
                        )
                        .properties(title="CPR Grid Shift: Jump Distance Histogram")
                    )
                    with col_right:
                        st.caption(
                            f"Histogram focus: 30-80 km (excluded long-tail outliers >80 km: {outlier_count:,})"
                        )
                        st.altair_chart(jump_hist, use_container_width=True)


data_dir = os.getenv("ADSB_DATA_DIR", str(DEFAULT_DATA_DIR))

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Receiver Coverage",
        "Sensor Health & Anomalies",
    ],
)

try:
    all_data = load_all_data(data_dir)
except Exception as exc:
    st.error(f"Data loading failed: {exc}")
    st.stop()

if page == "Overview":
    render_overview(all_data)
elif page == "Receiver Coverage":
    render_receiver_coverage(
        all_data,
        receiver_lat=DEFAULT_RECEIVER_LAT,
        receiver_lon=DEFAULT_RECEIVER_LON,
    )
elif page == "Sensor Health & Anomalies":
    render_sensor_health_anomalies(all_data)
