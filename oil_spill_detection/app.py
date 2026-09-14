"""
Smart India Hackathon (SIH) 2025 - Multi-Modal Oil Spill Detection & Vessel Attribution System
Prototype Dashboard: Satellite SAR Segmentation + AIS Trajectory Anomaly + Ocean Drift Physics + Explainable AI (XAI)
"""

import os
import glob
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium import plugins
from streamlit_folium import st_folium

from src.satellite.preprocessor import SatellitePreprocessor
from src.satellite.detector import SpillDetector
from src.ais.tracker import AISTracker
from src.fusion.drift import DriftModel
from src.fusion.attribution import MultiModalFusionEngine

# ----------------- PAGE CONFIGURATION & STYLING -----------------
st.set_page_config(
    page_title="SIH 2025 | AI Oil Spill Detection & Vessel Attribution",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Dark Maritime Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #080d1a 100%);
        color: #e2e8f0;
    }
    
    /* Sleek card container */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
    }
    
    /* Metric pill styling */
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-high {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
    }
    .badge-medium {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
    }
    .badge-low {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
    }
    
    /* Code/mono elements */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Header title gradient */
    .hero-title {
        background: white;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- HELPER FUNCTIONS -----------------
@st.cache_resource
def get_pipeline():
    preprocessor = SatellitePreprocessor(target_size=(512, 512))
    detector = SpillDetector(pixel_resolution_m=20.0)
    tracker = AISTracker(ref_lat=18.922, ref_lon=72.834)
    fusion_engine = MultiModalFusionEngine()
    return preprocessor, detector, tracker, fusion_engine

preprocessor, detector, tracker, fusion_engine = get_pipeline()

def scan_available_images():
    """Scan data/ais and data/satellite folders for test images."""
    paths = []
    for folder in ["data/ais", "data/satellite"]:
        if os.path.exists(folder):
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp"]:
                paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(list(set(paths)))


# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/satellite-sending-signal.png", width=64)
    st.markdown("### 🎛️ System Configuration")
    
    available_imgs = scan_available_images()
    
    st.markdown("#### 1. Ingest Satellite SAR Image")
    selected_source = st.radio(
        "Source Mode",
        ["Select Sample Dataset", "Upload SAR Image"],
        index=0
    )
    
    image_to_process = None
    image_name = "Sample SAR Image"
    
    if selected_source == "Select Sample Dataset":
        if available_imgs:
            selected_path = st.selectbox(
                "Select Available SAR Scene",
                available_imgs,
                format_func=lambda x: os.path.basename(x)
            )
            image_to_process = selected_path
            image_name = os.path.basename(selected_path)
        else:
            st.warning("No sample images found in data folders. Please upload one.")
    else:
        uploaded_file = st.file_uploader("Upload SAR Satellite Image", type=["jpg", "jpeg", "png", "tif"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_name = uploaded_file.name

    st.markdown("---")
    st.markdown("#### 2. Environmental Dynamics")
    wind_speed = st.slider("💨 Wind Speed (knots)", min_value=2.0, max_value=35.0, value=14.0, step=0.5)
    wind_dir = st.slider("🧭 Wind Direction (° from North)", min_value=0.0, max_value=360.0, value=220.0, step=5.0)
    current_speed = st.slider("🌊 Surface Current Speed (knots)", min_value=0.1, max_value=4.0, value=1.2, step=0.1)
    current_dir = st.slider("🧭 Current Direction (° from North)", min_value=0.0, max_value=360.0, value=65.0, step=5.0)

    st.markdown("---")
    st.markdown("#### 3. Model Tuning")
    detection_sensitivity = st.slider("SAR Spill Sensitivity", min_value=0.1, max_value=1.0, value=0.65, step=0.05)
    backtrack_hours = st.slider("Backtrack Time (Hours)", min_value=1.0, max_value=12.0, value=4.0, step=0.5)


# ----------------- MAIN PROCESSING PIPELINE -----------------

# Set default reference coordinates (e.g. offshore corridor)
SPILL_BASE_LAT = 18.9220
SPILL_BASE_LON = 72.8340

# 1. Image Preprocessing & Oil Spill Segmentation
if image_to_process is not None:
    prep_data = preprocessor.preprocess_pipeline(image_to_process)
    spill_res = detector.detect_spill(prep_data, sensitivity=detection_sensitivity)
else:
    # Fallback blank
    blank = np.zeros((512, 512, 3), dtype=np.uint8)
    prep_data = preprocessor.preprocess_pipeline(blank)
    spill_res = detector.detect_spill(prep_data, sensitivity=0.5)

# 2. Physics-based Drift Model & Backtracking
drift_model = DriftModel(
    wind_speed_knots=wind_speed,
    wind_dir_deg=wind_dir,
    current_speed_knots=current_speed,
    current_dir_deg=current_dir
)
drift_backtrack = drift_model.backtrack_spill_origin(
    detected_lat=SPILL_BASE_LAT,
    detected_lon=SPILL_BASE_LON,
    backtrack_hours=backtrack_hours
)
drift_forecast = drift_model.forecast_drift(
    start_lat=SPILL_BASE_LAT,
    start_lon=SPILL_BASE_LON,
    hours=48,
    step_hours=6
)

# 3. AIS Trajectory Simulation & Anomaly Detection
vessels = tracker.generate_synthetic_scenario(
    spill_center_lat=drift_backtrack["origin_lat"],
    spill_center_lon=drift_backtrack["origin_lon"]
)

# 4. Multi-Modal Fusion & Explainable AI (XAI) Attribution
attribution_results = fusion_engine.evaluate_attribution(
    vessels=vessels,
    spill_detection=spill_res,
    drift_info=drift_backtrack
)
top_suspect = attribution_results[0]


# ----------------- HERO HEADER -----------------
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown('<div class="hero-title"> AI Oil Spill Detection & Vessel Attribution System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Smart India Hackathon 2025 • Multi-Modal Fusion: Sentinel-1 SAR + AIS Kinematics + Ocean Drift Physics + Explainable AI</div>', unsafe_allow_html=True)

with col_head2:
    st.markdown(f"""
    <div style="text-align: right; padding-top: 10px;">
        <span class="metric-badge badge-{'high' if top_suspect.risk_level=='High' else 'medium' if top_suspect.risk_level=='Medium' else 'low'}">
            Risk: {top_suspect.risk_level} ({top_suspect.final_risk_score}%)
        </span>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Top Suspect: <b>{top_suspect.vessel.vessel_name}</b></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ----------------- QUICK METRICS ROW -----------------
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
with m_col1:
    st.metric(" Detected Slick Area", f"{spill_res['area_km2']} km²", f"{spill_res['total_pixels']:,} px")
with m_col2:
    st.metric(" Est. Spill Volume", f"{spill_res['estimated_volume_barrels']} bbl", f"{spill_res['estimated_volume_m3']} m³")
with m_col3:
    st.metric(" SAR Confidence", f"{spill_res['confidence_score']}%", f"{spill_res['aspect_ratio']} elongation")
with m_col4:
    st.metric(" Net Drift Speed", f"{drift_backtrack['net_drift_speed_knots']} kn", f"{drift_backtrack['net_drift_dir']}° Drift Heading")
with m_col5:
    st.metric(" Monitored Vessels", f"{len(vessels)} Tracks", f"{len([v for v in attribution_results if v.risk_level=='High'])} Flagged High Risk")


# ----------------- NAVIGATION TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Interactive Marine Map",
    " Satellite SAR Segmentation",
    " AIS Kinematics & Anomaly",
    " Physics Drift & Trajectory",
    " Multi-Modal Fusion & XAI"
])


# =========================================================================
# TAB 1: INTERACTIVE MARINE GIS MAP (FOLIUM)
# =========================================================================
with tab1:
    st.markdown("### Live Maritime Situation Map")
    st.markdown("Interactive geospatial map showing detected SAR oil slick footprint, AIS vessel corridors, backtracked discharge origin, and 48-hour forward drift projection cone.")

    # Create Folium Map centered on incident
    m = folium.Map(
        location=[SPILL_BASE_LAT, SPILL_BASE_LON],
        zoom_start=10,
        tiles="CartoDB dark_matter",
        control_scale=True
    )

    # 1. Add Detected Oil Spill Polygon/Circle Marker
    folium.Circle(
        location=[SPILL_BASE_LAT, SPILL_BASE_LON],
        radius=np.sqrt(spill_res['area_km2'] * 1_000_000 / np.pi) if spill_res['area_km2'] > 0 else 800,
        color="#ef4444",
        fill=True,
        fill_color="#ef4444",
        fill_opacity=0.6,
        popup=f"<b>Observed Oil Slick</b><br>Area: {spill_res['area_km2']} km²<br>Confidence: {spill_res['confidence_score']}%",
        tooltip="Observed Oil Slick Footprint"
    ).add_to(m)

    # 2. Add Backtracked Spill Origin Point
    folium.Marker(
        location=[drift_backtrack["origin_lat"], drift_backtrack["origin_lon"]],
        popup=f"<b>Estimated Discharge Origin</b><br>Backtracked: -{drift_backtrack['backtrack_hours']} hrs<br>Distance drifted: {drift_backtrack['drift_distance_km']} km",
        tooltip="Estimated Origin of Discharge",
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")
    ).add_to(m)

    # Line connecting backtracked origin to observed slick
    folium.PolyLine(
        locations=[
            [drift_backtrack["origin_lat"], drift_backtrack["origin_lon"]],
            [SPILL_BASE_LAT, SPILL_BASE_LON]
        ],
        color="#f87171",
        weight=2.5,
        dash_array="6, 6",
        tooltip="Backtracked Drift Vector"
    ).add_to(m)

    # 3. Add 48-hour Forward Drift Forecast Points
    drift_coords = [[pt["lat"], pt["lon"]] for pt in drift_forecast]
    folium.PolyLine(
        locations=drift_coords,
        color="#38bdf8",
        weight=3,
        tooltip="48-Hour Projected Drift Trajectory"
    ).add_to(m)

    for pt in drift_forecast[1:]:
        folium.CircleMarker(
            location=[pt["lat"], pt["lon"]],
            radius=6,
            color="#38bdf8",
            fill=True,
            fill_color="#0284c7",
            popup=f"<b>Forecast T+{pt['hour']}h</b><br>Distance: {pt['distance_km']} km<br>Spread Radius: {pt['spread_radius_km']} km",
            tooltip=f"T+{pt['hour']}h Forecast"
        ).add_to(m)

    # 4. Add AIS Vessel Trajectories
    colors = {
        "High": "#ef4444",    # Red
        "Medium": "#f59e0b",  # Amber
        "Low": "#10b981"      # Green
    }

    for att in attribution_results:
        v = att.vessel
        coords = v.dataframe[["lat", "lon"]].values.tolist()
        v_color = colors.get(att.risk_level, "#3b82f6")

        # Vessel Path Polyline
        folium.PolyLine(
            locations=coords,
            color=v_color,
            weight=4 if att.risk_level == "High" else 2,
            opacity=0.9 if att.risk_level == "High" else 0.6,
            tooltip=f"{v.vessel_name} ({v.vessel_type}) - Risk: {att.final_risk_score}%"
        ).add_to(m)

        # Vessel Last Known Position Marker
        last_pos = coords[-1]
        folium.CircleMarker(
            location=last_pos,
            radius=7 if att.risk_level == "High" else 5,
            color=v_color,
            fill=True,
            fill_color=v_color,
            popup=(
                f"<b>{v.vessel_name}</b> ({v.vessel_type})<br>"
                f"MMSI: {v.mmsi}<br>"
                f"Flag: {v.flag}<br>"
                f"Attribution Probability: <b>{att.attribution_probability}%</b><br>"
                f"Status: {att.risk_level} Risk"
            ),
            tooltip=f"{v.vessel_name} (Last Pos)"
        ).add_to(m)

        # Highlight Suspect Point where speed dropped / nearest approach
        if att.risk_level == "High":
            folium.Marker(
                location=[v.suspect_location[0], v.suspect_location[1]],
                popup=f"<b>Suspicious Slowdown Event</b><br>{v.vessel_name}<br>Time: {v.suspect_timestamp.strftime('%H:%M UTC')}<br>Speed: 3.1 kn",
                tooltip=f"Suspect Event: {v.vessel_name}",
                icon=folium.Icon(color="darkred", icon="ship", prefix="fa")
            ).add_to(m)

    # Render map in Streamlit
    st_folium(m, width="100%", height=560)


# =========================================================================
# TAB 2: SATELLITE SAR SEGMENTATION & PREPROCESSING
# =========================================================================
with tab2:
    st.markdown("### 🛰️ Sentinel-1 SAR Image Processing Pipeline")
    st.markdown("Demonstrating end-to-end processing: raw SAR input $\\rightarrow$ adaptive speckle filtering $\\rightarrow$ CLAHE enhancement $\\rightarrow$ dark slick segmentation mask & thermal heatmap.")

    sar_col1, sar_col2, sar_col3, sar_col4 = st.columns(4)
    with sar_col1:
        st.markdown("**1. Raw Input SAR Scene**")
        st.image(prep_data["original_rgb"], use_container_width=True, caption=image_name)

    with sar_col2:
        st.markdown("**2. Adaptive Speckle Filter**")
        st.image(prep_data["denoised"], use_container_width=True, caption="Bilateral + Median Filter")

    with sar_col3:
        st.markdown("**3. AI Spill Segmentation Mask**")
        st.image(spill_res["mask"], use_container_width=True, caption=f"Extracted Slick ({spill_res['area_km2']} km²)")

    with sar_col4:
        st.markdown("**4. Multi-Layer Composite Overlay**")
        st.image(spill_res["overlay"], use_container_width=True, caption="Slick Boundary & Spill Mask")

    st.markdown("---")
    
    # Morphological breakdown
    st.markdown("####  Morphological & Geometric Spill Signatures")
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        st.image(spill_res["heatmap"], use_container_width=True, caption="Distance Transform / Slick Density Heatmap")
    
    with p_col2:
        st.markdown(f"""
        <div class="glass-card">
            <h4>Quantitative Geometric Properties</h4>
            <ul>
                <li><b>Total Detected Surface Area:</b> <code>{spill_res['area_km2']} km²</code> ({spill_res['total_pixels']:,} pixels)</li>
                <li><b>Spill Perimeter:</b> <code>{spill_res['perimeter_km']} km</code></li>
                <li><b>Centroid Coordinates:</b> <code>Lat: {SPILL_BASE_LAT:.4f}, Lon: {SPILL_BASE_LON:.4f}</code></li>
                <li><b>Slick Elongation (Aspect Ratio):</b> <code>{spill_res['aspect_ratio']}:1</code> (Values &gt; 2.5 indicate vessel release wake)</li>
                <li><b>Slick Orientation Axis:</b> <code>{spill_res['orientation_deg']}°</code></li>
                <li><b>Estimated Volume:</b> <code>{spill_res['estimated_volume_m3']} m³</code> (~{spill_res['estimated_volume_barrels']} barrels)</li>
                <li><b>Segmentation Confidence:</b> <code>{spill_res['confidence_score']}%</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# =========================================================================
# TAB 3: AIS KINEMATICS & ANOMALY DETECTION
# =========================================================================
with tab3:
    st.markdown("###  AIS Vessel Trajectory & Kinematic Anomaly Analysis")
    st.markdown("Physics-based anomaly detection evaluating sudden speed drops, loitering patterns, and proximity to the spill zone.")

    # Vessel Selector
    vessel_names = [v.vessel_name for v in vessels]
    selected_v_name = st.selectbox("Select Vessel to Inspect Kinematics", vessel_names)
    selected_v = next(v for v in vessels if v.vessel_name == selected_v_name)
    selected_att = next(a for a in attribution_results if a.vessel.vessel_name == selected_v_name)

    v_meta1, v_meta2, v_meta3, v_meta4 = st.columns(4)
    with v_meta1:
        st.markdown(f"**Vessel Name:** `{selected_v.vessel_name}`")
        st.markdown(f"**MMSI:** `{selected_v.mmsi}`")
    with v_meta2:
        st.markdown(f"**Type:** `{selected_v.vessel_type}`")
        st.markdown(f"**Flag:** `{selected_v.flag}`")
    with v_meta3:
        st.markdown(f"**Anomaly Score:** `{selected_v.anomaly_score}/100`")
        st.markdown(f"**Status:** `{selected_v.anomaly_status}`")
    with v_meta4:
        st.markdown(f"**Attribution Probability:** `{selected_att.attribution_probability}%`")
        st.markdown(f"**Risk Level:** `{selected_att.risk_level}`")

    # Plot Speed Over Ground (SOG) and Course Over Ground (COG)
    fig_speed = px.line(
        selected_v.dataframe,
        x="timestamp",
        y="sog",
        title=f"Speed Over Ground (SOG) Timeline - {selected_v.vessel_name}",
        labels={"sog": "Speed (Knots)", "timestamp": "UTC Timestamp"},
        markers=True
    )
    fig_speed.update_traces(line_color="#38bdf8" if selected_att.risk_level != "High" else "#ef4444", line_width=3)
    fig_speed.update_layout(template="plotly_dark", height=320)
    st.plotly_chart(fig_speed, use_container_width=True)

    # Anomaly Flags
    st.markdown("####  Kinematic Anomaly Flags")
    if selected_v.anomaly_flags:
        for flag in selected_v.anomaly_flags:
            st.markdown(f"-  **{flag}**")
    else:
        st.markdown("✅ *No abnormal kinematic maneuvers detected. Vessel maintained uniform speed and transit heading.*")

    # Raw AIS Telemetry Table
    with st.expander(" View Raw Decoded AIS NMEA Data Frame"):
        st.dataframe(selected_v.dataframe, use_container_width=True)


# =========================================================================
# TAB 4: PHYSICS DRIFT MODEL & TRAJECTORY FORECAST
# =========================================================================
with tab4:
    st.markdown("###  Physics-Based Spill Drift & Trajectory Modeling")
    st.markdown("Advection modeling combining surface current vectors with 3% wind leeway and Coriolis boundary deflection.")

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown(f"""
        <div class="glass-card">
            <h4>⏮ Backtracked Spill Origin (Release Location)</h4>
            <ul>
                <li><b>Backtrack Duration:</b> <code>{drift_backtrack['backtrack_hours']} Hours</code></li>
                <li><b>Estimated Origin Latitude:</b> <code>{drift_backtrack['origin_lat']}° N</code></li>
                <li><b>Estimated Origin Longitude:</b> <code>{drift_backtrack['origin_lon']}° E</code></li>
                <li><b>Total Distance Drifted:</b> <code>{drift_backtrack['drift_distance_km']} km</code></li>
                <li><b>Net Drift Vector:</b> <code>{drift_backtrack['net_drift_speed_knots']} kn @ {drift_backtrack['net_drift_dir']}°</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with d_col2:
        st.markdown(f"""
        <div class="glass-card">
            <h4> Environmental Vector Components</h4>
            <ul>
                <li><b>Wind Vector:</b> <code>{wind_speed} knots @ {wind_dir}°</code></li>
                <li><b>Wind Leeway Contribution (3%):</b> <code>{wind_speed * 0.03:.2f} knots</code></li>
                <li><b>Ocean Current Vector:</b> <code>{current_speed} knots @ {current_dir}°</code></li>
                <li><b>Coriolis Deflection Angle:</b> <code>+20.0° (Northern Hemisphere)</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 48-Hour Forecast Table
    st.markdown("####  48-Hour Forward Spill Spread & Trajectory Forecast")
    df_forecast = pd.DataFrame(drift_forecast)
    st.dataframe(df_forecast, use_container_width=True)


# =========================================================================
# TAB 5: MULTI-MODAL FUSION & EXPLAINABLE AI (XAI)
# =========================================================================
with tab5:
    st.markdown("###  Multi-Modal Evidence Fusion & Explainable AI (XAI)")
    st.markdown("Combines satellite SAR detection evidence, AIS spatio-temporal correlation, kinematic anomaly scoring, and ocean drift trajectory matching.")

    # Attribution Ranking Table
    st.markdown("####  Suspect Vessel Attribution Leaderboard")
    
    summary_data = []
    for att in attribution_results:
        summary_data.append({
            "Vessel Name": att.vessel.vessel_name,
            "MMSI": att.vessel.mmsi,
            "Type": att.vessel.vessel_type,
            "Risk Level": att.risk_level,
            "Final Attribution Score": f"{att.final_risk_score}%",
            "Spatio-Temporal Match": f"{att.spatio_temporal_score}%",
            "Kinematic Anomaly": f"{att.kinematic_anomaly_score}%",
            "Drift Alignment": f"{att.drift_vector_alignment_score}%",
            "SAR Confidence": f"{att.sar_confidence_score}%"
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    st.markdown("---")

    # Primary Suspect XAI Card
    st.markdown("####  Explainable AI (XAI) Incident Audit")
    
    xai_col1, xai_col2 = st.columns([3, 2])
    with xai_col1:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid {'#ef4444' if top_suspect.risk_level=='High' else '#f59e0b'};">
            <h3>Suspect Case File: {top_suspect.vessel.vessel_name} (MMSI: {top_suspect.vessel.mmsi})</h3>
            <p><b>Executive Conclusion:</b> {top_suspect.xai_explanation_summary}</p>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <h4>Key Evidence Factors:</h4>
        </div>
        """, unsafe_allow_html=True)
        for point in top_suspect.xai_evidence_points:
            st.markdown(f"- {point}")

    with xai_col2:
        # Radar Chart of Evidence Dimensions
        categories = ['Spatio-Temporal Proximity', 'Kinematic Anomaly', 'Drift Vector Alignment', 'SAR Confidence']
        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=[
                top_suspect.spatio_temporal_score,
                top_suspect.kinematic_anomaly_score,
                top_suspect.drift_vector_alignment_score,
                top_suspect.sar_confidence_score
            ],
            theta=categories,
            fill='toself',
            name=top_suspect.vessel.vessel_name,
            line_color='#ef4444'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            template="plotly_dark",
            title="Multi-Modal Evidence Radar",
            height=320
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    
    # Exportable Incident Report
    st.markdown("####  Incident Investigation Export")
    report_json = {
        "incident_timestamp": datetime.now().isoformat(),
        "sar_image_analyzed": image_name,
        "spill_metrics": {
            "area_km2": spill_res["area_km2"],
            "perimeter_km": spill_res["perimeter_km"],
            "estimated_volume_barrels": spill_res["estimated_volume_barrels"],
            "confidence_score": spill_res["confidence_score"]
        },
        "backtracked_origin": drift_backtrack,
        "primary_suspect": {
            "vessel_name": top_suspect.vessel.vessel_name,
            "mmsi": top_suspect.vessel.mmsi,
            "attribution_score": top_suspect.final_risk_score,
            "risk_level": top_suspect.risk_level,
            "xai_summary": top_suspect.xai_explanation_summary
        }
    }
    
    st.download_button(
        label=" Download Official Incident Audit Report (JSON)",
        data=pd.Series(report_json).to_json(indent=2),
        file_name="oil_spill_incident_report.json",
        mime="application/json"
    )
