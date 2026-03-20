"""
Zambia Health Facility Accessibility Dashboard
Databricks Data App — Streamlit

Tables used:
  - prd_mega.sgbpi163.health_facilities_zmb        → existing facilities
  - prd_mega.sgpbpi163.lgu_accessibility_results_zmb → optimisation results

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from pyspark.sql import SparkSession

# ─── Databricks SparkSession ─────────────────────────────────────────────────
# In Databricks Apps the SparkSession is injected automatically.
# If running locally / for unit-testing, fall back to a builder.
try:
    spark  # type: ignore[name-defined]  # noqa: F821 – Databricks globals
except NameError:
    spark = SparkSession.builder.getOrCreate()

# ─── Constants ────────────────────────────────────────────────────────────────
EXISTING_TABLE   = "prd_mega.sgbpi163.health_facilities_zmb"
RESULTS_TABLE    = "prd_mega.sgpbpi163.lgu_accessibility_results_zmb"

# From notebook cell 14 transform output
CURRENT_ACCESS_PCT = 65.88          # override if known; queried below when available

ZAMBIA_CENTER     = (-13.5, 28.0)   # [lat, lon]
MAP_ZOOM          = 6
MAX_NEW_FACILITIES = 30

# ─── Colour palette ───────────────────────────────────────────────────────────
COLOUR_EXISTING  = "#F97316"   # warm orange  – existing facilities
COLOUR_NEW       = "#FFFFFF"   # white        – new / potential facilities
COLOUR_BORDER    = "#1E293B"   # dark slate   – marker border


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers (cached so Databricks only queries the warehouse once)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def load_existing_facilities() -> pd.DataFrame:
    """Return lat/lon of all existing health facilities."""
    df = (
        spark.table(EXISTING_TABLE)
        .select("id", "lat", "lon", "name")
        .toPandas()
    )
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["lat", "lon"])


@st.cache_data(ttl=600, show_spinner=False)
def load_accessibility_results() -> pd.DataFrame:
    """
    Return the full optimisation results table, sorted ascending by
    total_facilities so row 0 → first new facility added (total = 1259).
    """
    df = (
        spark.table(RESULTS_TABLE)
        .select(
            "total_facilities",
            "new_facility",
            "lat",
            "lon",
            "h3_index",
            "total_population_access_pct",
        )
        .toPandas()
        .sort_values("total_facilities", ascending=True)
        .reset_index(drop=True)
    )
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["total_population_access_pct"] = pd.to_numeric(
        df["total_population_access_pct"], errors="coerce"
    )
    return df


def get_new_facilities(results_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Return the first *n* rows from the optimisation table
    (total_facilities = 1259 … 1258+n).
    """
    return results_df.head(n)


def get_accessibility_pct(results_df: pd.DataFrame, n: int) -> float:
    """Accessibility % after adding n new facilities."""
    if n == 0 or results_df.empty:
        return CURRENT_ACCESS_PCT
    row = results_df[results_df["total_facilities"] == (len_existing + n)]
    if row.empty:
        row = results_df.head(n).tail(1)
    return float(row["total_population_access_pct"].values[0]) if not row.empty else CURRENT_ACCESS_PCT


# ══════════════════════════════════════════════════════════════════════════════
#  Map builder
# ══════════════════════════════════════════════════════════════════════════════

def build_map(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> folium.Map:
    """
    Build a Folium map with:
      • Orange circle markers = existing facilities
      • White circle markers  = new / potential facilities
    """
    fmap = folium.Map(
        location=list(ZAMBIA_CENTER),
        zoom_start=MAP_ZOOM,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # ── Existing facilities ───────────────────────────────────────────────────
    existing_group = folium.FeatureGroup(name="Existing Facilities", show=True)
    for _, row in existing_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color=COLOUR_EXISTING,
            weight=1.5,
            fill=True,
            fill_color=COLOUR_EXISTING,
            fill_opacity=0.85,
            tooltip=folium.Tooltip(
                f"<b>{row.get('name', 'Health Facility')}</b><br>"
                f"Lat: {row['lat']:.4f} | Lon: {row['lon']:.4f}",
                sticky=False,
            ),
        ).add_to(existing_group)
    existing_group.add_to(fmap)

    # ── New (potential) facilities ────────────────────────────────────────────
    if not new_df.empty:
        new_group = folium.FeatureGroup(name="New Facilities", show=True)
        for _, row in new_df.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=7,
                color="#0EA5E9",      # sky blue ring
                weight=2.5,
                fill=True,
                fill_color=COLOUR_NEW,
                fill_opacity=0.95,
                tooltip=folium.Tooltip(
                    f"<b>Proposed Facility</b><br>"
                    f"ID: {row.get('new_facility', 'N/A')}<br>"
                    f"Lat: {row['lat']:.4f} | Lon: {row['lon']:.4f}",
                    sticky=False,
                ),
            ).add_to(new_group)
        new_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


# ══════════════════════════════════════════════════════════════════════════════
#  Page configuration
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Zambia Health Access Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Import distinctive Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

    /* Global reset / dark background */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0A0F1E !important;
        color: #E2E8F0;
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background-color: #0D1321; }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Title bar */
    .dash-title {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .dash-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.88rem;
        color: #64748B;
        margin-top: 2px;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #111827 0%, #1E293B 100%);
        border: 1px solid #1E3A5F;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--accent);
        border-radius: 12px 12px 0 0;
    }
    .kpi-label {
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #64748B;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }
    .kpi-suffix {
        font-size: 0.9rem;
        color: #94A3B8;
        margin-top: 4px;
        font-weight: 400;
    }

    /* Colour theme variants */
    .kpi-orange { --accent: #F97316; }
    .kpi-orange .kpi-value { color: #F97316; }
    .kpi-white  { --accent: #FFFFFF; }
    .kpi-white  .kpi-value { color: #F8FAFC; }
    .kpi-green  { --accent: #10B981; }
    .kpi-green  .kpi-value { color: #10B981; }
    .kpi-sky    { --accent: #0EA5E9; }
    .kpi-sky    .kpi-value { color: #0EA5E9; }

    /* Slider label */
    .slider-wrap label {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        color: #94A3B8 !important;
        letter-spacing: 0.5px;
    }
    .stSlider > div > div > div > div {
        background: #0EA5E9 !important;
    }
    .stSlider > div > div > div {
        background: #1E293B !important;
    }

    /* Divider */
    hr { border-color: #1E293B; margin: 0.5rem 0; }

    /* Legend pill */
    .legend-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.78rem;
        color: #CBD5E1;
    }
    .dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }

    /* Map container */
    .map-wrap {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #1E293B;
    }

    /* Info banner */
    .info-banner {
        background: linear-gradient(90deg, #0A0F1E 0%, #0D1B33 50%, #0A0F1E 100%);
        border: 1px solid #1E3A5F;
        border-radius: 10px;
        padding: 10px 18px;
        font-size: 0.80rem;
        color: #64748B;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Load data
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading facility data from Databricks…"):
    existing_df   = load_existing_facilities()
    results_df    = load_accessibility_results()

len_existing = len(existing_df)

# ══════════════════════════════════════════════════════════════════════════════
#  Header row
# ══════════════════════════════════════════════════════════════════════════════

title_col, spacer, ctrl_col = st.columns([3, 1, 3])

with title_col:
    st.markdown(
        """
        <div class='dash-title'>🇿🇲 &nbsp;Zambia Health Access</div>
        <div class='dash-subtitle'>Facility placement optimisation &amp; population accessibility</div>
        """,
        unsafe_allow_html=True,
    )

with ctrl_col:
    st.markdown("<div class='slider-wrap'>", unsafe_allow_html=True)
    n_new = st.slider(
        "➕  New facilities to add",
        min_value=0,
        max_value=MAX_NEW_FACILITIES,
        value=0,
        step=1,
        help="Slide to simulate adding 1–30 optimally-placed new health facilities across Zambia.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  KPI cards
# ══════════════════════════════════════════════════════════════════════════════

new_df          = get_new_facilities(results_df, n_new)
access_pct      = get_accessibility_pct(results_df, n_new) if n_new > 0 else CURRENT_ACCESS_PCT
baseline_pct    = CURRENT_ACCESS_PCT
delta_pct       = round(access_pct - baseline_pct, 2) if n_new > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class='kpi-card kpi-orange'>
            <div class='kpi-label'>Existing Facilities</div>
            <div class='kpi-value'>{len_existing:,}</div>
            <div class='kpi-suffix'>health facilities</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class='kpi-card kpi-sky'>
            <div class='kpi-label'>New Facilities</div>
            <div class='kpi-value'>{n_new}</div>
            <div class='kpi-suffix'>proposed additions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class='kpi-card kpi-green'>
            <div class='kpi-label'>Population Access</div>
            <div class='kpi-value'>{access_pct:.1f}<span style='font-size:1.1rem'>%</span></div>
            <div class='kpi-suffix'>{'baseline' if n_new == 0 else f'+{delta_pct:.2f}% vs baseline'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    total_facilities = len_existing + n_new
    st.markdown(
        f"""
        <div class='kpi-card kpi-white'>
            <div class='kpi-label'>Total Facilities</div>
            <div class='kpi-value'>{total_facilities:,}</div>
            <div class='kpi-suffix'>after additions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Legend + info row
# ══════════════════════════════════════════════════════════════════════════════

leg1, leg2, leg3 = st.columns([2, 2, 4])

with leg1:
    st.markdown(
        "<div class='legend-pill'>"
        "<span class='dot' style='background:#F97316'></span>"
        "Existing facility"
        "</div>",
        unsafe_allow_html=True,
    )

with leg2:
    st.markdown(
        "<div class='legend-pill'>"
        "<span class='dot' style='background:#FFFFFF; border:2px solid #0EA5E9'></span>"
        "Proposed new facility"
        "</div>",
        unsafe_allow_html=True,
    )

with leg3:
    if n_new == 0:
        st.markdown(
            "<div class='info-banner'>"
            "Use the slider above to simulate adding new optimally-placed health facilities across Zambia."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='info-banner'>"
            f"Showing <b>{n_new}</b> new facilities (rows 1259–{1258 + n_new} of optimisation results). "
            f"Accessibility rises from <b>{baseline_pct}%</b> → <b>{access_pct:.2f}%</b>."
            "</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Map
# ══════════════════════════════════════════════════════════════════════════════

fmap = build_map(existing_df, new_df)

st.markdown("<div class='map-wrap'>", unsafe_allow_html=True)
st_folium(
    fmap,
    width="100%",
    height=620,
    returned_objects=[],   # disable expensive return data
    key=f"map_{n_new}",    # re-render when slider changes
)
st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Footer
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<br/>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center; font-size:0.72rem; color:#374151; font-family: Space Mono, monospace;'>
        ZAMBIA · HEALTH FACILITY ACCESSIBILITY · DATA APP · DATABRICKS
        &nbsp;|&nbsp; Population source: WorldPop 2025 · Facilities: OpenStreetMap
        &nbsp;|&nbsp; Optimisation: ILP / Gurobi
    </div>
    """,
    unsafe_allow_html=True,
)