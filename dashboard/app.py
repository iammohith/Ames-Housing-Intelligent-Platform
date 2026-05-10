"""
Ames Housing Intelligence Platform — Streamlit Multi-Page Dashboard
"""

import os

import requests
import streamlit as st

st.set_page_config(
    page_title="Ames Housing Intelligence Platform",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# Ames Housing Intelligence Platform\nProduction-grade ML pipeline with real-time monitoring.",
    },
)

API_URL = os.getenv("API_URL", "http://orchestration-api:8000")

# ── Global Design System ────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #334155;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: #F4F7FB; }
.main .block-container {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1440px;
}

/* ── Typography ── */
h1, h2, h3, h4, h5 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.015em;
    margin-bottom: 0.5rem;
}

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #F1F5F9;
    box-shadow: 4px 0 24px rgba(15, 23, 42, 0.02);
}
div[data-testid="stSidebar"] .stMarkdown p {
    color: #475569;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* ── Native Streamlit Metrics ── */
div[data-testid="stMetric"], div[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #F1F5F9;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover, div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(15, 23, 42, 0.08);
}
div[data-testid="stMetricLabel"] > div {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}
div[data-testid="stMetricValue"] > div {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.02em;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-family: 'Outfit', sans-serif;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3);
}
.stButton > button:active {
    transform: translateY(1px);
}
.stButton > button[kind="secondary"] {
    background: #FFFFFF;
    color: #334155;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}
.stButton > button[kind="secondary"]:hover {
    background: #F8FAFC;
    border-color: #CBD5E1;
    color: #0F172A;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #E2E8F0;
    background: transparent;
    padding-bottom: 2px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: #64748B;
    padding: 0.8rem 1.5rem;
    border-radius: 8px 8px 0 0;
    border-bottom: 3px solid transparent;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #334155;
    background: #F8FAFC;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #2563EB;
    border-bottom-color: #2563EB;
    background: transparent;
}

/* ── Tables & DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid #F1F5F9;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.02);
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    color: #1E293B;
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #F1F5F9;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02);
}

/* ── Alerts / info ── */
.stAlert {
    border-radius: 10px;
    border-left-width: 4px;
}

/* ── Shared card ── */
.platform-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

/* ── Page header band ── */
.page-header {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.page-header-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0;
}
.page-header-sub {
    font-size: 0.9rem;
    color: #64748B;
    margin-top: 0.25rem;
}
.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #0F172A;
    border-left: 3px solid #2563EB;
    padding-left: 0.75rem;
    margin: 2rem 0 1rem 0;
}
.insight-pill {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    font-size: 0.87rem;
    color: #1E40AF;
    line-height: 1.6;
    margin-bottom: 0.6rem;
}
.insight-pill b { color: #1D4ED8; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:700;color:#0F172A;">
                🏠 AHIP
            </div>
            <div style="font-size:0.78rem;color:#64748B;margin-top:2px;">
                Ames Housing Intelligence Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Navigation**")
    st.markdown(
        """
- 📡 **Pipeline Monitor**
- 📊 **Business Analytics**
- 🤖 **AI Insights Chatbot**
"""
    )
    st.divider()

    # Live system health
    st.markdown("**System Status**")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.markdown(
                '<span style="color:#059669;font-weight:600;font-size:0.85rem;">● API Healthy</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span style="color:#DC2626;font-size:0.85rem;">● API Unreachable</span>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            '<span style="color:#D97706;font-size:0.85rem;">● API Connecting…</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("v1.0.0 · 100% Offline · MIT License")

# ── Home Page ────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="page-header">
    <div class="page-header-title">🏠 Ames Housing Intelligence Platform</div>
    <div class="page-header-sub">
        Production-grade ML pipeline · 8-agent async DAG · Real-time monitoring · Fully offline
    </div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown(
        """
<div class="platform-card">
    <div style="font-size:1.75rem;margin-bottom:0.6rem;">📡</div>
    <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:1rem;color:#0F172A;margin-bottom:0.4rem;">Pipeline Monitor</div>
    <div style="font-size:0.875rem;color:#475569;line-height:1.6;">
        Trigger pipeline runs and watch all 8 agents execute in real-time.
        Live DAG visualization, progress tracking, and streaming diagnostics.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
<div class="platform-card">
    <div style="font-size:1.75rem;margin-bottom:0.6rem;">📊</div>
    <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:1rem;color:#0F172A;margin-bottom:0.4rem;">Business Analytics</div>
    <div style="font-size:0.875rem;color:#475569;line-height:1.6;">
        Executive KPIs, neighborhood segmentation, temporal trends, model leaderboard,
        and an interactive What-If pricing simulator with SHAP explanations.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
<div class="platform-card">
    <div style="font-size:1.75rem;margin-bottom:0.6rem;">🤖</div>
    <div style="font-family:'Outfit',sans-serif;font-weight:700;font-size:1rem;color:#0F172A;margin-bottom:0.4rem;">AI Insights Chatbot</div>
    <div style="font-size:0.875rem;color:#475569;line-height:1.6;">
        Ask questions about the dataset in plain English. Powered by a fully offline
        flan-t5-base RAG pipeline with hybrid BM25 + semantic retrieval.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.info(
    "**Getting started:** Select a page from the sidebar. "
    "Run the **Pipeline Monitor** first to train models — then all analytics and predictions become available.",
    icon="ℹ️",
)
