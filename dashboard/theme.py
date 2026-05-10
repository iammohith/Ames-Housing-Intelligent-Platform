import streamlit as st

def apply_theme():
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
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
div[data-testid="stMetricValue"] > div {
    font-family: 'Outfit', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.02em;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
div[data-testid="stMetricDelta"] > div {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: break-word !important;
    font-size: 0.85rem !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
}
.stButton > button[kind="primary"]:hover, .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3) !important;
}
.stButton > button[kind="primary"]:active, .stButton > button:active {
    transform: translateY(1px) !important;
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
    border-radius: 12px;
    border: none;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
}

/* ── Section Titles ── */
.page-header-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.2rem;
}
.page-header-sub {
    font-size: 1.05rem;
    color: #64748B;
    margin-bottom: 2rem;
}
.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #1E293B;
    margin-bottom: 1.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #F1F5F9;
}
</style>
""",
        unsafe_allow_html=True,
    )
