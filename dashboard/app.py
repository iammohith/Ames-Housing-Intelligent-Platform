"""
Ames Housing Intelligence Platform — Streamlit Multi-Page Dashboard
"""

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

# Premium Light Theme CSS
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #FAFAFC;
        color: #1A1F2B;
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container { 
        padding-top: 3rem; 
        max-width: 95%; 
    }

    h1, h2, h3, h4, h5 { 
        color: #0B1120; 
        font-family: 'Outfit', sans-serif; 
        font-weight: 700; 
        letter-spacing: -0.02em;
    }

    /* Premium Metrics */
    .stMetric { 
        background: rgba(255, 255, 255, 0.7); 
        border: 1px solid rgba(226, 232, 240, 0.8); 
        border-radius: 16px; 
        padding: 20px; 
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.03), 0 2px 6px -2px rgba(0, 0, 0, 0.02); 
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
    }
    .stMetric label { 
        color: #64748B; 
        font-size: 0.95rem; 
        font-weight: 600; 
        font-family: 'Outfit', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .stMetric .css-1xarl3l, .stMetric [data-testid="stMetricValue"] { 
        color: #0F172A; 
        font-size: 2.2rem; 
        font-weight: 700; 
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Refinement */
    div[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
        box-shadow: 2px 0 15px rgba(0, 0, 0, 0.02);
    }
    div[data-testid="stSidebar"] .stMarkdown { color: #334155; }
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.95rem;
    }

    /* Modern Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6, #2563EB);
        color: white; 
        border: none; 
        border-radius: 10px;
        font-weight: 600; 
        padding: 0.6rem 1.5rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.15), 0 2px 4px -1px rgba(37, 99, 235, 0.1);
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.01em;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.2), 0 4px 6px -2px rgba(37, 99, 235, 0.1);
        transform: translateY(-2px);
    }
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px -1px rgba(37, 99, 235, 0.15);
    }

    /* Status Badges */
    .success-badge { color: #059669; font-weight: 600; background: #D1FAE5; padding: 2px 8px; border-radius: 12px; font-size: 0.85rem;}
    .warning-badge { color: #D97706; font-weight: 600; background: #FEF3C7; padding: 2px 8px; border-radius: 12px; font-size: 0.85rem;}
    .error-badge { color: #DC2626; font-weight: 600; background: #FEE2E2; padding: 2px 8px; border-radius: 12px; font-size: 0.85rem;}
    .info-badge { color: #2563EB; font-weight: 600; background: #DBEAFE; padding: 2px 8px; border-radius: 12px; font-size: 0.85rem;}

    /* Dataframes and Tables */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #64748B;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #2563EB;
    }
    
    /* Inputs */
    .stTextInput input, .stSelectbox select {
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .stTextInput input:focus, .stSelectbox select:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("## 🏠 AHIP")
    st.markdown("**Ames Housing Intelligence Platform**")
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("""
    - 📡 **Pipeline Monitor** — Real-time DAG
    - 📊 **Business Analytics** — Data insights
    - 🤖 **AI Chatbot** — Ask anything
    """)
    st.markdown("---")
    st.markdown("### System Status")
    st.markdown("🟢 All services healthy")
    st.markdown("---")
    st.caption("v1.0.0 | 100% Offline")

# Main page
st.markdown("# 🏠 Ames Housing Intelligence Platform")
st.markdown("### Production-Grade ML Pipeline with Real-Time Monitoring")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    ### 📡 Pipeline Monitor
    Real-time DAG visualization with live agent status,
    streaming logs, and animated metrics.
    """)
with col2:
    st.markdown("""
    ### 📊 Business Analytics
    Interactive charts, neighborhood analysis,
    model comparison, and What-If simulator.
    """)
with col3:
    st.markdown("""
    ### 🤖 AI Insights
    Ask questions about the dataset in plain English.
    Powered by flan-t5-base RAG — fully offline.
    """)

st.markdown("---")
st.markdown(
    "**Select a page from the sidebar to begin.** Use `▶ RUN PIPELINE` on the Pipeline Monitor page to start processing."
)
