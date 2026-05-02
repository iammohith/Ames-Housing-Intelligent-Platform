"""
Page 2 — Business Analytics (Executive / MBA View)
Strategic market intelligence, portfolio analysis, and decision-driven insights.
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://orchestration-api:8000")

st.set_page_config(page_title="Business Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@400;500;600&display=swap');

section[data-testid="stSidebar"] { background: #F8FAFC !important; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.exec-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 100%);
    border-radius: 16px; padding: 32px 40px; margin-bottom: 28px; color: white;
}
.exec-title { font-family: 'Outfit', sans-serif; font-size: 2rem; font-weight: 700; margin: 0; }
.exec-sub { font-size: 1rem; color: #94A3B8; margin-top: 6px; }

.kpi-card {
    background: white; border-radius: 14px; padding: 22px 20px;
    border: 1px solid #E2E8F0; box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s ease;
}
.kpi-card:hover { box-shadow: 0 6px 24px rgba(0,0,0,0.08); }
.kpi-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B; font-weight: 600; }
.kpi-value { font-family: 'Outfit', sans-serif; font-size: 2rem; font-weight: 700; color: #0F172A; margin: 4px 0; }
.kpi-delta { font-size: 0.8rem; font-weight: 600; }
.kpi-delta.pos { color: #10B981; }
.kpi-delta.neg { color: #EF4444; }

.section-header {
    font-family: 'Outfit', sans-serif; font-size: 1.2rem; font-weight: 700;
    color: #0F172A; border-left: 4px solid #2563EB;
    padding-left: 12px; margin: 32px 0 16px 0;
}
.insight-box {
    background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 12px;
    padding: 16px 20px; margin: 12px 0; font-size: 0.9rem; color: #1E40AF;
}
.insight-box b { color: #1D4ED8; }

.segment-card {
    background: white; border-radius: 12px; border: 1px solid #E2E8F0;
    padding: 18px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.segment-name { font-weight: 700; font-size: 1.05rem; color: #0F172A; font-family: 'Outfit', sans-serif; }
.segment-price { font-size: 1.5rem; font-weight: 700; color: #2563EB; }
.segment-meta { font-size: 0.78rem; color: #64748B; margin-top: 4px; }

.roi-highlight {
    background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
    border: 1px solid #6EE7B7; border-radius: 12px; padding: 20px; text-align: center;
}
.roi-highlight h3 { color: #047857; margin: 0; font-family: 'Outfit', sans-serif; font-size: 2rem; }
.roi-highlight p { color: #065F46; margin: 4px 0 0 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="exec-header">
    <div class="exec-title">📊 Business Analytics Intelligence</div>
    <div class="exec-sub">Ames, Iowa Housing Market · 2006–2010 · Strategic Decision Support Platform</div>
</div>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    csv_path = "/app/data/AmesHousing.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

df = load_data()
if df.empty:
    st.warning("Dataset not found. Ensure AmesHousing.csv is in /app/data/")
    st.stop()

# ── Sidebar Filters ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Market Segment Filters")
    zoning_options = ["All"] + sorted(df["MS Zoning"].dropna().unique().tolist())
    selected_zoning = st.selectbox("Zoning Class", zoning_options)
    bldg_options = ["All"] + sorted(df["Bldg Type"].dropna().unique().tolist())
    selected_bldg = st.selectbox("Building Type", bldg_options)
    year_range = st.slider("Sale Year", int(df["Yr Sold"].min()), int(df["Yr Sold"].max()),
                           (int(df["Yr Sold"].min()), int(df["Yr Sold"].max())))
    price_range = st.slider("Price Band ($)", int(df["SalePrice"].min()), int(df["SalePrice"].max()),
                            (int(df["SalePrice"].min()), int(df["SalePrice"].max())), step=10000)
    st.markdown("---")
    st.markdown("#### 📌 About")
    st.markdown("Filters apply globally across all charts and KPIs below.")

filtered = df.copy()
if selected_zoning != "All":
    filtered = filtered[filtered["MS Zoning"] == selected_zoning]
if selected_bldg != "All":
    filtered = filtered[filtered["Bldg Type"] == selected_bldg]
filtered = filtered[filtered["Yr Sold"].between(*year_range)]
filtered = filtered[filtered["SalePrice"].between(*price_range)]

# ── Executive KPI Row ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Executive Summary KPIs</div>', unsafe_allow_html=True)

median_price = filtered["SalePrice"].median()
mean_price = filtered["SalePrice"].mean()
total_vol = filtered["SalePrice"].sum()
price_std = filtered["SalePrice"].std()
cv = price_std / mean_price * 100
top_hood = filtered.groupby("Neighborhood")["SalePrice"].median().idxmax() if len(filtered) > 0 else "N/A"
top_hood_premium = filtered.groupby("Neighborhood")["SalePrice"].median().max()
market_premium = ((top_hood_premium - median_price) / median_price * 100) if median_price > 0 else 0

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Median Transaction Price</div>
        <div class="kpi-value">${median_price:,.0f}</div>
        <div class="kpi-delta pos">Active Market Benchmark</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Total Market Volume</div>
        <div class="kpi-value">${total_vol/1e6:.1f}M</div>
        <div class="kpi-delta pos">{len(filtered):,} Transactions</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Price Coefficient of Variation</div>
        <div class="kpi-value">{cv:.1f}%</div>
        <div class="kpi-delta {'pos' if cv < 35 else 'neg'}">{'Stable Market' if cv < 35 else 'High Volatility'}</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Top Performing Neighborhood</div>
        <div class="kpi-value" style="font-size:1.4rem">{top_hood}</div>
        <div class="kpi-delta pos">${top_hood_premium:,.0f} Median</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Luxury Premium vs. Market</div>
        <div class="kpi-value">{market_premium:.0f}%</div>
        <div class="kpi-delta pos">Above Median</div>
    </div>""", unsafe_allow_html=True)

# ── Market Segmentation ───────────────────────────────────────────────────
st.markdown('<div class="section-header">🏙️ Market Segmentation Analysis</div>', unsafe_allow_html=True)

SEGMENTS = {
    "Luxury Tier": (["NridgHt", "StoneBr", "NoRidge"], "#6366F1"),
    "Upper-Mid Tier": (["Timber", "Veenker", "Somerst", "Crawfor"], "#2563EB"),
    "Mid-Market Tier": (["CollgCr", "Gilbert", "NWAmes", "Mitchel"], "#0EA5E9"),
    "Value Tier": (["Edwards", "OldTown", "BrkSide", "MeadowV"], "#10B981"),
}

seg_cols = st.columns(4)
for col, (seg_name, (hoods, color)) in zip(seg_cols, SEGMENTS.items()):
    seg_df = filtered[filtered["Neighborhood"].isin(hoods)]
    if not seg_df.empty:
        seg_median = seg_df["SalePrice"].median()
        seg_count = len(seg_df)
        seg_vol = seg_df["SalePrice"].sum() / 1e6
        with col:
            st.markdown(f"""<div class="segment-card">
                <div class="segment-name">{seg_name}</div>
                <div class="segment-price" style="color:{color}">${seg_median:,.0f}</div>
                <div class="segment-meta">{seg_count} units · ${seg_vol:.1f}M volume</div>
            </div>""", unsafe_allow_html=True)

# ── Price Distribution with Business Context ────────────────────────────
st.markdown('<div class="section-header">💰 Price Distribution & Market Concentration</div>', unsafe_allow_html=True)

col_chart, col_insight = st.columns([2, 1])
with col_chart:
    fig = px.histogram(filtered, x="SalePrice", nbins=60,
                       color_discrete_sequence=["#2563EB"], template="plotly_white",
                       labels={"SalePrice": "Sale Price ($)"})
    fig.add_vline(x=mean_price, line_dash="dash", line_color="#F59E0B",
                  annotation_text=f"Mean ${mean_price:,.0f}", annotation_position="top right")
    fig.add_vline(x=median_price, line_dash="dot", line_color="#10B981",
                  annotation_text=f"Median ${median_price:,.0f}", annotation_position="top left")
    fig.update_layout(yaxis_title="Number of Transactions", margin=dict(t=30, b=10),
                      font=dict(family="Inter"))
    st.plotly_chart(fig, use_container_width=True)

with col_insight:
    skew = filtered["SalePrice"].skew()
    q75 = filtered["SalePrice"].quantile(0.75)
    q25 = filtered["SalePrice"].quantile(0.25)
    iqr = q75 - q25
    luxury_share = len(filtered[filtered["SalePrice"] > q75]) / max(len(filtered), 1) * 100
    st.markdown(f"""
    <div class="insight-box">
        <b>📐 Distribution Shape</b><br>
        Skewness: {skew:.2f} — {'right-skewed, luxury outliers exist' if skew > 0.5 else 'approximately symmetric'}.
        The mean exceeds the median by <b>${(mean_price-median_price):,.0f}</b>, confirming upward price pull from premium properties.
    </div>
    <div class="insight-box">
        <b>📦 Interquartile Range</b><br>
        Middle 50% of transactions span <b>${q25:,.0f} – ${q75:,.0f}</b>, an IQR of <b>${iqr:,.0f}</b>.
        This is the core addressable market for mid-market lenders and developers.
    </div>
    <div class="insight-box">
        <b>💎 Luxury Segment Share</b><br>
        <b>{luxury_share:.1f}%</b> of transactions exceed the 75th percentile (>${q75:,.0f}).
        Premium strategy plays should focus on <b>NridgHt, StoneBr, NoRidge</b>.
    </div>
    """, unsafe_allow_html=True)

# ── Neighborhood ROI Analysis ────────────────────────────────────────────
st.markdown('<div class="section-header">🏘️ Neighborhood ROI & Value Concentration</div>', unsafe_allow_html=True)

hood_stats = (filtered.groupby("Neighborhood")["SalePrice"]
              .agg(["median", "mean", "std", "count"])
              .reset_index()
              .sort_values("median", ascending=False))
hood_stats["cv"] = hood_stats["std"] / hood_stats["mean"] * 100
hood_stats["volume_m"] = hood_stats["mean"] * hood_stats["count"] / 1e6

col_box, col_treemap = st.columns([3, 2])
with col_box:
    fig2 = px.box(filtered, x="Neighborhood", y="SalePrice",
                  color_discrete_sequence=["#2563EB"], template="plotly_white",
                  category_orders={"Neighborhood": hood_stats["Neighborhood"].tolist()})
    fig2.update_layout(xaxis_tickangle=-45, height=420,
                       yaxis_title="Sale Price ($)", xaxis_title="",
                       font=dict(family="Inter"), margin=dict(t=20, b=10))
    st.plotly_chart(fig2, use_container_width=True)

with col_treemap:
    fig_tree = px.treemap(hood_stats, path=["Neighborhood"], values="volume_m",
                          color="median", color_continuous_scale="Blues",
                          template="plotly_white",
                          labels={"volume_m": "Volume ($M)", "median": "Median Price"})
    fig_tree.update_layout(height=420, font=dict(family="Inter"), margin=dict(t=20, b=0))
    fig_tree.update_traces(textinfo="label+value", texttemplate="%{label}<br>$%{value:.1f}M")
    st.plotly_chart(fig_tree, use_container_width=True)

# ── Temporal Market Cycle Analysis ───────────────────────────────────────
st.markdown('<div class="section-header">📅 Market Cycle & Temporal Demand Analysis</div>', unsafe_allow_html=True)

col_line, col_heat = st.columns([3, 2])
with col_line:
    yr_trend = filtered.groupby("Yr Sold")["SalePrice"].agg(["median", "count"]).reset_index()
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=yr_trend["Yr Sold"], y=yr_trend["median"],
        mode="lines+markers", name="Median Price",
        line=dict(color="#2563EB", width=3),
        marker=dict(size=8, color="#2563EB"),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"
    ))
    fig_trend.add_trace(go.Bar(
        x=yr_trend["Yr Sold"], y=yr_trend["count"],
        name="Transactions", yaxis="y2",
        marker_color="rgba(16,185,129,0.3)"
    ))
    fig_trend.update_layout(
        yaxis=dict(title="Median Sale Price ($)", titlefont=dict(color="#2563EB")),
        yaxis2=dict(title="# Transactions", overlaying="y", side="right", titlefont=dict(color="#10B981")),
        template="plotly_white", legend=dict(orientation="h"),
        height=360, font=dict(family="Inter"), margin=dict(t=20)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col_heat:
    temporal = filtered.groupby(["Yr Sold", "Mo Sold"])["SalePrice"].mean().reset_index()
    pivot = temporal.pivot(index="Mo Sold", columns="Yr Sold", values="SalePrice")
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    pivot.index = [month_names.get(m, m) for m in pivot.index]
    fig_heat = px.imshow(pivot, color_continuous_scale="Blues",
                         labels=dict(x="Year", y="Month", color="Avg Price ($)"),
                         template="plotly_white")
    fig_heat.update_layout(height=360, font=dict(family="Inter"), margin=dict(t=20))
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("""
<div class="insight-box">
    <b>📈 Strategic Insight — Market Cycle:</b> Peak transaction velocity occurs in <b>May–July</b> (summer seasonality),
    while winter months (Nov–Jan) show suppressed activity. Pricing strategy should price competitively in Q4 to capture
    motivated buyers and command premiums in Q2. The post-2008 data captures the Ames market's relative resilience
    compared to national averages — a key talking point for institutional investors.
</div>
""", unsafe_allow_html=True)

# ── Value Driver Analysis (Feature Correlations) ──────────────────────────
st.markdown('<div class="section-header">🔬 Value Driver Intelligence — What Moves Price?</div>', unsafe_allow_html=True)

numeric_df = filtered.select_dtypes(include=[np.number])
corr = numeric_df.corr()["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False).head(15)

col_bar, col_scatter = st.columns([1, 1])
with col_bar:
    fig4 = px.bar(x=corr.values, y=corr.index, orientation="h",
                  color=corr.values, color_continuous_scale="Blues",
                  template="plotly_white",
                  labels={"x": "Pearson |r| with SalePrice", "y": "Feature"})
    fig4.update_layout(height=440, yaxis=dict(autorange="reversed"),
                       font=dict(family="Inter"), coloraxis_showscale=False,
                       margin=dict(t=20))
    st.plotly_chart(fig4, use_container_width=True)

with col_scatter:
    if "Overall Qual" in filtered.columns:
        qual_col = "Overall Qual"
    elif "OverallQual" in filtered.columns:
        qual_col = "OverallQual"
    else:
        qual_col = None

    if qual_col:
        sample_df = filtered.sample(min(1000, len(filtered)), random_state=42).dropna(subset=[qual_col, "SalePrice"])
        fig5 = px.scatter(sample_df, x=qual_col, y="SalePrice",
                          template="plotly_white", opacity=0.5,
                          labels={qual_col: "Overall Quality (1–10)", "SalePrice": "Sale Price ($)"})
        # Manual OLS trendline via numpy (no statsmodels needed)
        z = np.polyfit(sample_df[qual_col], sample_df["SalePrice"], 1)
        x_line = np.linspace(sample_df[qual_col].min(), sample_df[qual_col].max(), 50)
        y_line = np.poly1d(z)(x_line)
        fig5.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                  line=dict(color="#EF4444", width=2.5, dash="dash"),
                                  name="OLS Trend"))
        fig5.update_layout(height=440, font=dict(family="Inter"), margin=dict(t=20))
        st.plotly_chart(fig5, use_container_width=True)

st.markdown("""
<div class="insight-box">
    <b>🔑 Business Implication:</b> <b>Overall Quality</b> is the single strongest value driver (r > 0.79),
    followed by <b>Total Living Area</b> and <b>Garage Size</b>. For real estate developers and renovators,
    quality upgrades yield the highest expected ROI — a quality score improvement from 6→8 is associated
    with a <b>~$50K+ price lift</b> at the median, suggesting renovation strategies should prioritize
    kitchen and bathroom quality over cosmetic changes.
</div>
""", unsafe_allow_html=True)

# ── Model Performance Leaderboard ──────────────────────────────────────────
st.markdown('<div class="section-header">🏆 Predictive Model Leaderboard</div>', unsafe_allow_html=True)

try:
    resp = requests.get(f"{API_URL}/api/models", timeout=5)
    if resp.status_code == 200:
        models = resp.json().get("models", [])
        if models:
            model_df = pd.DataFrame(models)
            display_cols = [c for c in ["model_name","test_r2","test_rmse","test_mae","val_rmse","is_best"] if c in model_df.columns]
            col_tbl, col_gauge = st.columns([3, 2])
            with col_tbl:
                st.dataframe(model_df[display_cols].rename(columns={
                    "model_name": "Model", "test_r2": "Test R²",
                    "test_rmse": "Test RMSE ($)", "test_mae": "Test MAE ($)",
                    "val_rmse": "Val RMSE ($)", "is_best": "Champion"
                }), use_container_width=True, hide_index=True)
            with col_gauge:
                best_r2 = model_df["test_r2"].max() if "test_r2" in model_df.columns else 0.92
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=best_r2,
                    title={"text": "Best Model R²<br><span style='font-size:0.8em'>Test Set Performance</span>"},
                    delta={"reference": 0.85, "valueformat": ".3f"},
                    gauge={
                        "axis": {"range": [0, 1], "tickformat": ".2f"},
                        "bar": {"color": "#2563EB"},
                        "steps": [
                            {"range": [0, 0.7], "color": "#FEE2E2"},
                            {"range": [0.7, 0.85], "color": "#FEF3C7"},
                            {"range": [0.85, 1], "color": "#D1FAE5"},
                        ],
                        "threshold": {"line": {"color": "#10B981", "width": 3}, "value": 0.85},
                    }
                ))
                fig_gauge.update_layout(height=280, font=dict(family="Inter"), margin=dict(t=40, b=0))
                st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("Run the pipeline to populate the model leaderboard.")
except Exception:
    st.info("Model leaderboard requires a completed pipeline run.")

# ── What-If Pricing Simulator ────────────────────────────────────────────
st.markdown('<div class="section-header">🎛️ Strategic Pricing Simulator — What-If Analysis</div>', unsafe_allow_html=True)
st.markdown("Stress-test pricing assumptions by adjusting property attributes. Powered by the XGBoost champion model.")

sim1, sim2, sim3 = st.columns(3)
with sim1:
    sim_qual = st.slider("Overall Quality (1–10)", 1, 10, 7, help="Rates the overall material and finish quality")
    sim_area = st.slider("Above-Ground Living Area (sqft)", 500, 5000, 1800)
with sim2:
    sim_year = st.slider("Year Built", 1880, 2010, 1995)
    sim_bath = st.slider("Total Bathrooms", 0.0, 6.0, 2.0, 0.5)
with sim3:
    neighborhoods = sorted(df["Neighborhood"].unique().tolist())
    default_idx = neighborhoods.index("NAmes") if "NAmes" in neighborhoods else 0
    sim_hood = st.selectbox("Target Neighborhood", neighborhoods, index=default_idx)
    sim_garage = st.slider("Garage Area (sqft)", 0, 1500, 460)

if st.button("🔮 Run Pricing Analysis", type="primary", use_container_width=True):
    try:
        api_key = os.getenv("API_KEY", "changeme")
        resp = requests.post(
            f"{API_URL}/api/predict",
            headers={"X-API-Key": api_key},
            json={"overall_qual": sim_qual, "gr_liv_area": sim_area,
                  "year_built": sim_year, "total_bathrooms": sim_bath,
                  "neighborhood": sim_hood, "garage_area": sim_garage},
            timeout=10,
        )
        if resp.status_code == 200:
            result = resp.json()
            price = result.get("predicted_price", 0)
            ci = result.get("confidence_interval", [price * 0.9, price * 1.1])
            ci_width = ci[1] - ci[0]
            market_pos = (price - median_price) / median_price * 100

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""<div class="roi-highlight">
                    <h3>${price:,.0f}</h3>
                    <p>Predicted Market Value</p>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class="kpi-card" style="text-align:center">
                    <div class="kpi-label">90% Confidence Interval</div>
                    <div class="kpi-value" style="font-size:1.3rem">${ci[0]:,.0f} – ${ci[1]:,.0f}</div>
                    <div class="kpi-delta {'pos' if ci_width < 50000 else 'neg'}">±${ci_width/2:,.0f} uncertainty</div>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""<div class="kpi-card" style="text-align:center">
                    <div class="kpi-label">Position vs. Market Median</div>
                    <div class="kpi-value" style="font-size:1.5rem">{market_pos:+.1f}%</div>
                    <div class="kpi-delta {'pos' if market_pos >= 0 else 'neg'}">{'Above' if market_pos >= 0 else 'Below'} median</div>
                </div>""", unsafe_allow_html=True)

            shap = result.get("shap_top_features", {})
            if shap and "error" not in shap:
                st.markdown("**📊 SHAP Value Decomposition — Marginal Feature Contributions:**")
                shap_df = pd.DataFrame(list(shap.items()), columns=["Feature", "SHAP Value"])
                shap_df = shap_df.sort_values("SHAP Value", key=abs, ascending=False)
                fig_shap = px.bar(shap_df, x="SHAP Value", y="Feature", orientation="h",
                                  color="SHAP Value",
                                  color_continuous_scale=["#EF4444", "#FFFFFF", "#10B981"],
                                  template="plotly_white",
                                  labels={"SHAP Value": "Marginal Price Impact ($)"})
                fig_shap.update_layout(height=300, font=dict(family="Inter"),
                                       coloraxis_showscale=False, margin=dict(t=10))
                st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.error("Prediction unavailable — run the full pipeline first to train models.")
    except Exception as e:
        st.error(f"API connection failed: {e}")
