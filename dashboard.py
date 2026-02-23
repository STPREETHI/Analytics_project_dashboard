"""
dashboard.py
============
Main entry point for the Product Analytics Dashboard.

Run with:
    streamlit run dashboard.py

Architecture:
    data_generator.py  →  raw events DataFrame
    analytics.py       →  computed metrics & models
    utils.py           →  chart builders & formatters
    dashboard.py       →  UI layout & interactivity
"""

import streamlit as st
import pandas as pd
import numpy as np

from data_generator import load_data
from analytics import (
    compute_dau, compute_mau, compute_revenue_trend,
    compute_funnel, compute_cohort_retention,
    compute_rfm_segments, compute_ab_test, get_kpi_summary,
)
from utils import (
    chart_dau, chart_revenue, chart_funnel, chart_funnel_bars,
    chart_cohort_heatmap, chart_ab_test, chart_segmentation,
    chart_segment_pie, chart_channel_revenue, chart_device_conversion,
    fmt_number, fmt_currency, fmt_pct,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0D0F1A !important;
    color: #E8EAF6 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0A0C18 !important;
    border-right: 1px solid rgba(108,99,255,0.2) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #6C63FF !important;
}

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #13152A 0%, #1A1D35 100%);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 16px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.kpi-card:hover {
    border-color: rgba(108,99,255,0.6);
    transform: translateY(-2px);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 80px; height: 80px;
    background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.kpi-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6B7280;
    font-family: 'DM Mono', monospace;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #E8EAF6;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-sub {
    font-size: 12px;
    color: #6B7280;
    font-family: 'DM Mono', monospace;
}
.kpi-icon {
    font-size: 20px;
    margin-bottom: 10px;
}

/* ── Section headers ── */
.section-header {
    font-size: 18px;
    font-weight: 600;
    color: #E8EAF6;
    padding: 24px 0 12px 0;
    border-bottom: 1px solid rgba(108,99,255,0.15);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-pill {
    background: rgba(108,99,255,0.15);
    color: #6C63FF;
    font-size: 11px;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
}

/* ── Chart containers ── */
.chart-box {
    background: linear-gradient(135deg, #13152A 0%, #1A1D35 100%);
    border: 1px solid rgba(108,99,255,0.15);
    border-radius: 16px;
    padding: 4px;
    margin-bottom: 16px;
}

/* ── Insight cards ── */
.insight-card {
    background: rgba(108,99,255,0.08);
    border: 1px solid rgba(108,99,255,0.2);
    border-left: 3px solid #6C63FF;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #B0B3C5;
    margin: 8px 0;
    font-family: 'DM Mono', monospace;
}

/* ── AB result banner ── */
.ab-banner-win {
    background: linear-gradient(135deg, rgba(67,233,123,0.12), rgba(67,233,123,0.06));
    border: 1px solid rgba(67,233,123,0.3);
    border-radius: 12px;
    padding: 16px 24px;
    text-align: center;
}
.ab-banner-neutral {
    background: linear-gradient(135deg, rgba(107,114,128,0.12), rgba(107,114,128,0.06));
    border: 1px solid rgba(107,114,128,0.3);
    border-radius: 12px;
    padding: 16px 24px;
    text-align: center;
}

/* ── Stat table ── */
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 13px;
}
.stat-row:last-child { border-bottom: none; }
.stat-label { color: #6B7280; font-family: 'DM Mono', monospace; }
.stat-value { color: #E8EAF6; font-weight: 600; }

/* ── Funnel step table ── */
.funnel-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.funnel-step-name {
    flex: 1;
    font-weight: 500;
    text-transform: capitalize;
}
.funnel-bar-bg {
    flex: 2;
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    overflow: hidden;
}
.funnel-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #6C63FF, #43E97B);
}
.funnel-pct {
    width: 48px;
    text-align: right;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: #6C63FF;
}

/* ── Streamlit tweaks ── */
div[data-testid="metric-container"] {
    background: transparent;
}
.stPlotlyChart { border-radius: 12px; overflow: hidden; }
.stSelectbox label, .stDateInput label, .stMultiSelect label {
    color: #6B7280 !important;
    font-size: 12px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.stMarkdown hr { border-color: rgba(108,99,255,0.15) !important; }

/* Tab styling */
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    color: #6B7280 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #6C63FF !important;
    border-bottom-color: #6C63FF !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D0F1A; }
::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.3); border-radius: 3px; }

/* Logo */
.logo-text {
    font-size: 22px;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #43E97B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Space Grotesk', sans-serif;
}
.logo-sub {
    font-size: 11px;
    color: #6B7280;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    return load_data()

@st.cache_data(show_spinner=False)
def get_rfm(_df):
    return compute_rfm_segments(_df)

@st.cache_data(show_spinner=False)
def get_cohort(_df):
    return compute_cohort_retention(_df)


with st.spinner("⚙️  Generating 20,000-user dataset..."):
    raw_df = get_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo-text">◈ AnalyticOS</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">Product Intelligence Suite</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🎛 Filters")

    min_date = raw_df["event_date"].min().date()
    max_date = raw_df["event_date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    channels_available = sorted(raw_df["acquisition_channel"].unique())
    channels_sel = st.multiselect(
        "Acquisition Channel",
        options=channels_available,
        default=channels_available,
    )

    devices_available = sorted(raw_df["device_type"].unique())
    devices_sel = st.multiselect(
        "Device Type",
        options=devices_available,
        default=devices_available,
    )

    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    <div style='font-size:12px; color:#6B7280; font-family:"DM Mono",monospace; line-height:1.6;'>
    Built with Python & Streamlit.<br>
    Simulates realistic SaaS behavioral data for 20,000 users.<br><br>
    <b style='color:#6C63FF'>Key Concepts:</b><br>
    DAU/MAU · Funnel · Cohort<br>
    RFM Segments · A/B Testing<br>
    KMeans Clustering
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    total_events = len(raw_df)
    st.markdown(f"""
    <div style='font-size:11px; color:#6B7280; font-family:"DM Mono",monospace;'>
    📦 {total_events:,} events loaded<br>
    👥 {raw_df["user_id"].nunique():,} users<br>
    📅 {min_date} → {max_date}
    </div>
    """, unsafe_allow_html=True)


# ── Apply filters ─────────────────────────────────────────────────────────────
start_date, end_date = (date_range[0], date_range[1]) if len(date_range) == 2 else (min_date, max_date)

df = raw_df[
    (raw_df["event_date"].dt.date >= start_date) &
    (raw_df["event_date"].dt.date <= end_date) &
    (raw_df["acquisition_channel"].isin(channels_sel)) &
    (raw_df["device_type"].isin(devices_sel))
].copy()


# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("## 📊 Product Analytics Dashboard")
    st.markdown(
        f'<span style="font-size:13px;color:#6B7280;font-family:\'DM Mono\',monospace;">'
        f'Showing {df["user_id"].nunique():,} users · {len(df):,} events · '
        f'{start_date} to {end_date}</span>',
        unsafe_allow_html=True
    )
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Regenerate Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")


# ── KPI Cards ─────────────────────────────────────────────────────────────────
kpis = get_kpi_summary(df)

kpi_defs = [
    ("Total Users",     fmt_number(kpis["Total Users"]),    "👥", "Unique signups"),
    ("Total Revenue",   fmt_currency(kpis["Total Revenue"]), "💰", "From purchases"),
    ("Conversion Rate", fmt_pct(kpis["Conversion Rate"]),   "🎯", "Signup → Purchase"),
    ("Retention Rate",  fmt_pct(kpis["Retention Rate"]),    "🔄", "D30 retention"),
    ("Churn Rate",      fmt_pct(kpis["Churn Rate"]),        "📉", "60-day inactive"),
    ("ARPU",            fmt_currency(kpis["ARPU"]),         "💎", "Avg revenue/user"),
    ("Total Events",    fmt_number(kpis["Total Events"]),   "⚡", "All event types"),
]

cols = st.columns(7)
for col, (label, value, icon, sub) in zip(cols, kpi_defs):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Engagement",
    "🔀 Funnel",
    "🔁 Cohort",
    "🧪 A/B Test",
    "🎯 Segments",
    "📡 Channels",
])


# ─── Tab 1: Engagement ────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Daily & Monthly Engagement <span class="section-pill">Time Series</span></div>', unsafe_allow_html=True)

    dau = compute_dau(df)
    daily_rev, monthly_rev = compute_revenue_trend(df)
    mau = compute_mau(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(chart_dau(dau), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-card">
        💡 <b>Product Insight:</b> Track DAU 7-day MA to smooth daily noise.
        A rising trend indicates healthy acquisition or re-engagement campaigns working.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(chart_revenue(daily_rev, monthly_rev), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-card">
        💡 <b>Product Insight:</b> Revenue trend reveals monetization efficiency.
        If DAU rises but revenue stagnates, your ARPU or conversion funnel needs attention.
        </div>
        """, unsafe_allow_html=True)

    # MAU stats
    st.markdown('<div class="section-header">Monthly Active Users <span class="section-pill">MAU</span></div>', unsafe_allow_html=True)
    mau_df = mau.reset_index()
    mau_df["month"] = mau_df["month"].astype(str)
    import plotly.graph_objects as go
    fig_mau = go.Figure(go.Bar(
        x=mau_df["month"], y=mau_df["MAU"],
        marker=dict(
            color=mau_df["MAU"],
            colorscale=[[0,"#6C63FF"],[1,"#43E97B"]],
        ),
        text=mau_df["MAU"].apply(fmt_number),
        textposition="outside",
    ))
    fig_mau.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8EAF6"), margin=dict(l=20,r=20,t=10,b=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        height=280,
    )
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig_mau, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Tab 2: Funnel ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Conversion Funnel <span class="section-pill">Drop-off Analysis</span></div>', unsafe_allow_html=True)

    funnel_df = compute_funnel(df)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(chart_funnel(funnel_df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(chart_funnel_bars(funnel_df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Step-by-step breakdown
    st.markdown('<div class="section-header">Step Breakdown <span class="section-pill">Detail</span></div>', unsafe_allow_html=True)

    cols = st.columns(5)
    step_colors = ["#6C63FF", "#847CF8", "#A89CF9", "#FF6584", "#43E97B"]
    for i, (_, row) in enumerate(funnel_df.iterrows()):
        with cols[i]:
            prev_conv = f"↓ {row['dropoff_pct']:.1f}% drop" if i > 0 else "Entry point"
            color = step_colors[i]
            st.markdown(f"""
            <div class="kpi-card" style="border-color:{color}33;">
                <div class="kpi-label">{row['step'].replace('_',' ').upper()}</div>
                <div class="kpi-value" style="font-size:22px;">{fmt_number(row['users'])}</div>
                <div class="kpi-sub" style="color:{color};">{row['conversion_from_top']:.1f}% of total</div>
                <div class="kpi-sub" style="margin-top:4px;">{prev_conv}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Identify biggest bottleneck
    if len(funnel_df) > 1:
        bottleneck = funnel_df.iloc[1:]["dropoff_pct"].idxmax()
        bn_step = funnel_df.loc[bottleneck, "step"]
        bn_drop = funnel_df.loc[bottleneck, "dropoff_pct"]
        st.markdown(f"""
        <div class="insight-card" style="border-left-color:#FF6584;">
        🚨 <b>Biggest Bottleneck:</b> <span style="color:#FF6584;">{bn_step.replace('_',' ').title()}</span>
        has the highest drop-off at <b>{bn_drop:.1f}%</b>.
        This is your #1 optimization target — improving it by just 10% could yield significant revenue lift.
        </div>
        """, unsafe_allow_html=True)


# ─── Tab 3: Cohort ────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Cohort Retention Matrix <span class="section-pill">Monthly</span></div>', unsafe_allow_html=True)

    with st.spinner("Computing cohort retention..."):
        cohort_matrix = get_cohort(df)

    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(chart_cohort_heatmap(cohort_matrix), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-card">
        💡 <b>How to Read:</b> Each row is a signup cohort (month).
        Each column is months since signup. M0 = 100% (signup month).
        Darker = higher retention. Look for cohorts that retain better than average.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if not cohort_matrix.empty and "M1" in cohort_matrix.columns:
            avg_m1 = cohort_matrix["M1"].mean()
            st.markdown(f"""
            <div class="insight-card" style="border-left-color:#43E97B;">
            📊 <b>Avg M1 Retention:</b> <span style="color:#43E97B;">{avg_m1:.1f}%</span><br>
            Industry benchmark: 25–40% for SaaS.<br>
            Below 20% = onboarding problem. Above 40% = strong product-market fit.
            </div>
            """, unsafe_allow_html=True)

    # Raw matrix table
    with st.expander("📋 View Raw Cohort Data"):
        st.dataframe(
            cohort_matrix.style.background_gradient(cmap="Blues", axis=None).format("{:.1f}%"),
            use_container_width=True,
        )


# ─── Tab 4: A/B Testing ───────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">A/B Experiment Results <span class="section-pill">Statistical Testing</span></div>', unsafe_allow_html=True)

    ab = compute_ab_test(df)

    # Result banner
    if ab["significant"]:
        st.markdown(f"""
        <div class="ab-banner-win">
            <div style="font-size:28px; margin-bottom:8px;">🎉</div>
            <div style="font-size:20px; font-weight:700; color:#43E97B;">Statistically Significant!</div>
            <div style="font-size:14px; color:#9CA3AF; margin-top:4px;">
            Group B outperforms Group A by <b style="color:#43E97B;">{ab['lift_pct']:+.2f}%</b> lift.
            Safe to ship the variant.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ab-banner-neutral">
            <div style="font-size:28px; margin-bottom:8px;">⏳</div>
            <div style="font-size:20px; font-weight:700; color:#9CA3AF;">Not Significant Yet</div>
            <div style="font-size:14px; color:#6B7280; margin-top:4px;">
            Continue collecting data. Current lift: <b>{ab['lift_pct']:+.2f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(chart_ab_test(ab), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📋 Test Statistics")
        stats_rows = [
            ("Group A Users",       f"{ab['group_a_users']:,}"),
            ("Group B Users",       f"{ab['group_b_users']:,}"),
            ("Group A Conversions", f"{ab['group_a_conversions']:,}"),
            ("Group B Conversions", f"{ab['group_b_conversions']:,}"),
            ("Group A Conv. Rate",  fmt_pct(ab['rate_a'])),
            ("Group B Conv. Rate",  fmt_pct(ab['rate_b'])),
            ("Absolute Lift",       f"{ab['rate_b'] - ab['rate_a']:+.2f}pp"),
            ("Relative Lift",       f"{ab['lift_pct']:+.2f}%"),
            ("Chi² Statistic",      f"{ab['chi2']:.4f}"),
            ("p-value",             f"{ab['p_value']:.6f}"),
            ("Significance (α=0.05)", "✅ YES" if ab['significant'] else "❌ NO"),
        ]
        for label, value in stats_rows:
            color = "#43E97B" if label == "Significance (α=0.05)" and ab["significant"] else "#E8EAF6"
            st.markdown(f"""
            <div class="stat-row">
                <span class="stat-label">{label}</span>
                <span class="stat-value" style="color:{color};">{value}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
    🧪 <b>Test Design:</b> 50/50 random assignment at signup. Chi-square test on
    purchase conversion proportions. Group B simulates a checkout UX variant with
    10% uplift baked into data generation. p &lt; 0.05 confirms we can reject the
    null hypothesis (no difference between groups).
    </div>
    """, unsafe_allow_html=True)


# ─── Tab 5: Segments ─────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">User Segmentation <span class="section-pill">RFM + KMeans</span></div>', unsafe_allow_html=True)

    with st.spinner("Running KMeans clustering..."):
        rfm = get_rfm(df)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(chart_segmentation(rfm), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(chart_segment_pie(rfm), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Segment summary table
    st.markdown('<div class="section-header">Segment Profiles <span class="section-pill">Summary</span></div>', unsafe_allow_html=True)

    seg_summary = rfm.groupby("segment_label").agg(
        Users     = ("user_id",    "count"),
        Avg_Recency  = ("recency",  "mean"),
        Avg_Frequency = ("frequency","mean"),
        Avg_Revenue  = ("monetary", "mean"),
        Total_Revenue = ("monetary","sum"),
    ).round(1).reset_index()
    seg_summary["Total_Revenue"] = seg_summary["Total_Revenue"].apply(fmt_currency)
    seg_summary["Avg_Revenue"]   = seg_summary["Avg_Revenue"].apply(fmt_currency)
    seg_summary.columns = ["Segment", "Users", "Avg Recency (days)", "Avg Events", "Avg Revenue", "Total Revenue"]

    st.dataframe(seg_summary, use_container_width=True, hide_index=True)

    # Playbook
    playbook = {
        "Champions":  ("🏆", "#43E97B", "Reward with loyalty perks. Ask for reviews/referrals. Early access to new features."),
        "Loyal":      ("💜", "#6C63FF", "Upsell to premium tier. Cross-sell complementary products."),
        "At-Risk":    ("⚠️",  "#F9CA24", "Win-back email campaign. Offer discount. Personalized outreach."),
        "Low-Value":  ("📦", "#6B7280", "Low-cost nurture sequence. Identify if segment can be moved up."),
    }
    cols = st.columns(4)
    for col, (seg, (icon, color, action)) in zip(cols, playbook.items()):
        count = rfm[rfm["segment_label"] == seg].shape[0]
        pct   = count / len(rfm) * 100
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-color:{color}33; min-height:160px;">
                <div style="font-size:22px;">{icon}</div>
                <div class="kpi-label" style="color:{color};">{seg}</div>
                <div class="kpi-value" style="font-size:20px;">{fmt_number(count)}</div>
                <div class="kpi-sub">{pct:.1f}% of users</div>
                <div style="font-size:11px;color:#6B7280;margin-top:8px;line-height:1.5;">{action}</div>
            </div>
            """, unsafe_allow_html=True)


# ─── Tab 6: Channels ─────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">Acquisition Channel Analysis <span class="section-pill">Attribution</span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(chart_channel_revenue(df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(chart_device_conversion(df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Channel breakdown table
    st.markdown('<div class="section-header">Channel Performance Matrix <span class="section-pill">Detail</span></div>', unsafe_allow_html=True)

    ch_stats = df.groupby("acquisition_channel").agg(
        Total_Users  = ("user_id", "nunique"),
    ).reset_index()

    purchases_by_ch = df[df["event_type"] == "purchase"].groupby("acquisition_channel").agg(
        Conversions   = ("user_id", "nunique"),
        Total_Revenue = ("revenue", "sum"),
    ).reset_index()

    ch_stats = ch_stats.merge(purchases_by_ch, on="acquisition_channel", how="left").fillna(0)
    ch_stats["Conv. Rate"] = (ch_stats["Conversions"] / ch_stats["Total_Users"] * 100).round(1).astype(str) + "%"
    ch_stats["ARPU"] = (ch_stats["Total_Revenue"] / ch_stats["Total_Users"]).apply(fmt_currency)
    ch_stats["Total_Revenue"] = ch_stats["Total_Revenue"].apply(fmt_currency)
    ch_stats.columns = ["Channel", "Users", "Conversions", "Total Revenue", "Conv. Rate", "ARPU"]

    st.dataframe(ch_stats, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-card">
    💡 <b>Attribution Insight:</b> Organic and referral channels typically show higher LTV and ARPU
    because users arrive with intent. Paid channels drive volume but require careful CAC monitoring.
    A healthy business has &gt;30% revenue from organic/referral sources.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:11px; color:#6B7280; font-family:'DM Mono',monospace; padding:12px 0;">
◈ AnalyticOS · Built with Python, Streamlit & Plotly · 20,000 simulated users · Interview-ready Product Analytics System
</div>
""", unsafe_allow_html=True)
