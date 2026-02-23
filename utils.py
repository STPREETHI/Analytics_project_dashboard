"""
utils.py
========
Reusable chart builders and formatting helpers.
Centralizing chart logic keeps dashboard.py clean and charts consistent.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Brand palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "primary":    "#6C63FF",
    "secondary":  "#FF6584",
    "accent":     "#43E97B",
    "warning":    "#F9CA24",
    "dark":       "#0D0F1A",
    "card":       "#13152A",
    "text":       "#E8EAF6",
    "muted":      "#6B7280",
    "grid":       "rgba(255,255,255,0.05)",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'DM Mono', monospace", color=PALETTE["text"], size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor=PALETTE["grid"], linecolor=PALETTE["grid"]),
    yaxis=dict(gridcolor=PALETTE["grid"], linecolor=PALETTE["grid"]),
)


def _apply_layout(fig, title="") -> go.Figure:
    fig.update_layout(**CHART_LAYOUT, title=dict(
        text=title,
        font=dict(size=14, color=PALETTE["text"]),
        x=0.01
    ))
    return fig


# ── DAU Chart ─────────────────────────────────────────────────────────────────

def chart_dau(dau: pd.Series) -> go.Figure:
    dau_7d = dau.rolling(7).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dau.index, y=dau.values,
        mode="lines", name="DAU",
        line=dict(color=PALETTE["primary"], width=1),
        opacity=0.4,
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.06)"
    ))
    fig.add_trace(go.Scatter(
        x=dau_7d.index, y=dau_7d.values,
        mode="lines", name="7-Day MA",
        line=dict(color=PALETTE["accent"], width=2.5),
    ))
    _apply_layout(fig, "Daily Active Users")
    return fig


# ── Revenue Chart ─────────────────────────────────────────────────────────────

def chart_revenue(daily_rev: pd.DataFrame, monthly_rev: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=daily_rev["event_date"], y=daily_rev["daily_revenue"],
        name="Daily Revenue", marker_color=PALETTE["primary"],
        opacity=0.5,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=monthly_rev["month"], y=monthly_rev["monthly_revenue"],
        mode="lines+markers", name="Monthly Revenue",
        line=dict(color=PALETTE["secondary"], width=3),
        marker=dict(size=8),
    ), secondary_y=True)
    _apply_layout(fig, "Revenue Trend")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ── Funnel Chart ──────────────────────────────────────────────────────────────

def chart_funnel(funnel_df: pd.DataFrame) -> go.Figure:
    colors = [PALETTE["primary"], "#847CF8", "#A89CF9", PALETTE["secondary"], "#FF8FA3"]
    fig = go.Figure(go.Funnel(
        y=funnel_df["step"].str.replace("_", " ").str.title(),
        x=funnel_df["users"],
        textinfo="value+percent initial",
        marker=dict(color=colors),
        connector=dict(line=dict(color="rgba(255,255,255,0.1)", width=1)),
    ))
    _apply_layout(fig, "Conversion Funnel")
    return fig


# ── Funnel Step Bars ──────────────────────────────────────────────────────────

def chart_funnel_bars(funnel_df: pd.DataFrame) -> go.Figure:
    steps = funnel_df["step"].str.replace("_", " ").str.title()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=steps,
        y=funnel_df["conversion_from_top"],
        marker=dict(
            color=funnel_df["conversion_from_top"],
            colorscale=[[0, PALETTE["secondary"]], [1, PALETTE["accent"]]],
            showscale=False,
        ),
        text=funnel_df["conversion_from_top"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
    ))
    _apply_layout(fig, "Conversion % from Top of Funnel")
    fig.update_yaxes(range=[0, 115])
    return fig


# ── Cohort Heatmap ────────────────────────────────────────────────────────────

def chart_cohort_heatmap(matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale=[
            [0,    "rgba(108,99,255,0.05)"],
            [0.4,  "#6C63FF"],
            [0.7,  "#43E97B"],
            [1,    "#F9CA24"],
        ],
        text=matrix.values,
        texttemplate="%{text:.1f}%",
        textfont=dict(size=10),
        hovertemplate="Cohort: %{y}<br>Period: %{x}<br>Retention: %{z:.1f}%<extra></extra>",
    ))
    _apply_layout(fig, "Monthly Cohort Retention (%)")
    fig.update_layout(
        height=460,
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


# ── A/B Test Bar Chart ────────────────────────────────────────────────────────

def chart_ab_test(ab: dict) -> go.Figure:
    fig = go.Figure()
    groups = ["Group A (Control)", "Group B (Variant)"]
    rates  = [ab["rate_a"], ab["rate_b"]]
    colors = [PALETTE["muted"], PALETTE["accent"] if ab["significant"] else PALETTE["primary"]]

    fig.add_trace(go.Bar(
        x=groups, y=rates,
        marker_color=colors,
        text=[f"{r:.2f}%" for r in rates],
        textposition="outside",
        width=0.4,
    ))
    _apply_layout(fig, "A/B Test — Purchase Conversion Rate")
    fig.update_yaxes(range=[0, max(rates) * 1.3])
    return fig


# ── Segmentation Scatter ──────────────────────────────────────────────────────

def chart_segmentation(rfm: pd.DataFrame) -> go.Figure:
    seg_colors = {
        "Champions": PALETTE["accent"],
        "Loyal":     PALETTE["primary"],
        "At-Risk":   PALETTE["warning"],
        "Low-Value": PALETTE["secondary"],
    }
    fig = go.Figure()
    for seg, color in seg_colors.items():
        mask = rfm["segment_label"] == seg
        fig.add_trace(go.Scatter(
            x=rfm.loc[mask, "recency"],
            y=rfm.loc[mask, "frequency"],
            mode="markers",
            name=seg,
            marker=dict(
                color=color,
                size=rfm.loc[mask, "monetary"].clip(upper=1000) / 50 + 4,
                opacity=0.7,
                line=dict(width=0),
            ),
        ))
    _apply_layout(fig, "User Segments — Recency vs Frequency (size = Revenue)")
    fig.update_xaxes(title_text="Recency (days since last activity)")
    fig.update_yaxes(title_text="Frequency (total events)")
    return fig


# ── Segment Distribution Pie ──────────────────────────────────────────────────

def chart_segment_pie(rfm: pd.DataFrame) -> go.Figure:
    seg_counts = rfm["segment_label"].value_counts()
    fig = go.Figure(go.Pie(
        labels=seg_counts.index,
        values=seg_counts.values,
        hole=0.55,
        marker=dict(colors=[PALETTE["accent"], PALETTE["primary"], PALETTE["warning"], PALETTE["secondary"]]),
        textfont=dict(size=12),
    ))
    _apply_layout(fig, "Segment Distribution")
    return fig


# ── Channel Revenue Bar ────────────────────────────────────────────────────────

def chart_channel_revenue(df: pd.DataFrame) -> go.Figure:
    ch_rev = (
        df[df["event_type"] == "purchase"]
          .groupby("acquisition_channel")["revenue"]
          .sum()
          .sort_values(ascending=True)
    )
    fig = go.Figure(go.Bar(
        x=ch_rev.values, y=ch_rev.index,
        orientation="h",
        marker=dict(
            color=ch_rev.values,
            colorscale=[[0, PALETTE["primary"]], [1, PALETTE["accent"]]],
        ),
        text=[f"${v:,.0f}" for v in ch_rev.values],
        textposition="outside",
    ))
    _apply_layout(fig, "Revenue by Acquisition Channel")
    return fig


# ── Device Conversion Pie ─────────────────────────────────────────────────────

def chart_device_conversion(df: pd.DataFrame) -> go.Figure:
    device_conv = (
        df[df["event_type"] == "purchase"]
          .groupby("device_type")["user_id"]
          .nunique()
    )
    fig = go.Figure(go.Pie(
        labels=device_conv.index,
        values=device_conv.values,
        hole=0.5,
        marker=dict(colors=[PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]),
    ))
    _apply_layout(fig, "Conversions by Device")
    return fig


# ── Format helpers ────────────────────────────────────────────────────────────

def fmt_number(n: float) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:.0f}"

def fmt_currency(n: float) -> str:
    if n >= 1_000_000:
        return f"${n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"${n/1_000:.1f}K"
    return f"${n:.2f}"

def fmt_pct(n: float) -> str:
    return f"{n:.1f}%"
