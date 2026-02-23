"""
analytics.py
============
All analytical computations for the Product Analytics Dashboard.

Product Thinking:
- KPIs tell you WHERE you are; cohorts tell you WHY trends are changing
- Funnel analysis reveals the biggest leverage point for growth
- Segmentation enables personalized retention/upsell strategies
- A/B testing gives statistical confidence before shipping changes
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ── 1. KPI Calculations ───────────────────────────────────────────────────────

def compute_dau(df: pd.DataFrame) -> pd.Series:
    """
    Daily Active Users — unique users with ANY event per day.
    Product: rising DAU = healthy engagement; flat DAU with rising MAU = low stickiness.
    """
    return (
        df.groupby("event_date")["user_id"]
          .nunique()
          .rename("DAU")
    )


def compute_mau(df: pd.DataFrame) -> pd.Series:
    """
    Monthly Active Users — unique users per calendar month.
    DAU/MAU ratio = 'stickiness'. Good SaaS targets >20%.
    """
    df = df.copy()
    df["month"] = df["event_date"].dt.to_period("M")
    return (
        df.groupby("month")["user_id"]
          .nunique()
          .rename("MAU")
    )


def compute_revenue_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily & Monthly revenue from purchase events.
    """
    purchases = df[df["event_type"] == "purchase"].copy()
    daily = (
        purchases.groupby("event_date")["revenue"]
                 .sum()
                 .rename("daily_revenue")
                 .reset_index()
    )
    purchases["month"] = purchases["event_date"].dt.to_period("M")
    monthly = (
        purchases.groupby("month")["revenue"]
                 .sum()
                 .rename("monthly_revenue")
                 .reset_index()
    )
    monthly["month"] = monthly["month"].astype(str)
    return daily, monthly


def compute_conversion_rate(df: pd.DataFrame) -> float:
    """
    Overall funnel conversion: % of signed-up users who made a purchase.
    """
    total_users    = df["user_id"].nunique()
    converted      = df[df["event_type"] == "purchase"]["user_id"].nunique()
    return (converted / total_users) * 100


def compute_retention_rate(df: pd.DataFrame, days: int = 30) -> float:
    """
    D-N retention: % of users who return within N days of signup.
    Product: D30 retention > 25% is strong for SaaS.
    """
    user_first  = df.groupby("user_id")["event_date"].min().rename("signup")
    user_last   = df.groupby("user_id")["event_date"].max().rename("last_active")
    combined    = pd.concat([user_first, user_last], axis=1)
    combined["gap"] = (combined["last_active"] - combined["signup"]).dt.days
    retained    = (combined["gap"] >= days).sum()
    return (retained / len(combined)) * 100


def compute_churn_rate(df: pd.DataFrame, inactive_days: int = 60) -> float:
    """
    Churn: % of users with NO activity in the last 60 days of the dataset.
    Product: Monthly churn below 5% is considered healthy for B2C SaaS.
    """
    max_date    = df["event_date"].max()
    user_last   = df.groupby("user_id")["event_date"].max()
    churned     = ((max_date - user_last).dt.days >= inactive_days).sum()
    return (churned / df["user_id"].nunique()) * 100


def compute_arpu(df: pd.DataFrame) -> float:
    """
    Average Revenue Per User (total cohort).
    Product: ARPU growth = either better monetization or higher-LTV acquisition.
    """
    total_revenue = df[df["event_type"] == "purchase"]["revenue"].sum()
    total_users   = df["user_id"].nunique()
    return total_revenue / total_users


def get_kpi_summary(df: pd.DataFrame) -> dict:
    """Aggregates all top-level KPIs into a single dict for dashboard cards."""
    return {
        "Total Users":       df["user_id"].nunique(),
        "Total Revenue":     df[df["event_type"] == "purchase"]["revenue"].sum(),
        "Conversion Rate":   compute_conversion_rate(df),
        "Retention Rate":    compute_retention_rate(df),
        "Churn Rate":        compute_churn_rate(df),
        "ARPU":              compute_arpu(df),
        "Total Events":      len(df),
    }


# ── 2. Funnel Analysis ────────────────────────────────────────────────────────

FUNNEL_STEPS = ["signup", "login", "view_product", "add_to_cart", "purchase"]

def compute_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes absolute counts and conversion rates at each funnel step.

    Product Thinking:
    - The step with the largest drop-off is the prime optimization target
    - Small improvements at the top of funnel compound significantly
    """
    step_counts = []
    for step in FUNNEL_STEPS:
        count = df[df["event_type"] == step]["user_id"].nunique()
        step_counts.append({"step": step, "users": count})

    funnel_df = pd.DataFrame(step_counts)
    funnel_df["conversion_from_prev"] = (
        funnel_df["users"] / funnel_df["users"].shift(1) * 100
    ).round(1)
    funnel_df["conversion_from_top"] = (
        funnel_df["users"] / funnel_df["users"].iloc[0] * 100
    ).round(1)
    funnel_df["dropoff_pct"] = (100 - funnel_df["conversion_from_prev"]).round(1)
    funnel_df.loc[0, "conversion_from_prev"] = 100.0
    funnel_df.loc[0, "dropoff_pct"]          = 0.0
    return funnel_df


# ── 3. Cohort Analysis ────────────────────────────────────────────────────────

def compute_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a monthly cohort retention matrix.

    Each row = signup cohort month
    Each column = months since signup (0, 1, 2, ...)
    Cell = % of cohort users still active in that month

    Product: A flat retention curve = loyal product. Steep drop after M1 = onboarding problem.
    """
    df = df.copy()
    df["signup_month"]  = df["signup_date"].dt.to_period("M")
    df["event_month"]   = df["event_date"].dt.to_period("M")

    # Cohort size
    cohort_sizes = (
        df[df["event_type"] == "signup"]
          .groupby("signup_month")["user_id"]
          .nunique()
    )

    # Monthly activity per user
    user_activity = (
        df.groupby(["user_id", "signup_month", "event_month"])
          .size()
          .reset_index(name="events")
    )

    # Period index = months since signup
    user_activity["period"] = (
        (user_activity["event_month"] - user_activity["signup_month"])
        .apply(lambda x: x.n)
    )

    # Count active users per cohort × period
    cohort_data = (
        user_activity[user_activity["period"] >= 0]
          .groupby(["signup_month", "period"])["user_id"]
          .nunique()
          .reset_index(name="active_users")
    )

    # Merge cohort sizes
    cohort_data = cohort_data.merge(
        cohort_sizes.rename("cohort_size"),
        on="signup_month"
    )
    cohort_data["retention"] = (
        cohort_data["active_users"] / cohort_data["cohort_size"] * 100
    ).round(1)

    # Pivot to matrix
    matrix = cohort_data.pivot_table(
        index="signup_month",
        columns="period",
        values="retention"
    )
    matrix.index = matrix.index.astype(str)
    matrix.columns = [f"M{c}" for c in matrix.columns]
    return matrix


# ── 4. User Segmentation (RFM + KMeans) ───────────────────────────────────────

def compute_rfm_segments(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    RFM (Recency, Frequency, Monetary) segmentation using KMeans.

    Product Thinking:
    - Champions: recent, frequent, high spend → reward & ask for referrals
    - At-Risk: used to be good, now silent → win-back campaigns
    - New Users: recent, low frequency → onboarding nurture
    - Low-Value: infrequent, low spend → low marketing cost
    """
    max_date = df["event_date"].max()
    purchases = df[df["event_type"] == "purchase"]

    rfm = df.groupby("user_id").agg(
        recency   = ("event_date",    lambda x: (max_date - x.max()).days),
        frequency = ("event_date",    "count"),
        monetary  = ("revenue",        "sum"),
    ).reset_index()

    # Scale features for clustering
    scaler   = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    # KMeans clustering
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["segment"] = km.fit_predict(rfm_scaled)

    # Label segments by average monetary value
    seg_labels = {
        idx: label for idx, label in zip(
            rfm.groupby("segment")["monetary"].mean().sort_values(ascending=False).index,
            ["Champions", "Loyal", "At-Risk", "Low-Value"][:n_clusters]
        )
    }
    rfm["segment_label"] = rfm["segment"].map(seg_labels)
    return rfm


# ── 5. A/B Testing ────────────────────────────────────────────────────────────

def compute_ab_test(df: pd.DataFrame) -> dict:
    """
    Compares purchase conversion rate between experiment Group A and Group B.
    Uses Chi-square test for proportions.

    Product Thinking:
    - p < 0.05 → statistically significant; safe to ship Group B's variant
    - Always check practical significance (absolute lift) alongside statistical significance
    - Ensure sample sizes are large enough to avoid underpowered tests
    """
    ab = df.groupby(["user_id", "experiment_group"]).agg(
        converted = ("event_type", lambda x: int("purchase" in x.values))
    ).reset_index()

    group_a = ab[ab["experiment_group"] == "A"]
    group_b = ab[ab["experiment_group"] == "B"]

    conv_a = group_a["converted"].sum()
    conv_b = group_b["converted"].sum()
    n_a    = len(group_a)
    n_b    = len(group_b)

    rate_a = conv_a / n_a
    rate_b = conv_b / n_b
    lift   = ((rate_b - rate_a) / rate_a) * 100

    # Chi-square test: 2×2 contingency table
    contingency = np.array([
        [conv_a,       n_a - conv_a],
        [conv_b,       n_b - conv_b],
    ])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

    return {
        "group_a_users":      n_a,
        "group_b_users":      n_b,
        "group_a_conversions": conv_a,
        "group_b_conversions": conv_b,
        "rate_a":             rate_a * 100,
        "rate_b":             rate_b * 100,
        "lift_pct":           lift,
        "chi2":               chi2,
        "p_value":            p_value,
        "significant":        p_value < 0.05,
    }
